from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional
from xml.sax.saxutils import escape

from branca.element import Element, MacroElement, Template
from json import loads as json_loads

import contextily as cx
import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
# Configures Matplotlib with the bundled, free IPAex Japanese font so that
# title, legend and marker labels survive Streamlit's Linux rendering.
import japanize_matplotlib  # noqa: F401
import numpy as np
from shapely.geometry import Point


# Streamlit transports Folium's HTML over a WebSocket.  A large water system
# can exceed its message limit unless the display-only geometry is simplified.
# This is roughly 25 m at Japanese latitudes: visually faithful at the map's
# default scale while substantially smaller than the downloadable GeoJSON.
# Keep each GeoJSON payload comfortably below Streamlit's ~5 MB WebSocket
# message ceiling.  A combined map can contain two layers, hence the smaller
# per-layer budget.
WEB_GEOJSON_MAX_BYTES = 2_000_000
DOWNLOAD_GEOJSON_MAX_BYTES = 4_000_000
SIMPLIFY_TOLERANCES = (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
# A compact, labelled representation of GSI's relief palette.  The source
# tiles themselves are GSI's free relief tiles; this scale is drawn locally so
# it remains sharp in PNG exports and never depends on a remote legend image.
ELEVATION_MAX_METERS = 4000
ELEVATION_TICKS = (0, 1000, 2000, 3000, 4000)
ELEVATION_COLORS = (
    "#51c9dc",  # lowland / coast
    "#61b95a",
    "#c5c83a",
    "#e2ad43",
    "#bd7650",
    "#f7f7f2",  # high mountains
)
ELEVATION_CMAP = LinearSegmentedColormap.from_list("gsi_relief", ELEVATION_COLORS)


@dataclass
class Marker:
    lon: float
    lat: float
    label: Optional[str] = None


@dataclass
class StaticMapResult:
    buffer: BytesIO
    warning: Optional[str] = None


class ElevationColorbar(MacroElement):
    """A compact, readable elevation bar for the interactive GSI relief map."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: 'bottomright'});
        {{ this.get_name() }}.onAdd = function() {
            var div = L.DomUtil.create('div', 'river-elevation-legend leaflet-control');
            div.setAttribute('aria-label', '標高 0 メートルから 4000 メートルのカラーバー');
            div.innerHTML = '<div class="river-elevation-legend__title">標高（m）</div>'
                + '<div class="river-elevation-legend__bar"></div>'
                + '<div class="river-elevation-legend__ticks"><span>0</span><span>1,000</span><span>2,000</span><span>3,000</span><span>4,000</span></div>';
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            return div;
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """
    )

    def __init__(self) -> None:
        super().__init__()


def filter_geometries(
    gdf: gpd.GeoDataFrame,
    r_type: str,
    r_code: str,
) -> tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    if r_type == "a":
        main = gdf[gdf["W05_001"] == r_code[:6]]
        return main, None
    if r_type == "b":
        main = gdf[gdf["W05_002"] == r_code]
        return main, None
    water_system = gdf[gdf["W05_001"] == r_code[:6]]
    river = gdf[gdf["W05_002"] == r_code]
    return water_system, river


def expand_bounds(
    main: gpd.GeoDataFrame,
    sub: Optional[gpd.GeoDataFrame],
    margin_ratio: float = 0.12,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, ymin, xmax, ymax = main.total_bounds
    if sub is not None and not sub.empty:
        sxmin, symin, sxmax, symax = sub.total_bounds
        xmin = min(xmin, sxmin)
        ymin = min(ymin, symin)
        xmax = max(xmax, sxmax)
        ymax = max(ymax, symax)

    width = max(xmax - xmin, 1e-4)
    height = max(ymax - ymin, 1e-4)
    pad_x = width * margin_ratio
    pad_y = height * margin_ratio
    south = ymin - pad_y
    west = xmin - pad_x
    north = ymax + pad_y
    east = xmax + pad_x
    return (south, west), (north, east)


def _compact_geojson(
    gdf: gpd.GeoDataFrame,
    columns: list[str],
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Serialize GeoJSON within a transport-size budget.

    The source data is intentionally not altered.  This applies progressively
    stronger, display-scale simplification only when needed for a browser
    transfer, and returns whether any simplification was applied.
    """
    display = gdf[columns + ["geometry"]].copy()
    raw = display.to_json(drop_id=True).encode("utf-8")
    if len(raw) <= max_bytes:
        return raw, False

    candidate = display
    for tolerance in SIMPLIFY_TOLERANCES:
        candidate = display.copy()
        candidate["geometry"] = candidate.geometry.simplify(
            tolerance, preserve_topology=False
        )
        encoded = candidate.to_json(drop_id=True).encode("utf-8")
        if len(encoded) <= max_bytes:
            return encoded, True

    # The final candidate is still valid GeoJSON and is substantially smaller
    # than the source even for exceptionally detailed water systems.
    return encoded, True


def web_geojson(gdf: gpd.GeoDataFrame, code_column: str) -> dict:
    """Create compact GeoJSON for the interactive Folium map."""
    encoded, _ = _compact_geojson(gdf, [code_column], WEB_GEOJSON_MAX_BYTES)
    return json_loads(encoded)


def download_geojson(gdf: gpd.GeoDataFrame) -> tuple[bytes, bool]:
    """Create a downloadable GeoJSON that cannot overflow Streamlit's socket."""
    return _compact_geojson(gdf, ["W05_001", "W05_002"], DOWNLOAD_GEOJSON_MAX_BYTES)


def geojson_to_kml(geojson_bytes: bytes, document_name: str) -> bytes:
    """Convert line GeoJSON to a portable KML document without dependencies."""
    collection = json_loads(geojson_bytes)
    placemarks: list[str] = []
    for feature in collection.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        name = properties.get("W05_002") or properties.get("W05_001") or "河川ライン"
        if geometry.get("type") == "LineString":
            lines = [geometry.get("coordinates", [])]
        elif geometry.get("type") == "MultiLineString":
            lines = geometry.get("coordinates", [])
        else:
            continue
        line_xml = []
        for line in lines:
            coordinates = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat, *_ in line)
            if coordinates:
                line_xml.append(f"<LineString><tessellate>1</tessellate><coordinates>{coordinates}</coordinates></LineString>")
        if line_xml:
            shape = line_xml[0] if len(line_xml) == 1 else f"<MultiGeometry>{''.join(line_xml)}</MultiGeometry>"
            placemarks.append(f"<Placemark><name>{escape(str(name))}</name>{shape}</Placemark>")

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(document_name)}</name>"
        + "".join(placemarks)
        + "</Document></kml>"
    )
    return kml.encode("utf-8")


def build_map(
    main: gpd.GeoDataFrame,
    sub: Optional[gpd.GeoDataFrame],
    markers: list[Marker],
) -> folium.Map:
    (south, west), (north, east) = expand_bounds(main, sub)
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None, control_scale=True)
    # Keep the interactive and PNG maps on keyless GSI tiles.  CARTO now
    # returns an API-key-required image, and all layers used to be enabled at
    # once, allowing that error image to cover the map.
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        name="地理院タイル（標準地図）",
        attr="地理院タイル",
        max_zoom=18,
        show=False,
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/relief/{z}/{x}/{y}.png",
        name="GSI Relief",
        attr="地理院タイル（標高陰影図）",
        max_zoom=15,
        show=True,
    ).add_to(fmap)

    legend_style = MacroElement()
    legend_style._template = Template(
        """{% macro html(this, kwargs) %}
        <style>
          .river-elevation-legend { width: 228px; background: rgba(255,255,255,.96); padding: 7px 8px 6px;
            border: 1px solid #64748b; border-radius: 4px; box-shadow: 0 1px 5px rgba(0,0,0,.28); }
          .river-elevation-legend__title { font: 600 12px/1.2 sans-serif; margin: 0 0 4px; color: #0f172a; }
          .river-elevation-legend__bar { height: 13px; border: 1px solid #0f172a;
            background: linear-gradient(90deg, #51c9dc 0%, #61b95a 19%, #c5c83a 39%, #e2ad43 59%, #bd7650 79%, #f7f7f2 100%); }
          .river-elevation-legend__ticks { display: flex; justify-content: space-between; margin-top: 3px;
            font: 10px/1 sans-serif; color: #0f172a; }
        </style>
        {% endmacro %}"""
    )
    fmap.get_root().add_child(legend_style)
    ElevationColorbar().add_to(fmap)

    scale_style = Element(
        """
        <style>
        .leaflet-control-scale {
            font-size: 16px !important;
        }
        .leaflet-control-scale-line {
            padding: 10px 18px !important;
            font-size: 16px !important;
            border-width: 3px !important;
        }
        .leaflet-control-scale-line:first-child {
            border-top-left-radius: 8px !important;
            border-top-right-radius: 8px !important;
        }
        .leaflet-control-scale-line:last-child {
            border-bottom-left-radius: 8px !important;
            border-bottom-right-radius: 8px !important;
        }
        </style>
        """
    )
    fmap.get_root().html.add_child(scale_style)

    main_geojson = web_geojson(main, "W05_001")
    water_layer = folium.FeatureGroup(name="水系", show=True)
    folium.GeoJson(
        data=main_geojson,
        style_function=lambda _: {"color": "#1D4ED8", "weight": 4, "opacity": 0.9},
        highlight_function=lambda _: {"weight": 6, "color": "#2563EB"},
        tooltip=folium.features.GeoJsonTooltip(fields=["W05_001"], aliases=["水系コード"]),
    ).add_to(water_layer)
    water_layer.add_to(fmap)

    sub_geojson = None
    if sub is not None and not sub.empty:
        sub_geojson = web_geojson(sub, "W05_002")
        river_layer = folium.FeatureGroup(name="河川", show=True)
        folium.GeoJson(
            data=sub_geojson,
            style_function=lambda _: {"color": "#DC2626", "weight": 5, "opacity": 0.95},
            highlight_function=lambda _: {"weight": 7, "color": "#EF4444"},
            tooltip=folium.features.GeoJsonTooltip(fields=["W05_002"], aliases=["河川コード"]),
        ).add_to(river_layer)
        river_layer.add_to(fmap)

    for marker in markers:
        folium.Marker(
            location=[marker.lat, marker.lon],
            tooltip=marker.label or None,
            icon=folium.DivIcon(
                html='<div style="font-size:32px; color:#DC2626; font-weight:bold;">&times;</div>'
            ),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)

    # Reserve a safe viewing area around the data.  The lower-right space is
    # deliberately larger because the elevation bar lives there; fit_bounds
    # keeps every selected river line out from underneath that control.
    fmap.fit_bounds(
        [[south, west], [north, east]],
        padding_top_left=(56, 56),
        padding_bottom_right=(278, 112),
    )
    return fmap


def _round_scale_length(target_meters: float) -> float:
    if target_meters <= 0:
        return 0.0
    exponent = int(np.floor(np.log10(target_meters)))
    base = target_meters / (10 ** exponent)
    for candidate in (1, 2, 5, 10):
        if base <= candidate:
            return candidate * (10 ** exponent)
    return 10 ** (exponent + 1)


def _add_scale_bar(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    target_length = width / 5
    scale_length = _round_scale_length(target_length)
    if scale_length <= 0:
        return

    bar_height = height * 0.012
    x_start = xlim[0] + width * 0.05
    y_start = ylim[0] + height * 0.05

    first_segment = mpatches.Rectangle(
        (x_start, y_start),
        scale_length / 2,
        bar_height,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=7,
    )
    second_segment = mpatches.Rectangle(
        (x_start + scale_length / 2, y_start),
        scale_length / 2,
        bar_height,
        facecolor="black",
        edgecolor="black",
        linewidth=1.5,
        zorder=7,
    )
    ax.add_patch(first_segment)
    ax.add_patch(second_segment)

    if scale_length >= 1000:
        label_value = scale_length / 1000
        if label_value.is_integer():
            label = f"{int(label_value)} km"
        else:
            label = f"{label_value:.1f} km"
    else:
        label = f"{int(scale_length)} m"

    ax.text(
        x_start + scale_length / 2,
        y_start + bar_height * 1.9,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        zorder=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    )


def create_static_map_image(
    main: gpd.GeoDataFrame,
    sub: Optional[gpd.GeoDataFrame],
    markers: list[Marker],
    title: Optional[str] = None,
    legend: Optional[str] = None,
) -> StaticMapResult:
    main_3857 = main.to_crs(epsg=3857)
    sub_3857 = sub.to_crs(epsg=3857) if sub is not None and not sub.empty else None

    minx, miny, maxx, maxy = main_3857.total_bounds
    if sub_3857 is not None and not sub_3857.empty:
        sxmin, symin, sxmax, symax = sub_3857.total_bounds
        minx = min(minx, sxmin)
        miny = min(miny, symin)
        maxx = max(maxx, sxmax)
        maxy = max(maxy, symax)

    marker_series = None
    if markers:
        marker_series = gpd.GeoSeries(
            [Point(marker.lon, marker.lat) for marker in markers],
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        minx = min(minx, marker_series.x.min())
        maxx = max(maxx, marker_series.x.max())
        miny = min(miny, marker_series.y.min())
        maxy = max(maxy, marker_series.y.max())

    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad_x = width * 0.08
    pad_y = height * 0.08
    xlim = (minx - pad_x, maxx + pad_x)
    ylim = (miny - pad_y, maxy + pad_y)

    # Map annotations get their own margins.  Controls must never cover river
    # geometry, irrespective of where the selected river lies in its bounds.
    map_aspect = width / height
    # A tall watershed needs a portrait export; forcing it into a landscape
    # canvas leaves an unhelpful white field and makes the river details tiny.
    if map_aspect < 0.80:
        figure_size = (5.0, 9.8)
    elif map_aspect > 1.45:
        figure_size = (10.8, 6.6)
    else:
        figure_size = (9.6, 7.0)

    fig = plt.figure(figsize=figure_size, dpi=200)
    grid = fig.add_gridspec(3, 1, height_ratios=(0.10, 0.80, 0.10), hspace=0)
    title_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])
    footer_ax = fig.add_subplot(grid[2])
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    basemap_warning = None
    try:
        # OpenStreetMap blocks bulk server-side tile retrieval (HTTP 403) and
        # CARTO can return an "API key required" image instead of an error.
        # GSI's relief map is a public Japanese basemap and needs no key.
        cx.add_basemap(
            ax,
            source="https://cyberjapandata.gsi.go.jp/xyz/relief/{z}/{x}/{y}.png",
            crs="EPSG:3857",
            attribution=False,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        basemap_warning = f"ベースマップの取得に失敗したため、背景なしで画像を生成しました。（{exc}）"
        ax.set_facecolor("#f2f2f2")

    main_3857.plot(ax=ax, linewidth=2.0, color="#1D4ED8", alpha=0.95)
    if sub_3857 is not None and not sub_3857.empty:
        sub_3857.plot(ax=ax, linewidth=2.6, color="#DC2626", alpha=0.95)

    if marker_series is not None:
        ax.scatter(
            marker_series.x,
            marker_series.y,
            marker="x",
            s=160,
            linewidths=2.5,
            color="#DC2626",
            zorder=5,
        )
        for point, marker in zip(marker_series, markers):
            if marker.label:
                ax.annotate(
                    marker.label, (point.x, point.y), xytext=(6, 6),
                    textcoords="offset points", fontsize=8, color="#7F1D1D",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5), zorder=8,
                )

    title_ax.set_axis_off()
    if title:
        title_ax.text(0, 0.66, title, ha="left", va="center", fontsize=14, fontweight="bold", color="#111827")
    if legend:
        title_ax.text(0, 0.20, legend, ha="left", va="center", fontsize=9, color="#475569")

    # A horizontal, ticked colour bar belongs in the lower margin, not on the
    # geographic content.  The left side holds attribution for the free GSI
    # background tiles.
    footer_ax.set_axis_off()
    footer_ax.text(0, 0.50, "背景：地理院タイル（色別標高図）", ha="left", va="center", fontsize=7, color="#475569")
    legend_ax = footer_ax.inset_axes([0.58, 0.38, 0.40, 0.28])
    colorbar = matplotlib.colorbar.ColorbarBase(
        legend_ax,
        cmap=ELEVATION_CMAP,
        norm=Normalize(vmin=0, vmax=ELEVATION_MAX_METERS),
        orientation="horizontal",
        ticks=ELEVATION_TICKS,
    )
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(labelsize=7, length=3, pad=2)
    colorbar.ax.set_xticklabels([f"{tick:,}" for tick in ELEVATION_TICKS])
    colorbar.set_label("標高（m）", fontsize=8, labelpad=2)

    ax.set_axis_off()
    _add_scale_bar(ax, xlim, ylim)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    return StaticMapResult(buffer=buffer, warning=basemap_warning)


def parse_marker_input(text: str) -> list[Marker]:
    markers: list[Marker] = []
    if not text:
        return markers
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        line = line.replace("、", ",")
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"{idx}行目: '緯度,経度' または '緯度,経度,ラベル' の形式で入力してください。")
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"{idx}行目: 数値に変換できません。（{raw_line}）") from exc
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"{idx}行目: 緯度は-90〜90、経度は-180〜180の範囲で入力してください。")
        label = parts[2] if len(parts) == 3 else None
        markers.append(Marker(lon=lon, lat=lat, label=label or None))
    return markers


def geometry_summary(gdf: gpd.GeoDataFrame) -> dict[str, float | int]:
    """Summarise the rendered linework for a compact, honest river card."""
    projected = gdf.to_crs(epsg=3857)
    return {
        "line_count": int(len(gdf)),
        "length_km": float(projected.length.sum() / 1000),
    }


__all__ = [
    "Marker",
    "StaticMapResult",
    "build_map",
    "create_static_map_image",
    "download_geojson",
    "filter_geometries",
    "geojson_to_kml",
    "geometry_summary",
    "parse_marker_input",
]
