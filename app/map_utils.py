from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from branca.element import Element
from json import loads as json_loads

import contextily as cx
import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import Point


@dataclass
class Marker:
    lon: float
    lat: float


@dataclass
class StaticMapResult:
    buffer: BytesIO
    warning: Optional[str] = None


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


def build_map(
    main: gpd.GeoDataFrame,
    sub: Optional[gpd.GeoDataFrame],
    markers: list[Marker],
) -> folium.Map:
    (south, west), (north, east) = expand_bounds(main, sub)
    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        max_zoom=19,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        name="CartoDB Positron",
        attr="© OpenStreetMap contributors © CARTO",
        subdomains="abcd",
        max_zoom=19,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://cyberjapandata.gsi.go.jp/xyz/relief/{z}/{x}/{y}.png",
        name="GSI Relief",
        attr="地理院タイル（標高陰影図）",
        max_zoom=15,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        name="OSM Humanitarian",
        attr="© OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team",
        subdomains="abc",
        max_zoom=19,
    ).add_to(fmap)

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

    main_geojson = json_loads(main.to_json())
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
        sub_geojson = json_loads(sub.to_json())
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
            icon=folium.DivIcon(
                html='<div style="font-size:32px; color:#DC2626; font-weight:bold;">&times;</div>'
            ),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)

    fmap.fit_bounds([[south, west], [north, east]], padding=(28, 28))
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

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    basemap_warning = None
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs="EPSG:3857", attribution=False)
    except Exception as exc:  # pragma: no cover - network dependent
        basemap_warning = f"ベースマップの取得に失敗したため、背景なしで画像を生成しました。（{exc}）"
        ax.set_facecolor("#f2f2f2")

    ax.text(
        xlim[0] + (xlim[1] - xlim[0]) * 0.02,
        ylim[0] + (ylim[1] - ylim[0]) * 0.02,
        "© OpenStreetMap contributors | © CARTO",
        fontsize=6,
        color="#4b5563",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=2),
        zorder=6,
    )

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

    ax.set_axis_off()
    fig.tight_layout(pad=0)

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
        if len(parts) != 2:
            raise ValueError(f"{idx}行目: '緯度,経度' の形式で入力してください。")
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"{idx}行目: 数値に変換できません。（{raw_line}）") from exc
        markers.append(Marker(lon=lon, lat=lat))
    return markers


__all__ = [
    "Marker",
    "StaticMapResult",
    "build_map",
    "create_static_map_image",
    "filter_geometries",
    "parse_marker_input",
]
