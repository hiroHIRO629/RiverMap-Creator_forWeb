from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import contextily as cx
import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium
from json import loads as json_loads
from branca.element import Element


def _display_local_image(path: Path, caption: Optional[str] = None) -> None:
    """Safely display a local image file in Streamlit."""
    try:
        with path.open("rb") as fp:
            st.image(fp.read(), caption=caption)
    except FileNotFoundError:
        st.warning(f"画像ファイルが見つかりません: {path.name}")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"

RIVER_TABLE = {
    "a": DATA_DIR / "class_a2.csv",
    "b": DATA_DIR / "class_b.csv",
    "c": DATA_DIR / "class_b.csv",
}
GEOMETRY_PATH = DATA_DIR / "all_river_light.parquet"

MAP_TYPES = {"a": "水系全体", "b": "河川", "c": "水系全体と河川"}
MARKER_OPTIONS = {"none": "目印なし", "point": "指定地点に×印"}


@dataclass
class Marker:
    lon: float
    lat: float


@dataclass
class StaticMapResult:
    buffer: BytesIO
    warning: Optional[str] = None


@st.cache_data(show_spinner=False)
def load_river_table(r_type: str) -> pd.DataFrame:
    path = RIVER_TABLE[r_type]
    if not path.exists():
        raise FileNotFoundError(f"河川リストが見つかりません: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    df = df.set_index("河川名")
    return df


@st.cache_resource(show_spinner=False)
def load_river_geometries() -> gpd.GeoDataFrame:
    if not GEOMETRY_PATH.exists():
        raise FileNotFoundError(
            f"河川ジオメトリが見つかりません: {GEOMETRY_PATH}\n"
            "先に scripts/prepare_data.py で Shapefile を変換してください。"
        )
    if GEOMETRY_PATH.suffix == ".parquet":
        gdf = gpd.read_parquet(GEOMETRY_PATH)
    else:
        gdf = gpd.read_file(GEOMETRY_PATH)

    if {"W05_001", "W05_002", "geometry"} - set(gdf.columns):
        raise ValueError("ジオメトリファイルに必要な列 (W05_001, W05_002) が含まれていません。")

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        try:
            epsg = gdf.crs.to_epsg()
        except Exception:
            epsg = None

        if epsg == 4326:
            pass
        else:
            bounds = gdf.total_bounds
            lon_ok = np.all(np.abs(bounds[[0, 2]]) <= 180)
            lat_ok = np.all(np.abs(bounds[[1, 3]]) <= 90)
            if lon_ok and lat_ok:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            else:
                gdf = gdf.to_crs(epsg=4326)

    return gdf[["W05_001", "W05_002", "geometry"]].copy()


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
    marker: Optional[Marker],
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

    if marker is not None:
        folium.Marker(
            location=[marker.lat, marker.lon],
            icon=folium.DivIcon(
                html='<div style="font-size:32px; color:#DC2626; font-weight:bold;">&times;</div>'
            ),
        ).add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)

    fmap.fit_bounds([[south, west], [north, east]], padding=(28, 28))
    return fmap


def create_static_map_image(
    main: gpd.GeoDataFrame,
    sub: Optional[gpd.GeoDataFrame],
    marker: Optional[Marker],
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
    if marker is not None:
        marker_series = gpd.GeoSeries([Point(marker.lon, marker.lat)], crs="EPSG:4326").to_crs(epsg=3857)
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

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    return StaticMapResult(buffer=buffer, warning=basemap_warning)


def parse_coordinate(value: str, label: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        st.error(f"{label}は数値で入力してください（例: 132.77）。")
        return None


def render():
    st.set_page_config(page_title="RiverMap-Creator for Web", layout="wide")

    logo_path = ASSET_DIR / "RMC_long2.png"
    if logo_path.exists():
        _display_local_image(logo_path)
    st.markdown(
        """
### 指定された国内河川を検索して表示するアプリ  
表示結果は GeoJSON と PNG としてダウンロード可能。
        """.strip()
    )

    sidebar_logo = ASSET_DIR / "RMC.png"
    if sidebar_logo.exists():
        with sidebar_logo.open("rb") as fp:
            st.sidebar.image(fp.read())

    st.sidebar.header("使い方")
    st.sidebar.markdown(
        "1️⃣ 地図タイプを選択\n"
        "2️⃣ 河川名を正確に入力\n"
        "3️⃣ 複数候補がある場合は河川コードを選択\n"
        "4️⃣ 任意で座標を入力して目印を追加\n"
        "5️⃣ 「地図を表示」を押して結果を確認"
    )

    st.markdown("### 表示する地図タイプを選択")
    st.markdown("重信川（愛媛県）の例 ↓")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _display_local_image(ASSET_DIR / "a.png", caption="a：水系全体")
        st.caption("水系全域を青線で表示。")
    with col_b:
        _display_local_image(ASSET_DIR / "b.png", caption="b：河川")
        st.caption("選択した河川のみを表示。")
    with col_c:
        _display_local_image(ASSET_DIR / "c.png", caption="c：水系＋河川")
        st.caption("水系と河川を青線と赤線で重ね描き。")

    r_type = st.radio(
        "地図タイプ",
        options=list(MAP_TYPES.keys()),
        horizontal=True,
        format_func=lambda key: f"{key}：{MAP_TYPES[key]}",
        help="a=水系全体、b=河川のみ、c=水系＋河川を同時表示します。",
    )

    try:
        river_table = load_river_table(r_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
        st.session_state.form_signature = None

    with st.form("river_form", clear_on_submit=False):
        river_name = st.text_input(
            "河川名（例：重信川）",
            placeholder="正式名称を全角で入力してください",
        ).strip()

        marker_choice = st.radio(
            "目印の描画",
            options=list(MARKER_OPTIONS.keys()),
            format_func=lambda x: MARKER_OPTIONS[x],
            horizontal=True,
        )

        lon_input, lat_input = "", ""
        if marker_choice == "point":
            col_lon, col_lat = st.columns(2)
            with col_lon:
                lon_input = st.text_input("経度 (例: 132.77)")
            with col_lat:
                lat_input = st.text_input("緯度 (例: 33.85)")

        submitted = st.form_submit_button("地図を表示", type="primary")

    current_signature = (
        r_type,
        river_name,
        marker_choice,
        lon_input if marker_choice == "point" else None,
        lat_input if marker_choice == "point" else None,
    )

    if submitted:
        st.session_state.form_submitted = True
        st.session_state.form_signature = current_signature

    if st.session_state.form_signature != current_signature:
        st.session_state.form_submitted = False

    if not st.session_state.form_submitted:
        st.stop()

    if not river_name:
        st.warning("河川名を入力してください。")
        st.stop()

    try:
        selection = river_table.loc[[river_name]].copy()
    except KeyError:
        candidates = river_table[river_table.index.str.contains(river_name)]
        if candidates.empty:
            st.error(f"河川名「{river_name}」は登録されていません。正式名称で入力してください。")
        else:
            st.warning("入力された名称が完全一致しませんでした。候補を確認の上、正式名称を入力してください。")
            st.dataframe(candidates.astype(str), use_container_width=True)
        st.stop()

    st.write("河川情報")
    st.dataframe(selection.astype(str), use_container_width=True)

    codes = (
        selection["河川コード"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    if not codes:
        st.error("河川コードが見つかりません。CSV の内容を確認してください。")
        st.stop()

    if len(codes) == 1:
        river_code = st.selectbox(
            "河川コード",
            options=codes,
            index=0,
            help="候補が1件のみの場合は自動的に選択されます。",
        )
    else:
        sentinel = "__NONE__"
        selection = st.selectbox(
            "河川コード",
            options=[sentinel] + codes,
            index=0,
            format_func=lambda x: "河川コードを選択してください" if x == sentinel else x,
            help="複数候補の中から対象の河川コードを選択してください。",
        )
        if selection == sentinel:
            st.info("河川コードを選択すると地図が表示されます。")
            st.stop()
        river_code = selection

    marker: Optional[Marker] = None
    if marker_choice == "point":
        lon = parse_coordinate(lon_input, "経度")
        lat = parse_coordinate(lat_input, "緯度")
        if lon is not None and lat is not None:
            marker = Marker(lon=lon, lat=lat)
        else:
            st.info("有効な座標が入力されるまで目印は追加されません。")

    try:
        gdf = load_river_geometries()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    main_geom, sub_geom = filter_geometries(gdf, r_type, river_code)
    if main_geom.empty:
        st.error("該当する河川ジオメトリが見つかりませんでした。CSV とデータの整合性を確認してください。")
        st.stop()

    river_map = build_map(main_geom, sub_geom, marker)
    st_folium(
        river_map,
        width=None,
        height=700,
        returned_objects=[],
        key=f"river_map_{r_type}_{river_code}",
    )

    with st.spinner("ダウンロード用の静的画像を生成しています..."):
        static_result = create_static_map_image(main_geom, sub_geom, marker)

    static_bytes = static_result.buffer.getvalue()
    st.image(static_bytes, caption=f"{river_name}（{river_code}）")

    export_frames = [main_geom]
    if sub_geom is not None and not sub_geom.empty:
        export_frames.append(sub_geom)
    export_geojson = pd.concat(export_frames).to_json()

    col_img, col_geo = st.columns(2)
    with col_img:
        st.download_button(
            label="マップ画像をダウンロード (PNG)",
            data=static_bytes,
            file_name=f"{river_name}_{river_code}_{r_type}.png",
            mime="image/png",
            key=f"download_png_{r_type}_{river_code}",
        )
    with col_geo:
        st.download_button(
            label="GeoJSON をダウンロード",
            data=export_geojson.encode("utf-8"),
            file_name=f"{river_name}_{river_code}_{r_type}.geojson",
            mime="application/geo+json",
            key=f"download_geojson_{r_type}_{river_code}",
        )

    if static_result.warning:
        st.info(static_result.warning)


if __name__ == "__main__":
    render()
