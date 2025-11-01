from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from .map_utils import (
    Marker,
    build_map,
    create_static_map_image,
    filter_geometries,
    parse_marker_input,
)
from .data import ASSET_DIR, load_river_geometries, load_river_table, load_water_systems


MAP_TYPES = {"a": "水系全体", "b": "河川", "c": "水系全体と河川"}
MARKER_OPTIONS = {"none": "目印なし", "point": "指定地点に×印"}


def _display_local_image(path: Path, caption: Optional[str] = None) -> None:
    """Safely display a local image file in Streamlit."""
    try:
        with path.open("rb") as fp:
            st.image(fp.read(), caption=caption)
    except FileNotFoundError:
        st.warning(f"画像ファイルが見つかりません: {path.name}")


def render() -> None:
    st.set_page_config(page_title="RiverMap-Creator for Web", layout="wide")

    logo_path = ASSET_DIR / "RMC_long2.png"
    if logo_path.exists():
        _display_local_image(logo_path)
    st.markdown(
        """
### 指定された国内河川を検索して地図上に表示するアプリ  
        """.strip()
    )

    sidebar_logo = ASSET_DIR / "RMC.png"
    if sidebar_logo.exists():
        with sidebar_logo.open("rb") as fp:
            st.sidebar.image(fp.read())

    st.sidebar.header("使い方")
    st.sidebar.markdown(
        "1️⃣ 河川タイプを選択（a,b,cから選択）<br>"
        "2️⃣ 河川名を正確に入力<br>"
        "3️⃣ 複数候補がある場合は河川コードを選択<br>"
        "4️⃣ 任意で座標を入力して目印を追加<br>"
        "5️⃣ 「地図を表示」を押して結果を確認",
        unsafe_allow_html=True,
    )


    st.sidebar.header("水系とは")
    st.sidebar.markdown(
        "河川(本川)に流れ込む支川や派川を全て合わせたもの．全国で109の一級水系が存在．"
    )

    water_systems = load_water_systems()
    with st.sidebar.expander(f"一級水系一覧 ({len(water_systems)})", expanded=False):
        for system in water_systems:
            st.markdown(system)

    st.sidebar.header("データ提供元・利用条件")
    st.sidebar.markdown(
        """
        出典：国土数値情報（河川データ）（国土交通省）
       (https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html)   
        本アプリは上記データを加工して作成しています。   
        データには空間的・時間的な誤差が含まれる可能性があり、正確な測量・証明・ナビゲーション等の用途には適していません。  
        利用は自己責任で行ってください。  
        本データは国土数値情報ダウンロードサービスの利用規約に基づいて利用しています。  
        [利用規約はこちら](https://nlftp.mlit.go.jp/ksj/other/agreement_02.html)
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ↓ 表示可能な河川タイプには以下の三つがある（例：重信川(愛媛県)の場合）
        """
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _display_local_image(ASSET_DIR / "a.png", caption="a：水系全体（重信川水系）")
        st.caption("水系全域を青線で表示。一級河川水系（109水系）を指定可能。")
    with col_b:
        _display_local_image(ASSET_DIR / "b.png", caption="b：河川（重信川）")
        st.caption("指定した河川（本川または支川）のみを表示。")
    with col_c:
        _display_local_image(ASSET_DIR / "c.png", caption="c：水系全体と河川（重信川水系＋重信川）")
        st.caption("指定した河川（赤線）とそれが含まれる水系（青線）を重ね描き。")
    st.markdown("</div>", unsafe_allow_html=True)

    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
        st.session_state.form_signature = None

    marker_input = ""

    with st.form("river_form", clear_on_submit=False):
        r_type = st.radio(
            "河川タイプを選択",
            options=list(MAP_TYPES.keys()),
            horizontal=True,
            format_func=lambda key: f"{key}：{MAP_TYPES[key]}",
            help="a=水系全体、b=河川のみ、c=水系＋河川を同時表示します。",
        )

        river_name = st.text_input(
            "河川/水系名",
            placeholder="正式名称を全角で入力してください",
            help="「〜川」まで入力すること．a:水系全体を選択した場合でも「〜川」と入力．",
        ).strip()

        marker_choice = st.radio(
            "目印の描画",
            options=list(MARKER_OPTIONS.keys()),
            format_func=lambda x: MARKER_OPTIONS[x],
            horizontal=True,
        )

        if marker_choice == "point":
            marker_input = st.text_area(
                "座標リスト（緯度,経度 を1行ごとに）",
                placeholder="33.85,132.77\n33.80,132.70",
                help="緯度,経度 の形式で1行につき1地点を入力。複数行入力すると複数地点にマーカーが描画されます。",
            )

        submitted = st.form_submit_button("地図を表示", type="primary")

    current_signature = (
        r_type,
        river_name,
        marker_choice,
        marker_input if marker_choice == "point" else None,
    )

    if submitted:
        st.session_state.form_submitted = True
        st.session_state.form_signature = current_signature

    if st.session_state.form_signature != current_signature:
        st.session_state.form_submitted = False

    if not st.session_state.form_submitted:
        st.stop()

    if marker_choice == "point":
        try:
            markers = parse_marker_input(marker_input)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if not markers:
            st.warning("座標を少なくとも1行入力してください。")
            st.stop()
    else:
        markers: list[Marker] = []

    try:
        river_table = load_river_table(r_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
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

    display_selection = selection.copy()
    if "都道府県" in display_selection.columns:
        display_selection["都道府県"] = display_selection["都道府県"].apply(
            lambda value: (
                "未登録"
                if pd.isna(value)
                or (isinstance(value, str) and value.strip().lower() in {"", "nan"})
                else value
            )
        )

    st.write("河川情報")
    st.dataframe(display_selection.astype(str), use_container_width=True)

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
        st.error("申し訳ありません。選択された河川はアプリ内の河川データベースに存在しません。")
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

    try:
        gdf = load_river_geometries()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    main_geom, sub_geom = filter_geometries(gdf, r_type, river_code)
    if main_geom.empty:
        st.error("申し訳ありません。選択された河川はアプリ内の河川データベースに存在しません。")
        st.stop()

    river_map = build_map(main_geom, sub_geom, markers)
    st_folium(
        river_map,
        width=None,
        height=700,
        returned_objects=[],
        key=f"river_map_{r_type}_{river_code}",
    )

    with st.spinner("ダウンロード用の静的画像を生成しています..."):
        static_result = create_static_map_image(main_geom, sub_geom, markers)

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


__all__ = ["render"]
