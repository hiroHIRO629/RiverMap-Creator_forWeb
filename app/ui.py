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
    download_geojson,
    filter_geometries,
    geometry_summary,
    parse_marker_input,
)
from .data import ASSET_DIR, load_river_geometries, load_river_table, load_water_systems, search_rivers


MAP_TYPES = {"a": "水系全体", "b": "河川", "c": "水系全体と河川"}
MARKER_OPTIONS = {"none": "目印なし", "point": "指定地点に×印"}


def _display_local_image(path: Path, caption: Optional[str] = None) -> None:
    """Safely display a local image file in Streamlit."""
    try:
        with path.open("rb") as fp:
            st.image(fp.read(), caption=caption)
    except FileNotFoundError:
        st.warning(f"画像ファイルが見つかりません: {path.name}")


def _candidate_label(row: pd.Series) -> str:
    """Create an unambiguous, compact label for the river picker."""
    prefecture = row.get("都道府県")
    prefecture_text = "未登録" if pd.isna(prefecture) or not str(prefecture).strip() else str(prefecture)
    return f"{row['河川名']} ｜ {row['水系']}水系 ｜ {prefecture_text} ｜ {row['河川コード']}"


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

    st.header("河川を探す")
    st.caption("一部の名称、読みの一部、水系名、都道府県名から探せます。候補から選ぶため、同名河川も取り違えません。")
    r_type = st.radio(
        "表示する範囲",
        options=list(MAP_TYPES.keys()), horizontal=True,
        format_func=lambda key: f"{key}：{MAP_TYPES[key]}",
    )
    try:
        river_table = load_river_table(r_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    filter_left, filter_middle, filter_right = st.columns((2, 1, 1))
    with filter_left:
        query = st.text_input("名前・水系・都道府県で検索", placeholder="例：重信、利根川、愛媛")
    with filter_middle:
        systems = ["すべて"] + sorted(river_table["水系"].dropna().unique().tolist())
        selected_system = st.selectbox("水系で絞り込み", systems)
    with filter_right:
        prefectures = ["すべて"] + sorted(
            {part.strip() for value in river_table["都道府県"].dropna() for part in str(value).replace("、", ",").split(",") if part.strip()}
        )
        selected_prefecture = st.selectbox("都道府県で絞り込み", prefectures)

    candidates = search_rivers(river_table, query, selected_system, selected_prefecture)
    if candidates.empty:
        st.info("該当する河川がありません。別の表記や絞り込み条件をお試しください。")
        st.stop()
    if len(candidates) == 200:
        st.caption("候補は先頭200件を表示しています。さらに名前や水系で絞り込めます。")

    candidate_rows = candidates.reset_index(drop=True)
    chosen_index = st.selectbox(
        f"候補から選択（{len(candidate_rows)}件）",
        options=list(candidate_rows.index),
        format_func=lambda index: _candidate_label(candidate_rows.iloc[index]),
    )
    chosen = candidate_rows.iloc[chosen_index]
    river_name = str(chosen["河川名"])
    river_code = str(chosen["河川コード"]).strip()

    marker_choice = st.radio(
        "目印の描画", options=list(MARKER_OPTIONS.keys()), horizontal=True,
        format_func=lambda x: MARKER_OPTIONS[x],
    )
    marker_input = ""
    if marker_choice == "point":
        marker_input = st.text_area(
            "座標リスト（緯度,経度 を1行ごとに）", placeholder="33.85,132.77\n33.80,132.70",
            help="複数地点を指定できます。緯度は-90〜90、経度は-180〜180です。",
        )

    signature = (r_type, river_code, marker_choice, marker_input if marker_choice == "point" else "")
    if st.button("地図を表示", type="primary", use_container_width=True):
        if marker_choice == "point":
            try:
                markers = parse_marker_input(marker_input)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            if not markers:
                st.warning("座標を少なくとも1行入力してください。")
                st.stop()
        st.session_state.active_map_signature = signature

    if st.session_state.get("active_map_signature") != signature:
        st.info("候補または目印を選び、「地図を表示」を押してください。")
        st.stop()

    markers = parse_marker_input(marker_input) if marker_choice == "point" else []
    selection = chosen.to_frame().T
    display_selection = selection.copy()
    if "都道府県" in display_selection:
        display_selection["都道府県"] = display_selection["都道府県"].fillna("未登録")
    st.write("河川情報")
    st.dataframe(display_selection.astype(str), use_container_width=True, hide_index=True)

    try:
        gdf = load_river_geometries()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    main_geom, sub_geom = filter_geometries(gdf, r_type, river_code)
    if main_geom.empty:
        st.error("申し訳ありません。選択された河川はアプリ内の河川データベースに存在しません。")
        st.stop()

    summary_target = sub_geom if sub_geom is not None and not sub_geom.empty else main_geom
    summary = geometry_summary(summary_target)
    metric_left, metric_right = st.columns(2)
    metric_left.metric("収録ライン数", f"{summary['line_count']:,} 本")
    metric_right.metric("収録ライン長（概算）", f"{summary['length_km']:.1f} km")
    st.caption("国土数値情報を簡略化して収録したラインの合計です。河川の公称延長とは一致しない場合があります。")

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
    export_gdf = pd.concat(export_frames)
    export_geojson, geojson_simplified = download_geojson(export_gdf)

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
            data=export_geojson,
            file_name=f"{river_name}_{river_code}_{r_type}.geojson",
            mime="application/geo+json",
            key=f"download_geojson_{r_type}_{river_code}",
        )

    if geojson_simplified:
        st.caption("GeoJSON はブラウザ上で安全に扱えるサイズに調整しています。")

    if static_result.warning:
        st.info(static_result.warning)


__all__ = ["render"]
