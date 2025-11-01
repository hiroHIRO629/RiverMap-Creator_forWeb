from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"

RIVER_TABLE = {
    "a": DATA_DIR / "class_a2.csv",
    "b": DATA_DIR / "class_b.csv",
    "c": DATA_DIR / "class_b.csv",
}
GEOMETRY_PATH = DATA_DIR / "all_river_light.parquet"


@st.cache_data(show_spinner=False)
def load_river_table(r_type: str) -> pd.DataFrame:
    path = RIVER_TABLE[r_type]
    if not path.exists():
        raise FileNotFoundError(f"河川リストが見つかりません: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    return df.set_index("河川名")


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


@st.cache_data(show_spinner=False)
def load_water_systems() -> list[str]:
    df = load_river_table("a")
    return [
        f"{row['水系']}（{row['開発局']}）"
        for _, row in df.drop_duplicates(subset=["水系", "開発局"]).iterrows()
    ]


__all__ = [
    "ASSET_DIR",
    "DATA_DIR",
    "BASE_DIR",
    "load_river_table",
    "load_river_geometries",
    "load_water_systems",
]
