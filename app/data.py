from __future__ import annotations

from pathlib import Path
import re
import unicodedata

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


def normalize_search_text(value: object) -> str:
    """Return a comparison-friendly Japanese search key.

    River lists contain a mixture of full-width spaces and occasional spacing
    differences.  Normalising those details makes partial-name search useful
    without changing the original labels shown to users.
    """
    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[\s\u3000]+", "", text)


def search_rivers(
    river_table: pd.DataFrame,
    query: str = "",
    water_system: str | None = None,
    prefecture: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Find river records by partial name and optional free local filters."""
    result = river_table.reset_index().copy()
    if water_system and water_system != "すべて":
        result = result[result["水系"].fillna("") == water_system]
    if prefecture and prefecture != "すべて":
        # A record can cover more than one prefecture, so exact equality is
        # intentionally not required here.
        result = result[result["都道府県"].fillna("").str.contains(prefecture, regex=False)]

    key = normalize_search_text(query)
    if key:
        searchable_columns = [column for column in ("河川名", "水系", "都道府県") if column in result]
        matches = pd.Series(False, index=result.index)
        for column in searchable_columns:
            matches |= result[column].map(normalize_search_text).str.contains(key, regex=False)
        result = result[matches]

    # Duplicate records occasionally exist in the source table.  Showing one
    # choice per river code prevents an ambiguous select box.
    result = result.drop_duplicates(subset=["河川コード"])
    return result.sort_values(["水系", "河川名", "河川コード"], kind="stable").head(limit)


__all__ = [
    "ASSET_DIR",
    "DATA_DIR",
    "BASE_DIR",
    "load_river_table",
    "load_river_geometries",
    "load_water_systems",
    "normalize_search_text",
    "search_rivers",
]
