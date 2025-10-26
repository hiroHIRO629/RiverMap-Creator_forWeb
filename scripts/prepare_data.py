from pathlib import Path
import geopandas as gpd

SRC = Path("../RiverMap-Creator/riverline/all_river/all_river.shp")
DEST = Path("../RiverMap-Creator_forWeb/data/all_river_light.parquet")

def main() -> None:
    gdf = gpd.read_file(SRC, columns=["W05_001", "W05_002", "geometry"])
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf["geometry"] = gdf.geometry.simplify(0.0005, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf.to_parquet(DEST, compression="snappy")
    size_mb = DEST.stat().st_size / 1024 ** 2
    print(f"saved: {DEST} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
