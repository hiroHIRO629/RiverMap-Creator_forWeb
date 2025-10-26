from pathlib import Path
import geopandas as gpd

SRC = Path("../RiverMap-Creator/riverline/all_river/all_river.shp")
DEST = Path("../RiverMap-Creator_forWeb/data/all_river_light.parquet")

def main() -> None:
    gdf = gpd.read_file(SRC, columns=["W05_001", "W05_002", "geometry"])
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
            lon_ok = (abs(bounds[[0, 2]]) <= 180).all()
            lat_ok = (abs(bounds[[1, 3]]) <= 90).all()
            if lon_ok and lat_ok:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            else:
                gdf = gdf.to_crs(epsg=4326)

    gdf["geometry"] = gdf.geometry.simplify(5e-5, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf.to_parquet(DEST, compression="snappy")
    size_mb = DEST.stat().st_size / 1024 ** 2
    print(f"saved: {DEST} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
