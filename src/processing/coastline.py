import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from rasterio import mask
from rasterio.features import shapes
from scipy.ndimage import label, binary_erosion, distance_transform_edt
from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_coastline(
    water_mask: np.ndarray,
    aoi_path: str = None,
    reference_raster: str = None,
    aoi_margin_m: float = 50.0,
    min_blob_px: int = 200
) -> np.ndarray:
    """
    Estima la línia de costa a partir d'una màscara binària d'aigua (1=aigua, 0=terra).

    Passos:
    1️⃣ Talla el marge exterior (fora AOI) si es proporciona AOI.
    2️⃣ Elimina masses petites d’aigua (soroll).
    3️⃣ Extreu la vora d’aigua d’1 píxel.
    4️⃣ Es queda només amb la vora més propera a terra.
    """
    water = (water_mask > 0).astype(np.uint8)

    # --- 1️⃣ Aplicar AOI erosionat per eliminar marges del retall
    if aoi_path and reference_raster:
        aoi = gpd.read_file(aoi_path)
        with rasterio.open(reference_raster) as src:
            aoi_mask, _ = mask.mask(src, aoi.geometry, crop=False)
            aoi_mask = aoi_mask[0].astype(bool)
            px_size = abs(src.transform.a)
            iters = max(1, int(round(aoi_margin_m / px_size)))

        aoi_inner = binary_erosion(aoi_mask, iterations=iters)
        water &= aoi_inner
        logger.info(f"Aplicat AOI erosionat {iters} píxels (~{aoi_margin_m} m).")

    # --- 2️⃣ Neteja de masses petites (soroll)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    labeled, n = label(water, structure=structure)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small_labels = np.where(sizes < min_blob_px)[0]
        water[np.isin(labeled, small_labels)] = 0

    # --- 3️⃣ Extreure vora d’aigua (1 píxel de gruix)
    eroded = binary_erosion(water, iterations=1)
    edge = (water == 1) & (eroded == 0)

    # --- 4️⃣ Mantenir només la vora més propera a terra
    land = (water == 0)
    d_land = distance_transform_edt(~land)
    d_water = distance_transform_edt(water)
    keep_land_side = d_land <= d_water
    coastline = edge & keep_land_side

    return coastline.astype(bool)


def export_coastline_geojson(coastline_mask: np.ndarray, reference_raster: str, output_path: str):
    """
    Exporta la línia de costa com a fitxer GeoJSON (LineString).

    Paràmetres
    ----------
    coastline_mask : np.ndarray
        Màscara binària de la línia de costa.
    reference_raster : str
        Ruta del raster de referència (per coordenades i transformació).
    output_path : str
        Ruta de sortida del fitxer GeoJSON.
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        # Vectoritzar la màscara i extreure només les línies
        shapes_gen = shapes(coastline_mask.astype(np.uint8), mask=coastline_mask, transform=transform)
        line_geoms = [shape(geom).boundary for geom, val in shapes_gen if val == 1]

        if not line_geoms:
            logger.warning("No s'han trobat píxels de costa per exportar.")
            return

        gdf = gpd.GeoDataFrame(geometry=line_geoms)
        gdf.set_crs(crs or "EPSG:32631", inplace=True)
        gdf = gdf.to_crs(epsg=4326)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Línia de costa exportada a {output_path} (EPSG:4326).")


def export_coastline_csv(coastline_mask: np.ndarray, reference_raster: str, output_path: str, date=None):
    """
    Exporta la línia de costa com a CSV amb coordenades lon/lat i, opcionalment, data.
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        # Coordenades dels píxels de costa
        y_idx, x_idx = np.where(coastline_mask)
        xs, ys = rasterio.transform.xy(transform, y_idx, x_idx)

        gpts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys))
        gpts.set_crs(crs or "EPSG:32631", inplace=True)
        gpts = gpts.to_crs(epsg=4326)

        df = pd.DataFrame({
            "lon": gpts.geometry.x,
            "lat": gpts.geometry.y
        })
        if date:
            df["Date"] = date

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Coordenades de costa exportades a CSV a: {output_path}")
