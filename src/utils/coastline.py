import numpy as np
from scipy.ndimage import convolve
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import os

from shapely.geometry import shape
from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_coastline(water_mask: np.ndarray) -> np.ndarray:
    """
    Detecta la línia de costa a partir d'una màscara binària (1=aigua, 0=terra),
    utilitzant veïns 8-connectats.

    Parameters
    ----------
    water_mask : np.ndarray
        Màscara binària 2D (1=aigua, 0=terra).

    Returns
    -------
    np.ndarray
        Màscara 2D amb píxels de costa (True = costa).
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_sum = convolve(water_mask.astype(np.uint8), kernel, mode="constant", cval=0)
    coastline = np.logical_and(water_mask == 1, neighbor_sum < kernel.sum())

    logger.info("Línia de costa estimada (8 veïns)")
    return coastline


def export_coastline_geojson(coastline_mask: np.ndarray, reference_raster: str, output_path: str):
    """
    Exporta la línia de costa com a GeoJSON (vectoritzat).
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        shapes_gen = shapes(coastline_mask.astype(np.uint8), mask=coastline_mask, transform=transform)
        geoms = [shape(geom) for geom, val in shapes_gen if val == 1]

        if not geoms:
            logger.warning("No s'han trobat píxels de costa per exportar.")
            return

        gdf = gpd.GeoDataFrame(geometry=geoms)

        # 🔧 Forcem CRS si no està definit al raster
        if crs is not None:
            gdf.set_crs(crs, inplace=True)
        else:
            gdf.set_crs("EPSG:32631", inplace=True)

        # 🔧 Reprojectem a WGS84 (lon/lat) per compatibilitat amb geojson.io, Google Earth...
        gdf = gdf.to_crs(epsg=4326)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Línia de costa exportada a {output_path} en EPSG:4326")


def export_coastline_csv(coastline_mask: np.ndarray, reference_raster: str, output_path: str, date=None):
    """
    Exporta la línia de costa com a CSV de coordenades (lon, lat, Date).
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        y_idx, x_idx = np.where(coastline_mask)
        xs, ys = rasterio.transform.xy(transform, y_idx, x_idx)

        gpts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys))

        # 🔧 Forcem CRS si no està definit al raster
        if crs is not None:
            gpts.set_crs(crs, inplace=True)
        else:
            gpts.set_crs("EPSG:32631", inplace=True)

        gpts = gpts.to_crs(epsg=4326)

        df = pd.DataFrame({
            "lon": gpts.geometry.x,
            "lat": gpts.geometry.y
        })
        if date:
            df["Date"] = date

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Coordenades de costa exportades a {output_path} en EPSG:4326")
