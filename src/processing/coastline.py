import os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label, binary_erosion
import geopandas as gpd
from shapely.geometry import shape
from rasterio import features, mask
from src.utils.logger import get_logger
from scipy.ndimage import label, binary_erosion

logger = get_logger(__name__)


def estimate_coastline(water_mask: np.ndarray) -> np.ndarray:
    """
    Detecta la línia de costa neta, eliminant vores artificials i dobles.
    """
    # --- 0) Assegurem que és binària
    water = (water_mask > 0).astype(np.uint8)

    # --- 1️⃣ Etiquetem masses d’aigua (8-connectivitat)
    structure = np.ones((3, 3), dtype=bool)
    labeled, n = label(water, structure=structure)

    # --- 2️⃣ Eliminem components que toquen la vora del raster
    border_labels = np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ]))
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    keep[border_labels] = False
    clean_water = keep[labeled]

    # --- 3️⃣ Elimina masses petites (soroll)
    sizes = np.bincount(labeled.ravel())
    small_labels = np.where(sizes < 200)[0]  # ajustable
    clean_water[np.isin(labeled, small_labels)] = False

    # --- 4️⃣ Erosiona per extreure una línia de 1 píxel
    eroded = binary_erosion(clean_water, iterations=1)
    coastline = np.logical_and(clean_water, np.logical_not(eroded))

    return coastline


def export_coastline_geojson(coastline_mask: np.ndarray, reference_raster: str, output_path: str):
    """
    Exporta la línia de costa com a fitxer GeoJSON.

    La línia s'exporta com a geometries LineString per evitar franges dobles.

    Paràmetres
    ----------
    coastline_mask : np.ndarray
        Màscara binària amb la línia de costa.
    reference_raster : str
        Raster de referència per obtenir transformació i CRS.
    output_path : str
        Ruta de sortida del fitxer GeoJSON.
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        # Vectoritzar la màscara de costa
        shapes_gen = features.shapes(coastline_mask.astype(np.uint8),
                                     mask=coastline_mask,
                                     transform=transform)

        # Convertir cada polígon en línia
        line_geoms = [shape(geom).boundary for geom, val in shapes_gen if val == 1]

        if not line_geoms:
            logger.warning("No s'han trobat píxels de costa per exportar.")
            return

        gdf = gpd.GeoDataFrame(geometry=line_geoms)

        # Assegurar CRS
        if crs is not None:
            gdf.set_crs(crs, inplace=True)
        else:
            gdf.set_crs("EPSG:32631", inplace=True)

        # Reprojectar a WGS84 per compatibilitat
        gdf = gdf.to_crs(epsg=4326)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Línia de costa exportada correctament a: {output_path}")


def export_coastline_csv(coastline_mask: np.ndarray, reference_raster: str, output_path: str, date=None):
    """
    Exporta la línia de costa com a CSV amb coordenades lon/lat i, opcionalment, data.

    Paràmetres
    ----------
    coastline_mask : np.ndarray
        Màscara binària amb la línia de costa.
    reference_raster : str
        Raster de referència per obtenir coordenades.
    output_path : str
        Ruta de sortida del CSV.
    date : str, opcional
        Data associada a la línia (si es vol incloure al fitxer).
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        # Coordenades dels píxels de costa
        y_idx, x_idx = np.where(coastline_mask)
        xs, ys = rasterio.transform.xy(transform, y_idx, x_idx)

        gpts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys))

        # Assegurar CRS
        if crs is not None:
            gpts.set_crs(crs, inplace=True)
        else:
            gpts.set_crs("EPSG:32631", inplace=True)

        # Convertir a WGS84
        gpts = gpts.to_crs(epsg=4326)

        # Crear DataFrame final
        df = pd.DataFrame({
            "lon": gpts.geometry.x,
            "lat": gpts.geometry.y
        })
        if date:
            df["Date"] = date

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Coordenades de costa exportades a CSV a: {output_path}")
