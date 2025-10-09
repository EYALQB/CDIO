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
    1) Elimina l'aigua connectada a les vores del raster (retall) sense moure la costa.
    2) (Opcional) Elimina masses petites de soroll.
    3) Extreu la línia de costa com a frontera d'1 píxel.
    """
    import numpy as np
    from scipy.ndimage import label, binary_erosion

    # 0) Assegura màscara binària 0/1
    water = (water_mask > 0).astype(np.uint8)

    # 1) ELIMINA AIGUA QUE TOCA LA VORA (robust, sense desplaçar la costa)
    #    - etiquetem components 8-connectats
    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]], dtype=bool)
    labeled, n = label(water, structure=structure)

    #    - trobem quines etiquetes toquen la vora
    border_labels = np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ]))

    #    - marquem per conservar només components que NO toquen la vora
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False                 # 0 = fons
    keep[border_labels] = False     # fora tot el que toca la vora
    clean_water = keep[labeled]

    # 2) (OPCIONAL) Treu soroll: elimina masses d'aigua petites per mida
    #    ajusta 'min_size' segons resolució; a Sentinel-2 (10 m), 500 píxels ≈ 5 ha
    sizes = np.bincount(labeled.ravel())
    min_size = 500                   # puja/baixa si cal
    small_labels = np.where(sizes < min_size)[0]
    clean_water[np.isin(labeled, small_labels)] = False

    # 3) COSTA = aigua neta - aigua erodida (1 píxel de gruix, sense moure la posició)
    eroded = binary_erosion(clean_water, iterations=1)
    coastline = np.logical_and(clean_water, np.logical_not(eroded))

    return coastline


def export_coastline_geojson(coastline_mask: np.ndarray, reference_raster: str, output_path: str):
    """
    Exporta la línia de costa com a GeoJSON, convertint la frontera
    d'àrea (polígon) a una línia (LineString) per evitar franges dobles.
    """
    from shapely.geometry import shape
    import geopandas as gpd
    from rasterio.features import shapes
    import os

    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        # 🔹 Vectoritza el raster (obté geometries dels píxels de costa)
        shapes_gen = shapes(coastline_mask.astype(np.uint8), mask=coastline_mask, transform=transform)

        # 🔹 Converteix cada polígon a la seva vora (LineString)
        line_geoms = [shape(geom).boundary for geom, val in shapes_gen if val == 1]

        if not line_geoms:
            logger.warning("No s'han trobat píxels de costa per exportar.")
            return

        # 🔹 Crea el GeoDataFrame amb línies, no polígons
        gdf = gpd.GeoDataFrame(geometry=line_geoms)

        # 🔧 Assegura CRS (si no ve definit al raster)
        if crs is not None:
            gdf.set_crs(crs, inplace=True)
        else:
            gdf.set_crs("EPSG:32631", inplace=True)

        # 🔧 Reprojecta a WGS84 per compatibilitat amb geojson.io
        gdf = gdf.to_crs(epsg=4326)

        # 🔹 Crea carpetes si no existeixen i exporta
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Línia de costa exportada com LineString a {output_path} en EPSG:4326")



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
