import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from rasterio import mask
from rasterio.features import shapes
from scipy.ndimage import (
    label, binary_erosion, binary_opening, binary_closing,
    binary_fill_holes, distance_transform_edt
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Funció auxiliar: mantenir només la línia superior
# ----------------------------------------------------------------------
def keep_topmost_line(
    coast_mask: np.ndarray,
    margin_left_px: int = 0,
    margin_right_px: int = 0,
    margin_top_px: int = 0,
    margin_bottom_px: int = 0,
) -> np.ndarray:
    """
    Deixa només el primer píxel de costa per cada columna (la línia 'superior')
    i aplica un retall simple de marges en píxels.
    """
    h, w = coast_mask.shape
    out = np.zeros_like(coast_mask, dtype=bool)

    # Definir àrea de treball dins dels marges
    x0 = max(0, margin_left_px)
    x1 = max(x0, w - margin_right_px)
    y0 = max(0, margin_top_px)
    y1 = max(y0, h - margin_bottom_px)
    work = coast_mask[y0:y1, x0:x1]

    # Per cada columna, selecciona el primer píxel True (el més amunt)
    for j in range(work.shape[1]):
        col = work[:, j]
        if col.any():
            idx = np.argmax(col)
            out[y0 + idx, x0 + j] = True

    return out


# ----------------------------------------------------------------------
# Estimar la línia de costa
# ----------------------------------------------------------------------
def estimate_coastline(
    water_mask: np.ndarray,
    aoi_path: str = None,
    reference_raster: str = None,
    aoi_margin_m: float = 50.0,
    min_blob_px: int = 500,
) -> np.ndarray:
    """
    Estima la línia de costa a partir d'una màscara binària d'aigua (1=aigua, 0=terra).

    Passos:
    1️⃣ (Opcional) Talla el marge exterior amb AOI.
    2️⃣ Elimina masses petites i soroll amb operacions morfològiques.
    3️⃣ Extreu la vora d’aigua d’1 píxel.
    4️⃣ Es queda només amb la línia superior (la que toca la platja).
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

    # --- 2️⃣ Neteja avançada de soroll i coherència espacial
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    labeled, n = label(water, structure=structure)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small_labels = np.where(sizes < min_blob_px)[0]
        water[np.isin(labeled, small_labels)] = 0

    # Operacions morfològiques per suavitzar i eliminar forats petits
    water = binary_opening(water, structure=structure, iterations=1)
    water = binary_closing(water, structure=structure, iterations=2)
    water = binary_fill_holes(water)
    water = water.astype(np.uint8)

    # --- 3️⃣ Extreure vora d’aigua (1 píxel de gruix)
    eroded = binary_erosion(water, iterations=1)
    edge = (water == 1) & (eroded == 0)

    # --- 4️⃣ Mantenir només la línia superior
    coastline = keep_topmost_line(
        edge,
        margin_left_px=5,
        margin_right_px=5,
        margin_top_px=0,
        margin_bottom_px=0
    )

    return coastline.astype(bool)


# ----------------------------------------------------------------------
# Exportar línia de costa a GeoJSON
# ----------------------------------------------------------------------
def export_coastline_geojson(coastline_mask: np.ndarray, reference_raster: str, output_path: str):
    """
    Exporta la línia de costa com a fitxer GeoJSON (LineString).
    Manté només la línia superior de la costa (ja filtrada abans).
    """
    with rasterio.open(reference_raster) as src:
        transform = src.transform
        crs = src.crs

        # Vectoritzar la màscara i extreure les línies
        shapes_gen = shapes(coastline_mask.astype(np.uint8), mask=coastline_mask, transform=transform)
        line_geoms = [shape(geom).boundary for geom, val in shapes_gen if val == 1]

        if not line_geoms:
            logger.warning("No s'han trobat píxels de costa per exportar.")
            return

        gdf = gpd.GeoDataFrame(geometry=line_geoms, crs=crs or "EPSG:32631")

        # Reprojectar a WGS84 per compatibilitat amb geojson.io
        gdf = gdf.to_crs(epsg=4326)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Línia de costa (filtrada i neta) exportada a {output_path} (EPSG:4326).")


# ----------------------------------------------------------------------
# Exportar línia de costa a CSV
# ----------------------------------------------------------------------
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
