import os
import rasterio
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_geotiff(output_path: str, array: np.ndarray, transform, crs):
    """
    Desa un array 2D com a fitxer GeoTIFF amb la seva referència espacial.

    Els valors NaN es converteixen automàticament en un valor 'nodata' (-9999)
    i es marca al fitxer perquè altres programes (com QGIS) el reconeguin.

    Paràmetres
    ----------
    output_path : str
        Ruta completa on es desarà el fitxer GeoTIFF.
    array : np.ndarray
        Matriu 2D amb els valors a escriure.
    transform : Affine
        Transformació afí del raster (coordenades).
    crs : dict o str
        Sistema de coordenades (p. ex. 'EPSG:32631').
    """

    # --- 1️⃣ Crear carpeta de sortida si no existeix
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- 2️⃣ Definir valor 'nodata' per substituir els NaN
    nodata_value = -9999

    # --- 3️⃣ Substituir NaN per nodata_value
    array_to_save = np.where(np.isnan(array), nodata_value, array)

    # --- 4️⃣ Escriure el GeoTIFF amb Rasterio
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array_to_save.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(array_to_save, 1)

    # --- 5️⃣ Registrar missatge de confirmació
    logger.info(f"Fitxer GeoTIFF desat correctament a: {output_path}")
