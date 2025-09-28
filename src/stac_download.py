"""
Mòdul per descarregar imatges Sentinel-2 des d'un STAC API.
"""

import json
import os
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd

from src.utils.logger import get_logger
logger = get_logger(__name__)

STAC_API_URL = "https://earth-search.aws.element84.com/v1"


# ----------------------------------------------------------------------
# 1. Carregar AOI
# ----------------------------------------------------------------------
import geopandas as gpd

def load_aoi(aoi_file: str) -> dict:
    """
    Carrega un AOI des d'un fitxer .geojson i l'assegura en EPSG:4326 (lat/lon).

    :param aoi_file: Path al fitxer .geojson
    :return: AOI en format GeoJSON (dict) amb CRS EPSG:4326
    """
    gdf = gpd.read_file(aoi_file)

    # Si el CRS no és 4326, el convertim
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    return gdf.__geo_interface__

# ----------------------------------------------------------------------
# 2. Query STAC
# ----------------------------------------------------------------------
def query_stac(
    aoi: Dict,
    start_date: str,
    end_date: str,
    limit: int = 10,
    max_cloud: int = 20
) -> List[Dict]:
    """
    Fa una query a la STAC API i retorna una llista d'items disponibles.

    :param aoi: AOI en format GeoJSON
    :param start_date: Data inicial (YYYY-MM-DD)
    :param end_date: Data final (YYYY-MM-DD)
    :param limit: Nombre màxim d'items a retornar
    :param max_cloud: Percentatge màxim de núvols permès
    """
    search_url = f"{STAC_API_URL}/search"

    # Calculem el bounding box del AOI en EPSG:4326
    import geopandas as gpd
    gdf = gpd.GeoDataFrame.from_features(aoi["features"], crs="EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds

    params = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [minx, miny, maxx, maxy],
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lt": max_cloud}
        }
    }

    logger.debug(f"Query STAC params: {json.dumps(params, indent=2)}")

    response = requests.post(search_url, json=params)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])
    if not features:
        logger.warning("No s'han trobat imatges amb els criteris donats.")
    return features


# ----------------------------------------------------------------------
# 3. Selecció d'items
# ----------------------------------------------------------------------
def select_items(items: List[Dict], n: int) -> List[Dict]:
    """
    Selecciona els primers N items de la llista.
    """
    return items[:n]


# ----------------------------------------------------------------------
# 4. Descarregar un únic asset amb reintents i validació
# ----------------------------------------------------------------------
def download_asset(url: str, out_path: str, retries: int = 3, min_size: int = 10000) -> None:
    """
    Descarrega un únic asset (fitxer .tif) amb reintents i validació de mida.

    :param url: URL de l'asset
    :param out_path: Ruta de sortida
    :param retries: Nombre de reintents si falla la descàrrega
    :param min_size: Mida mínima del fitxer en bytes per considerar-lo vàlid
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) >= min_size:
        logger.info(f"Ja existeix, es salta: {out_path}")
        return

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Validar mida del fitxer
            if os.path.getsize(out_path) < min_size:
                raise ValueError("Fitxer massa petit, pot estar corrupte")

            logger.info(f"[OK] Descarregat: {out_path}")
            return

        except Exception as e:
            logger.error(f"Error descarregant {url} (intent {attempt}/{retries}): {e}")
            if attempt == retries:
                logger.error(f"Error permanent: no s'ha pogut descarregar {url}")
                if os.path.exists(out_path):
                    os.remove(out_path)  # esborrem el fitxer corrupte


# ----------------------------------------------------------------------
# 5. Descarregar diversos items en paral·lel
# ----------------------------------------------------------------------
def download_images_multithread(items: List[Dict], out_dir: str, max_workers: int = 5) -> None:
    """
    Descarrega les bandes red, green, blue i nir de múltiples items en paral·lel.
    """
    os.makedirs(out_dir, exist_ok=True)
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in items:
            date = item["properties"]["datetime"][:10]
            for band in ["red", "green", "blue", "nir"]:
                asset = item["assets"].get(band)
                if asset is None:
                    logger.warning(f"No existeix asset {band} per {date}")
                    continue
                url = asset["href"]
                out_path = os.path.join(out_dir, f"{date}_{band}.tif")
                tasks.append(executor.submit(download_asset, url, out_path))

        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error en un task de descàrrega: {e}")
