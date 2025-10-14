"""
Mòdul per descarregar imatges Sentinel-2 des d'una STAC API.
Permet:
  - Llegir una AOI (.geojson)
  - Fer una consulta STAC filtrant per dates i núvols
  - Descarregar bandes en paral·lel (red, green, blue, nir)
"""

import os
import json
import requests
import geopandas as gpd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
logger = get_logger(__name__)

STAC_API_URL = "https://earth-search.aws.element84.com/v1"


# ----------------------------------------------------------------------
# 1️⃣ Carregar AOI
# ----------------------------------------------------------------------
def load_aoi(aoi_file: str) -> dict:
    """
    Carrega un AOI des d'un fitxer GeoJSON i assegura que està en EPSG:4326.

    Paràmetres
    ----------
    aoi_file : str
        Ruta al fitxer .geojson

    Retorna
    -------
    dict
        AOI en format GeoJSON (llest per usar en una query STAC)
    """
    gdf = gpd.read_file(aoi_file)

    # Si el CRS no és 4326, el convertim
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    logger.info(f"AOI carregat correctament: {aoi_file}")
    return gdf.__geo_interface__


# ----------------------------------------------------------------------
# 2️⃣ Fer consulta a STAC API
# ----------------------------------------------------------------------
def query_stac(
    aoi: Dict,
    start_date: str,
    end_date: str,
    limit: int = 10,
    max_cloud: int = 20
) -> List[Dict]:
    """
    Fa una consulta a l'API STAC d'Element84 (AWS) i retorna una llista d'items disponibles.

    Paràmetres
    ----------
    aoi : dict
        AOI en format GeoJSON (EPSG:4326).
    start_date : str
        Data inicial (YYYY-MM-DD).
    end_date : str
        Data final (YYYY-MM-DD).
    limit : int, opcional
        Nombre màxim d'imatges a retornar (per defecte 10).
    max_cloud : int, opcional
        Percentatge màxim de núvols permès (per defecte 20).

    Retorna
    -------
    list[dict]
        Llista d'items STAC trobats.
    """
    search_url = f"{STAC_API_URL}/search"

    # Calculem el bounding box de l'AOI
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

    logger.info("Consultant API STAC...")
    response = requests.post(search_url, json=params)
    response.raise_for_status()
    data = response.json()

    features = data.get("features", [])
    if not features:
        logger.warning(" No s'han trobat imatges amb els criteris donats.")
    else:
        logger.info(f"{len(features)} imatges trobades a la STAC API.")

    return features


# ----------------------------------------------------------------------
# 3️⃣ Selecció d'items
# ----------------------------------------------------------------------
def select_items(items: List[Dict], n: int) -> List[Dict]:
    """
    Selecciona els primers N items de la llista retornada per la STAC API.

    Paràmetres
    ----------
    items : list[dict]
        Llista d'items STAC.
    n : int
        Nombre d'items a seleccionar.

    Retorna
    -------
    list[dict]
        Llista reduïda amb els N primers items.
    """
    return items[:n]


# ----------------------------------------------------------------------
# 4️⃣ Descarregar un únic asset amb reintents i validació
# ----------------------------------------------------------------------
def download_asset(url: str, out_path: str, retries: int = 3, min_size: int = 10000) -> None:
    """
    Descarrega un únic asset (.tif) amb reintents i validació de mida mínima.

    Paràmetres
    ----------
    url : str
        URL de l'asset.
    out_path : str
        Ruta local on es desarà el fitxer.
    retries : int, opcional
        Nombre màxim de reintents si falla (per defecte 3).
    min_size : int, opcional
        Mida mínima en bytes per considerar el fitxer vàlid (per defecte 10000).
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) >= min_size:
        logger.info(f" Ja existeix, es salta: {out_path}")
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
                    os.remove(out_path)  # Esborrem fitxer corrupte


# ----------------------------------------------------------------------
# 5️⃣ Descarregar múltiples imatges en paral·lel
# ----------------------------------------------------------------------
def download_images_multithread(items: List[Dict], out_dir: str, max_workers: int = 5) -> None:
    """
    Descarrega les bandes (red, green, blue, nir) de múltiples items en paral·lel.

    Paràmetres
    ----------
    items : list[dict]
        Llista d'items STAC a descarregar.
    out_dir : str
        Carpeta de sortida.
    max_workers : int, opcional
        Nombre màxim de threads en paral·lel (per defecte 5).
    """
    os.makedirs(out_dir, exist_ok=True)
    tasks = []

    logger.info(f"Iniciant descàrrega multithread amb {max_workers} fils...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in items:
            date = item["properties"]["datetime"][:10]
            for band in ["red", "green", "blue", "nir"]:
                asset = item["assets"].get(band)
                if asset is None:
                    logger.warning(f"No existeix asset '{band}' per a la data {date}")
                    continue
                url = asset["href"]
                out_path = os.path.join(out_dir, f"{date}_{band}.tif")
                tasks.append(executor.submit(download_asset, url, out_path))

        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error en un task de descàrrega: {e}")

    logger.info("Descàrrega finalitzada correctament")
