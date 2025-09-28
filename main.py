import os
import yaml
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask

from src.visualize import show_image, show_rgb
from src.indices import compute_ndwi
from src.stac_download import load_aoi, query_stac, select_items, download_images_multithread
from src.utils.logger import get_logger

logger = get_logger("main")

def clip_to_aoi(band_path, aoi_file):
    """Retalla una banda Sentinel-2 al AOI donat (geojson), amb reprojecció."""
    with rasterio.open(band_path) as src:
        # Llegim AOI i el convertim al CRS del raster
        aoi = gpd.read_file(aoi_file)
        aoi = aoi.to_crs(src.crs)

        out_image, out_transform = mask(src, aoi.geometry, crop=True)
    return out_image[0]

def main():
    # 0. Carregar configuració des de config.yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Carregar AOI
    aoi = load_aoi(config["aoi_file"])

    # 2. Fer query STAC
    items = query_stac(
        aoi,
        config["date_start"],
        config["date_end"],
        limit=20,
        max_cloud=config["max_cloud"]
    )
    logger.info(f"S'han trobat {len(items)} imatges")

    # 3. Seleccionar N imatges
    selected = select_items(items, n=config["n_images"])
    logger.info(f"S'han seleccionat {len(selected)} imatges")

    # 4. Descarregar en paral·lel
    out_dir = config["out_dir"]
    download_images_multithread(
        selected,
        out_dir=out_dir,
        max_workers=config["max_workers"]
    )

    # 5. Processar cada imatge seleccionada
    for item in selected:
        date = item["properties"]["datetime"][:10]
        logger.info(f"Processant imatge del {date}")

        band_paths = {
            "red": os.path.join(out_dir, f"{date}_red.tif"),
            "green": os.path.join(out_dir, f"{date}_green.tif"),
            "blue": os.path.join(out_dir, f"{date}_blue.tif"),
            "nir": os.path.join(out_dir, f"{date}_nir.tif"),
        }

        # Llegir bandes amb validació i retallar al AOI
        bands = {}
        for name, path in band_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Falta la banda {name.upper()} ({path}), es salta aquesta imatge.")
                bands[name] = None
                continue

            try:
                bands[name] = clip_to_aoi(path, config["aoi_file"]).astype("float32")
            except Exception as e:
                logger.error(f"Error retallant {name.upper()} ({path}): {e}")
                bands[name] = None

        if any(band is None for band in bands.values()):
            logger.warning(f"Imatge {date} incompleta o corrupta, es salta.")
            continue  

        red, green, blue, nir = bands["red"], bands["green"], bands["blue"], bands["nir"]

        # Debug opcional
        for name, arr in bands.items():
            if arr is not None:
                print(
                    f"{date} - {name.upper()} -> shape={arr.shape}, "
                    f"min={arr.min()}, max={arr.max()}, mean={arr.mean()}"
                )

        # Visualització de bandes individuals
        show_image(red,   title=f"{date} - Banda Vermella (RED)", cmap="Reds")
        show_image(green, title=f"{date} - Banda Verda (GREEN)",  cmap="Greens")
        show_image(blue,  title=f"{date} - Banda Blava (BLUE)",   cmap="Blues")

        # Composició RGB
        show_rgb(red, green, blue, title=f"{date} - Composició RGB (True Color)")

        # NDWI
        ndwi = compute_ndwi(green, nir)
        show_image(ndwi, title=f"{date} - NDWI (Verd vs NIR)", cmap="RdYlBu")


if __name__ == "__main__":
    main()
