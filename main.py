"""
Script principal del projecte CDIO.

Flux complet:
1️⃣ Carrega l'AOI i la configuració des de 'config.yaml'.
2️⃣ Consulta i descarrega imatges Sentinel-2 des de l'API STAC.
3️⃣ Calcula NDWI i genera màscares d’aigua.
4️⃣ Estima la línia de costa i exporta resultats a GeoJSON i CSV.
"""

import os
import yaml
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.colors as mcolors
from rasterio.mask import mask

# Imports propis del projecte
from src.visualization.visualize import show_image, show_rgb
from src.processing.indices import compute_ndwi, detect_waterbody
from src.processing.stac_download import load_aoi, query_stac, select_items, download_images_multithread
from src.processing.coastline import estimate_coastline, export_coastline_geojson, export_coastline_csv
from src.utils.save_geotiff import save_geotiff
from src.utils.logger import get_logger

# Configuració del logger principal
logger = get_logger("main")

# Colormap personalitzat: terra (negre) i aigua (blau)
water_cmap = mcolors.ListedColormap(["black", "blue"])


# ----------------------------------------------------------------------
# Funció auxiliar per retallar una banda al AOI
# ----------------------------------------------------------------------
def clip_to_aoi(band_path: str, aoi_file: str):
    """
    Retalla una banda Sentinel-2 segons l'AOI especificada.

    Paràmetres
    ----------
    band_path : str
        Ruta a la banda Sentinel-2 (.tif).
    aoi_file : str
        Ruta al fitxer .geojson amb l'AOI.

    Retorna
    -------
    tuple
        (imatge retallada, transformació, CRS, màscara)
    """
    with rasterio.open(band_path) as src:
        # Llegim AOI i la reproyectem al CRS del raster
        aoi = gpd.read_file(aoi_file)
        aoi = aoi.to_crs(src.crs)

        # Retalla la imatge al AOI
        out_image, out_transform = mask(src, aoi.geometry, crop=True, filled=False)

    return out_image[0].data, out_transform, src.crs, out_image[0].mask


# ----------------------------------------------------------------------
# Funció principal
# ----------------------------------------------------------------------
def main():
    """Execució principal del pipeline."""
    # 0️⃣ Carregar configuració del fitxer YAML
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1️⃣ Carregar AOI
    aoi = load_aoi(config["aoi_file"])

    # 2️⃣ Fer consulta STAC
    items = query_stac(
        aoi,
        config["date_start"],
        config["date_end"],
        limit=20,
        max_cloud=config["max_cloud"]
    )
    logger.info(f"S'han trobat {len(items)} imatges disponibles.")

    # 3️⃣ Seleccionar N imatges
    selected = select_items(items, n=config["n_images"])
    logger.info(f"S'han seleccionat {len(selected)} imatges per al processament.")

    # 4️⃣ Descarregar bandes en paral·lel
    out_dir = config["out_dir"]
    download_images_multithread(
        selected,
        out_dir=out_dir,
        max_workers=config["max_workers"]
    )

    # Carpeta per resultats de línia de costa
    coast_dir = config.get("coastline_out", "outputs")
    os.makedirs(coast_dir, exist_ok=True)

    # 5️⃣ Processar cada imatge seleccionada
    for item in selected:
        date = item["properties"]["datetime"][:10]
        logger.info(f"Processant imatge del {date}")

        # Rutes de bandes necessàries
        band_paths = {
            "red": os.path.join(out_dir, f"{date}_red.tif"),
            "green": os.path.join(out_dir, f"{date}_green.tif"),
            "blue": os.path.join(out_dir, f"{date}_blue.tif"),
            "nir": os.path.join(out_dir, f"{date}_nir.tif"),
        }

        # Llegir i retallar bandes al AOI
        bands = {}
        for name, path in band_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Falta la banda {name.upper()} ({path}), es salta aquesta imatge.")
                bands[name] = None
                continue

            try:
                bands[name], transform, crs, mask_array = clip_to_aoi(path, config["aoi_file"])
            except Exception as e:
                logger.error(f"Error retallant {name.upper()} ({path}): {e}")
                bands[name] = None

        # Si falta alguna banda, saltem la imatge
        if any(band is None for band in bands.values()):
            logger.warning(f"Imatge {date} incompleta o corrupta, es salta.")
            continue

        red, green, blue, nir = bands["red"], bands["green"], bands["blue"], bands["nir"]

        # Visualització de bandes individuals
        show_image(red,   title=f"{date} - Banda Vermella (RED)", cmap="Reds")
        show_image(green, title=f"{date} - Banda Verda (GREEN)",  cmap="Greens")
        show_image(blue,  title=f"{date} - Banda Blava (BLUE)",   cmap="Blues")

        # Composició RGB
        show_rgb(red, green, blue, title=f"{date} - Composició RGB (True Color)")

        # Càlcul NDWI (aigua vs terra)
        ndwi = compute_ndwi(green, nir)
        show_image(ndwi, title=f"{date} - NDWI (Verd vs NIR)", cmap="RdYlBu")

        # Crear màscara binària d'aigua
        waterbody = detect_waterbody(ndwi).astype("float32")

        # Fora AOI = NaN (per transparència a la gràfica)
        waterbody[mask_array] = np.nan
        show_image(waterbody, title=f"{date} - Waterbody (Aigua en blau, Terra en negre)", cmap=water_cmap)

        # Guardar GeoTIFF de màscara
        water_path = os.path.join(out_dir, f"{date}_waterbody.tif")
        save_geotiff(water_path, waterbody, transform, crs)
        logger.info(f"Waterbody guardat a {water_path}")

        # --------------------------------------------------------
        # NOVETAT: retallem segons AOI per evitar línies del retall
        # --------------------------------------------------------
        aoi_gdf = gpd.read_file(config["aoi_file"])
        with rasterio.open(water_path) as src:
            aoi_mask, _ = mask(src, aoi_gdf.geometry, crop=False)
            aoi_mask = aoi_mask[0].astype(bool)
        waterbody = np.where(aoi_mask, waterbody, np.nan)
        # --------------------------------------------------------

        # Estimar línia de costa
        water_mask = np.nan_to_num(waterbody, nan=0).astype("uint8")
        coastline_mask = estimate_coastline(
            water_mask,
            aoi_path=config["aoi_file"],
            reference_raster=water_path
        )
        logger.info(f"Línia de costa estimada correctament per {date}")

        # Exportar resultats
        geojson_out = os.path.join(coast_dir, f"{date}_coastline.geojson")
        csv_out = os.path.join(coast_dir, f"{date}_coastline.csv")

        export_coastline_geojson(coastline_mask, reference_raster=water_path, output_path=geojson_out)
        export_coastline_csv(coastline_mask, reference_raster=water_path, output_path=csv_out, date=date)

        logger.info(f"Resultats exportats a {coast_dir}")


# ----------------------------------------------------------------------
# Execució
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
