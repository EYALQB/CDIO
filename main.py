"""
Script principal del projecte CDIO.

Flux complet:
1️⃣ Carrega l'AOI i la configuració des de 'config.yaml'.
2️⃣ Consulta i descarrega imatges Sentinel-2 des de l'API STAC.
3️⃣ Calcula NDWI i genera màscares d’aigua.
4️⃣ Estima la línia de costa i exporta resultats a GeoJSON i CSV.
5️⃣ Calcula l'error RMSE entre línies de costa i mostra una gràfica.
"""

import os
import glob
import yaml
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.mask import mask
from shapely.ops import nearest_points

# Imports propis del projecte
from src.visualization.visualize import show_image, show_rgb
from src.processing.indices import compute_ndwi, detect_waterbody
from src.processing.stac_download import load_aoi, query_stac, select_items, download_images_multithread
from src.processing.coastline import estimate_coastline, export_coastline_geojson, export_coastline_csv
from src.processing.filters import clean_water_mask
from src.utils.save_geotiff import save_geotiff
from src.utils.logger import get_logger
from src.analysis.shoreline_change import (
    load_coastlines_as_lines,
    make_reference_samples,
    compute_transect_intersections,
    analyze_erosion_accretion
)

logger = get_logger("main")
water_cmap = mcolors.ListedColormap(["black", "blue"])

coastlines = load_coastlines_as_lines("data/outputs")  # o la carpeta on tens els geojson
date_ref, ref_line, ref_pts = make_reference_samples(coastlines, spacing_m=10, smooth_window=5)

results = compute_transect_intersections(ref_line, ref_pts, coastlines, transect_len=100)
analyze_erosion_accretion(results, "outputs/erosion_analysis")

# ----------------------------------------------------------------------
# Funció auxiliar per retallar una banda al AOI
# ----------------------------------------------------------------------
def clip_to_aoi(band_path: str, aoi_file: str):
    """Retalla una banda Sentinel-2 segons l'AOI especificada."""
    with rasterio.open(band_path) as src:
        aoi = gpd.read_file(aoi_file).to_crs(src.crs)
        out_image, out_transform = mask(src, aoi.geometry, crop=True, filled=False)
    return out_image[0].data, out_transform, src.crs, out_image[0].mask


# ----------------------------------------------------------------------
# Funció auxiliar per aplicar un marge interior (padding)
# ----------------------------------------------------------------------
def apply_inner_padding_nan(arr: np.ndarray, px_top=5, px_left=5, px_right=5):
    """Posa NaN a una franja interior (superior i laterals) per eliminar contorns."""
    out = arr.copy()
    h, w = out.shape
    if px_top > 0:
        out[:px_top, :] = np.nan
    if px_left > 0:
        out[:, :px_left] = np.nan
    if px_right > 0:
        out[:, w - px_right:] = np.nan
    return out


# ----------------------------------------------------------------------
# Anàlisi de línies de costa (RMSE)
# ----------------------------------------------------------------------
def analyze_coastlines(coastlines_dir: str, output_csv: str):
    """Analitza totes les línies de costa en una carpeta, calcula RMSE i genera un gràfic."""
    files = sorted(glob.glob(os.path.join(coastlines_dir, "*_coastline.geojson")))
    if len(files) < 2:
        logger.warning("No hi ha prou línies de costa per calcular RMSE.")
        return

    coastlines = [gpd.read_file(f) for f in files]
    dates = [os.path.basename(f).split("_")[0] for f in files]
    base = coastlines[0].unary_union  # línia de referència (primer dia)
    rmse_list = []

    for i, gdf in enumerate(coastlines):
        distances = []
        for geom in gdf.geometry:
            pts = geom.coords
            for x, y in pts:
                p = gpd.GeoSeries([gpd.points_from_xy([x], [y])[0]])
                nearest = nearest_points(p.iloc[0], base)[1]
                dist = p.distance(nearest).values[0]
                distances.append(dist)
        rmse = np.sqrt(np.mean(np.square(distances)))
        rmse_list.append(rmse)

    # Crear DataFrame i ordenar cronològicament
    df = pd.DataFrame({"Date": dates, "RMSE_m": rmse_list}).sort_values("Date")
    df.to_csv(output_csv, index=False)
    logger.info(f"Resultats RMSE exportats a {output_csv}")

    # Gràfic
    plt.figure(figsize=(8, 4))
    plt.plot(df["Date"], df["RMSE_m"], marker="o")
    plt.title("Evolució de l'error RMSE de la línia de costa")
    plt.xlabel("Data")
    plt.ylabel("RMSE (metres)")
    plt.grid(True)
    plt.tight_layout()


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
        aoi, config["date_start"], config["date_end"],
        limit=20, max_cloud=config["max_cloud"]
    )
    logger.info(f"S'han trobat {len(items)} imatges disponibles.")

    # 3️⃣ Seleccionar N imatges
    selected = select_items(items, n=config["n_images"])
    logger.info(f"S'han seleccionat {len(selected)} imatges per al processament.")

    # 4️⃣ Descarregar bandes
    out_dir = config["out_dir"]
    download_images_multithread(selected, out_dir=out_dir, max_workers=config["max_workers"])

    # 5️⃣ Carpeta de sortida per línies de costa
    coast_dir = config.get("coastline_out", "outputs")
    os.makedirs(coast_dir, exist_ok=True)

    # 6️⃣ Processar cada imatge
    for item in selected:
        date = item["properties"]["datetime"][:10]
        logger.info(f"Processant imatge del {date}")

        band_paths = {
            "red": os.path.join(out_dir, f"{date}_red.tif"),
            "green": os.path.join(out_dir, f"{date}_green.tif"),
            "blue": os.path.join(out_dir, f"{date}_blue.tif"),
            "nir": os.path.join(out_dir, f"{date}_nir.tif"),
        }

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

        if any(b is None for b in bands.values()):
            continue

        red, green, blue, nir = bands["red"], bands["green"], bands["blue"], bands["nir"]

        # NDWI
        ndwi = compute_ndwi(green, nir)
        show_image(ndwi, title=f"{date} - NDWI", cmap="RdYlBu")

        # Waterbody netejat
        waterbody = detect_waterbody(ndwi).astype(np.uint8)
        waterbody = clean_water_mask(waterbody, min_blob_px=500).astype("float32")
        waterbody[mask_array] = np.nan
        show_image(waterbody, title=f"{date} - Waterbody net", cmap=water_cmap)

        # Guardar màscara
        water_path = os.path.join(out_dir, f"{date}_waterbody.tif")
        save_geotiff(water_path, waterbody, transform, crs)

        # Aplicar marge interior
        with rasterio.open(water_path) as src:
            px_size = abs(src.transform.a)
        waterbody = apply_inner_padding_nan(
            waterbody,
            px_top=int(round(40 / px_size)),
            px_left=int(round(25 / px_size)),
            px_right=int(round(25 / px_size)),
        )

        # Línia de costa
        water_mask = np.nan_to_num(waterbody, nan=0).astype("uint8")
        coastline_mask = estimate_coastline(water_mask, aoi_path=config["aoi_file"], reference_raster=water_path)

        # Exportar resultats
        geojson_out = os.path.join(coast_dir, f"{date}_coastline.geojson")
        csv_out = os.path.join(coast_dir, f"{date}_coastline.csv")
        export_coastline_geojson(coastline_mask, reference_raster=water_path, output_path=geojson_out)
        export_coastline_csv(coastline_mask, reference_raster=water_path, output_path=csv_out, date=date)
        logger.info(f"Resultats exportats a {coast_dir}")

    # 7️⃣ Analitzar RMSE entre línies de costa
    output_rmse_csv = os.path.join(coast_dir, "coastline_rmse.csv")
    analyze_coastlines(coast_dir, output_rmse_csv)

    # 8️⃣ Mostrar totes les figures (imatges + gràfiques)
    plt.show()


# ----------------------------------------------------------------------
# Execució
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
