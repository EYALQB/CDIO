import os
import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, mapping
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np

GROUND_TRUTH_LAT = 17.034261
GROUND_TRUTH_LON = 78.183078
MAX_CLOUDS = 5
AREA_SIZE_M = 100
OUTPUT_DIR = "output_stac"
BANDS = ["green", "blue", "red", "nir"]
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dates = [
    "2024-01-15T00:00:00Z",
    "2024-02-15T00:00:00Z",
    "2024-03-15T00:00:00Z",
    "2024-04-15T00:00:00Z",
    "2024-05-15T00:00:00Z",
    "2024-06-15T00:00:00Z",
    "2024-07-15T00:00:00Z",
    "2024-08-15T00:00:00Z",
    "2024-09-15T00:00:00Z",
    "2024-10-15T00:00:00Z"
]

platforms = ["sentinel-2a", "sentinel-2b", "sentinel-2c"]

def search_s2_l2a(lat, lon, date_str, max_clouds, platform_name=None):
    body = {
        "collections": ["sentinel-2-l2a"],
        "filter": {
            "op": "and",
            "args": [
                {"op": "lt", "args":[{"property": "eo:cloud_cover"}, max_clouds]},
                {"op": "eq", "args":[{"property":"platform"}, platform_name]} if platform_name else {},
                {"op": "between", "args":[{"property": "datetime"}, date_str, date_str]}
            ]
        },
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "limit": 1
    }
    r = requests.post(STAC_URL, json=body)
    r.raise_for_status()
    data = r.json()
    if "features" not in data or len(data["features"]) == 0:
        return None
    return data["features"][0]

def download_asset_http(url, out_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def crop_tiff(input_tif, output_tif, lat, lon, size_m):
    with rasterio.open(input_tif) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        h = size_m / 2
        poly = Polygon([(x-h, y-h), (x+h, y-h), (x+h, y+h), (x-h, y+h)])
        geo = [mapping(poly)]
        try:
            out_img, out_tr = mask(src, geo, crop=True)
        except ValueError:
            out_img = src.read()
            out_tr = src.transform
        meta = src.meta.copy()
        meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_tr})
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(out_img)

def show_geotiff(tif_path, title=""):
    with rasterio.open(tif_path) as src:
        img = src.read(1)
        transform = src.transform
        crs_src = src.crs
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap='gray')
        plt.colorbar(im)
        plt.title(title)
        clicked_coords = []
        def onclick(event):
            if event.inaxes != ax:
                return
            x_pix, y_pix = int(event.xdata), int(event.ydata)
            val = img[y_pix, x_pix]
            avg_val = np.mean(img[max(0, y_pix-1):y_pix+2, max(0, x_pix-1):x_pix+2])
            x_ras, y_ras = rasterio.transform.xy(transform, y_pix, x_pix)
            transformer = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x_ras, y_ras)
            print(f"Clicked pixel: ({x_pix}, {y_pix})")
            print(f"Coordinates (EPSG:4326): ({lat:.6f}, {lon:.6f})")
            print(f"Pixel Value: {val}")
            print(f"Average Value (3x3 window): {avg_val:.2f}")
            clicked_coords.append((lon, lat, val))
            plt.close(fig)
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        if clicked_coords:
            return clicked_coords[0]
        return None

for i, date in enumerate(dates):
    platform_name = platforms[i] if i < len(platforms) else None
    print(f"\nProcesando fecha {date} - Plataforma: {platform_name}")
    item = search_s2_l2a(GROUND_TRUTH_LAT, GROUND_TRUTH_LON, date, MAX_CLOUDS, platform_name)
    if item is None:
        print("No se encontró imagen sin nubes para esta fecha.")
        continue
    date_dir = os.path.join(OUTPUT_DIR, f"date_{i+1}")
    os.makedirs(date_dir, exist_ok=True)
    local_files = {}
    for b in BANDS:
        if b not in item["assets"]:
            continue
        href = item["assets"][b]["href"]
        if href.startswith("s3://"):
            href = href.replace("s3://sentinel-s2-l2a/", "https://sentinel-s2-l2a.s3.amazonaws.com/")
        out = os.path.join(date_dir, f"{b}.jp2" if href.endswith(".jp2") else f"{b}.tif")
        download_asset_http(href, out)
        local_files[b] = out
    for b, path in local_files.items():
        out_path = path.replace(".jp2", "_crop.tif").replace(".tif", "_crop.tif")
        crop_tiff(path, out_path, GROUND_TRUTH_LAT, GROUND_TRUTH_LON, AREA_SIZE_M)
    first_band_crop = list(local_files.values())[0].replace(".jp2", "_crop.tif").replace(".tif", "_crop.tif")
    clicked = show_geotiff(first_band_crop, title=f"Fecha {date} - Banda para clic")
    if clicked:
        clicked_lon, clicked_lat, val = clicked
        delta_lat = abs(clicked_lat - GROUND_TRUTH_LAT)
        delta_lon = abs(clicked_lon - GROUND_TRUTH_LON)
        print(f"Ground truth: lat={GROUND_TRUTH_LAT}, lon={GROUND_TRUTH_LON}")
        print(f"Coordenada clicada: lat={clicked_lat:.6f}, lon={clicked_lon:.6f}")
        print(f"Diferencia: Δlat={delta_lat:.6f}, Δlon={delta_lon:.6f}")
