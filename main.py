import os
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from pystac_client import Client

# Import our NDWI function
from src.ndwi import compute_ndwi


def main():
    # -------------------------------
    # Step 1. Load AOI
    # -------------------------------
    aoi_path = "data/polygon.geojson"
    aoi = gpd.read_file(aoi_path)
    bbox = aoi.total_bounds  # minx, miny, maxx, maxy

    # -------------------------------
    # Step 2. Connect to STAC API
    # -------------------------------
    api_url = "https://earth-search.aws.element84.com/v1"
    client = Client.open(api_url)

    search = client.search(
        collections=["sentinel-2-l2a"],
        datetime="2025-01-01/2025-07-01",
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": 10}},  # filter clouds < 10%
        limit=1                                # just 1 image for now
    )

    items = list(search.get_items())
    print(f"Found {len(items)} images")

    if not items:
        raise ValueError("No images found for AOI + filters")

    item = items[0]
    print("Using image:", item.id)

    # -------------------------------
    # Step 3. Download Green & NIR
    # -------------------------------
    def download_band(url, out_path, bbox):

        with rasterio.open(url) as src:
            # Crear ventana con el bounding box del AOI
            window = from_bounds(*bbox, transform=src.transform)
            data = src.read(1, window=window)

            # Actualizar metadatos para el recorte
            profile = src.profile
            profile.update({
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": src.window_transform(window)
            })

        # Guardar el recorte
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        return out_path

    os.makedirs("data/bands", exist_ok=True)

    green_url = item.assets["green"].href
    nir_url   = item.assets["nir"].href

    green_path = download_band(green_url, "data/bands/green.tif")
    nir_path   = download_band(nir_url, "data/bands/nir.tif")

    print("Bands saved:", green_path, nir_path)

    # -------------------------------
    # Step 4. Compute NDWI (with our function)
    # -------------------------------
    ndwi_path = "data/bands/ndwi.tif"
    compute_ndwi(green_path, nir_path, ndwi_path)

    print("NDWI saved at:", ndwi_path)

    # -------------------------------
    # Step 5. Visualize NDWI
    # -------------------------------
    with rasterio.open(ndwi_path) as src:
        ndwi = src.read(1)

    plt.imshow(ndwi, cmap="RdYlBu")
    plt.colorbar(label="NDWI")
    plt.title("NDWI (Green vs NIR)")
    plt.show()


if __name__ == "__main__":
    main()