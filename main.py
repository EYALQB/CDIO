import rasterio
from src.visualize import show_image
from src.indices import compute_ndwi

def main():
    # Cargar imagen TIFF RGB (cambia el nombre del archivo si es distinto)
    path_imagen = "data/2025-09-23-00_00_2025-09-23-23_59_Sentinel-2_L2A_True_color.tiff"

    with rasterio.open(path_imagen) as src:
        # En un TIFF RGB típico: banda 1=Red, 2=Green, 3=Blue
        red   = src.read(1).astype("float32")
        green = src.read(2).astype("float32")
        blue  = src.read(3).astype("float32")

    # Visualizar bandas individuales
    show_image(red,   title="Banda Vermella",   cmap="Reds")
    show_image(green, title="Banda Verda", cmap="Greens")
    show_image(blue,  title="Banda blava",  cmap="Blues")

    # Calcular NDWI (experimental, usando Red como NIR falso)
    ndwi = compute_ndwi(green, red)
    show_image(ndwi, title="NDWI (Verd vs Vermell)", cmap="RdYlBu")

if __name__ == "__main__":
    main()
