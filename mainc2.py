import rasterio
from rasterio.transform import xy
from pyproj import Transformer
import matplotlib.pyplot as plt

# ============================================================
#   FUNCIÓN PRINCIPAL PARA HACER 9 CLICKS EN LA IMAGEN
# ============================================================

def show_geotiff_9_clicks(image_path, title="Selecciona 9 puntos"):
    """
    Muestra una imagen GeoTIFF completa, permite seleccionar 9 puntos con el ratón,
    convierte píxeles → coordenadas geográficas, y devuelve la lista de puntos.
    """

    with rasterio.open(image_path) as src:
        img = src.read(1)
        transform = src.transform
        crs_src = src.crs

    # Ajustar rango para mostrar la imagen correctamente
    vmin, vmax = img.min(), img.max()
    if vmin == vmax:
        vmin, vmax = 0, 1  # evitar imagen completamente negra

    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"{title}\nHaz 9 clicks")
    plt.colorbar(im)

    clicked_geo_points = []
    transformer = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)

    def onclick(event):
        nonlocal clicked_geo_points
        if event.inaxes != ax:
            return
        x_pix, y_pix = int(event.xdata), int(event.ydata)
        x_map, y_map = xy(transform, y_pix, x_pix)
        lon, lat = transformer.transform(x_map, y_map)
        clicked_geo_points.append((lon, lat))
        print(f" Punto {len(clicked_geo_points)} → lon={lon:.6f}, lat={lat:.6f}")

        if len(clicked_geo_points) == 9:
            print("\n ✔ Se han registrado los 9 puntos. Cerrando ventana...")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    return clicked_geo_points

# ============================================================
#                   PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # ❗ Cambia esto por la ruta a tu TIFF completo ❗
    IMAGE_PATH = "S2L2Ax10_T31TDF-559314ea8-20251127_MS.tif"

    puntos = show_geotiff_9_clicks(IMAGE_PATH, title="Selecciona los 9 cuadrados EETAC")

    print("\n=== PUNTOS REGISTRADOS (lon, lat) ===")
    for i, p in enumerate(puntos):
        print(f"{i+1}: {p}")
