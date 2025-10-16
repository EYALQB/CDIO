"""
Mòdul de visualització d’imatges i índexs NDWI/NDVI.

Conté funcions per:
- Normalitzar bandes Sentinel-2 per visualització
- Mostrar imatges individuals, composicions RGB i màscares binàries
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Paleta de colors per aigua (blau) i terra (negre)
land_color = "black"
water_cmap = mcolors.ListedColormap([land_color, "blue"])


def normalize_band(band: np.ndarray, reflectance_scale: float = 10000.0, clip_pct: tuple = (2, 98)) -> np.ndarray:
    """
    Normalitza una banda Sentinel-2 per visualització RGB.

    Escala els valors de reflectància i retalla els extrems
    segons els percentils especificats per millorar el contrast.

    Paràmetres
    ----------
    band : np.ndarray
        Banda d’entrada (reflectància o valors digitals).
    reflectance_scale : float, opcional
        Factor d’escala per a reflectància (per defecte 10000.0).
    clip_pct : tuple(int, int), opcional
        Percentils per retallar els valors extrems (per defecte 2–98).

    Retorna
    -------
    np.ndarray
        Banda normalitzada amb valors entre 0 i 1.
    """
    # Escalar reflectància
    band = band / reflectance_scale

    # Retallar valors extrems per millorar contrast
    low, high = np.percentile(band, clip_pct)
    if high <= low:
        return np.clip(band, 0, 1)

    band = np.clip(band, low, high)
    return (band - low) / (high - low)


def show_image(image: np.ndarray, title: str = "Imatge", cmap: str = "gray",show_now: bool = False) -> None:
    """
    Mostra una sola banda o índex (NDWI, NDVI...) amb barra de color.

    Paràmetres
    ----------
    image : np.ndarray
        Imatge 2D a mostrar.
    title : str, opcional
        Títol del gràfic.
    cmap : str, opcional
        Colormap de Matplotlib (per defecte 'gray').
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="Valors de píxel")
    plt.axis("off")
    plt.tight_layout()
    #plt.show()
    #if show_now:
        #plt.show()

def show_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, title: str = "Composició RGB (True Color)",show_now=False) -> None:
    """
    Mostra una composició RGB a partir de bandes R, G i B Sentinel-2.

    Paràmetres
    ----------
    r, g, b : np.ndarray
        Bandes vermella, verda i blava de la imatge.
    title : str, opcional
        Títol del gràfic.
    """
    rn = normalize_band(r)
    gn = normalize_band(g)
    bn = normalize_band(b)
    rgb = np.dstack([rn, gn, bn])

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    #if show_now:
     #   plt.show()


def show_waterbody(array: np.ndarray, title: str = "Màscara d’aigua (terra/aigua)") -> None:
    """
    Mostra una màscara binària on 0 = terra i 1 = aigua.

    Paràmetres
    ----------
    array : np.ndarray
        Màscara binària d’aigua/terra.
    title : str, opcional
        Títol del gràfic.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=water_cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    #plt.show()
