import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


nigga = 'black'
water_cmap = mcolors.ListedColormap([nigga, "blue"])


def normalize_band(band, reflectance_scale=10000.0, clip_pct=(2, 98)):
    """Normalitza una banda Sentinel-2 per visualització RGB."""
    # Escalar reflectància
    band = band / reflectance_scale

    # Percentils per retallar valors extrems
    low, high = np.percentile(band, clip_pct)
    if high <= low:
        return np.clip(band, 0, 1)

    # Retallar i escalar
    band = np.clip(band, low, high)
    return (band - low) / (high - low)

def show_image(image: np.ndarray, title: str = "Image", cmap: str = "gray"):
    """Mostra una sola banda o índex amb colorbar."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="Pixel values")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, title: str = "Composició RGB (True Color)"):
    """Mostra composició RGB a partir de bandes R, G, B normalitzades."""
    rn = normalize_band(r)
    gn = normalize_band(g)
    bn = normalize_band(b)
    rgb = np.dstack([rn, gn, bn])

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_waterbody(array, title="Waterbody (terra/aigua)"):
    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=water_cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

