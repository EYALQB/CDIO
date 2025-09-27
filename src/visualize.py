import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # o "Qt5Agg" si tens Qt


def show_image(image: np.ndarray, title: str = "Image", cmap: str = "gray"):
    """
    Visualiza una imagen 2D con Matplotlib.

    Parameters
    ----------
    image : np.ndarray
        Imagen a mostrar.
    title : str
        Título del gráfico.
    cmap : str
        Paleta de colores.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="Pixel values")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
