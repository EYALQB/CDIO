import numpy as np

def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Compute NDWI = (Green - NIR) / (Green + NIR)

    Paràmetres
    ----------
    green_band : np.ndarray
        Banda verda de la imatge (reflectance o digital values).
    nir_band : np.ndarray
        Banda NIR (Near InfraRed) de la imatge.

    Retorna
    -------
    np.ndarray
        Imatge NDWI amb valors entre -1 i 1.
        Retorna NaN si el denominador és 0 o si hi ha valors faltants (missing values).
    """

    g = green_band.astype(np.float32, copy=True)
    n = nir_band.astype(np.float32, copy=True)

    denom = g + n
    num = g - n

    # División segura (evita dividir por cero)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denom != 0.0, num / denom, np.nan)

    # Propaga NaNs de entrada a la salida
    mask_nan = np.isnan(g) | np.isnan(n)
    ndwi[mask_nan] = np.nan

    return ndwi
