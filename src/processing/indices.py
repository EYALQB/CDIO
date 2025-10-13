import numpy as np

def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calcula l'índex NDWI (Normalized Difference Water Index).

    Fórmula:
        NDWI = (BandaVerda - BandaNIR) / (BandaVerda + BandaNIR)

    Paràmetres
    ----------
    green_band : np.ndarray
        Banda verda de la imatge (reflectància o valors digitals).
    nir_band : np.ndarray
        Banda NIR (infraroig proper) de la imatge.

    Retorna
    -------
    np.ndarray
        Matriu NDWI amb valors entre -1 i 1.
        Retorna NaN si el denominador és 0 o si hi ha valors faltants.
    """

    # --- 1️⃣ Comprovació bàsica de dimensions
    if green_band.shape != nir_band.shape:
        raise ValueError("Les dues bandes (Green i NIR) han de tenir la mateixa mida.")

    # --- 2️⃣ Conversió a float32 per evitar saturacions
    g = green_band.astype(np.float32, copy=True)
    n = nir_band.astype(np.float32, copy=True)

    # --- 3️⃣ Càlcul del numerador i denominador
    denom = g + n
    num = g - n

    # --- 4️⃣ Divisió segura (evita dividir per zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denom != 0.0, num / denom, np.nan)

    # --- 5️⃣ Propagar NaN si alguna de les bandes té valors faltants
    mask_nan = np.isnan(g) | np.isnan(n)
    ndwi[mask_nan] = np.nan

    return ndwi


def detect_waterbody(ndwi_array: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Converteix una imatge NDWI en una màscara binària d'aigua/terra.

    Paràmetres
    ----------
    ndwi_array : np.ndarray
        Matriu NDWI (valors entre -1 i 1).
    threshold : float, opcional
        Valor llindar per decidir què és aigua (per defecte = 0.0).

    Retorna
    -------
    np.ndarray
        Màscara binària (1 = aigua, 0 = terra).
    """
    return (ndwi_array > threshold).astype(np.uint8)
