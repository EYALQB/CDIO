import numpy as np
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label

def clean_water_mask(mask: np.ndarray, min_blob_px: int = 500) -> np.ndarray:
    """
    Neteja una màscara binària d'aigua (1=aigua, 0=terra) eliminant soroll,
    forats petits i vores irregulars.

    Paràmetres
    ----------
    mask : np.ndarray
        Màscara binària (0 i 1) d’aigua.
    min_blob_px : int
        Nombre mínim de píxels per mantenir una massa d’aigua.

    Retorna
    -------
    np.ndarray
        Màscara neta i suau (uint8).
    """
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)

    mask = (mask > 0).astype(np.uint8)

    # 1️⃣ Eliminar blobs petits
    labeled, n = label(mask, structure=structure)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = np.where(sizes < min_blob_px)[0]
        mask[np.isin(labeled, small)] = 0

    # 2️⃣ Suavitzar i omplir forats petits
    mask = binary_opening(mask, structure=structure, iterations=1)
    mask = binary_closing(mask, structure=structure, iterations=2)
    mask = binary_fill_holes(mask)

    return mask.astype(np.uint8)
