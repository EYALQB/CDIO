import numpy as np
import rasterio

def compute_ndwi(green_path, nir_path, output_path):
    """
    Compute NDWI from Sentinel-2 Green and NIR bands.
    
    Parameters
    ----------
    green_path : str
        Path to the Green band GeoTIFF.
    nir_path : str
        Path to the NIR band GeoTIFF.
    output_path : str
        Where to save the NDWI raster.

    Returns
    -------
    str
        Path to the saved NDWI raster.
    """
    # Open bands
    with rasterio.open(green_path) as src_green:
        green = src_green.read(1).astype("float32")
        profile = src_green.profile  # copy metadata

    with rasterio.open(nir_path) as src_nir:
        nir = src_nir.read(1).astype("float32")

    # NDWI formula
    ndwi = (green - nir) / (green + nir)
    ndwi = np.nan_to_num(ndwi, nan=-1)  # avoid NaN issues

    # Save to GeoTIFF
    profile.update(dtype="float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ndwi, 1)

    return output_path