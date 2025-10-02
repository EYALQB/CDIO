import rasterio
import numpy as np

def save_geotiff(output_path, array, transform, crs):
    """
    Desa un array 2D com a GeoTIFF amb la mateixa referència espacial.
    Els valors NaN es guarden com a 'nodata'.
    """
    # Definir el valor de nodata (per als NaN)
    nodata_value = -9999  

    # Si hi ha NaN, substituïm temporalment per nodata_value
    array_to_save = np.where(np.isnan(array), nodata_value, array)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_value   # <- marca oficialment el valor com a nodata
    ) as dst:
        dst.write(array_to_save, 1)
