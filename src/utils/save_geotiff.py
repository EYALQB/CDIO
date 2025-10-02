import rasterio

def save_geotiff(output_path, array, transform, crs):
    """
    Desa un array 2D com a GeoTIFF amb la mateixa referència espacial.
    """
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
    ) as dst:
        dst.write(array, 1)
