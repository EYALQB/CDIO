import os
import glob
import numpy as np
import geopandas as gpd
from shapely.ops import linemerge, unary_union
from shapely.geometry import LineString, MultiLineString, Point
from typing import List, Tuple

# ---- helpers geomètrics ------------------------------------------------

def _to_single_linestring(gdf: gpd.GeoDataFrame) -> LineString:
    """
    Converteix (Multi)LineString(s) a una única LineString contínua.
    - Uneix/merja segments si cal.
    - Assumeix que la geometria resultant segueix la costa en ordre.
    """
    geom = gdf.unary_union
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        # linemerge pot retornar MultiLineString si no hi ha connexió topològica
        if isinstance(merged, LineString):
            return merged
        # escollim el més llarg si en queden diversos
        longest = max(merged.geoms, key=lambda ls: ls.length)
        return longest
    # si fos un polígon accidentalment, en traiem la frontera
    if hasattr(geom, "boundary"):
        b = geom.boundary
        if isinstance(b, LineString):
            return b
        if isinstance(b, MultiLineString):
            return max(b.geoms, key=lambda ls: ls.length)
    raise ValueError("No s'ha pogut convertir la geometria a LineString.")

def _densify(linestring: LineString, spacing: float) -> List[Point]:
    """
    Genera punts equiespaiats (cada 'spacing' metres) al llarg d'una LineString.
    Retorna llista de shapely Point en el mateix CRS de la línia.
    """
    L = linestring.length
    if L == 0:
        return [linestring.interpolate(0.0)]
    dists = np.arange(0.0, L, spacing)
    pts = [linestring.interpolate(d) for d in dists]
    # garanteix afegir l’últim punt si no cau exactament
    if dists.size == 0 or dists[-1] < L:
        pts.append(linestring.interpolate(L))
    return pts

# ---- API principal (PAS 1) ---------------------------------------------

def load_coastlines_as_lines(
    coast_dir: str,
    crs_metric: str = "EPSG:32631",
) -> List[Tuple[str, LineString]]:
    """
    Llegeix tots els *_coastline.geojson, projecta a CRS mètric i retorna [(data, LineString)].
    """
    files = sorted(glob.glob(os.path.join(coast_dir, "*_coastline.geojson")))
    if not files:
        raise FileNotFoundError(f"No s'han trobat geojson a: {coast_dir}")

    out = []
    for f in files:
        date = os.path.basename(f).split("_")[0]
        gdf = gpd.read_file(f)
        # assegurem CRS -> mètric
        if gdf.crs is None:
            # si vinguessin en WGS84 sense declarar, assumim EPSG:4326
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf = gdf.to_crs(crs_metric)
        line = _to_single_linestring(gdf)
        out.append((date, line))
    return out

def make_reference_samples(
    lines: List[Tuple[str, LineString]],
    spacing_m: float = 10.0,
    smooth_window: int = 0,
) -> Tuple[str, LineString, List[Point]]:
    """
    Tria la primera línia com a referència, la (opcionalment) suavitza i en treu punts equiespaiats.
    Torna: (date_ref, ref_line, ref_points)
    """
    date_ref, ref_line = lines[0]

    # Suavitzat simple opcional (moving average sobre vertices)
    if smooth_window and smooth_window > 2:
        xs, ys = np.array(ref_line.coords)[:, 0], np.array(ref_line.coords)[:, 1]
        k = smooth_window
        # padding als extrems per no escurçar
        pad = k // 2
        xs_pad = np.pad(xs, (pad, pad), mode="edge")
        ys_pad = np.pad(ys, (pad, pad), mode="edge")
        xs_s = np.convolve(xs_pad, np.ones(k)/k, mode="valid")
        ys_s = np.convolve(ys_pad, np.ones(k)/k, mode="valid")
        ref_line = LineString(np.column_stack([xs_s, ys_s]))

    ref_points = _densify(ref_line, spacing_m)
    return date_ref, ref_line, ref_points

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import math


def _get_perpendicular_transect(line: LineString, point: Point, length: float = 100.0) -> LineString:
    """
    Crea un transecte perpendicular a la línia en un punt donat.

    Paràmetres
    ----------
    line : LineString
        Línia de referència.
    point : Point
        Punt sobre la línia on es vol el transecte.
    length : float
        Longitud total del transecte (m).
    """
    d = line.project(point)
    if d < 1 or d > line.length - 1:
        # Evita extrems (on no hi ha direcció ben definida)
        return None

    # Punt lleugerament abans i després per estimar el vector tangent
    p1 = line.interpolate(d - 1)
    p2 = line.interpolate(d + 1)
    dx, dy = p2.x - p1.x, p2.y - p1.y

    # Vector perpendicular normalitzat
    mag = math.sqrt(dx**2 + dy**2)
    nx, ny = -dy / mag, dx / mag

    half = length / 2.0
    pA = Point(point.x + nx * half, point.y + ny * half)
    pB = Point(point.x - nx * half, point.y - ny * half)

    return LineString([pA, pB])


def compute_transect_intersections(
    ref_line: LineString,
    ref_points: list[Point],
    coastlines: list[tuple[str, LineString]],
    transect_len: float = 100.0,
) -> list[dict]:
    """
    Calcula les interseccions entre transectes perpendiculars i totes les línies de costa.

    Retorna
    -------
    List[dict] amb:
        {
            "transect_id": i,
            "date": data,
            "distance_m": distància signada (+erosió / -acreció),
            "transect": LineString
        }
    """
    results = []
    ref_date, ref_geom = coastlines[0]

    for i, pt in enumerate(ref_points):
        transect = _get_perpendicular_transect(ref_line, pt, length=transect_len)
        if transect is None:
            continue

        # Intersecció amb la línia de referència (sempre 0)
        base_inter = transect.intersection(ref_geom)
        if base_inter.is_empty:
            continue
        base_p = base_inter if isinstance(base_inter, Point) else list(base_inter.geoms)[0]

        # Vector normal del transecte (direcció +y → terra)
        p0, p1 = list(transect.coords)
        nx, ny = p1[0] - p0[0], p1[1] - p0[1]
        norm_vec = np.array([nx, ny]) / np.linalg.norm([nx, ny])

        for date, line in coastlines[1:]:
            inter = transect.intersection(line)
            if inter.is_empty:
                continue
            p = inter if isinstance(inter, Point) else list(inter.geoms)[0]
            dist = base_p.distance(p)

            # signe segons posició (terra o mar)
            v = np.array([p.x - base_p.x, p.y - base_p.y])
            sign = np.sign(np.dot(v, norm_vec))
            dist_signed = dist * sign

            results.append({
                "transect_id": i,
                "date": date,
                "distance_m": dist_signed,
                "transect": transect
            })

    return results

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

def analyze_erosion_accretion(results: list[dict], output_dir: str):
    """
    Guarda resultats d’erosió/acreció i mostra un gràfic.

    Paràmetres
    ----------
    results : list[dict]
        Sortida de compute_transect_intersections().
    output_dir : str
        Carpeta per guardar CSV i figures.
    """
    if not results:
        print("⚠️ Cap resultat per analitzar.")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df_summary = df.groupby("date")["distance_m"].mean().reset_index()
    df_summary.rename(columns={"distance_m": "mean_shift_m"}, inplace=True)

    # Guardar CSV
    csv_out = os.path.join(output_dir, "erosion_accretion_summary.csv")
    df_summary.to_csv(csv_out, index=False)
    print(f"✅ Resultats exportats a {csv_out}")

    # Gràfic temporal (mitjana per data)
    plt.figure(figsize=(8,4))
    plt.plot(df_summary["date"], df_summary["mean_shift_m"], marker="o")
    plt.title("Evolució mitjana de l'erosió/acreció")
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylabel("Desplaçament mitjà (m)")
    plt.xlabel("Data")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Guardar transectes a GeoJSON (opcional)
    transects = gpd.GeoDataFrame(geometry=[r["transect"] for r in results])
    transects.set_crs("EPSG:32631", inplace=True)
    transects.to_file(os.path.join(output_dir, "transects.geojson"), driver="GeoJSON")
