import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # carreguem els csv amb les dades originals
    zip_path = Path("data/shoreline_distances_castefa_gava_prat_2017_2024.csv")
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {zip_path}")

    # llegim el csv
    df = pd.read_csv(zip_path)

    # Convertir la columna de fechas a tipo datetime
    df["date"] = pd.to_datetime(df["date"])

    # Crear tabla pivot: filas = fechas, columnas = transectos, valores = distancia
    pivot_df = df.pivot(index="date", columns="transect_id", values="distance_m")

    # Calcular porcentaje total de valores NaN
    total_nan_pct = pivot_df.isna().mean().mean() * 100
    print(f"\nPorcentaje medio total de NaN: {total_nan_pct:.2f}%")

    # Crear carpetas necesarias antes de guardar figuras o resultados
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # Calcular porcentaje de NaN por transecto
    nan_by_transect = pivot_df.isna().mean() * 100

    # Graficar distribución de NaN por transecto
    plt.figure(figsize=(10, 4))
    nan_by_transect.plot(kind="bar", color="skyblue")
    plt.title("Porcentaje de NaN per transecte")
    plt.ylabel("% de NaN")
    plt.tight_layout()
    plt.savefig("outputs/figures/nan_distribution_by_transect.png", dpi=130)
    plt.close()

    # Filtrar transectes amb menys del 20% de NaN
    filtered_df = pivot_df.loc[:, pivot_df.isna().mean() < 0.2]
    print(f"\nNúmero de transectes retinguts (<20% NaN): {filtered_df.shape[1]}")

    # Calcular la sèrie mitjana entre transectes vàlids
    mean_series = filtered_df.mean(axis=1)

    # mostregem mensualmente y aplicar interpolación lineal
    monthly_series = mean_series.resample("MS").mean().interpolate("linear")

    # Creem DataFrame final amb columnes
    result_df = pd.DataFrame({
        "date": monthly_series.index,
        "shoreline": monthly_series.values
    })

    # Guardar resultat
    output_path = Path("data/shoreline_monthly.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\nArchivo guardado correctamente en: {output_path}")

    #resumen 
    print("\nResum final")
    print(result_df.head())
    print(f"\nNúmero total de registros mensuales: {len(result_df)}")

if __name__ == "__main__":
    main()
