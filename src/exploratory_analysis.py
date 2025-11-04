import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def main():
    # Cargar la serie mensual generada en el codi de data_preparation
    data_path = Path("data/shoreline_monthly.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    # Crear carpetas de salida si no existen
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    # Exploratory visualization: serie temporal
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["shoreline"], color="steelblue")
    plt.title("Evolución temporal de la línea de costa")
    plt.xlabel("Fecha")
    plt.ylabel("Distancia de la línea de costa (m)")
    plt.tight_layout()
    plt.savefig("outputs/figures/shoreline_over_time.png", dpi=130)
    plt.close()

    # Estadística descriptiva
    print("\nEstadistiques inicials de la sèrie de línia de costa:")
    print(df["shoreline"].describe())

    # Boxplot mensual
    df["month"] = df.index.month
    plt.figure(figsize=(10, 4))
    df.boxplot(column="shoreline", by="month", grid=False)
    plt.title("Distribución mensual de la línea de costa")
    plt.suptitle("")
    plt.xlabel("Mes")
    plt.ylabel("Distancia (m)")
    plt.tight_layout()
    plt.savefig("outputs/figures/monthly_boxplot.png", dpi=130)
    plt.close()

    # Boxplot anual
    df["year"] = df.index.year
    plt.figure(figsize=(10, 4))
    df.boxplot(column="shoreline", by="year", grid=False)
    plt.title("Distribución anual de la línea de costa")
    plt.suptitle("")
    plt.xlabel("Año")
    plt.ylabel("Distancia (m)")
    plt.tight_layout()
    plt.savefig("outputs/figures/yearly_boxplot.png", dpi=130)
    plt.close()

    # Trend analysis: medias móviles
    df["MA_6"] = df["shoreline"].rolling(window=6, center=True).mean()
    df["MA_12"] = df["shoreline"].rolling(window=12, center=True).mean()

    # Regresión lineal simple
    t = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, df["shoreline"])
    trend_line = intercept + slope * t

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["shoreline"], label="Serie original", color="lightgray")
    plt.plot(df.index, df["MA_6"], label="Media móvil 6 meses", color="steelblue")
    plt.plot(df.index, df["MA_12"], label="Media móvil 12 meses", color="orange")
    plt.plot(df.index, trend_line, label="Tendencia lineal", color="red")
    plt.title("Análisis de tendencia")
    plt.xlabel("Fecha")
    plt.ylabel("Distancia (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/trend_analysis.png", dpi=130)
    plt.close()

    # Frequency domain analysis (FFT)
    y = df["shoreline"].values - np.mean(df["shoreline"].values)
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1)  # frecuencia en ciclos/mes
    fft_magnitude = np.abs(np.fft.rfft(y))

    # Identificar frecuencias principales
    dominant_indices = np.argsort(fft_magnitude)[::-1][:5]
    dominant_freqs = freq[dominant_indices]
    dominant_periods = 1 / dominant_freqs[1:]  # evitar el componente DC (freq=0)
    
    print("\nFrecuencias dominantes (FFT)")
    for f, p in zip(dominant_freqs[1:], dominant_periods):
        print(f"Frecuencia: {f:.4f} ciclos/mes  →  Periodo: {p:.2f} meses")

    plt.figure(figsize=(10, 4))
    plt.plot(freq[1:], fft_magnitude[1:], color="purple")
    plt.title("Espectro de frecuencias (FFT)")
    plt.xlabel("Frecuencia (ciclos/mes)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.savefig("outputs/figures/fft_spectrum.png", dpi=130)
    plt.close()

    # Conclusiones exploratorias
    print("\nAnálisis")
    print("• Observem la variabilitat a curt i llarg termini en la figura 'shoreline_over_time.png'.")
    print("• Els boxplots ens mostren si existeixen patrons estacionals (mensuals o anuals).")
    print("• La gràfica 'trend_analysis.png' ens permet identificar una tendència global.")
    print("• L'espectre FFT revela possibles cicles estacionals dominants.")

if __name__ == "__main__":
    main()
