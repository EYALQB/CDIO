import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# utilitats

def ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def picp(y_obs, lower, upper):
    inside = (y_obs >= lower) & (y_obs <= upper)
    return float(np.mean(inside) * 100.0)

# visualizagion

def plot_validation(dates, y_obs, y1, lo1, up1, y2, lo2, up2, outpath):
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates, y_obs, label="Observado 2025", color="black", linewidth=1.8)
    plt.plot(dates, y1, label="Predicción Modelo 1 (step)", color="tab:blue")
    plt.fill_between(dates, lo1, up1, color="tab:blue", alpha=0.2, label="IC 95% step")
    plt.plot(dates, y2, label="Predicción Modelo 2 (sigmoid)", color="tab:orange")
    plt.fill_between(dates, lo2, up2, color="tab:orange", alpha=0.2, label="IC 95% sigmoid")
    plt.title("Validación: predicciones vs observaciones (ene–jun 2025)")
    plt.xlabel("Fecha")
    plt.ylabel("Distancia (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

# main

def main():
    ensure_dirs()

    pred_path = Path("outputs/predictions_2025_H1.csv")
    obs_path = Path("data/shoreline_distances_castefa_gava_prat_h1_2025_ref2017.csv")

    if not pred_path.exists():
        raise FileNotFoundError("No se encontró outputs/predictions_2025_H1.csv. Ejecuta antes el paso 5.")
    if not obs_path.exists():
        raise FileNotFoundError(f"No se encontró {obs_path}. Coloca el archivo de observaciones reales en 'data/'.")

    # prediccions
    pred = pd.read_csv(pred_path)
    pred["date"] = pd.to_datetime(pred["date"])

    # observacons reals
    obs = pd.read_csv(obs_path)

    # columnes rellevants
    possible_date_cols = [c for c in obs.columns if "date" in c.lower()]
    possible_dist_cols = [c for c in obs.columns if "dist" in c.lower()]

    if not possible_date_cols or not possible_dist_cols:
        raise ValueError("No se encontraron columnas de fecha o distancia en el archivo de observaciones.")

    obs = obs[[possible_date_cols[0], possible_dist_cols[0]]].copy()
    obs.columns = ["date", "shoreline_obs"]
    obs["date"] = pd.to_datetime(obs["date"])

    # Promediar si hi ha més d'un transecto per data
    obs = obs.groupby("date", as_index=False)["shoreline_obs"].mean()

    # Alinear fechas (mensualment) 
    pred["date"] = pd.to_datetime(pred["date"]).dt.to_period("M").dt.to_timestamp()
    obs["date"] = pd.to_datetime(obs["date"]).dt.to_period("M").dt.to_timestamp()

    df = pred.merge(obs, on="date", how="inner", suffixes=("", "_obs"))

    if df.empty:
        raise ValueError("No hay fechas coincidentes entre predicciones y observaciones tras normalizar al primer día del mes.")

    y_obs = df["shoreline_obs"].values
    dates = df["date"].values

    y1, lo1, up1 = df["model1_pred"].values, df["model1_lower"].values, df["model1_upper"].values
    y2, lo2, up2 = df["model2_pred"].values, df["model2_lower"].values, df["model2_upper"].values

    # calcul mètriques
    rmse1, mae1, picp1 = rmse(y_obs, y1), mae(y_obs, y1), picp(y_obs, lo1, up1)
    rmse2, mae2, picp2 = rmse(y_obs, y2), mae(y_obs, y2), picp(y_obs, lo2, up2)

    # grafiquem
    plot_validation(dates, y_obs, y1, lo1, up1, y2, lo2, up2, "outputs/figures/validation_vs_observed_2025.png")

    # resum
    lines = []
    lines.append("Validation & Discussion (Jan - Jun 2025)")
    lines.append("")
    lines.append("Model 1 (step):")
    lines.append(f"  RMSE = {rmse1:.4f}")
    lines.append(f"  MAE  = {mae1:.4f}")
    lines.append(f"  PICP = {picp1:.2f}%")
    lines.append("")
    lines.append("Model 2 (sigmoid):")
    lines.append(f"  RMSE = {rmse2:.4f}")
    lines.append(f"  MAE  = {mae2:.4f}")
    lines.append(f"  PICP = {picp2:.2f}%")
    lines.append("")
    better_rmse = "Model 1 (step)" if rmse1 < rmse2 else "Model 2 (sigmoid)"
    lines.append(f"→ {better_rmse} presenta menor RMSE.")
    

    with open("outputs/validation_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nFigura guardada en outputs/figures/validation_vs_observed_2025.png")
    print("Resumen guardado en outputs/validation_summary.txt")

if __name__ == "__main__":
    main()
