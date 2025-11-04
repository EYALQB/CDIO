import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

# utilitats

def ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def detect_top_frequencies(y, top_k=3):
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    n = len(y)
    if n < 8:
        return []
    freq = np.fft.rfftfreq(n, d=1.0)          # cicles/mes
    spec = np.fft.rfft(y)
    power = np.abs(spec)
    mask = freq > 0
    freq_nz, power_nz = freq[mask], power[mask]
    if len(freq_nz) == 0:
        return []
    idx = np.argsort(power_nz)[::-1][:top_k]
    return list(freq_nz[idx])

def seasonal_design_matrix(t, freqs):
    if len(freqs) == 0:
        return np.zeros((len(t), 0))
    cols = []
    for f in freqs:
        w = 2 * np.pi * f
        cols.append(np.sin(w * t))
        cols.append(np.cos(w * t))
    return np.column_stack(cols)

# models i prediccioo

def fit_step_model(t, y, freqs, grid_points=60):
    X_lin = np.column_stack([np.ones_like(t), t])
    X_sea = seasonal_design_matrix(t, freqs)
    X_base = np.column_stack([X_lin, X_sea]) if X_sea.size else X_lin

    lo, hi = float(np.min(t)), float(np.max(t))
    span = hi - lo
    eps = 0.05 * span if span > 0 else 1.0
    ts_grid = np.linspace(lo + eps, hi - eps, grid_points) if span > 0 else np.array([lo + 1e-6])

    best = {"rmse": np.inf}
    for ts in ts_grid:
        H = (t >= ts).astype(float).reshape(-1, 1)
        X = np.column_stack([X_base, H])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        e = rmse(y, yhat)
        if e < best["rmse"]:
            best = {"coef": coef, "t_step": float(ts), "yhat": yhat, "rmse": e}
    return best

def predict_step(t_future, fit, freqs):
    a0 = fit["coef"][0]
    a1 = fit["coef"][1]
    sea_coef = fit["coef"][2:-1] if len(freqs) > 0 else np.array([])
    A = fit["coef"][-1]
    t_step = fit["t_step"]

    X_sea_f = seasonal_design_matrix(t_future, freqs)
    sea_f = X_sea_f @ sea_coef if X_sea_f.size else 0.0
    Hf = (t_future >= t_step).astype(float)
    return a0 + a1 * t_future + A * Hf + sea_f

def fit_sigmoid_model(t, y, freqs):
    X_lin = np.column_stack([np.ones_like(t), t])
    X_sea = seasonal_design_matrix(t, freqs)
    X_ls = np.column_stack([X_lin, X_sea]) if X_sea.size else X_lin

    t0_init = float(np.median(t))
    w_init = max(1.0, 0.1 * (np.max(t) - np.min(t)))

    def residuals(theta):
        t0, logw = theta
        w = np.exp(logw)
        s = 1.0 / (1.0 + np.exp(-(t - t0) / w))
        X = np.column_stack([X_ls, s.reshape(-1, 1)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - (X @ coef)

    theta0 = np.array([t0_init, np.log(w_init)], dtype=float)
    res = least_squares(residuals, theta0, method="trf")
    t0_hat, logw_hat = res.x
    w_hat = float(np.exp(logw_hat))

    s = 1.0 / (1.0 + np.exp(-(t - t0_hat) / w_hat))
    X = np.column_stack([X_ls, s.reshape(-1, 1)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef

    return {"coef": coef, "t0": float(t0_hat), "w": float(w_hat), "yhat": yhat}

def predict_sigmoid(t_future, fit, freqs):
    a0 = fit["coef"][0]
    a1 = fit["coef"][1]
    sea_coef = fit["coef"][2:-1] if len(freqs) > 0 else np.array([])
    A = fit["coef"][-1]
    t0 = fit["t0"]
    w = fit["w"]

    X_sea_f = seasonal_design_matrix(t_future, freqs)
    sea_f = X_sea_f @ sea_coef if X_sea_f.size else 0.0
    sig_f = 1.0 / (1.0 + np.exp(-(t_future - t0) / w))
    return a0 + a1 * t_future + A * sig_f + sea_f

# intervals

def residual_based_ci(y, yhat, y_fore, alpha=0.05):
    # Asume residuos iid. Intervalo 95%: y_fore ± 1.96 * sigma_res
    res = y - yhat
    sigma = float(np.std(res, ddof=1)) if len(res) > 1 else 0.0
    z = 1.96
    lower = y_fore - z * sigma
    upper = y_fore + z * sigma
    return lower, upper, sigma

# visualization

def plot_forecast(dates_hist, y_hist, dates_fore, y1, lo1, up1, y2, lo2, up2, outpath):
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates_hist, y_hist, label="Observado", linewidth=1.5, color="black")
    plt.plot(dates_fore, y1, label="Modelo 1 (step)", linewidth=1.8, color="tab:blue")
    plt.fill_between(dates_fore, lo1, up1, alpha=0.2, label="IC 95% step", color="tab:blue")
    plt.plot(dates_fore, y2, label="Modelo 2 (sigmoid)", linewidth=1.8, color="tab:orange")
    plt.fill_between(dates_fore, lo2, up2, alpha=0.2, label="IC 95% sigmoid", color="tab:orange")
    plt.title("Pronóstico ene–jun 2025 con bandas de confianza")
    plt.xlabel("Fecha")
    plt.ylabel("Distancia (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

# main

def main():
    ensure_dirs()

    data_path = Path("data/shoreline_monthly.csv")
    if not data_path.exists():
        raise FileNotFoundError("No se encontró 'data/shoreline_monthly.csv'.")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    y = df["shoreline"].values.astype(float)
    dates = df["date"].values
    t = np.arange(len(y), dtype=float)

    freqs = detect_top_frequencies(y, top_k=3)

    m1 = fit_step_model(t, y, freqs, grid_points=60)
    m2 = fit_sigmoid_model(t, y, freqs)

    # Definir pronòstic gener - juny 2025
    start_fore = pd.Timestamp("2025-01-01")
    end_fore = pd.Timestamp("2025-06-01")
    dates_fore = pd.date_range(start_fore, end_fore, freq="MS")
    # Construir malla temporal extendida en mesos desde el inicio
    # Suponemos que df es mensual tipo 'MS' desde el inicio
    last_date = pd.to_datetime(df["date"].iloc[-1])
    # Numero de meses observados
    n_obs = len(df)
    # index futurs
    t_future = np.arange(n_obs, n_obs + len(dates_fore), dtype=float)

    y1_fore = predict_step(t_future, m1, freqs)
    y2_fore = predict_sigmoid(t_future, m2, freqs)

    lo1, up1, sigma1 = residual_based_ci(y, m1["yhat"], y1_fore, alpha=0.05)
    lo2, up2, sigma2 = residual_based_ci(y, m2["yhat"], y2_fore, alpha=0.05)

    # Guardarem csv amb prediccions i intervals
    out_csv = Path("outputs/predictions_2025_H1.csv")
    out_df = pd.DataFrame({
        "date": dates_fore,
        "model1_pred": y1_fore,
        "model1_lower": lo1,
        "model1_upper": up1,
        "model2_pred": y2_fore,
        "model2_lower": lo2,
        "model2_upper": up2
    })
    out_df.to_csv(out_csv, index=False)

    # Visualitzem
    plot_forecast(
        dates_hist=df["date"].values,
        y_hist=y,
        dates_fore=dates_fore,
        y1=y1_fore, lo1=lo1, up1=up1,
        y2=y2_fore, lo2=lo2, up2=up2,
        outpath="outputs/figures/forecast_2025H1.png"
    )

    # Resum
    rmse1 = rmse(y, m1["yhat"])
    rmse2 = rmse(y, m2["yhat"])
    r21 = r2_score(y, m1["yhat"])
    r22 = r2_score(y, m2["yhat"])

    lines = []
    lines.append("Forecasting Jan - Jun 2025")
    lines.append(f"Model 1 (step): RMSE={rmse1:.4f}, R2={r21:.4f}, sigma_res={sigma1:.4f}")
    lines.append(f"Model 2 (sigmoid): RMSE={rmse2:.4f}, R2={r22:.4f}, sigma_res={sigma2:.4f}")
    lines.append("Arxiu CSV amb prediccions: outputs/predictions_2025_H1.csv")
    lines.append("Figura: outputs/figures/forecast_2025H1.png")
    

    with open("outputs/forecasting_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nFigures guardades en: outputs/figures/")
    print("CSV guardat en: outputs/predictions_2025_H1.csv")

if __name__ == "__main__":
    main()
