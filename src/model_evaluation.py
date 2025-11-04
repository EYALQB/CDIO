import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from scipy.optimize import least_squares

# utilitats

def ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def detect_top_frequencies(y, top_k=3):
    y = y - np.mean(y)
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1.0)
    spec = np.fft.rfft(y)
    power = np.abs(spec)
    mask = freq > 0
    freq_nz, power_nz = freq[mask], power[mask]
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

# models

def fit_step_model(t, y, freqs):
    X_lin = np.column_stack([np.ones_like(t), t])
    X_sea = seasonal_design_matrix(t, freqs)
    X_base = np.column_stack([X_lin, X_sea]) if X_sea.size else X_lin
    lo, hi = float(np.min(t)), float(np.max(t))
    grid = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 60)
    best = {"rmse": np.inf}
    for ts in grid:
        H = (t >= ts).astype(float).reshape(-1, 1)
        X = np.column_stack([X_base, H])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        e = rmse(y, yhat)
        if e < best["rmse"]:
            best = {"coef": coef, "t_step": ts, "yhat": yhat, "rmse": e}
    return best

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
        X = np.column_stack([X_ls, s])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - (X @ coef)
    theta0 = np.array([t0_init, np.log(w_init)])
    res = least_squares(residuals, theta0, method="trf")
    t0_hat, logw_hat = res.x
    w_hat = float(np.exp(logw_hat))
    s = 1.0 / (1.0 + np.exp(-(t - t0_hat) / w_hat))
    X = np.column_stack([X_ls, s])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return {"coef": coef, "t0": t0_hat, "w": w_hat, "yhat": yhat}

# gràfiques

def plot_comparison(dates, y, y1, y2, outpath):
    plt.figure(figsize=(10, 4.2))
    plt.plot(dates, y, label="Observat", linewidth=1.5, color="black")
    plt.plot(dates, y1, label="Model 1 (step)", linewidth=1.5, color="tab:blue")
    plt.plot(dates, y2, label="Model 2 (sigmoid)", linewidth=1.5, color="tab:orange")
    plt.title("Comparació de models")
    plt.xlabel("Data")
    plt.ylabel("Distància (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

def plot_residuals(dates, res, title, outpath):
    plt.figure(figsize=(10, 3.5))
    plt.plot(dates, res, color="gray")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

def plot_residual_hist(res, title, outpath):
    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=20, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequència")
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

def plot_qq(res, title, outpath):
    plt.figure(figsize=(5, 5))
    stats.probplot(res, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130)
    plt.close()

# main

def main():
    ensure_dirs()

    data_path = Path("data/shoreline_monthly.csv")
    if not data_path.exists():
        raise FileNotFoundError("No se encontró el archivo de datos procesados.")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    y = df["shoreline"].values
    t = np.arange(len(y), dtype=float)
    freqs = detect_top_frequencies(y, top_k=3)

    # ajustar modelos
    m1 = fit_step_model(t, y, freqs)
    m2 = fit_sigmoid_model(t, y, freqs)

    # calcular métricas
    rmse1, r21 = rmse(y, m1["yhat"]), r2_score(y, m1["yhat"])
    rmse2, r22 = rmse(y, m2["yhat"]), r2_score(y, m2["yhat"])

    # residuos
    res1 = y - m1["yhat"]
    res2 = y - m2["yhat"]

    # gráficas
    plot_comparison(df["date"], y, m1["yhat"], m2["yhat"], "outputs/figures/model_comparison.png")
    plot_residuals(df["date"], res1, "Residuos Modelo 1 (Step)", "outputs/figures/residuals_model1.png")
    plot_residuals(df["date"], res2, "Residuos Modelo 2 (Sigmoid)", "outputs/figures/residuals_model2.png")
    plot_residual_hist(res1, "Histograma Residuos Modelo 1", "outputs/figures/residual_hist_model1.png")
    plot_residual_hist(res2, "Histograma Residuos Modelo 2", "outputs/figures/residual_hist_model2.png")
    plot_qq(res1, "Q–Q plot Modelo 1", "outputs/figures/qq_model1.png")
    plot_qq(res2, "Q–Q plot Modelo 2", "outputs/figures/qq_model2.png")

    # guardar métricas
    lines = []
    lines.append("Model evaluation (monthly series)")
    lines.append(f"Model 1 (step): RMSE = {rmse1:.4f}, R2 = {r21:.4f}")
    lines.append(f"Model 2 (sigmoid): RMSE = {rmse2:.4f}, R2 = {r22:.4f}")
    lines.append("")
    if rmse1 < rmse2:
        lines.append("→ El Model 1 tiene menor RMSE (mejor ajuste).")
    else:
        lines.append("→ El Model 2 tiene menor RMSE (mejor ajuste).")
   
    with open("outputs/metrics_model_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nFigures guardades en: outputs/figures/")
    print("Mètriques guardades en: outputs/metrics_model_evaluation.txt")

if __name__ == "__main__":
    main()
