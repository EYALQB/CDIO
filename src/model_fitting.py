import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# FFT i estacionalitat

def detect_top_frequencies(y, top_k=3):
    # i ha de estar muestreada mensualment (d=1 mes)
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    n = len(y)
    if n < 8:
        return [], []

    freq = np.fft.rfftfreq(n, d=1.0)        # cicles/mes
    spec = np.fft.rfft(y)
    power = np.abs(spec)

    mask = freq > 0                          # treure component DC
    freq_nz = freq[mask]
    power_nz = power[mask]

    if len(freq_nz) == 0:
        return [], []

    idx = np.argsort(power_nz)[::-1][:top_k]
    dom_freqs = freq_nz[idx]
    dom_periods = 1.0 / dom_freqs            # meses
    return list(dom_freqs), list(dom_periods)

def seasonal_design_matrix(t, freqs):
    # columnes: sin(2π f t), cos(2π f t) per a cada freqüència
    if len(freqs) == 0:
        return np.zeros((len(t), 0))
    cols = []
    for f in freqs:
        w = 2 * np.pi * f
        cols.append(np.sin(w * t))
        cols.append(np.cos(w * t))
    return np.column_stack(cols)

# models

def fit_base_model(t, y, freqs):
    # y = a0 + a1 t + sum_k (b_k sin + c_k cos)
    X_lin = np.column_stack([np.ones_like(t), t])
    X_sea = seasonal_design_matrix(t, freqs)
    X = np.column_stack([X_lin, X_sea]) if X_sea.size else X_lin

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef

    return {
        "coef": coef,
        "freqs": freqs,
        "yhat": yhat,
        "rmse": rmse(y, yhat),
        "r2": r2_score(y, yhat),
        "slope": float(coef[1]),
    }

def fit_step_model(t, y, freqs, grid_points=60):
    # y = a0 + a1 t + seasonal + A * H(t - t_s)
    X_lin = np.column_stack([np.ones_like(t), t])
    X_sea = seasonal_design_matrix(t, freqs)
    X_base = np.column_stack([X_lin, X_sea]) if X_sea.size else X_lin

    lo, hi = float(np.min(t)), float(np.max(t))
    span = hi - lo
    eps = 0.05 * span
    ts_grid = np.linspace(lo + eps, hi - eps, grid_points) if span > 0 else np.array([lo + 1e-6])

    best = {"rmse": np.inf, "t_step": None, "coef": None, "yhat": None}
    for ts in ts_grid:
        H = (t >= ts).astype(float).reshape(-1, 1)
        X = np.column_stack([X_base, H])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        e = rmse(y, yhat)
        if e < best["rmse"]:
            best = {"rmse": e, "t_step": float(ts), "coef": coef, "yhat": yhat}

    return {
        "coef": best["coef"],
        "freqs": freqs,
        "t_step": best["t_step"],
        "yhat": best["yhat"],
        "rmse": best["rmse"],
        "r2": r2_score(y, best["yhat"]),
        "slope": float(best["coef"][1]),
    }

def fit_sigmoid_model(t, y, freqs):
    # y ~ a0 + a1 t + seasonal + A * sig(t; t0, w)
    # sig(t) = 1 / (1 + exp(-(t - t0)/w))
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

    return {
        "coef": coef,
        "freqs": freqs,
        "t0": float(t0_hat),
        "w": float(w_hat),
        "yhat": yhat,
        "rmse": rmse(y, yhat),
        "r2": r2_score(y, yhat),
        "slope": float(coef[1]),
    }

# gràfiques

def plot_fit(dates, y, yhat, title, outpath):
    plt.figure(figsize=(10, 4.2))
    plt.plot(dates, y, label="Serie", linewidth=1.5)
    plt.plot(dates, yhat, label="Reconstrucció", linewidth=1.5)
    plt.title(title)
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
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    dates = df["date"].values
    y = df["shoreline"].values.astype(float)

    # temps en mesos des del inici
    t = np.arange(len(y), dtype=float)

    # detecció de freqències dominants
    dom_freqs, dom_periods = detect_top_frequencies(y, top_k=3)

    # ajuste de models
    base = fit_base_model(t, y, dom_freqs)
    step = fit_step_model(t, y, dom_freqs, grid_points=60)
    sigm = fit_sigmoid_model(t, y, dom_freqs)

    # gràfiques de reconstrucció
    plot_fit(dates, y, base["yhat"], "Model base: tendencia + estacional (FFT)", "outputs/figures/model_base_fit.png")
    plot_fit(dates, y, step["yhat"], "Model 1: tendencia + esgraó + estacional", "outputs/figures/model1_step_fit.png")
    plot_fit(dates, y, sigm["yhat"], "Model 2: tendencia + sigmoide + estacional", "outputs/figures/model2_sigmoid_fit.png")

    # mètriques i resum
    metrics_lines = []

    metrics_lines.append("Model fitting (monthly series)")
    metrics_lines.append("Freqències dominants (cicles/mes) i períodes (mesos):")
    if dom_freqs:
        for f, p in zip(dom_freqs, dom_periods):
            metrics_lines.append(f"  f = {f:.4f}  ->  periodo ≈ {p:.2f} mesos")
    else:
        metrics_lines.append("No se detectaron frecuencias dominantes.")

    metrics_lines.append("")
    metrics_lines.append("Model base: tendencia + estacional (FFT)")
    metrics_lines.append(f"  Pendiente (a1): {base['slope']:.6f} unidades/mes")
    metrics_lines.append(f"  RMSE: {base['rmse']:.4f}")
    metrics_lines.append(f"  R2: {base['r2']:.4f}")

    metrics_lines.append("")
    metrics_lines.append("Model 1: tendencia + esgraó + estacional")
    metrics_lines.append(f"  Pendiente (a1): {step['slope']:.6f} unidades/mes")
    metrics_lines.append(f"  t_step (índice de mes): {step['t_step']:.2f}")
    metrics_lines.append(f"  RMSE: {step['rmse']:.4f}")
    metrics_lines.append(f"  R2: {step['r2']:.4f}")

    metrics_lines.append("")
    metrics_lines.append("Model 2: tendencia + sigmoide + estacional")
    metrics_lines.append(f"  Pendiente (a1): {sigm['slope']:.6f} unidades/mes")
    metrics_lines.append(f"  t0 (centro sigmoide, índice de mes): {sigm['t0']:.2f}")
    metrics_lines.append(f"  w (ancho sigmoide, meses): {sigm['w']:.2f}")
    metrics_lines.append(f"  RMSE: {sigm['rmse']:.4f}")
    metrics_lines.append(f"  R2: {sigm['r2']:.4f}")

    metrics_lines.append("")
    metrics_lines.append(f"• ¿La tendencia es positiva o negativa? (pendiente base = {base['slope']:.6f})")
    if dom_periods:
        dom_txt = ", ".join([f"~{p:.1f}m" for p in dom_periods])
        metrics_lines.append(f"• Periodos dominantes detectados: {dom_txt}")
    else:
        metrics_lines.append("• No se detectaron periodos dominantes claros.")
    

    with open("outputs/metrics_model_fitting.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines))

    print("\n".join(metrics_lines))
    print("\nFiguras guardadas en: outputs/figures/")
    print("Mètriques guardadas en: outputs/metrics_model_fitting.txt")

if __name__ == "__main__":
    main()
