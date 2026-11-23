"""Calibration of the out-of-fold dropout probabilities in the binary task.

    python -m src.calibration

Reads results/binary/oof_predictions.csv (every student predicted once, by a model fitted
without them) and writes results/calibration/calibration_summary.csv,
results/calibration/reliability_bins.csv and figures/calibration_binary.{png,pdf,svg}.

Per stage and model, on the pooled out-of-fold predictions:
  Brier score            mean squared error of the probability (lower is better); the
                         reference is the Brier score of always predicting the base rate
  ECE                    expected calibration error over 10 equal-count bins
  calibration intercept  logistic recalibration with logit(p) as an offset (ideal 0;
                         > 0 means the model under-predicts risk overall)
  calibration slope      logistic regression of the outcome on logit(p) (ideal 1; < 1 means
                         predictions are too extreme, > 1 too timid)
"""
import numpy as np
import pandas as pd

from . import config as C
from . import figstyle as fs

import matplotlib.pyplot as plt  # noqa: E402  (figstyle selects the Agg backend)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

OUT = C.RESULTS_DIR / "calibration"
STAGES = ["T0", "T1", "T2"]
MODELS = {"XGBoost": (fs.PALETTE["teal"], "o"), "RandomForest": (fs.PALETTE["violet"], "s"),
          "LogisticRegression": ("#767676", "^")}
N_BINS = 10


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _newton_logistic(X, y, offset, iters=50):
    """Unpenalised logistic regression by Newton-Raphson; returns the coefficients."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ beta + offset)))
        grad = X.T @ (y - mu)
        hess = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_stats(p, y):
    lp = _logit(p)
    ones = np.ones((len(y), 1))
    intercept = _newton_logistic(ones, y, offset=lp)[0]
    slope = _newton_logistic(np.column_stack([ones[:, 0], lp]), y, offset=np.zeros(len(y)))[1]
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, N_BINS)
    ece = sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins)
    base = y.mean()
    return {"brier": float(np.mean((p - y) ** 2)), "brier_base_rate": float(base * (1 - base)),
            "ece": float(ece), "calibration_intercept": float(intercept),
            "calibration_slope": float(slope), "mean_predicted": float(p.mean()),
            "observed_rate": float(base)}, bins


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def main():
    fs.apply_publication_style(fs.SLIDE_STYLE)
    OUT.mkdir(parents=True, exist_ok=True)
    o = pd.read_csv(C.RESULTS_DIR / "binary" / "oof_predictions.csv")
    rows, bin_rows = [], []
    for (stage, model), g in o.groupby(["stage", "model"], sort=False):
        p = g["p_Dropout"].to_numpy()
        y = (g["y_true"] == "Dropout").to_numpy(int)
        stats, bins = calibration_stats(p, y)
        rows.append({"stage": stage, "model": model, "n": len(y), **stats})
        for i, b in enumerate(bins):
            lo, hi = wilson(int(y[b].sum()), len(b))
            bin_rows.append({"stage": stage, "model": model, "bin": i + 1, "n": len(b),
                             "mean_predicted": p[b].mean(), "observed_rate": y[b].mean(),
                             "ci_lo": lo, "ci_hi": hi})
    summ, rel = pd.DataFrame(rows), pd.DataFrame(bin_rows)
    summ.to_csv(OUT / "calibration_summary.csv", index=False, float_format="%.4f")
    rel.to_csv(OUT / "reliability_bins.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(2, 3, figsize=(fs.SLIDE_WIDTH, 6.6), layout="constrained",
                             height_ratios=[3, 1], sharex=True)
    fig.get_layout_engine().set(h_pad=0.12)          # room between the reliability and count rows
    for j, stage in enumerate(STAGES):
        ax, hx = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color="#4D4D4D", ls="--", lw=1.2)
        for model, (color, mk) in MODELS.items():
            r = rel[(rel.stage == stage) & (rel.model == model)]
            if model == "XGBoost":
                ax.vlines(r.mean_predicted, r.ci_lo, r.ci_hi, color=color, lw=2)
            ax.plot(r.mean_predicted, r.observed_rate, marker=mk, ms=8 if model == "XGBoost" else 6,
                    color=color, lw=1.6 if model == "XGBoost" else 1.0,
                    alpha=1.0 if model == "XGBoost" else 0.8)
        s = summ[(summ.stage == stage) & (summ.model == "XGBoost")].iloc[0]
        ax.text(0.97, 0.03, f"XGBoost: Brier {s.brier:.3f}\nECE {s.ece:.3f}, slope {s.calibration_slope:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title({"T0": "T0: at enrollment", "T1": "T1: after semester 1",
                      "T2": "T2: after semester 2"}[stage], fontsize=15)
        if j == 0:
            ax.set_ylabel("Observed dropout rate")
        else:
            ax.tick_params(labelleft=False)
        fs.panel_label(ax, "abc"[j])
        g = o[(o.stage == stage) & (o.model == "XGBoost")]
        hx.hist(g["p_Dropout"], bins=np.linspace(0, 1, 21), color=fs.PALETTE["teal"], alpha=0.8)
        hx.set_xlabel("Predicted P(dropout)")
        hx.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))   # no "0" at the corner
        if j == 0:
            hx.set_ylabel("Students")
        else:
            hx.tick_params(labelleft=False)
    handles = [Line2D([], [], color=c, marker=m, label=n.replace("LogisticRegression", "Logistic Regression")
                      .replace("RandomForest", "Random Forest")) for n, (c, m) in MODELS.items()]
    handles.append(Line2D([], [], color="#4D4D4D", ls="--", label="Perfect calibration"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=12)
    fs.save_figure(fig, C.FIGURES_DIR / "calibration_binary")

    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
