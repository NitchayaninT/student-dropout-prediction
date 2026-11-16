"""Stage-comparison figure from results/<task>/cv_summary.csv.
Usage: python -m src.plot_results --task multiclass"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#eb6834", "XGBoost": "#1baf7a"}
MARKERS = {"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^"}
PANELS = {
    "multiclass": [("f1_macro", "Macro F1 (3 classes)"),
                   ("dropout_roc_auc", "Dropout ROC-AUC (vs rest)"),
                   ("dropout_recall", "Dropout recall")],
    "binary": [("dropout_roc_auc", "ROC-AUC"), ("dropout_pr_auc", "PR-AUC"),
               ("dropout_recall", "Dropout recall")],
}
STAGE_LABELS = ["T0\nenrollment", "T1\nafter sem 1", "T2\nafter sem 2"]


def main(task):
    s = pd.read_csv(C.RESULTS_DIR / task / "cv_summary.csv")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.edgecolor": "#c3c2b7", "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#898781"})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor="#fcfcfb")
    for ax, (metric, title) in zip(axes, PANELS[task]):
        ax.set_facecolor("#fcfcfb")
        for off, (model, g) in zip((-0.06, 0, 0.06), s.groupby("model", sort=False)):
            x = [i + off for i in range(len(g))]
            ax.errorbar(x, g[f"{metric}_mean"], yerr=g[f"{metric}_std"], lw=2,
                        marker=MARKERS[model], ms=6, mec="#fcfcfb", mew=1.2,
                        capsize=2.5, elinewidth=1, color=COLORS[model], label=model)
        ax.set_xticks(range(3), STAGE_LABELS)
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e1e0d9", lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right", labelcolor="#52514e")
    name = "3-class (Dropout / Enrolled / Graduate)" if task == "multiclass" \
        else "binary (Dropout vs Graduate)"
    fig.suptitle(f"Predictive performance by chronological stage, {name}",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.01, "Mean ± SD over stratified 5-fold CV (hyperparameters tuned in "
             "inner folds). Separate y-scale per panel.", fontsize=8.5, color="#898781")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    C.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"stage_comparison_{task}.png", dpi=180)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="multiclass")
    main(ap.parse_args().task)
