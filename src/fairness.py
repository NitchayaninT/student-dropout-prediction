"""Subgroup & intersectional fairness audit from out-of-fold predictions.

Evaluates Dropout-class fairness across Gender and Scholarship holder (plus
their 2x2 intersection) at T0, T1, and T2 for every model in
results/<task>/oof_predictions.csv. Also summarizes Dropout risk distributions
for the 794 Currently Enrolled students.

Usage:
  python -m src.fairness --task multiclass
  python -m src.fairness --task binary
"""
import argparse
import textwrap

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from . import config as C
from .features import add_domain_features, load_data, stage_features

# ---- style tokens (matched to plot_results.py & explain.py) -----------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {
    "LogisticRegression": "#2a78d6",
    "RandomForest": "#eb6834",
    "XGBoost": "#1baf7a",
}
MARKERS = {"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^"}
STAGE_LABELS = ["T0\nenrollment", "T1\nafter sem 1", "T2\nafter sem 2"]
STAGE_ORDER = ["T0", "T1", "T2"]

# Reference vs comparison group for signed gap = group_a - group_b
SENSITIVE_ATTRS = {
    "Gender": {
        "labels": {0: "Female", 1: "Male"},
        "pair": ("Male", "Female"),          # higher-dropout group minus lower
    },
    "Scholarship holder": {
        "labels": {0: "Non-Scholarship", 1: "Scholarship"},
        "pair": ("Non-Scholarship", "Scholarship"),
    },
}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.edgecolor": AXIS, "text.color": INK,
        "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def _clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _group_metrics(y_true_d: np.ndarray, y_pred_d: np.ndarray,
                   p_d: np.ndarray) -> dict:
    """Binary Dropout-vs-rest metrics for one subgroup."""
    n = int(len(y_true_d))
    n_pos = int(y_true_d.sum())
    n_neg = n - n_pos
    fp = int(((y_pred_d == 1) & (y_true_d == 0)).sum())
    tp = int(((y_pred_d == 1) & (y_true_d == 1)).sum())
    tpr = tp / n_pos if n_pos > 0 else np.nan
    fpr = fp / n_neg if n_neg > 0 else np.nan
    has_both = (n_pos > 0) and (n_neg > 0)
    return {
        "n": n,
        "n_dropout": n_pos,
        "dropout_rate": n_pos / n if n > 0 else np.nan,
        "selection_rate": float(y_pred_d.mean()) if n > 0 else np.nan,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision_score(y_true_d, y_pred_d, zero_division=0) if n > 0 else np.nan,
        "f1": f1_score(y_true_d, y_pred_d, zero_division=0) if n > 0 else np.nan,
        "roc_auc": roc_auc_score(y_true_d, p_d) if has_both else np.nan,
        "pr_auc": average_precision_score(y_true_d, p_d) if has_both else np.nan,
        "brier": brier_score_loss(y_true_d, p_d) if n > 0 else np.nan,
    }


def _bootstrap_gap_ci(g_a: pd.DataFrame, g_b: pd.DataFrame,
                      n_boot: int = 1000, seed: int = C.SEED) -> dict:
    """Stratified bootstrap 95% CIs for signed gaps (group_a - group_b)."""
    rng = np.random.default_rng(seed)
    ya, pa, pra = g_a["is_dropout"].to_numpy(), g_a["pred_dropout"].to_numpy(), g_a["p_Dropout"].to_numpy()
    yb, pb, prb = g_b["is_dropout"].to_numpy(), g_b["pred_dropout"].to_numpy(), g_b["p_Dropout"].to_numpy()
    na, nb = len(ya), len(yb)
    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))

    def _vec_rates(y, p, idx):
        ys, ps = y[idx], p[idx]
        sel = ps.mean(axis=1)
        pos = ys.sum(axis=1)
        neg = (1 - ys).sum(axis=1)
        tp = ((ps == 1) & (ys == 1)).sum(axis=1)
        fp = ((ps == 1) & (ys == 0)).sum(axis=1)
        tpr = np.where(pos > 0, tp / np.where(pos > 0, pos, 1), np.nan)
        fpr = np.where(neg > 0, fp / np.where(neg > 0, neg, 1), np.nan)
        return tpr, fpr, sel

    tpr_a, fpr_a, sel_a = _vec_rates(ya, pa, idx_a)
    tpr_b, fpr_b, sel_b = _vec_rates(yb, pb, idx_b)
    out = {}
    for name, diff in [("tpr", tpr_a - tpr_b),
                       ("fpr", fpr_a - fpr_b),
                       ("selection_rate", sel_a - sel_b)]:
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        out[f"gap_{name}_ci_lo"] = float(lo)
        out[f"gap_{name}_ci_hi"] = float(hi)
    return out


def attach_attributes(oof: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join OOF predictions on original row index with sensitive attributes."""
    attrs = df[["Gender", "Scholarship holder"]].copy()
    attrs.index.name = "row"
    merged = oof.merge(attrs.reset_index(), on="row", how="left", validate="many_to_one")
    assert merged["Gender"].notna().all(), "Row index mismatch joining data.csv"
    merged["is_dropout"] = (merged["y_true"] == "Dropout").astype(int)
    merged["pred_dropout"] = (merged["y_pred"] == "Dropout").astype(int)
    merged["Gender_label"] = merged["Gender"].map(SENSITIVE_ATTRS["Gender"]["labels"])
    merged["Scholarship_label"] = merged["Scholarship holder"].map(
        SENSITIVE_ATTRS["Scholarship holder"]["labels"]
    )
    merged["Intersectional"] = (
        merged["Gender_label"] + " / " + merged["Scholarship_label"]
    )
    return merged


def evaluate_oof_fairness(oof: pd.DataFrame, df: pd.DataFrame,
                          n_boot: int = 1000):
    """Compute subgroup metrics, disparity gaps, and intersectional table.

    Reusable by src/ablation.py so baseline and ablated models use identical
    fairness definitions.
    """
    m = attach_attributes(oof, df)
    sub_rows, gap_rows, inter_rows = [], [], []

    group_cols = ["stage", "model"]
    if "variant" in m.columns:
        group_cols = ["variant"] + group_cols

    for keys, sub_df in m.groupby(group_cols, sort=False):
        meta = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

        # ---- 1. Single-attribute subgroup & gap metrics ---------------------
        for attr, spec in SENSITIVE_ATTRS.items():
            label_col = "Gender_label" if attr == "Gender" else "Scholarship_label"
            pooled_by_grp = {}
            fold_by_grp = {}

            for grp_name in spec["pair"]:
                gdf = sub_df[sub_df[label_col] == grp_name]
                pooled = _group_metrics(
                    gdf["is_dropout"].to_numpy(),
                    gdf["pred_dropout"].to_numpy(),
                    gdf["p_Dropout"].to_numpy(),
                )
                pooled_by_grp[grp_name] = pooled

                # Fold-level SD for uncertainty bars across the 5 CV folds
                fmetrics = []
                for _, fg in gdf.groupby("fold"):
                    fmetrics.append(_group_metrics(
                        fg["is_dropout"].to_numpy(),
                        fg["pred_dropout"].to_numpy(),
                        fg["p_Dropout"].to_numpy(),
                    ))
                fdf = pd.DataFrame(fmetrics)
                fold_by_grp[grp_name] = fdf

                row = {**meta, "attribute": attr, "group": grp_name, **pooled}
                for col in ("tpr", "fpr", "selection_rate", "f1", "roc_auc"):
                    row[f"{col}_fold_mean"] = float(fdf[col].mean())
                    row[f"{col}_fold_std"] = float(fdf[col].std(ddof=1))
                sub_rows.append(row)

            ga, gb = spec["pair"]
            pa_m, pb_m = pooled_by_grp[ga], pooled_by_grp[gb]
            fa_df, fb_df = fold_by_grp[ga], fold_by_grp[gb]

            gap_row = {
                **meta,
                "attribute": attr,
                "group_a": ga,
                "group_b": gb,
                "gap_definition": f"{ga} - {gb}",
            }
            for col in ("tpr", "fpr", "selection_rate", "f1", "roc_auc", "precision", "brier"):
                diff_pooled = pa_m[col] - pb_m[col]
                gap_row[f"gap_{col}"] = diff_pooled
                gap_row[f"abs_gap_{col}"] = abs(diff_pooled)
                if col in ("tpr", "fpr", "selection_rate", "roc_auc"):
                    fold_diff = fa_df[col].to_numpy() - fb_df[col].to_numpy()
                    gap_row[f"gap_{col}_fold_mean"] = float(np.nanmean(fold_diff))
                    gap_row[f"gap_{col}_fold_std"] = float(np.nanstd(fold_diff, ddof=1))
                    gap_row[f"abs_gap_{col}_fold_mean"] = float(np.nanmean(np.abs(fold_diff)))
                    gap_row[f"abs_gap_{col}_fold_std"] = float(np.nanstd(np.abs(fold_diff), ddof=1))

            if n_boot > 0:
                ci = _bootstrap_gap_ci(
                    sub_df[sub_df[label_col] == ga],
                    sub_df[sub_df[label_col] == gb],
                    n_boot=n_boot,
                )
                gap_row.update(ci)
            gap_rows.append(gap_row)

        # ---- 2. Intersectional (Gender x Scholarship) -----------------------
        for inter_grp, igdf in sub_df.groupby("Intersectional", sort=True):
            im = _group_metrics(
                igdf["is_dropout"].to_numpy(),
                igdf["pred_dropout"].to_numpy(),
                igdf["p_Dropout"].to_numpy(),
            )
            inter_rows.append({**meta, "intersectional_group": inter_grp, **im})

    return pd.DataFrame(sub_rows), pd.DataFrame(gap_rows), pd.DataFrame(inter_rows)


def evaluate_enrolled_risk(task: str, df: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Summarize Dropout risk distribution for the 794 Currently Enrolled students."""
    rows = []
    if task == "multiclass":
        enr = attach_attributes(oof, df)
        enr = enr[enr["y_true"] == "Enrolled"]
        for (stage, model), g in enr.groupby(["stage", "model"], sort=False):
            p = g["p_Dropout"].to_numpy()
            rows.append({
                "task": task, "source": "oof_multiclass",
                "stage": stage, "model": model, "n_enrolled": len(g),
                "p_dropout_mean": float(p.mean()),
                "p_dropout_median": float(np.median(p)),
                "p_dropout_q25": float(np.percentile(p, 25)),
                "p_dropout_q75": float(np.percentile(p, 75)),
                "pct_pred_dropout": float((g["y_pred"] == "Dropout").mean()),
                "pct_p_ge_050": float((p >= 0.50).mean()),
                "pct_p_ge_070": float((p >= 0.70).mean()),
                "mean_p_male": float(g.loc[g["Gender"] == 1, "p_Dropout"].mean()),
                "mean_p_female": float(g.loc[g["Gender"] == 0, "p_Dropout"].mean()),
                "mean_p_no_scholar": float(g.loc[g["Scholarship holder"] == 0, "p_Dropout"].mean()),
                "mean_p_scholar": float(g.loc[g["Scholarship holder"] == 1, "p_Dropout"].mean()),
            })
    else:
        df_eng = add_domain_features(df)
        enr_df = df_eng[df_eng[C.TARGET] == "Enrolled"].copy()
        model_dir = C.MODELS_DIR / "binary"
        for stage in STAGE_ORDER:
            X_enr = enr_df[stage_features(stage)]
            for model_name in COLORS:
                mpath = model_dir / f"{stage}_{model_name}.joblib"
                if not mpath.exists():
                    continue
                bundle = joblib.load(mpath)
                classes = bundle["classes"]
                d_idx = classes.index("Dropout")
                p = bundle["model"].predict_proba(X_enr)[:, d_idx]
                rows.append({
                    "task": task, "source": "final_binary_model",
                    "stage": stage, "model": model_name, "n_enrolled": len(enr_df),
                    "p_dropout_mean": float(p.mean()),
                    "p_dropout_median": float(np.median(p)),
                    "p_dropout_q25": float(np.percentile(p, 25)),
                    "p_dropout_q75": float(np.percentile(p, 75)),
                    "pct_pred_dropout": float((p >= 0.50).mean()),
                    "pct_p_ge_050": float((p >= 0.50).mean()),
                    "pct_p_ge_070": float((p >= 0.70).mean()),
                    "mean_p_male": float(p[enr_df["Gender"].to_numpy() == 1].mean()),
                    "mean_p_female": float(p[enr_df["Gender"].to_numpy() == 0].mean()),
                    "mean_p_no_scholar": float(p[enr_df["Scholarship holder"].to_numpy() == 0].mean()),
                    "mean_p_scholar": float(p[enr_df["Scholarship holder"].to_numpy() == 1].mean()),
                })
    return pd.DataFrame(rows)


# ---- plotting ---------------------------------------------------------------
def plot_fairness_gaps(task: str, gaps: pd.DataFrame):
    """2x3 grid: rows = Gender / Scholarship, cols = ΔTPR, ΔFPR, ΔSelection Rate."""
    _style()
    metrics = [
        ("gap_tpr", "Equal Opportunity gap (Δ TPR / Recall)"),
        ("gap_fpr", "Predictive Equality gap (Δ FPR)"),
        ("gap_selection_rate", "Demographic Parity gap (Δ Selection Rate)"),
    ]
    attrs = list(SENSITIVE_ATTRS.keys())
    fig, axes = plt.subplots(len(attrs), len(metrics), figsize=(12.2, 6.8))

    for r, attr in enumerate(attrs):
        ga, gb = SENSITIVE_ATTRS[attr]["pair"]
        sub = gaps[gaps["attribute"] == attr]
        for c, (mcol, title) in enumerate(metrics):
            ax = axes[r, c]
            ax.axhline(0.0, color=AXIS, lw=1.1, ls="--", zorder=1)
            for off, model in zip((-0.07, 0.0, 0.07), COLORS):
                g = sub[sub["model"] == model].set_index("stage").reindex(STAGE_ORDER)
                x = [i + off for i in range(len(STAGE_ORDER))]
                y = g[f"{mcol}_fold_mean"].to_numpy()
                yerr = g[f"{mcol}_fold_std"].to_numpy()
                ax.errorbar(
                    x, y, yerr=yerr, lw=2, marker=MARKERS[model], ms=6,
                    mec=SURFACE, mew=1.2, capsize=2.5, elinewidth=1,
                    color=COLORS[model], label=model, zorder=3,
                )
            ax.set_xticks(range(3), STAGE_LABELS)
            ax.set_xlim(-0.32, 2.32)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{attr}\n({ga} − {gb})", fontsize=9.5, fontweight="bold")
            _clean(ax)

    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8.8, labelcolor=INK2)
    tlabel = "3-class (Dropout / Enrolled / Graduate)" if task == "multiclass" \
        else "binary (Dropout vs Graduate)"
    fig.suptitle(f"Subgroup fairness gaps across chronological stages, {tlabel}",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(
        0.012, 0.012,
        "Signed gap = group A − group B (0 = parity). Points and error bars show mean ± SD "
        "across stratified 5-fold out-of-fold predictions.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    C.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"fairness_gaps_{task}.png", dpi=180)
    plt.close(fig)


def plot_subgroup_rates(task: str, sub: pd.DataFrame, model_name: str = "XGBoost"):
    """2x2 panel for one highlight model: TPR and FPR per subgroup across T0..T2."""
    _style()
    sub_m = sub[sub["model"] == model_name]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), sharex=True)
    pair_colors = {
        "Female": "#2a78d6", "Male": "#e34948",
        "Scholarship": "#1baf7a", "Non-Scholarship": "#eb6834",
    }
    for r, attr in enumerate(SENSITIVE_ATTRS):
        sa = sub_m[sub_m["attribute"] == attr]
        for c, (mcol, mtitle) in enumerate([("tpr", "Dropout Recall (TPR)"),
                                            ("fpr", "False Positive Rate (FPR)")]):
            ax = axes[r, c]
            for off, grp in zip((-0.05, 0.05), SENSITIVE_ATTRS[attr]["pair"]):
                g = sa[sa["group"] == grp].set_index("stage").reindex(STAGE_ORDER)
                x = [i + off for i in range(3)]
                ax.errorbar(
                    x, g[f"{mcol}_fold_mean"], yerr=g[f"{mcol}_fold_std"],
                    lw=2.2, marker="o", ms=6.5, mec=SURFACE, mew=1.2,
                    capsize=2.5, color=pair_colors[grp], label=grp,
                )
            ax.set_xticks(range(3), STAGE_LABELS)
            ax.set_xlim(-0.3, 2.3)
            if r == 0:
                ax.set_title(mtitle, loc="left", fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(attr, fontsize=10, fontweight="bold")
            ax.legend(frameon=False, loc="best", fontsize=9, labelcolor=INK2)
            _clean(ax)

    tlabel = "3-class" if task == "multiclass" else "binary"
    fig.suptitle(f"Subgroup Dropout TPR and FPR by stage ({model_name}, {tlabel})",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.012, "Out-of-fold mean ± SD across stratified 5 folds.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(C.FIGURES_DIR / f"fairness_subgroups_{task}.png", dpi=180)
    plt.close(fig)


def plot_intersectional(task: str, inter: pd.DataFrame, model_name: str = "XGBoost"):
    """Side-by-side heatmaps of TPR and FPR across Gender x Scholarship (T0..T2)."""
    _style()
    sub = inter[inter["model"] == model_name]
    groups = sorted(sub["intersectional_group"].unique())
    counts = sub[sub["stage"] == "T0"].set_index("intersectional_group")["n"]
    ylabels = [f"{g}  (n={int(counts[g])})" for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    panels = [("tpr", "Dropout Recall (TPR)", "Blues", 0.0, 1.0),
              ("fpr", "False Positive Rate (FPR)", "OrRd", 0.0, 0.45)]

    for ax, (col, title, cmap, vmin, vmax) in zip(axes, panels):
        mat = np.zeros((len(groups), len(STAGE_ORDER)))
        for i, grp in enumerate(groups):
            for j, st in enumerate(STAGE_ORDER):
                v = sub[(sub["intersectional_group"] == grp) & (sub["stage"] == st)][col]
                mat[i, j] = v.iloc[0] if len(v) else np.nan
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3), STAGE_ORDER, fontsize=10)
        ax.set_yticks(range(len(groups)), ylabels, fontsize=9.5)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for (i, j), val in np.ndenumerate(mat):
            if not np.isnan(val):
                thresh = vmin + 0.55 * (vmax - vmin)
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color="#ffffff" if val > thresh else INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8.5)

    tlabel = "3-class" if task == "multiclass" else "binary"
    fig.suptitle(f"Intersectional fairness (Gender × Scholarship), {model_name} ({tlabel})",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.015,
             textwrap.fill("Pooled out-of-fold Dropout Recall (TPR) and False Positive Rate "
                           "(FPR) across all 4 Gender × Scholarship subgroups.", 120),
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(C.FIGURES_DIR / f"fairness_intersectional_{task}.png", dpi=180)
    plt.close(fig)


def run(task: str, n_boot: int = 1000, highlight_model: str = "XGBoost"):
    out = C.RESULTS_DIR / task
    oof_path = out / "oof_predictions.csv"
    assert oof_path.exists(), f"Missing {oof_path}. Run `python -m src.train --task {task}` first."

    df = load_data()
    oof = pd.read_csv(oof_path)
    sub_df, gap_df, inter_df = evaluate_oof_fairness(oof, df, n_boot=n_boot)
    enr_df = evaluate_enrolled_risk(task, df, oof)

    sub_df.to_csv(out / "fairness_subgroups.csv", index=False, float_format="%.5f")
    gap_df.to_csv(out / "fairness_gaps.csv", index=False, float_format="%.5f")
    inter_df.to_csv(out / "fairness_intersectional.csv", index=False, float_format="%.5f")
    if len(enr_df):
        enr_df.to_csv(out / "enrolled_risk_summary.csv", index=False, float_format="%.5f")

    plot_fairness_gaps(task, gap_df)
    plot_subgroup_rates(task, sub_df, model_name=highlight_model)
    plot_intersectional(task, inter_df, model_name=highlight_model)

    print(f"[{task}] Saved fairness tables to {out}/ and figures to {C.FIGURES_DIR}/")
    show = gap_df[gap_df["model"] == highlight_model][
        ["stage", "attribute", "gap_tpr", "gap_fpr", "gap_selection_rate"]
    ]
    print(f"\n--- {highlight_model} signed fairness gaps ({task}) ---")
    print(show.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for 95% gap CIs (0 to disable)")
    ap.add_argument("--highlight-model", default="XGBoost", choices=list(COLORS))
    args = ap.parse_args()
    run(args.task, n_boot=args.n_boot, highlight_model=args.highlight_model)
