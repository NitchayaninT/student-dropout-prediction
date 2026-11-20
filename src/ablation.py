"""Demographic ablation study: accuracy-fairness trade-off across T0 / T1 / T2.

Reuses the exact outer/inner Stratified CV protocol (SEED=42) and model
pipelines from Part 1 so comparisons against results/<task>/cv_fold_metrics.csv
and results/<task>/oof_predictions.csv are strictly paired.

Variants evaluated:
  full           Baseline loaded directly from results/<task>/ (not re-run)
  no_sensitive   Drops Gender and Scholarship holder (2 features)
  no_demographic Drops all 7 personal demographic features
  no_demo_socio  Drops all 12 personal demographic + socioeconomic features

Usage:
  python -m src.ablation --task multiclass
  python -m src.ablation --task binary
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from .fairness import (AXIS, GRID, INK, INK2, MUTED, STAGE_LABELS, STAGE_ORDER,
                       SURFACE, _clean, _style, evaluate_oof_fairness)
from .features import get_xy, load_data
from .models import MODEL_NAMES, build
from .train import fold_metrics

ABLATION_DROPS = {
    "no_sensitive": [
        "Gender", "Scholarship holder",
    ],
    "no_demographic": [
        "Gender", "Age at enrollment", "Marital status",
        "Nacionality", "International", "Displaced",
        "Educational special needs",
    ],
    "no_demo_socio": [
        "Gender", "Age at enrollment", "Marital status",
        "Nacionality", "International", "Displaced",
        "Educational special needs",
        "Scholarship holder",
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation",
    ],
}

VARIANT_LABELS = {
    "full": "Full model (baseline)",
    "no_sensitive": "No Gender & Scholarship",
    "no_demographic": "No demographics (7)",
    "no_demo_socio": "No demo + socio (12)",
}
VARIANT_COLORS = {
    "full": "#0b0b0b",
    "no_sensitive": "#2a78d6",
    "no_demographic": "#eb6834",
    "no_demo_socio": "#1baf7a",
}
VARIANT_MARKERS = {
    "full": "o",
    "no_sensitive": "s",
    "no_demographic": "^",
    "no_demo_socio": "D",
}


def plot_ablation_tradeoff(task: str, summary: pd.DataFrame,
                           gaps: pd.DataFrame, highlight_model: str = "XGBoost"):
    """3-panel stage comparison: Performance, |ΔTPR| (Gender), |ΔTPR| (Scholarship)."""
    _style()
    perf_col = "dropout_roc_auc"
    perf_title = "Dropout ROC-AUC (vs rest)" if task == "multiclass" else "ROC-AUC (Binary)"

    s_mod = summary[summary["model"] == highlight_model]
    g_mod = gaps[gaps["model"] == highlight_model]
    variants = [v for v in VARIANT_LABELS if v in s_mod["variant"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    offsets = np.linspace(-0.09, 0.09, len(variants))

    # Panel 1: Performance
    ax = axes[0]
    for off, v in zip(offsets, variants):
        gv = s_mod[s_mod["variant"] == v].set_index("stage").reindex(STAGE_ORDER)
        x = [i + off for i in range(3)]
        ax.errorbar(
            x, gv[f"{perf_col}_mean"], yerr=gv[f"{perf_col}_std"],
            lw=2, marker=VARIANT_MARKERS[v], ms=5.8, mec=SURFACE, mew=1.1,
            capsize=2.5, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
        )
    ax.set_xticks(range(3), STAGE_LABELS)
    ax.set_xlim(-0.35, 2.35)
    ax.set_title(perf_title, loc="left", fontsize=10.5, fontweight="bold")
    _clean(ax)

    # Panel 2 & 3: |ΔTPR| for Gender and Scholarship
    for ax, attr, title in [
        (axes[1], "Gender", "Gender |Δ TPR| (Equal Opportunity)"),
        (axes[2], "Scholarship holder", "Scholarship |Δ TPR| (Equal Opportunity)"),
    ]:
        ga = g_mod[g_mod["attribute"] == attr]
        for off, v in zip(offsets, variants):
            gv = ga[ga["variant"] == v].set_index("stage").reindex(STAGE_ORDER)
            x = [i + off for i in range(3)]
            ax.errorbar(
                x, gv["abs_gap_tpr_fold_mean"], yerr=gv["abs_gap_tpr_fold_std"],
                lw=2, marker=VARIANT_MARKERS[v], ms=5.8, mec=SURFACE, mew=1.1,
                capsize=2.5, color=VARIANT_COLORS[v], label=VARIANT_LABELS[v],
            )
        ax.set_xticks(range(3), STAGE_LABELS)
        ax.set_xlim(-0.35, 2.35)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        _clean(ax)

    axes[0].legend(frameon=False, loc="lower right", fontsize=8.3, labelcolor=INK2)
    tlabel = "3-class" if task == "multiclass" else "binary"
    fig.suptitle(f"Demographic ablation trade-off: accuracy vs fairness ({highlight_model}, {tlabel})",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(
        0.012, 0.012,
        "Paired nested stratified 5-fold CV (seed 42). Fairness gaps are evaluated against true "
        "Gender and Scholarship labels even when excluded from training.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    C.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"ablation_tradeoff_{task}.png", dpi=180)
    plt.close(fig)


def run(task: str, stages: list, models: list, variants: list,
        n_iter: int, n_jobs: int, n_boot: int, highlight_model: str):
    out = C.RESULTS_DIR / task
    base_folds_path = out / "cv_fold_metrics.csv"
    base_oof_path = out / "oof_predictions.csv"
    assert base_folds_path.exists() and base_oof_path.exists(), (
        f"Missing Part 1 baseline outputs in {out}/. Run `python -m src.train --task {task}` first."
    )

    df = load_data()
    scoring = "f1_macro" if task == "multiclass" else "roc_auc"

    # Load Part 1 baseline so 'full' is strictly identical to Part 1
    base_folds = pd.read_csv(base_folds_path)
    base_folds = base_folds[
        base_folds["stage"].isin(stages) & base_folds["model"].isin(models)
    ].copy()
    base_folds.insert(0, "variant", "full")

    base_oof = pd.read_csv(base_oof_path)
    base_oof = base_oof[
        base_oof["stage"].isin(stages) & base_oof["model"].isin(models)
    ].copy()
    base_oof.insert(0, "variant", "full")

    rows = []
    oof_frames = [base_oof]

    for variant in variants:
        drop_cols = ABLATION_DROPS[variant]
        for stage in stages:
            X_full, y, idx, classes = get_xy(df, stage, task)
            present_drops = [c for c in drop_cols if c in X_full.columns]
            X = X_full.drop(columns=present_drops)
            assert not set(present_drops).intersection(X.columns), "Ablation leakage!"

            # Identical outer CV splits as src/train.py
            outer = StratifiedKFold(C.N_OUTER, shuffle=True, random_state=C.SEED)
            splits = list(outer.split(X, y))

            for name in models:
                t0 = time.time()
                oof = np.zeros((len(y), len(classes)))
                fold_id = np.zeros(len(y), dtype=int)

                for k, (tr, te) in enumerate(splits):
                    pipe, space = build(name, list(X.columns), len(classes))
                    search = RandomizedSearchCV(
                        pipe, space, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs,
                        cv=StratifiedKFold(C.N_INNER, shuffle=True, random_state=C.SEED),
                        random_state=C.SEED + k, refit=True,
                    )
                    search.fit(X.iloc[tr], y[tr])
                    oof[te] = search.predict_proba(X.iloc[te])
                    fold_id[te] = k
                    rows.append({
                        "variant": variant, "stage": stage, "model": name,
                        "fold": k, "n_features": X.shape[1],
                        **fold_metrics(y[te], oof[te], classes),
                    })

                f = pd.DataFrame(oof, columns=[f"p_{c}" for c in classes])
                f.insert(0, "variant", variant)
                f.insert(1, "row", idx)
                f.insert(2, "stage", stage)
                f.insert(3, "model", name)
                f.insert(4, "fold", fold_id)
                f["y_true"] = [classes[i] for i in y]
                f["y_pred"] = [classes[i] for i in oof.argmax(1)]
                oof_frames.append(f)

                last = pd.DataFrame(rows[-C.N_OUTER:])
                print(
                    f"[{task}] {variant:<15} {stage} {name:<18} "
                    f"n_feat={X.shape[1]:2d} f1_macro={last.f1_macro.mean():.3f} "
                    f"dropout_auc={last.dropout_roc_auc.mean():.3f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    all_folds = pd.concat([base_folds, pd.DataFrame(rows)], ignore_index=True)
    all_oof = pd.concat(oof_frames, ignore_index=True)

    metric_cols = [c for c in all_folds.columns
                   if c not in ("variant", "stage", "model", "fold", "n_features")]
    summ = (
        all_folds.groupby(["variant", "stage", "model", "n_features"], sort=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()

    # Add paired delta vs 'full' baseline
    base_ref = summ[summ["variant"] == "full"].set_index(["stage", "model"])
    for key_m in ("dropout_roc_auc_mean", "f1_macro_mean", "dropout_f1_mean", "dropout_recall_mean"):
        ref_vals = summ.set_index(["stage", "model"]).index.map(base_ref[key_m])
        summ[f"delta_{key_m}"] = summ[key_m] - ref_vals

    # Compute fairness across full + all ablated variants
    _, gap_df, _ = evaluate_oof_fairness(all_oof, df, n_boot=n_boot)

    all_folds.to_csv(out / "ablation_fold_metrics.csv", index=False, float_format="%.5f")
    summ.to_csv(out / "ablation_summary.csv", index=False, float_format="%.5f")
    all_oof.to_csv(out / "ablation_oof_predictions.csv", index=False)
    gap_df.to_csv(out / "ablation_fairness_gaps.csv", index=False, float_format="%.5f")

    if highlight_model in models and set(STAGE_ORDER).issubset(set(stages)):
        plot_ablation_tradeoff(task, summ, gap_df, highlight_model=highlight_model)

    print(f"\n[{task}] Saved ablation results to {out}/ablation_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--stages", nargs="+", default=STAGE_ORDER)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--variants", nargs="+", default=list(ABLATION_DROPS.keys()),
                    choices=list(ABLATION_DROPS.keys()))
    ap.add_argument("--n-iter", type=int, default=C.N_ITER)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--highlight-model", default="XGBoost", choices=MODEL_NAMES)
    a = ap.parse_args()
    run(a.task, a.stages, a.models, a.variants, a.n_iter, a.n_jobs,
        a.n_boot, a.highlight_model)
