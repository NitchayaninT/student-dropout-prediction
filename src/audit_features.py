"""Data quality, leakage control verification, and feature sensitivity audit.

Directly addresses Member 2's review items from README.md:
  1. Programmatic data & leakage audit -> results/data_audit_report.json
  2. Domain-ratio refinement (`ratio_flagged`): adds explicit binary indicators
     for zero-enrolled students (n=180) so Approval ratio = 0 does not conflate
     un-enrolled students with students who enrolled and failed all units (n=538).
  3. Financial proxy/retrospective leakage check (`no_financial_flags`): tests
     T1 and T2 performance when Debtor & Tuition fees up to date are removed.

Usage:
  python -m src.audit_features --task multiclass
  python -m src.audit_features --task binary
  python -m src.audit_features --audit-only
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from .fairness import (COLORS, INK2, MUTED, SURFACE, _clean, _style)
from .features import get_xy, load_data
from .models import MODEL_NAMES, build
from .train import fold_metrics

FINANCIAL_COLS = ["Debtor", "Tuition fees up to date"]

EXP_LABELS = {
    "baseline": "Baseline (Part 1)",
    "ratio_flagged": "+ Zero-enrolled flags",
    "no_financial_flags": "No financial flags",
}
EXP_COLORS = {
    "baseline": "#0b0b0b",
    "ratio_flagged": "#1baf7a",
    "no_financial_flags": "#e34948",
}


def run_structural_audit(df: pd.DataFrame) -> dict:
    """Verify schema, missing values, leakage guards, scale mismatch, edge cases."""
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": df[C.TARGET].value_counts().to_dict(),
        "binary_cohort_n": int((df[C.TARGET] != "Enrolled").sum()),
        "enrolled_cohort_n": int((df[C.TARGET] == "Enrolled").sum()),
        "scale_ranges": {
            "Previous qualification (grade)": [
                float(df["Previous qualification (grade)"].min()),
                float(df["Previous qualification (grade)"].max()),
            ],
            "Admission grade": [
                float(df["Admission grade"].min()),
                float(df["Admission grade"].max()),
            ],
            "Curricular units 1st sem (grade)": [
                float(df["Curricular units 1st sem (grade)"].min()),
                float(df["Curricular units 1st sem (grade)"].max()),
            ],
            "Curricular units 2nd sem (grade)": [
                float(df["Curricular units 2nd sem (grade)"].min()),
                float(df["Curricular units 2nd sem (grade)"].max()),
            ],
        },
        "macro_unique_cohort_profiles": int(
            len(df[["Unemployment rate", "Inflation rate", "GDP"]].drop_duplicates())
        ),
        "edge_cases": {},
        "leakage_guard_checks": {},
    }

    for sem in ("1st", "2nd"):
        enr = df[f"Curricular units {sem} sem (enrolled)"]
        app = df[f"Curricular units {sem} sem (approved)"]
        grd = df[f"Curricular units {sem} sem (grade)"]
        no_enr = df[enr == 0]
        fail_all = df[(enr > 0) & (app == 0)]
        report["edge_cases"][f"{sem}_sem"] = {
            "enrolled_eq_0_n": int(len(no_enr)),
            "enrolled_eq_0_targets": no_enr[C.TARGET].value_counts().to_dict(),
            "enrolled_eq_0_courses": {
                str(k): int(v) for k, v in no_enr["Course"].value_counts().head(5).items()
            },
            "enrolled_gt_0_approved_eq_0_n": int(len(fail_all)),
            "enrolled_gt_0_approved_eq_0_targets": fail_all[C.TARGET].value_counts().to_dict(),
            "grade_eq_0_n": int((grd == 0).sum()),
        }

    same_zero = (
        (df["Curricular units 1st sem (enrolled)"] == 0)
        == (df["Curricular units 2nd sem (enrolled)"] == 0)
    ).all()
    report["edge_cases"]["same_students_zero_enrolled_both_sems"] = bool(same_zero)

    # Check Grade change anomalies where one sem grade == 0 and the other > 0
    g1 = df["Curricular units 1st sem (grade)"]
    g2 = df["Curricular units 2nd sem (grade)"]
    report["edge_cases"]["grade_change_one_sem_zero_n"] = int(
        (((g1 == 0) & (g2 > 0)) | ((g1 > 0) & (g2 == 0))).sum()
    )

    # Verify chronological guards across tasks and stages
    for task in ("multiclass", "binary"):
        checks = {}
        for stage in ("T0", "T1", "T2"):
            X, y, idx, classes = get_xy(df, stage, task)
            checks[stage] = {
                "n_rows": int(len(X)),
                "n_features": int(X.shape[1]),
                "has_sem1_cols": any("1st sem" in c for c in X.columns),
                "has_sem2_cols": any("2nd sem" in c or c == "Grade change" for c in X.columns),
                "has_financial_cols": any(c in FINANCIAL_COLS for c in X.columns),
            }
        assert not checks["T0"]["has_sem1_cols"] and not checks["T0"]["has_financial_cols"]
        assert not checks["T0"]["has_sem2_cols"] and not checks["T1"]["has_sem2_cols"]
        report["leakage_guard_checks"][task] = checks

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / "data_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _modify_features(X: pd.DataFrame, stage: str, experiment: str) -> pd.DataFrame:
    """Apply leakage-safe feature transformation for one sensitivity experiment."""
    X = X.copy()
    if experiment == "ratio_flagged":
        # Keep Approval ratio numeric (0.0 when enrolled==0) so StandardScaler in LR works,
        # and add explicit binary flags distinguishing zero-enrolled from zero-approved.
        if stage in ("T1", "T2"):
            X["No units enrolled 1st sem"] = (
                X["Curricular units 1st sem (enrolled)"] == 0
            ).astype(int)
        if stage == "T2":
            X["No units enrolled 2nd sem"] = (
                X["Curricular units 2nd sem (enrolled)"] == 0
            ).astype(int)
            X["Zero grade either sem"] = (
                (X["Curricular units 1st sem (grade)"] == 0)
                | (X["Curricular units 2nd sem (grade)"] == 0)
            ).astype(int)
    elif experiment == "no_financial_flags":
        X = X.drop(columns=[c for c in FINANCIAL_COLS if c in X.columns])
        assert not set(FINANCIAL_COLS).intersection(X.columns)
    else:
        raise ValueError(experiment)
    return X


def evaluate_zero_enrolled_subgroup(oof_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Compare Dropout predictions specifically on the 180 enrolled==0 students."""
    enr1 = df["Curricular units 1st sem (enrolled)"]
    app1 = df["Curricular units 1st sem (approved)"]
    tag = np.where(enr1 == 0, "No units enrolled (n=180)",
                   np.where(app1 == 0, "Enrolled >0, Approved =0 (n=538)",
                            "Approved >0"))
    meta = pd.DataFrame({"row": df.index, "enrollment_group": tag})
    m = oof_df.merge(meta, on="row", how="left", validate="many_to_one")
    m["is_d"] = (m["y_true"] == "Dropout").astype(int)
    m["pred_d"] = (m["y_pred"] == "Dropout").astype(int)

    rows = []
    for (exp, stage, model, egrp), g in m.groupby(
        ["experiment", "stage", "model", "enrollment_group"], sort=False
    ):
        if egrp == "Approved >0":
            continue
        has_both = g["is_d"].nunique() == 2
        rows.append({
            "experiment": exp, "stage": stage, "model": model,
            "enrollment_group": egrp, "n": len(g),
            "actual_dropout_rate": float(g["is_d"].mean()),
            "predicted_dropout_rate": float(g["pred_d"].mean()),
            "dropout_recall": float(recall_score(g["is_d"], g["pred_d"], zero_division=0)),
            "dropout_f1": float(f1_score(g["is_d"], g["pred_d"], zero_division=0)),
            "dropout_roc_auc": float(roc_auc_score(g["is_d"], g["p_Dropout"])) if has_both else np.nan,
            "brier": float(brier_score_loss(g["is_d"], g["p_Dropout"])),
        })
    return pd.DataFrame(rows)


def plot_sensitivity(task: str, summary: pd.DataFrame):
    """2-panel (T1, T2) grouped bar comparison of Dropout ROC-AUC across experiments."""
    _style()
    stages = [s for s in ("T1", "T2") if s in summary["stage"].unique()]
    models = [m for m in MODEL_NAMES if m in summary["model"].unique()]
    exps = [e for e in EXP_LABELS if e in summary["experiment"].unique()]

    fig, axes = plt.subplots(1, len(stages), figsize=(5.4 * len(stages), 4.2), sharey=True,
                             squeeze=False)
    width = 0.24
    x = np.arange(len(models))

    for ax, stage in zip(axes[0], stages):
        s = summary[summary["stage"] == stage]
        for idx_e, exp in enumerate(exps):
            se = s[s["experiment"] == exp].set_index("model").reindex(models)
            pos = x + (idx_e - (len(exps) - 1) / 2) * width
            means = se["dropout_roc_auc_mean"].to_numpy()
            stds = se["dropout_roc_auc_std"].to_numpy()
            ax.bar(pos, means, width=width * 0.92, yerr=stds, capsize=2.5,
                   color=EXP_COLORS[exp], label=EXP_LABELS[exp], zorder=3)
            for px, val in zip(pos, means):
                if not np.isnan(val):
                    ax.text(px, val + 0.004, f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7.8, color=INK2)
        ax.set_xticks(x, models, fontsize=9.5)
        ax.set_title(f"{stage}: {'after semester 1' if stage == 'T1' else 'after semester 2'}",
                     loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Dropout ROC-AUC" if stage == stages[0] else "")
        lo = max(0.70, float(summary["dropout_roc_auc_mean"].min()) - 0.05)
        ax.set_ylim(lo, 1.0)
        _clean(ax)

    axes[0][0].legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK2)
    tlabel = "3-class" if task == "multiclass" else "binary"
    fig.suptitle(f"Feature engineering & financial leakage sensitivity ({tlabel})",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(
        0.012, 0.012,
        "Mean ± SD over paired nested stratified 5-fold CV (seed 42). Compares Part 1 baseline "
        "against adding zero-enrolled flags and removing in-year financial flags.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    C.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIGURES_DIR / f"audit_sensitivity_{task}.png", dpi=180)
    plt.close(fig)


def run(task: str, models: list, n_iter: int, n_jobs: int, audit_only: bool):
    df = load_data()
    report = run_structural_audit(df)
    print(f"[audit] Wrote {C.RESULTS_DIR / 'data_audit_report.json'} "
          f"(rows={report['n_rows']}, missing={report['missing_values']}, "
          f"macro_cohorts={report['macro_unique_cohort_profiles']}, "
          f"zero_enrolled_sem1={report['edge_cases']['1st_sem']['enrolled_eq_0_n']})")
    if audit_only:
        return

    out = C.RESULTS_DIR / task
    base_folds = pd.read_csv(out / "cv_fold_metrics.csv")
    base_oof = pd.read_csv(out / "oof_predictions.csv")

    stages = ["T1", "T2"]
    base_folds = base_folds[
        base_folds["stage"].isin(stages) & base_folds["model"].isin(models)
    ].copy()
    base_folds.insert(0, "experiment", "baseline")

    base_oof = base_oof[
        base_oof["stage"].isin(stages) & base_oof["model"].isin(models)
    ].copy()
    base_oof.insert(0, "experiment", "baseline")

    scoring = "f1_macro" if task == "multiclass" else "roc_auc"
    rows = []
    oof_frames = [base_oof]

    for exp in ("ratio_flagged", "no_financial_flags"):
        for stage in stages:
            X_base, y, idx, classes = get_xy(df, stage, task)
            X = _modify_features(X_base, stage, exp)

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
                        "experiment": exp, "stage": stage, "model": name,
                        "fold": k, "n_features": X.shape[1],
                        **fold_metrics(y[te], oof[te], classes),
                    })

                f = pd.DataFrame(oof, columns=[f"p_{c}" for c in classes])
                f.insert(0, "experiment", exp)
                f.insert(1, "row", idx)
                f.insert(2, "stage", stage)
                f.insert(3, "model", name)
                f.insert(4, "fold", fold_id)
                f["y_true"] = [classes[i] for i in y]
                f["y_pred"] = [classes[i] for i in oof.argmax(1)]
                oof_frames.append(f)

                last = pd.DataFrame(rows[-C.N_OUTER:])
                print(
                    f"[{task}] {exp:<18} {stage} {name:<18} "
                    f"n_feat={X.shape[1]:2d} f1_macro={last.f1_macro.mean():.3f} "
                    f"dropout_auc={last.dropout_roc_auc.mean():.3f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    all_folds = pd.concat([base_folds, pd.DataFrame(rows)], ignore_index=True)
    all_oof = pd.concat(oof_frames, ignore_index=True)

    metric_cols = [c for c in all_folds.columns
                   if c not in ("experiment", "stage", "model", "fold", "n_features")]
    summ = (
        all_folds.groupby(["experiment", "stage", "model", "n_features"], sort=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()

    base_ref = summ[summ["experiment"] == "baseline"].set_index(["stage", "model"])
    for key_m in ("dropout_roc_auc_mean", "f1_macro_mean", "dropout_f1_mean", "dropout_recall_mean"):
        ref_vals = summ.set_index(["stage", "model"]).index.map(base_ref[key_m])
        summ[f"delta_{key_m}"] = summ[key_m] - ref_vals

    zero_sub = evaluate_zero_enrolled_subgroup(all_oof, df)

    all_folds.to_csv(out / "audit_fold_metrics.csv", index=False, float_format="%.5f")
    summ.to_csv(out / "audit_summary.csv", index=False, float_format="%.5f")
    all_oof.to_csv(out / "audit_oof_predictions.csv", index=False)
    zero_sub.to_csv(out / "audit_zero_enrolled_subgroup.csv", index=False, float_format="%.5f")

    plot_sensitivity(task, summ)
    print(f"\n[{task}] Saved sensitivity audit results to {out}/audit_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--n-iter", type=int, default=C.N_ITER)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--audit-only", action="store_true",
                    help="only run structural & leakage verification without CV retraining")
    a = ap.parse_args()
    run(a.task, a.models, a.n_iter, a.n_jobs, a.audit_only)
