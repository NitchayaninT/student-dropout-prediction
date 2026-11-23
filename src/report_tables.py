"""Print the tables and numbers quoted in the README, computed from data/ and results/.

    python -m src.report_tables > docs/results_tables.md

Nothing here retrains a model. Everything is read from the outputs of src.pipeline, plus
three summaries the proposal planned that the other scripts do not compute:
  - paired stage gains: per-fold change in Dropout ROC-AUC from T0 to T1 and T1 to T2
    (the outer folds are identical across stages, so the differences are paired)
  - Recall@K: share of all dropouts found in the K% of students with the highest
    out-of-fold Dropout probability, computed within each outer fold
  - paired bootstrap of the change in Equal Opportunity gap between the full model and
    each demographic-ablation variant
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import add_domain_features, load_data

MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
SHORT = {"LogisticRegression": "LR", "RandomForest": "RF", "XGBoost": "XGBoost"}
TASK_NAME = {"binary": "Binary (Dropout vs Graduate, n = 3,630)",
             "multiclass": "3-class (Dropout / Enrolled / Graduate, n = 4,424)"}


def ms(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join(lines)


def section(title):
    print(f"\n## {title}\n")


# ---- data ---------------------------------------------------------------------
def data_facts(df):
    section("Data")
    n = len(df)
    counts = df[C.TARGET].value_counts().reindex(C.CLASSES)
    print(md(pd.DataFrame({"Outcome": counts.index, "Students": counts.values,
                           "Share": [f"{v / n:.1%}" for v in counts.values]})))
    print(f"\nProgrammes: {df['Course'].nunique()}. Macro-economic profiles "
          f"(Unemployment, Inflation, GDP): "
          f"{df[['Unemployment rate', 'Inflation rate', 'GDP']].drop_duplicates().shape[0]}.")

    e = add_domain_features(df)
    d = (e[C.TARGET] == "Dropout")

    def rate(mask):
        return f"{d[mask].mean():.1%} (n = {int(mask.sum())})"

    age = e["Age at enrollment"]
    r1 = e["Approval ratio 1st sem"]
    enr1 = e["Curricular units 1st sem (enrolled)"]
    facts = [
        ("Women", rate(e["Gender"] == 0)), ("Men", rate(e["Gender"] == 1)),
        ("Scholarship holders", rate(e["Scholarship holder"] == 1)),
        ("No scholarship", rate(e["Scholarship holder"] == 0)),
        ("Age at enrollment 20 or under", rate(age <= 20)),
        ("Age at enrollment 25 to 29", rate((age >= 25) & (age <= 29))),
        ("Tuition fees not up to date", rate(e["Tuition fees up to date"] == 0)),
        ("Tuition fees up to date", rate(e["Tuition fees up to date"] == 1)),
        ("Sem 1: no units enrolled", rate(enr1 == 0)),
        ("Sem 1: enrolled, passed none", rate((enr1 > 0) & (r1 == 0))),
        ("Sem 1: passed more than 0 and up to half", rate((r1 > 0) & (r1 <= 0.5))),
        ("Sem 1: passed more than 80%", rate(r1 > 0.8)),
    ]
    print("\nObserved dropout rate by group (all 4,424 students):\n")
    print(md(pd.DataFrame(facts, columns=["Group", "Dropout rate"])))

    by_course = (e.assign(d=d).groupby("Course")["d"].agg(["mean", "size"])
                 .sort_values("mean"))
    big = by_course[by_course["size"] > 100]
    print(f"\nDropout rate by programme: {big['mean'].min():.1%} to {big['mean'].max():.1%} "
          f"among the {len(big)} programmes with more than 100 students.")


# ---- RQ1 ----------------------------------------------------------------------
def performance(task):
    s = pd.read_csv(C.RESULTS_DIR / task / "cv_summary.csv")
    if task == "binary":
        cols = [("dropout_roc_auc", "ROC-AUC"), ("dropout_pr_auc", "PR-AUC"),
                ("dropout_f1", "Dropout F1"), ("dropout_recall", "Dropout recall"),
                ("dropout_precision", "Dropout precision")]
    else:
        cols = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"),
                ("dropout_roc_auc", "Dropout ROC-AUC"), ("dropout_pr_auc", "Dropout PR-AUC"),
                ("dropout_f1", "Dropout F1"), ("dropout_recall", "Dropout recall")]
    rows = []
    for _, r in s.iterrows():
        row = {"Stage": r["stage"], "Model": SHORT[r["model"]], "Features": int(r["n_features"])}
        for c, name in cols:
            row[name] = ms(r[f"{c}_mean"], r[f"{c}_std"])
        rows.append(row)
    section(f"Cross-validated performance: {TASK_NAME[task]}")
    print("Mean ± SD over the 5 outer folds. Dropout is the positive class.\n")
    print(md(pd.DataFrame(rows)))


def paired_gains(task):
    f = pd.read_csv(C.RESULTS_DIR / task / "cv_fold_metrics.csv")
    p = f.pivot_table(index=["model", "fold"], columns="stage", values="dropout_roc_auc")
    rows = []
    for m in MODELS:
        g = p.loc[m]
        for a, b in (("T0", "T1"), ("T1", "T2")):
            d = g[b] - g[a]
            rows.append({"Model": SHORT[m], "Change": f"{a} to {b}",
                         "Mean": f"{d.mean():+.3f}", "Smallest fold": f"{d.min():+.3f}",
                         "Largest fold": f"{d.max():+.3f}"})
    section(f"Paired stage gains in Dropout ROC-AUC: {TASK_NAME[task]}")
    print("Same 5 outer folds at every stage, so each fold gives one paired difference.\n")
    print(md(pd.DataFrame(rows)))


def recall_at_k(task, ks=(0.10, 0.20, 0.30)):
    o = pd.read_csv(C.RESULTS_DIR / task / "oof_predictions.csv")
    base = (o.drop_duplicates("row")["y_true"] == "Dropout").mean()
    rows = []
    for (stage, model), g in o.groupby(["stage", "model"], sort=False):
        row = {"Stage": stage, "Model": SHORT[model]}
        for k in ks:
            rec, prec, cap = [], [], []
            for _, fg in g.groupby("fold"):
                y = (fg["y_true"] == "Dropout").to_numpy()
                n_top = int(round(k * len(fg)))
                top = np.argsort(-fg["p_Dropout"].to_numpy(), kind="stable")[:n_top]
                rec.append(y[top].sum() / y.sum())
                prec.append(y[top].mean())
                cap.append(min(1.0, n_top / y.sum()))
            # folds are identical across stages and models, so the cap is the same in every row
            row[f"Top {int(k * 100)}%: recall (max {np.mean(cap):.3f})"] = \
                ms(np.mean(rec), np.std(rec, ddof=1))
            row[f"Top {int(k * 100)}%: precision"] = ms(np.mean(prec), np.std(prec, ddof=1))
        rows.append(row)
    section(f"Recall@K and precision@K: {TASK_NAME[task]}")
    print("Students are ranked by out-of-fold Dropout probability within each outer fold. "
          "Recall@K is the share of the fold's dropouts found in its top K%; precision@K is the "
          f"share of that list who dropped out (mean ± SD over folds). Dropouts are {base:.1%} "
          "of students, so a list of K% cannot hold more than the 'max' share of them.\n")
    print(md(pd.DataFrame(rows)))


def already_left_check(df, model="XGBoost"):
    """How much of the stage gain survives without the students who passed nothing in
    semester 1 (many of whom had already stopped attending)? Pooled out-of-fold ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    o = pd.read_csv(C.RESULTS_DIR / "binary" / "oof_predictions.csv")
    o = o[o["model"] == model]
    enr = df["Curricular units 1st sem (enrolled)"]
    passed_none = ((enr > 0) & (df["Curricular units 1st sem (approved)"] == 0))
    no_exam = ((enr > 0) & (df["Curricular units 1st sem (evaluations)"] == 0))
    rows = []
    for stage, g in o.groupby("stage", sort=False):
        y = (g["y_true"] == "Dropout").to_numpy()
        keep = ~passed_none.reindex(g["row"]).to_numpy()
        rows.append({"Stage": stage,
                     "All students": f"{roc_auc_score(y, g['p_Dropout']):.3f}",
                     "Without students who passed nothing in semester 1":
                         f"{roc_auc_score(y[keep], g['p_Dropout'].to_numpy()[keep]):.3f}",
                     "Students kept": int(keep.sum())})
    section(f"Sensitivity: students who had already stopped attending, binary {model}")
    print(f"{int(no_exam.sum())} students enrolled in semester-1 units but sat no exam; "
          f"{(df.loc[no_exam, C.TARGET] == 'Dropout').mean():.0%} of them dropped out. "
          f"{int(passed_none.sum())} enrolled and passed nothing. Pooled out-of-fold ROC-AUC "
          "with and without them:\n")
    print(md(pd.DataFrame(rows)))


# ---- RQ2 ----------------------------------------------------------------------
def shap_tables(task, top=5):
    imp = pd.read_csv(C.RESULTS_DIR / task / "shap_importance.csv")
    inter = pd.read_csv(C.RESULTS_DIR / task / "shap_interactions.csv")
    rows = []
    for stage, g in imp.groupby("stage", sort=False):
        g = g.sort_values("rank").head(top)
        rows.append({"Stage": stage, **{f"#{int(r['rank'])}": f"{r['feature']} ({r['mean_abs_shap_Dropout']:.2f})"
                                       for _, r in g.iterrows()}})
    section(f"SHAP, top {top} features by mean |SHAP| on the Dropout log-odds: {TASK_NAME[task]}")
    print("Tuned XGBoost, fitted on all students of the task.\n")
    print(md(pd.DataFrame(rows)))

    share = imp.groupby(["stage", "available_from"])["mean_abs_shap_Dropout"].sum()
    tot = imp.groupby("stage")["mean_abs_shap_Dropout"].sum()
    srows = []
    for stage in ("T0", "T1", "T2"):
        srows.append({"Stage": stage, **{f"Features from {g}": f"{share.get((stage, g), 0) / tot[stage]:.0%}"
                                         for g in ("T0", "T1", "T2")}})
    print("\nShare of total mean |SHAP| by the stage at which a feature becomes available:\n")
    print(md(pd.DataFrame(srows)))

    prow = []
    for stage, g in inter.groupby("stage", sort=False):
        for _, r in g.sort_values("rank").head(3).iterrows():
            prow.append({"Stage": stage, "Rank": int(r["rank"]),
                         "Pair": f"{r['feature_a']} × {r['feature_b']}",
                         "Mean abs. interaction": f"{r['mean_abs_interaction']:.3f}"})
    print("\nStrongest pairwise SHAP interactions:\n")
    print(md(pd.DataFrame(prow)))


def fairness_table(task, model="XGBoost"):
    g = pd.read_csv(C.RESULTS_DIR / task / "fairness_gaps.csv")
    g = g[g["model"] == model]
    rows = []
    for _, r in g.iterrows():
        rows.append({
            "Stage": r["stage"], "Gap": f"{r['attribute']} ({r['gap_definition']})",
            "ΔTPR [95% CI]": f"{r['gap_tpr']:+.3f} [{r['gap_tpr_ci_lo']:+.3f}, {r['gap_tpr_ci_hi']:+.3f}]",
            "ΔFPR [95% CI]": f"{r['gap_fpr']:+.3f} [{r['gap_fpr_ci_lo']:+.3f}, {r['gap_fpr_ci_hi']:+.3f}]",
            "Δ selection rate": f"{r['gap_selection_rate']:+.3f}",
        })
    rows.sort(key=lambda x: (x["Gap"], x["Stage"]))
    section(f"Subgroup error-rate gaps, {model} out-of-fold: {TASK_NAME[task]}")
    print("Signed gap = group A minus group B; 0 is parity. Bootstrap 95% intervals "
          "(1,000 resamples within each group).\n")
    print(md(pd.DataFrame(rows)))


def _tpr_gap(y, pred, a, b):
    return pred[a & y].mean() - pred[b & y].mean()


def ablation_tables(task, df, model="XGBoost", n_boot=2000):
    s = pd.read_csv(C.RESULTS_DIR / task / "ablation_summary.csv")
    s = s[s["model"] == model]
    # ablation_summary.csv is rounded to 5 decimals; take the full-model value from
    # cv_summary.csv so it prints exactly as in the main table (0.80750 vs 0.8075022)
    cv = pd.read_csv(C.RESULTS_DIR / task / "cv_summary.csv")
    cv = cv[cv["model"] == model].set_index("stage")["dropout_roc_auc_mean"]
    rows = []
    for _, r in s.iterrows():
        auc = cv[r["stage"]] if r["variant"] == "full" else r["dropout_roc_auc_mean"]
        rows.append({"Stage": r["stage"], "Variant": r["variant"], "Features": int(r["n_features"]),
                     "Dropout ROC-AUC": f"{auc:.3f}",
                     "Δ vs full": "" if r["variant"] == "full" else f"{r['delta_dropout_roc_auc_mean']:+.3f}"})
    rows.sort(key=lambda x: (x["Stage"], list(("full", "no_sensitive", "no_demographic",
                                                "no_demo_socio")).index(x["Variant"])))
    section(f"Demographic ablation, {model}: {TASK_NAME[task]}")
    print("Removed features: `no_sensitive` = Gender, Scholarship holder; `no_demographic` = "
          "Gender, Age at enrollment, Marital status, Nacionality, International, Displaced, "
          "Educational special needs (Scholarship holder stays in); `no_demo_socio` = both lists "
          "plus the parents' qualifications and occupations (12 features).\n")
    print(md(pd.DataFrame(rows)))

    o = pd.read_csv(C.RESULTS_DIR / task / "ablation_oof_predictions.csv")
    o = o[o["model"] == model]
    groups = {"Gender": (df["Gender"].to_numpy() == 1, df["Gender"].to_numpy() == 0),
              "Scholarship holder": (df["Scholarship holder"].to_numpy() == 0,
                                     df["Scholarship holder"].to_numpy() == 1)}
    rng = np.random.default_rng(C.SEED)
    prow = []
    for stage in ("T0", "T1", "T2"):
        st = o[o["stage"] == stage]
        pred = st.pivot(index="row", columns="variant", values="y_pred")
        y_true = st[st["variant"] == "full"].set_index("row")["y_true"].reindex(pred.index)
        idx = pred.index.to_numpy()
        y = (y_true.to_numpy() == "Dropout")
        boot = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        full = pred["full"].to_numpy() == "Dropout"
        for attr, (ga, gb) in groups.items():
            a, b = ga[idx], gb[idx]
            bs_full = np.array([_tpr_gap(y[k], full[k], a[k], b[k]) for k in boot])
            for var in ("no_sensitive", "no_demographic", "no_demo_socio"):
                vp = pred[var].to_numpy() == "Dropout"
                change = _tpr_gap(y, vp, a, b) - _tpr_gap(y, full, a, b)
                bs = np.array([_tpr_gap(y[k], vp[k], a[k], b[k]) for k in boot]) - bs_full
                lo, hi = np.percentile(bs, [2.5, 97.5])
                prow.append({"Stage": stage, "Gap": attr, "Variant": var,
                             "Full": f"{_tpr_gap(y, full, a, b):+.3f}",
                             "Variant ΔTPR": f"{_tpr_gap(y, vp, a, b):+.3f}",
                             "Change [95% CI]": f"{change:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                             "Interval excludes 0": "yes" if (hi < 0 or lo > 0) else "no"})
    print(f"\nChange in the Equal Opportunity gap (ΔTPR) when features are removed. Paired "
          f"bootstrap over students ({n_boot:,} resamples; both models scored on the same "
          "resample; trained models held fixed):\n")
    print(md(pd.DataFrame(prow)))


def audit_table(task):
    s = pd.read_csv(C.RESULTS_DIR / task / "audit_summary.csv")
    rows = []
    for _, r in s[s["experiment"] != "baseline"].iterrows():
        rows.append({"Stage": r["stage"], "Model": SHORT[r["model"]], "Experiment": r["experiment"],
                     "Features": int(r["n_features"]),
                     "Macro F1 (Δ)": f"{r['f1_macro_mean']:.3f} ({r['delta_f1_macro_mean']:+.3f})",
                     "Dropout ROC-AUC (Δ)": f"{r['dropout_roc_auc_mean']:.3f} ({r['delta_dropout_roc_auc_mean']:+.3f})",
                     "Dropout F1 (Δ)": f"{r['dropout_f1_mean']:.3f} ({r['delta_dropout_f1_mean']:+.3f})"})
    section(f"Feature-timing and zero-enrolment checks: {TASK_NAME[task]}")
    print("`ratio_flagged` adds 0/1 flags for students with no enrolled units; "
          "`no_financial_flags` removes Debtor and Tuition fees up to date. Δ is against the "
          "same model and stage in the main run.\n")
    print(md(pd.DataFrame(rows)))


def main():
    df = load_data()
    print("# Result tables\n\nGenerated by `python -m src.report_tables` from `data/` and "
          "`results/`. Do not edit by hand; rerun the script after rerunning the pipeline.")
    data_facts(df)
    for task in ("binary", "multiclass"):
        performance(task)
        paired_gains(task)
        recall_at_k(task)
    already_left_check(df)
    for task in ("binary", "multiclass"):
        shap_tables(task)
    for task in ("binary", "multiclass"):
        fairness_table(task)
        ablation_tables(task, df)
        audit_table(task)


if __name__ == "__main__":
    main()
