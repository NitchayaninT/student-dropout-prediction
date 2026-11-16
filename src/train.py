"""3-stage experiment: tune + evaluate LR / RF / XGBoost at T0, T1, T2.

Nested cross-validation:
  outer  Stratified 5-fold  -> unbiased performance estimate (same folds for
                               every stage and model, so results are paired)
  inner  Stratified 3-fold  -> RandomizedSearchCV picks hyperparameters using
                               the outer-training part only

Usage:  python -m src.train --task multiclass      (or: --task binary)
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from .features import get_xy, load_data
from .models import MODEL_NAMES, build


def fold_metrics(y_true, proba, classes):
    d = classes.index("Dropout")
    pred = proba.argmax(1)
    is_d, p_d = (y_true == d).astype(int), proba[:, d]
    m = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        # Dropout-class metrics (Dropout vs rest)
        "dropout_roc_auc": roc_auc_score(is_d, p_d),
        "dropout_pr_auc": average_precision_score(is_d, p_d),
        "dropout_f1": f1_score(is_d, pred == d),
        "dropout_recall": recall_score(is_d, pred == d),
        "dropout_precision": precision_score(is_d, pred == d, zero_division=0),
    }
    if len(classes) > 2:
        m["roc_auc_macro_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                               average="macro")
        for i, c in enumerate(classes):
            m[f"f1_{c.lower()}"] = f1_score(y_true == i, pred == i)
    return m


def run(task, stages, models, n_iter, n_jobs, save_models):
    out = C.RESULTS_DIR / task
    out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    scoring = "f1_macro" if task == "multiclass" else "roc_auc"
    rows, oof_frames, best, cms = [], [], {}, {}

    for stage in stages:
        X, y, idx, classes = get_xy(df, stage, task)
        outer = StratifiedKFold(C.N_OUTER, shuffle=True, random_state=C.SEED)
        splits = list(outer.split(X, y))
        for name in models:
            t0 = time.time()
            oof = np.zeros((len(y), len(classes)))
            fold_id = np.zeros(len(y), dtype=int)
            best[f"{stage}/{name}"] = {"outer_folds": []}
            for k, (tr, te) in enumerate(splits):
                pipe, space = build(name, list(X.columns), len(classes))
                search = RandomizedSearchCV(
                    pipe, space, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs,
                    cv=StratifiedKFold(C.N_INNER, shuffle=True, random_state=C.SEED),
                    random_state=C.SEED + k, refit=True)
                search.fit(X.iloc[tr], y[tr])
                oof[te] = search.predict_proba(X.iloc[te])
                fold_id[te] = k
                rows.append({"stage": stage, "model": name, "fold": k,
                             "n_features": X.shape[1],
                             **fold_metrics(y[te], oof[te], classes)})
                best[f"{stage}/{name}"]["outer_folds"].append(
                    {kk: (vv.item() if hasattr(vv, "item") else vv)
                     for kk, vv in search.best_params_.items()})
            cms[f"{stage}/{name}"] = confusion_matrix(y, oof.argmax(1)).tolist()
            f = pd.DataFrame(oof, columns=[f"p_{c}" for c in classes])
            f.insert(0, "row", idx); f.insert(1, "stage", stage)
            f.insert(2, "model", name); f.insert(3, "fold", fold_id)
            f["y_true"] = [classes[i] for i in y]
            f["y_pred"] = [classes[i] for i in oof.argmax(1)]
            oof_frames.append(f)

            if save_models:   # final model on all data, for SHAP / deployment
                pipe, space = build(name, list(X.columns), len(classes))
                final = RandomizedSearchCV(
                    pipe, space, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs,
                    cv=StratifiedKFold(C.N_OUTER, shuffle=True, random_state=C.SEED),
                    random_state=C.SEED).fit(X, y)
                best[f"{stage}/{name}"]["final_model"] = {
                    kk: (vv.item() if hasattr(vv, "item") else vv)
                    for kk, vv in final.best_params_.items()}
                (C.MODELS_DIR / task).mkdir(parents=True, exist_ok=True)
                joblib.dump({"model": final.best_estimator_, "features": list(X.columns),
                             "classes": classes},
                            C.MODELS_DIR / task / f"{stage}_{name}.joblib")
            last = pd.DataFrame(rows[-C.N_OUTER:])
            print(f"[{task}] {stage} {name:<18} f1_macro={last.f1_macro.mean():.3f} "
                  f"dropout_auc={last.dropout_roc_auc.mean():.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    folds = pd.DataFrame(rows)
    folds.to_csv(out / "cv_fold_metrics.csv", index=False)
    metric_cols = [c for c in folds.columns if c not in ("stage", "model", "fold", "n_features")]
    summ = folds.groupby(["stage", "model", "n_features"], sort=False)[metric_cols] \
                .agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.reset_index().to_csv(out / "cv_summary.csv", index=False)
    pd.concat(oof_frames).to_csv(out / "oof_predictions.csv", index=False)
    json.dump(best, open(out / "best_params.json", "w"), indent=2)
    json.dump({"classes": classes, "matrices": cms},
              open(out / "confusion_matrices.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--stages", nargs="+", default=["T0", "T1", "T2"])
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--n-iter", type=int, default=C.N_ITER)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-save-models", action="store_true")
    a = ap.parse_args()
    run(a.task, a.stages, a.models, a.n_iter, a.n_jobs, not a.no_save_models)
