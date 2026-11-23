"""Compare regenerated outputs (results/, figures/) with a reference copy of the same tree.

    python -m src.verify_reproduction reference/ .     # reference root, new root

For every file in either tree it reports one status:
  identical      same bytes
  equal          different bytes, same values (CSV/JSON compared cell by cell)
  differs        values differ: the largest absolute difference and how many cells differ
  image differs  PNG pixels differ: the share of pixels that changed
  missing        in the reference but not regenerated
  extra          regenerated but not in the reference

It also prints the headline cross-validation table for each task next to the reference,
and writes everything to reproduction_report.csv in the new root.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMPARED_DIRS = ["results", "figures"]
HEADLINE = {  # columns of cv_summary.csv shown side by side with the reference
    "binary": ["dropout_roc_auc_mean", "dropout_pr_auc_mean", "dropout_f1_mean",
               "dropout_recall_mean"],
    "multiclass": ["f1_macro_mean", "dropout_roc_auc_mean", "dropout_pr_auc_mean",
                   "dropout_recall_mean"],
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root):
    out = set()
    for d in COMPARED_DIRS:
        base = root / d
        if base.exists():
            out |= {p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file()}
    return out


def compare_csv(a, b):
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        return {"status": "differs", "detail": f"shape/columns {x.shape} vs {y.shape}"}
    max_diff, n_diff, worst = 0.0, 0, ""
    for col in x.columns:
        xc, yc = x[col], y[col]
        if pd.api.types.is_numeric_dtype(xc) and pd.api.types.is_numeric_dtype(yc):
            xv, yv = xc.to_numpy(float), yc.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.where(both_nan, 0.0, np.abs(xv - yv))
            d = np.where(np.isnan(d), np.inf, d)          # NaN on one side only
            k = int((d > 0).sum())
            if k:
                n_diff += k
                if d.max() > max_diff:
                    max_diff, worst = float(d.max()), col
        else:                                              # labels such as y_pred
            k = int((xc.astype(str) != yc.astype(str)).sum())
            if k:
                n_diff += k
                worst = worst or col
    if n_diff == 0:
        return {"status": "equal", "detail": ""}
    return {"status": "differs", "max_abs_diff": max_diff, "cells_differ": n_diff,
            "cells": int(x.size), "detail": f"largest change in column '{worst}'"}


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_json(a, b):
    x = dict(_flatten(json.loads(a.read_text())))
    y = dict(_flatten(json.loads(b.read_text())))
    keys = set(x) | set(y)
    diff = sorted(k for k in keys if x.get(k, "<missing>") != y.get(k, "<missing>"))
    if not diff:
        return {"status": "equal", "detail": ""}
    num = [abs(x[k] - y[k]) for k in diff
           if isinstance(x.get(k), (int, float)) and isinstance(y.get(k), (int, float))]
    return {"status": "differs", "max_abs_diff": max(num) if num else np.nan,
            "cells_differ": len(diff), "cells": len(keys),
            "detail": "first differing keys: " + ", ".join(diff[:3])}


def compare_png(a, b):
    import matplotlib.image as mpimg
    x, y = mpimg.imread(a), mpimg.imread(b)
    if x.shape != y.shape:
        return {"status": "image differs", "detail": f"size {x.shape[:2]} vs {y.shape[:2]}"}
    changed = np.any(np.abs(x - y) > 1e-6, axis=-1) if x.ndim == 3 else np.abs(x - y) > 1e-6
    share = float(changed.mean())
    if share == 0:
        return {"status": "equal", "detail": "same pixels"}
    return {"status": "image differs", "max_abs_diff": share,
            "detail": f"{share:.2%} of pixels changed"}


def compare_file(rel, ref_root, new_root):
    a, b = ref_root / rel, new_root / rel
    if _sha(a) == _sha(b):
        return {"status": "identical", "detail": ""}
    suffix = a.suffix.lower()
    if suffix == ".csv":
        return compare_csv(a, b)
    if suffix == ".json":
        return compare_json(a, b)
    if suffix == ".png":
        return compare_png(a, b)
    return {"status": "differs", "detail": "bytes differ"}


def headline_table(ref_root, new_root, task):
    path = Path("results") / task / "cv_summary.csv"
    if not (ref_root / path).exists() or not (new_root / path).exists():
        return
    r = pd.read_csv(ref_root / path).set_index(["stage", "model"])
    n = pd.read_csv(new_root / path).set_index(["stage", "model"])
    cols = HEADLINE[task]
    print(f"--- {task}: cross-validated means, committed -> regenerated ---")
    rows = []
    for idx in r.index:
        row = {"stage": idx[0], "model": idx[1]}
        for c in cols:
            row[c.replace("_mean", "")] = f"{r.loc[idx, c]:.3f} -> {n.loc[idx, c]:.3f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    d = (n[cols] - r[cols]).abs().to_numpy().max()
    print(f"largest absolute change in these columns: {d:.2e}\n")


def verify(ref_root, new_root, tasks=("binary", "multiclass")):
    ref_root, new_root = Path(ref_root), Path(new_root)
    ref_files, new_files = _files(ref_root), _files(new_root)
    rows = []
    for rel in sorted(ref_files | new_files):
        if rel not in new_files:
            res = {"status": "missing", "detail": "not regenerated"}
        elif rel not in ref_files:
            res = {"status": "extra", "detail": "not in reference"}
        else:
            res = compare_file(rel, ref_root, new_root)
        rows.append({"file": rel, **res})
    rep = pd.DataFrame(rows)
    for col in ("max_abs_diff", "cells_differ", "cells"):
        if col not in rep:
            rep[col] = np.nan
    rep = rep[["file", "status", "max_abs_diff", "cells_differ", "cells", "detail"]]
    rep.to_csv(new_root / "reproduction_report.csv", index=False)

    print("=== reproduction check ===")
    for task in tasks:
        headline_table(ref_root, new_root, task)
    counts = rep["status"].value_counts()
    print("file status counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    bad = rep[~rep["status"].isin(["identical", "equal"])]
    if len(bad):
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 70)
        print("\nfiles that are not value-identical:")
        print(bad.to_string(index=False))
    data_like = rep[rep["file"].str.endswith((".csv", ".json"))]
    n_data_ok = data_like["status"].isin(["identical", "equal"]).sum()
    print(f"\nVERDICT: {n_data_ok} of {len(data_like)} result tables (CSV/JSON) match the "
          f"reference exactly; {int((rep['file'].str.endswith('.png') & rep['status'].isin(['identical', 'equal'])).sum())}"
          f" of {int(rep['file'].str.endswith('.png').sum())} figures are pixel-identical.")
    print(f"Full report: {new_root / 'reproduction_report.csv'}", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_root", type=Path)
    ap.add_argument("new_root", type=Path)
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"])
    a = ap.parse_args()
    verify(a.reference_root, a.new_root, a.tasks)
    sys.exit(0)
