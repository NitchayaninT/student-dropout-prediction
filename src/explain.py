"""SHAP explainability for the tuned XGBoost model of each stage (T0 / T1 / T2).

What is explained: the model's Dropout score on the log-odds scale (binary task:
log-odds of Dropout vs Graduate; 3-class task: the Dropout margin before softmax).
A positive SHAP value pushes a student towards Dropout.

SHAP values are exact TreeSHAP, computed with XGBoost's built-in implementation
(`pred_contribs` / `pred_interactions`). These are the numbers
`shap.TreeExplainer(model.named_steps["clf"])` returns for an XGBoost model, so the
`shap` package (and its numba dependency) is not needed to run this script.

Outputs
  figures/shap_bar_<task>.png                       global importance, T0 | T1 | T2
  figures/shap_beeswarm_<task>_<stage>.png          global beeswarm per stage
  figures/shap_group_share_<task>.png               attribution share by feature group
  figures/shap_interaction_heatmap_<task>_<stage>.png
  figures/shap_interaction_dependence_<task>_<stage>.png
  results/<task>/shap_importance.csv                mean |SHAP| per stage, feature, class
  results/<task>/shap_interactions.csv              mean |interaction| per stage, pair

Usage:  python -m src.explain --task multiclass      (or: --task binary)
"""
import argparse
import json
import os
import textwrap

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D

from . import config as C
from .features import get_xy, load_data, stage_features
from .models import build

# ---- style (same tokens as plot_results.py) --------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NOMINAL_GREY = "#a3a199"
# feature value low -> high (blue <-> red around a neutral midpoint)
VALUE_CMAP = LinearSegmentedColormap.from_list("value", [BLUE, "#d9d8d0", RED])
# magnitude: one hue, light -> dark
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"])
# when a feature first becomes available
GROUPS = {"T0": ("Known at enrollment (T0)", BLUE),
          "T1": ("Semester 1 records + fees (T1)", ORANGE),
          "T2": ("Semester 2 records (T2)", AQUA)}
STAGE_TITLES = {"T0": "T0: at enrollment", "T1": "T1: after semester 1",
                "T2": "T2: after semester 2"}
BINARY_LABELS = {"Gender": ("Female", "Male"),
                 "Daytime/evening attendance": ("Evening", "Daytime")}


def _style():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.edgecolor": AXIS, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _clean(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)


def short(name: str) -> str:
    return name.replace("Curricular units", "Units")


def feature_group(name: str) -> str:
    """Earliest stage at which a feature is available."""
    return next(s for s in ("T0", "T1", "T2") if name in stage_features(s))


def task_label(task):
    return "3-class model" if task == "multiclass" else "binary model"


def task_note(task):
    """Footer sentence that spells out the binary task."""
    if task == "multiclass":
        return ""
    return " Binary model: Dropout vs Graduate, Enrolled students excluded."


# ---- model + SHAP values ----------------------------------------------------
def load_model(task, stage, X, y, classes):
    """Final tuned XGBoost pipeline. models/ is git-ignored, so when the joblib
    is absent the model is re-fitted on all data with the tuned hyperparameters
    stored in results/<task>/best_params.json (what train.py's refit does)."""
    path = C.MODELS_DIR / task / f"{stage}_XGBoost.joblib"
    if path.exists():
        bundle = joblib.load(path)
        assert bundle["features"] == list(X.columns), "feature list changed; re-run train.py"
        return bundle["model"]
    params = json.load(open(C.RESULTS_DIR / task / "best_params.json"))
    pipe, _ = build("XGBoost", list(X.columns), len(classes))
    pipe.set_params(**params[f"{stage}/XGBoost"]["final_model"]).fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "features": list(X.columns), "classes": classes}, path)
    return pipe


def tree_shap(task, stage, model, X, classes, recompute=False):
    """Exact TreeSHAP values per explained class, and interaction values for Dropout.

    Returns (phi {class: [n, f]}, inter [n, f, f]). Rows of `inter` sum to
    phi["Dropout"]; the diagonal holds main effects and each off-diagonal cell holds
    half of the pairwise interaction effect (full effect = inter[i, j] + inter[j, i]).
    Interaction values are the slow part, so the arrays are cached next to the model
    (models/ is git-ignored). The cache is refreshed when the model file is newer,
    or with --recompute.
    """
    names = [classes[1]] if len(classes) == 2 else classes   # binary: log-odds of class 1
    cache = C.MODELS_DIR / task / f"{stage}_XGBoost_shap.npz"
    model_file = C.MODELS_DIR / task / f"{stage}_XGBoost.joblib"
    fresh = cache.exists() and cache.stat().st_mtime >= model_file.stat().st_mtime
    if fresh and not recompute:
        z = np.load(cache)
        if z["phi"].shape == (len(X), len(names), X.shape[1]):
            return dict(zip(names, z["phi"].transpose(1, 0, 2))), z["inter"]
    booster = model.named_steps["clf"].get_booster()
    booster.set_param({"nthread": os.cpu_count()})   # the model was fitted with n_jobs=1
    dm = xgb.DMatrix(X)
    k = names.index("Dropout")
    phi = booster.predict(dm, pred_contribs=True).reshape(len(X), len(names), -1)
    inter = booster.predict(dm, pred_interactions=True).reshape(
        len(X), len(names), X.shape[1] + 1, -1)[:, k]
    margin = booster.predict(dm, output_margin=True).reshape(len(X), len(names))
    # local accuracy: contributions + bias reproduce the model output exactly
    assert np.allclose(phi.sum(-1), margin, atol=1e-2), "SHAP additivity check failed"
    assert np.allclose(inter.sum(-1), phi[:, k], atol=1e-2), "interaction additivity failed"
    phi, inter = phi[:, :, :-1], inter[:, :-1, :-1]          # drop the bias term
    np.savez_compressed(cache, phi=phi, inter=inter)
    return dict(zip(names, phi.transpose(1, 0, 2))), inter


def pair_strength(inter):
    """Mean |full pairwise interaction effect| (symmetric, zero diagonal)."""
    m = np.abs(inter + inter.transpose(0, 2, 1)).mean(0)
    np.fill_diagonal(m, 0.0)
    return m


def top_pairs(X, strength):
    feats = list(X.columns)
    iu = np.triu_indices(len(feats), 1)
    t = pd.DataFrame({"feature_a": [feats[i] for i in iu[0]],
                      "feature_b": [feats[j] for j in iu[1]],
                      "mean_abs_interaction": strength[iu]})
    t = t.sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)
    t.insert(0, "rank", t.index + 1)
    return t


# ---- global: bar ------------------------------------------------------------
def plot_bar(task, res, top=10):
    stages = list(res)
    fig, axes = plt.subplots(1, len(stages), figsize=(4.6 * len(stages) + 0.4, 4.9),
                             sharex=True, squeeze=False)
    xmax = max(r["imp"].max() for r in res.values())
    for ax, stage in zip(axes[0], stages):
        imp = res[stage]["imp"].sort_values(ascending=False)
        head = imp.head(top)[::-1]
        ax.barh(range(len(head)), head.values, height=0.62,
                color=[GROUPS[feature_group(f)][1] for f in head.index])
        for i, v in enumerate(head.values):
            ax.text(v + xmax * 0.015, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(head)), [short(f) for f in head.index], fontsize=9)
        ax.set_xlim(0, xmax * 1.14)
        ax.set_title(f"{STAGE_TITLES[stage]}  ({len(imp)} features)", loc="left",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("mean |SHAP value|  (log-odds of Dropout)", fontsize=9)
        _clean(ax)
    used = [g for g in GROUPS if any(feature_group(f) == g for r in res.values()
                                     for f in r["imp"].index)]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=GROUPS[g][1]) for g in used],
               [GROUPS[g][0] for g in used], frameon=False, ncol=len(used),
               loc="upper left", bbox_to_anchor=(0.008, 0.925), fontsize=9,
               labelcolor=INK2, handlelength=1.1, handleheight=1.0, columnspacing=1.6)
    fig.suptitle(f"What drives predicted dropout at each stage: top {top} features by "
                 f"global SHAP importance, {task_label(task)}",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.012, "Tuned XGBoost per stage, exact TreeSHAP on all students. "
             "Common x-scale, so bar lengths compare across stages. "
             "Bar colour = stage at which the feature first becomes available."
             + task_note(task), fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.9))
    fig.savefig(C.FIGURES_DIR / f"shap_bar_{task}.png", dpi=180)
    plt.close(fig)


# ---- global: beeswarm -------------------------------------------------------
def _swarm_offsets(v, rng, nbins=110, row_height=0.4):
    """Vertical offsets that stack points sharing an x-bin (as shap's beeswarm)."""
    quant = np.round(nbins * (v - v.min()) / (v.max() - v.min() + 1e-8))
    order = np.argsort(quant + rng.normal(size=len(v)) * 1e-6)
    ys, layer, last = np.zeros(len(v)), 0, -1
    for i in order:
        if quant[i] != last:
            layer = 0
        ys[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer, last = layer + 1, quant[i]
    return ys * (0.9 * row_height / np.max(np.abs(ys) + 1))


def _value_colors(x):
    """Feature value -> colour, scaled between the 5th and 95th percentile."""
    lo, hi = np.nanpercentile(x, [5, 95])
    if lo == hi:
        lo, hi = np.nanmin(x), np.nanmax(x)
    return VALUE_CMAP(np.clip((x - lo) / (hi - lo + 1e-12), 0, 1))


def plot_beeswarm(task, stage, r, top=15):
    rng = np.random.default_rng(C.SEED)
    X, phi, imp = r["X"], r["phi"], r["imp"].sort_values(ascending=False)
    head = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(head) + 1.9))
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    has_nominal = False
    for row, f in enumerate(head.index):
        j = X.columns.get_loc(f)
        v, x = phi[:, j], X[f].to_numpy(dtype=float)
        if f in C.NOMINAL:           # integer codes with no order: colour would mislead
            colors, has_nominal = np.tile(to_rgba(NOMINAL_GREY), (len(v), 1)), True
        else:
            colors = _value_colors(x)
        p = rng.permutation(len(v))  # random draw order so no colour hides the other
        ax.scatter(v[p], row + _swarm_offsets(v, rng)[p], c=colors[p], s=7, lw=0,
                   alpha=0.9, rasterized=True, zorder=2)
    ax.set_yticks(range(len(head)), [short(f) for f in head.index], fontsize=9.5)
    ax.set_ylim(-0.7, len(head) - 0.3)
    ax.set_xlabel("SHAP value: impact on log-odds of Dropout   "
                  "(← lowers dropout risk  |  raises dropout risk →)", fontsize=9)
    _clean(ax, grid="y")
    ax.grid(axis="y", color=GRID, lw=0.6, ls=(0, (1, 3)))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=VALUE_CMAP), ax=ax, ticks=[0, 1],
                      aspect=38, pad=0.02, fraction=0.03)
    cb.ax.set_yticklabels(["Low", "High"], fontsize=9)
    cb.set_label("Feature value", fontsize=9, labelpad=-14, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    if has_nominal:
        ax.legend([Line2D([], [], marker="o", ls="", color=NOMINAL_GREY, ms=5)],
                  ["nominal code (categories have no low/high order)"], frameon=False,
                  loc="lower right", bbox_to_anchor=(1.0, 0.995), fontsize=8.5,
                  labelcolor=INK2, handletextpad=0.2, borderaxespad=0)
    covered = head.sum() / imp.sum()
    fig.suptitle(f"SHAP summary, {STAGE_TITLES[stage]}: top {len(head)} of {len(imp)} "
                 f"features, {task_label(task)}", x=0.012, ha="left", fontsize=12.5,
                 fontweight="bold")
    fig.text(0.012, 0.012, f"One dot per student (n = {len(X):,}); features ordered by mean "
             f"|SHAP|. The features shown carry {covered:.0%} of total attribution.\n"
             "Tuned XGBoost, exact TreeSHAP." + task_note(task), fontsize=8.5, color=MUTED,
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(C.FIGURES_DIR / f"shap_beeswarm_{task}_{stage}.png", dpi=180)
    plt.close(fig)


# ---- global: share of attribution by feature group --------------------------
def plot_group_share(task, res):
    stages = list(res)
    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(stages) + 1.9))
    for row, stage in enumerate(stages[::-1]):
        imp = res[stage]["imp"]
        share = imp.groupby([feature_group(f) for f in imp.index]).sum() / imp.sum()
        left = 0.0
        for g, (_, color) in GROUPS.items():
            w = share.get(g, 0.0)
            if w <= 0:
                continue
            ax.barh(row, w, left=left, height=0.58, color=color, edgecolor=SURFACE, lw=2)
            if w > 0.045:
                ax.text(left + w / 2, row, f"{w:.0%}", ha="center", va="center",
                        color="#ffffff", fontsize=10, fontweight="bold")
            left += w
    ax.set_yticks(range(len(stages)), [STAGE_TITLES[s] for s in stages[::-1]], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of total mean |SHAP|", fontsize=9)
    _clean(ax)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in GROUPS.values()],
              [n for n, _ in GROUPS.values()], frameon=False, ncol=3, loc="lower left",
              bbox_to_anchor=(-0.01, 1.0), fontsize=9, labelcolor=INK2, handlelength=1.1,
              handleheight=1.0, columnspacing=1.6)
    fig.suptitle("Where the model's dropout signal comes from as records accumulate, "
                 f"{task_label(task)}", x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.015, "Each bar: one stage's tuned XGBoost; mean |SHAP| (log-odds of "
             "Dropout) summed within the stage at which a feature first becomes available."
             + task_note(task).replace(" Binary", "\nBinary"), fontsize=8.5,
             color=MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.045 if task == "multiclass" else 0.09, 1, 0.93))
    fig.savefig(C.FIGURES_DIR / f"shap_group_share_{task}.png", dpi=180)
    plt.close(fig)


# ---- interactions -----------------------------------------------------------
def plot_interaction_heatmap(task, stage, r, top=10):
    X, strength = r["X"], r["pairs"]
    feats = list(X.columns)
    picked = []                      # features of the strongest pairs, in rank order
    for p in r["pair_table"].itertuples():
        picked += [f for f in (p.feature_a, p.feature_b) if f not in picked]
        if len(picked) >= top:
            break
    keep = np.array([feats.index(f) for f in picked[:top]])
    keep = keep[np.argsort(-strength[np.ix_(keep, keep)].sum(0))]
    m = strength[np.ix_(keep, keep)]
    shown = np.where(np.tril(np.ones_like(m, dtype=bool), -1), m, np.nan)[1:, :-1]
    labels = [short(feats[k]) for k in keep]
    n = len(labels) - 1
    fig, ax = plt.subplots(figsize=(9.2, 6.9))
    im = ax.imshow(shown, cmap=SEQ_CMAP, vmin=0, vmax=np.nanmax(shown))
    # 2px surface gap between cells
    ax.set_xticks(np.arange(1, n) - 0.5, minor=True)
    ax.set_yticks(np.arange(1, n) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(range(n), labels[:-1], rotation=38, ha="right",
                  rotation_mode="anchor", fontsize=9)
    ax.set_yticks(range(n), labels[1:], fontsize=9)
    for (i, j), v in np.ndenumerate(shown):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > 0.5 * np.nanmax(shown) else INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=30)
    cb.set_label("mean |SHAP interaction value|  (log-odds of Dropout)", fontsize=9,
                 color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5)
    main = np.abs(np.einsum("nii->ni", r["inter"])).mean(0).sum()
    off = strength.sum() / 2
    fig.suptitle(f"Pairwise feature interactions, {STAGE_TITLES[stage]}, {task_label(task)}",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    note = (f"Grid: the {len(labels)} features that form the strongest pairs (each cell = one "
            f"pair). Summed over all {len(feats)} features, pairwise interactions make up "
            f"{off / (main + off):.0%} of mean absolute attribution and main effects the rest. "
            "Tuned XGBoost, exact TreeSHAP interaction values on all students."
            + task_note(task))
    fig.text(0.012, 0.012, textwrap.fill(note, 132), fontsize=8.5, color=MUTED,
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(C.FIGURES_DIR / f"shap_interaction_heatmap_{task}_{stage}.png", dpi=180)
    plt.close(fig)


def _orient(X, a, b):
    """x-axis = the ordered, finer-grained feature of a pair; the other is colour.
    None when neither feature is ordered with more than two values."""
    axis_ok = [f for f in (a, b) if f not in C.NOMINAL and X[f].nunique() > 2]
    if not axis_ok:
        return None
    x = max(axis_ok, key=lambda f: X[f].nunique())
    return x, (b if x == a else a)


def plot_interaction_dependence(task, stage, r, n_panels=4):
    """The strongest pairs, one panel each: the pair's interaction effect against
    the value of one feature, coloured by the other."""
    rng = np.random.default_rng(C.SEED)
    X, inter, cols = r["X"], r["inter"], list(r["X"].columns)
    panels = []
    for p in r["pair_table"].itertuples():
        o = _orient(X, p.feature_a, p.feature_b)
        if o:
            panels.append((*o, p.rank, p.mean_abs_interaction))
        if len(panels) == n_panels:
            break
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.6, 3.9 * nrow + 1.3), squeeze=False)
    dot = dict(s=9, lw=0, rasterized=True)
    for ax, (fx, fc, rank, strength) in zip(axes.ravel(), panels):
        i, j = cols.index(fx), cols.index(fc)
        y = inter[:, i, j] + inter[:, j, i]            # the pair's full effect
        x, c = X[fx].to_numpy(dtype=float), X[fc].to_numpy(dtype=float)
        discrete = X[fx].nunique() <= 30
        if discrete:                                   # spread integer x-values
            x = x + rng.uniform(-0.28, 0.28, len(x)) * np.min(np.diff(np.unique(x)))
        o = rng.permutation(len(x))                    # random draw order
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        if X[fc].nunique() == 2 or fc in C.NOMINAL:
            if X[fc].nunique() == 2:
                levels = list(np.unique(c))
                names = [f"{short(fc)}: {n}" for n in BINARY_LABELS.get(fc, ("No", "Yes"))]
            else:   # nominal code: highlight the two categories that drive the effect
                levels = list(pd.Series(np.abs(y)).groupby(c).sum().nlargest(2).index)
                names = [f"{short(fc)} code {int(v)}" for v in levels]
                rest = ~np.isin(c, levels)
                ax.scatter(x[rest], y[rest], color=NOMINAL_GREY, alpha=0.35, **dot)
                names.append(f"other {short(fc)} codes")
            sel = o[np.isin(c[o], levels)]
            ax.scatter(x[sel], y[sel], c=np.where(c[sel] == levels[0], BLUE, ORANGE),
                       alpha=0.75, **dot)
            ax.legend([Line2D([], [], marker="o", ls="", color=k, ms=5)
                       for k in (BLUE, ORANGE, NOMINAL_GREY)[:len(names)]], names,
                      frameon=False, fontsize=8.5, labelcolor=INK2, handletextpad=0.2,
                      loc="best", borderpad=0.2)
        else:
            ax.scatter(x[o], y[o], c=_value_colors(c)[o], alpha=0.8, **dot)
            lo, hi = np.nanpercentile(c, [5, 95])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=VALUE_CMAP), ax=ax, ticks=[0, 1],
                              fraction=0.04, pad=0.02, aspect=26)
            cb.ax.set_yticklabels([f"≤ {lo:.3g}", f"≥ {hi:.3g}"], fontsize=8.5)
            cb.set_label(short(fc), fontsize=8.5, labelpad=-12, color=INK2)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=0)
        if not discrete:                               # trim the axis to where the data are
            a, b = np.percentile(x, [0.5, 99.5])
            ax.set_xlim(a - 0.04 * (b - a), b + 0.04 * (b - a))
        ax.set_xlabel(short(fx), fontsize=9.5)
        ax.set_ylabel("interaction effect (log-odds of Dropout)", fontsize=8.5)
        ax.set_title(f"{short(fx)}  ×  {short(fc)}", loc="left", fontsize=10,
                     fontweight="bold", pad=21)
        ax.text(0, 1.03, f"rank {rank} of {len(r['pair_table'])} pairs  ·  "
                f"mean |interaction| {strength:.3f}", transform=ax.transAxes, fontsize=8.5,
                color=MUTED)
        _clean(ax, grid="both")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"Strongest feature interactions, {STAGE_TITLES[stage]}, {task_label(task)}",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    note = ("Each dot is a student. y: SHAP interaction value of the pair, i.e. how much the "
            "two features together shift the Dropout log-odds beyond their separate main "
            "effects. Above zero raises dropout risk. Separate y-scale per panel; integer "
            "x-values are jittered. Pairs of two unordered or two-valued features cannot be "
            f"drawn this way and are listed in results/{task}/shap_interactions.csv only."
            + task_note(task))
    fig.text(0.012, 0.01, textwrap.fill(note, 158), fontsize=8.5, color=MUTED,
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.07, 1, 0.955), h_pad=2.2, w_pad=2.0)
    fig.savefig(C.FIGURES_DIR / f"shap_interaction_dependence_{task}_{stage}.png", dpi=180)
    plt.close(fig)


# ---- driver -----------------------------------------------------------------
def run(task, stages, top, recompute=False):
    _style()
    C.FIGURES_DIR.mkdir(exist_ok=True)
    out = C.RESULTS_DIR / task
    out.mkdir(parents=True, exist_ok=True)
    df = load_data()
    res, imp_rows, pair_frames = {}, [], []
    for stage in stages:
        X, y, _, classes = get_xy(df, stage, task)
        model = load_model(task, stage, X, y, classes)
        by_class, inter = tree_shap(task, stage, model, X, classes, recompute)
        phi = by_class["Dropout"]
        r = {"X": X, "phi": phi, "inter": inter, "pairs": pair_strength(inter),
             "imp": pd.Series(np.abs(phi).mean(0), index=X.columns)}
        r["pair_table"] = top_pairs(X, r["pairs"])
        res[stage] = r

        t = pd.DataFrame({"stage": stage, "feature": X.columns,
                          "available_from": [feature_group(f) for f in X.columns]})
        for c, p in by_class.items():
            t[f"mean_abs_shap_{c}"] = np.abs(p).mean(0)
        t["share_Dropout"] = t["mean_abs_shap_Dropout"] / t["mean_abs_shap_Dropout"].sum()
        # direction: does a higher feature value raise (+) or lower (-) the Dropout score?
        t["corr_value_shap_Dropout"] = [
            np.nan if f in C.NOMINAL or np.std(phi[:, j]) == 0
            else np.corrcoef(X[f].to_numpy(dtype=float), phi[:, j])[0, 1]
            for j, f in enumerate(X.columns)]
        t = t.sort_values("mean_abs_shap_Dropout", ascending=False)
        t.insert(1, "rank", range(1, len(t) + 1))
        imp_rows.append(t)
        pair_frames.append(r["pair_table"].assign(stage=stage).iloc[:, [4, 0, 1, 2, 3]])

        plot_beeswarm(task, stage, r, top)
        plot_interaction_heatmap(task, stage, r)
        plot_interaction_dependence(task, stage, r)
        print(f"[{task}] {stage}: top features  "
              + ", ".join(f"{f} ({v:.2f})" for f, v in t.set_index("feature")
                          ["mean_abs_shap_Dropout"].head(5).items()), flush=True)
        print(f"[{task}] {stage}: top pairs     "
              + "; ".join(f"{p.feature_a} x {p.feature_b} ({p.mean_abs_interaction:.3f})"
                          for p in r["pair_table"].head(3).itertuples()), flush=True)

    plot_bar(task, res)
    plot_group_share(task, res)
    pd.concat(imp_rows).to_csv(out / "shap_importance.csv", index=False, float_format="%.5f")
    pd.concat(pair_frames).to_csv(out / "shap_interactions.csv", index=False,
                                  float_format="%.5f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--stages", nargs="+", default=["T0", "T1", "T2"])
    ap.add_argument("--top", type=int, default=15, help="features shown in each beeswarm")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore cached SHAP arrays in models/<task>/")
    a = ap.parse_args()
    run(a.task, a.stages, a.top, a.recompute)
