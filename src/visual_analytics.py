"""Part 3: visual analytics of the students at T0, T1 and T2.

Reads data/data.csv and the out-of-fold predictions already in results/ (it retrains no
predictive model), and writes figures/va_*.{png,pdf,svg} and tables in results/visual/.

    python -m src.visual_analytics            # everything, a few minutes
    python -m src.visual_analytics --quick    # one UMAP setting, no stability runs

What it does
  1. Maps. For each stage, the stage's features are encoded as in the Logistic Regression
     pipeline (nominal codes one-hot, levels with fewer than 10 students grouped), with two
     changes for distances: 0/1 flags stay 0/1 instead of being standardised (a rare flag
     would otherwise put its few students far from everyone), and the macro-economic
     columns are left out (they only mark the 10 intake cohorts). The matrix is projected
     with PCA and with UMAP (n_neighbors 15, 30, 50; min_dist 0.1; seed 42). All 4,424
     students are mapped; no outcome is used to build the maps.
  2. Map quality. Trustworthiness at k = 10, the share of each student's 10 nearest
     neighbours with the same outcome (in 2-D and in the full space), and, for UMAP, the
     overlap of 10-nearest-neighbour sets across seeds.
  3. VAE rule, fixed before running: a tabular VAE is built only if every PCA and UMAP map
     of some stage has trustworthiness below 0.90, i.e. if no map keeps neighbourhoods.
  4. Clusters. k-means on the first 10 principal components, k = 2..8 chosen by
     silhouette; stability as the adjusted Rand index across seeds and 80% subsamples.
  5. Risk by programme and by group: observed dropout rate with a Wilson 95% interval
     next to the mean out-of-fold predicted risk of the binary XGBoost model.
  6. Semester transitions: dropout rate by semester-1 x semester-2 approval band.
  7. Risk tiers from T0 to T2 (low < 0.2, medium 0.2 to 0.5, high >= 0.5).
Rates and risks use the binary cohort (Dropout vs Graduate, 3,630 students). The 794
Enrolled students are in the maps but have no out-of-fold risk in the binary task.
"""
import argparse
import time

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import figstyle as fs
from .features import add_domain_features, load_data, stage_features

import matplotlib.pyplot as plt  # noqa: E402  (figstyle selects the Agg backend)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = C.RESULTS_DIR / "visual"
STAGES = ["T0", "T1", "T2"]
STAGE_NAMES = {"T0": "T0: at enrollment", "T1": "T1: after semester 1",
               "T2": "T2: after semester 2"}
MACRO = ["Unemployment rate", "Inflation rate", "GDP"]
NN_MAIN, NN_ALL, K = 30, (15, 30, 50), 10
VAE_THRESHOLD = 0.90
TIER_CUTS, TIER_NAMES = [0.2, 0.5], ["Low (< 0.2)", "Medium (0.2 to 0.5)", "High (≥ 0.5)"]
TIER_COLORS = ["#E3E3E3", fs.PALETTE["red_2"], fs.PALETTE["red_strong"]]
COURSE_NAMES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Mgmt",
    9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening)"}
APPLICATION_MODES = {
    1: "1st phase, general contingent", 2: "Ordinance 612/93",
    5: "1st phase, special contingent (Azores)", 7: "holders of other higher courses",
    10: "Ordinance 854-B/99", 15: "international student", 16: "1st phase, special contingent (Madeira)",
    17: "2nd phase, general contingent", 18: "3rd phase, general contingent",
    26: "Ordinance 533-A/99 b2 (different plan)", 27: "Ordinance 533-A/99 b3 (other institution)",
    39: "over 23 years old", 42: "transfer", 43: "change of course",
    44: "technological specialization diploma", 51: "change of institution/course",
    53: "short cycle diploma", 57: "change of institution/course (international)"}
QUALIFICATIONS = {  # UCI codes shared by previous and parents' qualification
    1: "secondary education (12th year)", 2: "bachelor's degree", 3: "higher education degree",
    4: "master's", 5: "doctorate", 6: "frequency of higher education",
    9: "12th year not completed", 10: "11th year not completed", 11: "7th year (old)",
    12: "other, 11th year", 13: "2nd year complementary high school", 14: "10th year",
    15: "10th year not completed", 18: "general commerce course",
    19: "basic education 3rd cycle (9th-11th year)", 20: "complementary high school",
    22: "technical-professional course", 25: "complementary high school, not concluded",
    26: "7th year", 27: "2nd cycle of general high school", 29: "9th year not completed",
    30: "8th year", 31: "general course of administration and commerce",
    33: "supplementary accounting and administration", 34: "unknown",
    35: "can't read or write", 36: "can read without 4th year",
    37: "basic education 1st cycle (4th/5th year)", 38: "basic education 2nd cycle (6th-8th year)",
    39: "technological specialization course", 40: "degree (1st cycle)",
    41: "specialized higher studies", 42: "professional higher technical course",
    43: "master (2nd cycle)", 44: "doctorate (3rd cycle)"}
CODE_NAMES = {"Course": COURSE_NAMES, "Application mode": APPLICATION_MODES,
              "Previous qualification": QUALIFICATIONS, "Mother's qualification": QUALIFICATIONS,
              "Father's qualification": QUALIFICATIONS}
FLAGS = ["Gender", "Scholarship holder", "Debtor", "Tuition fees up to date", "Displaced",
         "Daytime/evening attendance", "International", "Educational special needs"]
GROUPS = [  # attribute, {code: label}; Age is banded separately
    ("Gender", {0: "Female", 1: "Male"}),
    ("Scholarship holder", {1: "Scholarship", 0: "No scholarship"}),
    ("Tuition fees up to date", {1: "Fees up to date", 0: "Fees behind"}),
    ("Debtor", {0: "Not in debt", 1: "Debtor"}),
    ("Displaced", {0: "Not displaced", 1: "Displaced"}),
    ("Daytime/evening attendance", {1: "Daytime", 0: "Evening"}),
    ("International", {0: "Not international", 1: "International"}),
]
BANDS = ["No units\nenrolled", "Enrolled,\nno exams", "Exams,\nnone passed",
         "Passed\n≤ 50%", "Passed\n50 to 80%", "Passed\n> 80%"]


# ---- helpers --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Proportion and Wilson 95% interval."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, mid - half, mid + half


def knn(Z, k=K):
    """Indices of each row's k nearest neighbours, the row itself excluded."""
    return NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)


def agreement(labels, nbrs):
    labels = np.asarray(labels)
    return float((labels[nbrs] == labels[:, None]).mean())


def chance(labels):
    _, c = np.unique(np.asarray(labels), return_counts=True)
    p = c / c.sum()
    return float((p ** 2).sum())


def jaccard(a, b):
    inter = np.array([len(set(x) & set(y)) for x, y in zip(a, b)])
    return float((inter / (2 * a.shape[1] - inter)).mean())


def short(name):
    return (name.replace("Curricular units", "Units").replace(" (grade)", " grade")
            .replace("Daytime/evening attendance", "Daytime attendance"))


def load_inputs():
    e = add_domain_features(load_data())
    oof = pd.read_csv(C.RESULTS_DIR / "binary" / "oof_predictions.csv")
    oof = oof[oof["model"] == "XGBoost"]
    risk = oof.pivot(index="row", columns="stage", values="p_Dropout").reindex(e.index)
    return e, e[C.TARGET].to_numpy(), risk


def embedding_input(e, stage):
    cols = [c for c in stage_features(stage) if c not in MACRO]
    nominal = [c for c in cols if c in C.NOMINAL]
    binary = [c for c in cols if c not in C.NOMINAL and set(e[c].unique()) <= {0, 1}]
    numeric = [c for c in cols if c not in nominal and c not in binary]
    ct = ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                              sparse_output=False), nominal),
        ("bin", "passthrough", binary),
        ("num", StandardScaler(), numeric)])
    return ct.fit_transform(e[cols]).astype(float), cols


def fit_umap(X, nn, seed):
    U = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="euclidean",
                  random_state=seed).fit_transform(X)
    if not np.isfinite(U).all():
        raise ValueError("UMAP returned non-finite coordinates")
    return U


# ---- 1-3: maps, quality, VAE rule ---------------------------------------------------------
def run_maps(e, outcome, quick):
    cohort = pd.factorize(e[MACRO].astype(str).agg("|".join, axis=1))[0]
    course = e["Course"].to_numpy()
    rows, maps = [], {}
    for stage in STAGES:
        t0 = time.time()
        X, cols = embedding_input(e, stage)
        pca = PCA(n_components=10, svd_solver="full", random_state=C.SEED).fit(X)
        P = pca.transform(X)
        evr = pca.explained_variance_ratio_
        maps[stage] = {"X_dims": X.shape[1], "evr": evr, "PCA10": P, "PCA": P[:, :2]}
        for nn in ((NN_MAIN,) if quick else NN_ALL):
            maps[stage][f"UMAP-{nn}"] = fit_umap(X, nn, C.SEED)
        seed_jac = np.nan
        if not quick:
            nb_main = knn(maps[stage][f"UMAP-{NN_MAIN}"])
            extra = [fit_umap(X, NN_MAIN, C.SEED + s) for s in (1, 2)]
            maps[stage][f"UMAP-{NN_MAIN}-seed{C.SEED + 1}"] = extra[0]
            seed_jac = float(np.mean([jaccard(nb_main, knn(U)) for U in extra]))
        full_agree = agreement(outcome, knn(X))
        for name in [m for m in maps[stage] if m == "PCA" or m.startswith("UMAP")]:
            if "seed" in name:
                continue
            Y = maps[stage][name]
            nb = knn(Y)
            lifts = {}
            for f in FLAGS:
                if f in cols:
                    v = e[f].to_numpy()
                    ch = chance(v)
                    lifts[f] = (agreement(v, nb) - ch) / (1 - ch)
            top = sorted(lifts.items(), key=lambda kv: -kv[1])[:3]
            rows.append({
                "stage": stage, "map": name, "features": len(cols), "encoded_dims": X.shape[1],
                "trustworthiness_k10": trustworthiness(X, Y, n_neighbors=K),
                "outcome_agreement_2d": agreement(outcome, nb),
                "outcome_agreement_full": full_agree, "outcome_chance": chance(outcome),
                "cohort_agreement_2d": agreement(cohort, nb), "cohort_chance": chance(cohort),
                "programme_agreement_2d": agreement(course, nb), "programme_chance": chance(course),
                "pca_variance_2": evr[:2].sum(), "pca_variance_10": evr.sum(),
                "umap_seed_knn_jaccard": seed_jac if name == f"UMAP-{NN_MAIN}" else np.nan,
                "flags_organising_map": "; ".join(f"{k} {v:.2f}" for k, v in top),
            })
        print(f"[visual] {stage}: maps done in {time.time() - t0:.0f}s "
              f"({len(cols)} features, {X.shape[1]} encoded columns)", flush=True)
    return pd.DataFrame(rows), maps


def vae_decision(quality):
    lines = [f"Rule (fixed before running): build a tabular VAE only if every PCA and UMAP map "
             f"of some stage has trustworthiness (k = {K}) below {VAE_THRESHOLD:.2f}."]
    needed = False
    for stage, g in quality.groupby("stage", sort=False):
        best = g.loc[g["trustworthiness_k10"].idxmax()]
        gap = best["trustworthiness_k10"] < VAE_THRESHOLD
        needed |= gap
        lines.append(f"{stage}: best map {best['map']} with trustworthiness "
                     f"{best['trustworthiness_k10']:.3f}; PCA {g.loc[g['map'] == 'PCA', 'trustworthiness_k10'].iloc[0]:.3f}"
                     f" -> {'no map keeps neighbourhoods' if gap else 'at least one map keeps neighbourhoods'}")
    lines.append("Decision: VAE " + ("justified; build and compare it with PCA at equal dimension."
                                     if needed else "not justified; PCA and UMAP are kept."))
    return "\n".join(lines)


# ---- 4: clusters ---------------------------------------------------------------------------
def describe(e, stage, mask):
    """Up to four features that set a cluster apart, largest standardised difference first."""
    out = []
    for c in [c for c in stage_features(stage) if c not in MACRO]:
        if c in C.NOMINAL:
            share_all = e[c].value_counts(normalize=True)
            share_in = e.loc[mask, c].value_counts(normalize=True)
            top = share_in.index[0]
            if share_in.iloc[0] >= 0.25 and share_in.iloc[0] >= 2 * share_all[top]:
                label = CODE_NAMES.get(c, {}).get(top, f"code {top}")
                out.append((2.0 + share_in.iloc[0] - share_all[top],
                            f"{c} {label}: {share_in.iloc[0]:.0%} vs {share_all[top]:.0%}"))
            continue
        sd = e[c].std()
        if sd == 0:
            continue
        a, b = e.loc[mask, c].mean(), e[c].mean()
        smd = (a - b) / sd
        if set(e[c].unique()) <= {0, 1}:
            out.append((abs(smd), f"{short(c)}: {a:.0%} vs {b:.0%}"))
        else:
            out.append((abs(smd), f"{short(c)}: {a:.2f} vs {b:.2f}"))
    out.sort(key=lambda t: -t[0])
    return "; ".join(s for _, s in out[:4])


def run_clusters(e, outcome, risk, maps, quick):
    resolved = outcome != "Enrolled"
    sel, prof, labels = [], [], {}
    for stage in STAGES:
        P = maps[stage]["PCA10"]
        sil = {}
        for k in range(2, 9):
            lab = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit_predict(P)
            sil[k] = silhouette_score(P, lab)
        k_best = max(sil, key=sil.get)
        lab = KMeans(n_clusters=k_best, n_init=10, random_state=C.SEED).fit_predict(P)
        order = np.argsort(-np.bincount(lab))             # cluster 1 = largest
        lab = np.argsort(order)[lab]
        labels[stage] = lab
        ari_seed = ari_sub = np.nan
        if not quick:
            ari_seed = float(np.mean([adjusted_rand_score(
                lab, KMeans(n_clusters=k_best, n_init=10, random_state=C.SEED + s).fit_predict(P))
                for s in range(1, 6)]))
            rng = np.random.default_rng(C.SEED)
            subs = []
            for _ in range(10):
                idx = rng.choice(len(P), size=int(0.8 * len(P)), replace=False)
                km = KMeans(n_clusters=k_best, n_init=10, random_state=C.SEED).fit(P[idx])
                subs.append(adjusted_rand_score(lab, km.predict(P)))
            ari_sub = float(np.mean(subs))
        for k, s in sil.items():
            sel.append({"stage": stage, "k": k, "silhouette": s, "chosen": k == k_best,
                        "ari_across_seeds": ari_seed if k == k_best else np.nan,
                        "ari_across_80pct_subsamples": ari_sub if k == k_best else np.nan})
        for c in range(k_best):
            m = lab == c
            r = m & resolved
            p, lo, hi = wilson(int((outcome[r] == "Dropout").sum()), int(r.sum()))
            prof.append({"stage": stage, "cluster": c + 1, "n": int(m.sum()),
                         "share_of_students": m.mean(), "n_resolved": int(r.sum()),
                         "dropout_rate": p, "ci_lo": lo, "ci_hi": hi,
                         "enrolled_share": float((outcome[m] == "Enrolled").mean()),
                         "mean_oof_risk": float(risk.loc[r, stage].mean()),
                         "distinguishing_features": describe(e, stage, m)})
        print(f"[visual] {stage}: k = {k_best} clusters (silhouette {sil[k_best]:.3f}, "
              f"ARI seeds {ari_seed:.3f}, subsamples {ari_sub:.3f})", flush=True)
    return pd.DataFrame(sel), pd.DataFrame(prof), labels


# ---- 5-7: tables -----------------------------------------------------------------------------
def programme_table(e, outcome, risk):
    rows, t1 = [], {}
    for code, g in e[outcome != "Enrolled"].groupby("Course"):
        idx = g.index
        p, lo, hi = wilson(int((outcome[idx] == "Dropout").sum()), len(idx))
        row = {"course": code, "programme": COURSE_NAMES[code], "n_resolved": len(idx),
               "observed_dropout_rate": p, "ci_lo": lo, "ci_hi": hi}
        for s in STAGES:
            row[f"mean_risk_{s}"] = risk.loc[idx, s].mean()
        rows.append(row)
        t1[code] = risk.loc[idx, "T1"].to_numpy()
    return pd.DataFrame(rows).sort_values("observed_dropout_rate").reset_index(drop=True), t1


def group_table(e, outcome, risk):
    res = e[outcome != "Enrolled"]
    age = pd.cut(res["Age at enrollment"], [0, 20, 24, 29, 200],
                 labels=["Age 20 or under", "Age 21 to 24", "Age 25 to 29", "Age 30 or over"])
    specs = [("Age at enrollment", age.astype(str), list(age.cat.categories))]
    for col, names in GROUPS:
        specs.append((col, res[col].map(names), list(names.values())))
    rows = []
    for attr, values, levels in specs:
        for lev in levels:
            idx = values.index[values == lev]
            p, lo, hi = wilson(int((outcome[idx] == "Dropout").sum()), len(idx))
            row = {"attribute": attr, "group": lev, "n_resolved": len(idx),
                   "observed_dropout_rate": p, "ci_lo": lo, "ci_hi": hi}
            for s in STAGES:
                row[f"mean_risk_{s}"] = risk.loc[idx, s].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def semester_band(e, sem):
    enr = e[f"Curricular units {sem} sem (enrolled)"]
    ev = e[f"Curricular units {sem} sem (evaluations)"]
    ap = e[f"Curricular units {sem} sem (approved)"]
    ratio = ap / enr.where(enr > 0)
    return np.select([enr == 0, (ev == 0) & (ap == 0), ap == 0, ratio <= 0.5, ratio <= 0.8],
                     [0, 1, 2, 3, 4], default=5)


def transition_table(e, outcome):
    res = outcome != "Enrolled"
    b1, b2 = semester_band(e, "1st")[res], semester_band(e, "2nd")[res]
    d = outcome[res] == "Dropout"
    rows = []
    for i in range(6):
        for j in range(6):
            m = (b1 == i) & (b2 == j)
            n = int(m.sum())
            rows.append({"sem1_band": BANDS[i].replace("\n", " "), "sem2_band": BANDS[j].replace("\n", " "),
                         "i": i, "j": j, "n": n, "dropouts": int(d[m].sum()),
                         "dropout_rate": d[m].mean() if n else np.nan})
    return pd.DataFrame(rows)


def tier(p):
    return np.digitize(p, TIER_CUTS)


def risk_flow_table(outcome, risk):
    res = outcome != "Enrolled"
    rows = []
    for out in ("Dropout", "Graduate"):
        m = res & (outcome == out)
        for s in STAGES:
            t = tier(risk.loc[m, s].to_numpy())
            for k, name in enumerate(TIER_NAMES):
                rows.append({"outcome": out, "stage": s, "tier": name, "n": int((t == k).sum()),
                             "share": float((t == k).mean())})
    return pd.DataFrame(rows)


# ---- figures ---------------------------------------------------------------------------------
def _map_axes(ax, name, evr=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if name == "PCA":
        ax.set_xlabel(f"PC 1 ({evr[0]:.0%} of variance)")
        ax.set_ylabel(f"PC 2 ({evr[1]:.0%})")
    else:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")


def fig_maps_outcome(maps, outcome, n_counts):
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(len(outcome))
    cols = np.array([fs.OUTCOME_COLORS[o] for o in outcome])
    fig, axes = plt.subplots(2, 3, figsize=(fs.SLIDE_WIDTH, 8.0), layout="constrained")
    for j, stage in enumerate(STAGES):
        for i, name in enumerate(["PCA", f"UMAP-{NN_MAIN}"]):
            ax = axes[i, j]
            Y = maps[stage][name]
            ax.scatter(Y[order, 0], Y[order, 1], c=cols[order], s=4, alpha=0.6,
                       linewidths=0, rasterized=True)
            _map_axes(ax, "PCA" if i == 0 else "UMAP", maps[stage]["evr"])
            if i == 0:
                ax.set_title(STAGE_NAMES[stage], fontsize=15)
            fs.panel_label(ax, "abcdef"[i * 3 + j])
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=fs.OUTCOME_COLORS[o],
                      label=f"{o} (n = {n_counts[o]:,})") for o in ("Dropout", "Enrolled", "Graduate")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return fig


def fig_maps_risk(maps, outcome, risk):
    rng = np.random.default_rng(C.SEED)
    res = outcome != "Enrolled"
    idx_res = rng.permutation(np.flatnonzero(res))
    fig, axes = plt.subplots(2, 3, figsize=(fs.SLIDE_WIDTH, 7.6), layout="constrained")
    sc = None
    for j, stage in enumerate(STAGES):
        r = risk[stage].to_numpy()
        for i, name in enumerate(["PCA", f"UMAP-{NN_MAIN}"]):
            ax = axes[i, j]
            Y = maps[stage][name]
            sc = ax.scatter(Y[idx_res, 0], Y[idx_res, 1], c=r[idx_res], cmap=fs.RISK_CMAP,
                            vmin=0, vmax=1, s=4, alpha=0.8, linewidths=0, rasterized=True)
            _map_axes(ax, "PCA" if i == 0 else "UMAP", maps[stage]["evr"])
            if i == 0:
                ax.set_title(STAGE_NAMES[stage], fontsize=15)
            fs.panel_label(ax, "abcdef"[i * 3 + j])
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Out-of-fold P(dropout) at that stage")
    fig.text(0.5, 0.005, f"Students with a known outcome (n = {res.sum():,}). The {(~res).sum():,} "
             "Enrolled students have no out-of-fold risk in the binary task and are not shown.",
             ha="center", va="bottom", fontsize=12, color="#4D4D4D")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    return fig


def fig_quality(quality):
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.SLIDE_WIDTH, 4.8), layout="constrained")
    x = np.arange(len(STAGES))
    styles = {"PCA": ("#767676", "s"), "UMAP-15": (fs.PALETTE["green_3"], "o"),
              "UMAP-30": (fs.PALETTE["teal"], "o"), "UMAP-50": (fs.PALETTE["violet"], "o")}
    present = [m for m in styles if m in set(quality["map"])]
    offs = np.linspace(-0.18, 0.18, len(present))
    for off, m in zip(offs, present):
        g = quality[quality["map"] == m].set_index("stage").reindex(STAGES)
        a.plot(x + off, g["trustworthiness_k10"], ls="", marker=styles[m][1], ms=10,
               color=styles[m][0], label=m.replace("-", ", k = ") if m != "PCA" else "PCA")
    a.axhline(VAE_THRESHOLD, color="#4D4D4D", ls="--", lw=1.4)
    a.text(x[-1] + 0.35, VAE_THRESHOLD - 0.006, "VAE rule", ha="right", va="top", fontsize=12,
           color="#4D4D4D")
    a.set_xticks(x, STAGES)
    a.set_xlim(-0.45, len(STAGES) - 0.55)
    lo = min(quality["trustworthiness_k10"].min(), VAE_THRESHOLD) - 0.03
    a.set_ylim(lo, 1.0)
    a.set_ylabel(f"Trustworthiness (k = {K})")
    a.set_xlabel("Stage")
    a.legend(frameon=False, fontsize=12, loc="upper left", ncol=2)
    fs.panel_label(a, "a")
    cols = {"Full feature space": ("black", "D", "outcome_agreement_full", "PCA"),
            "PCA (2-D)": ("#767676", "s", "outcome_agreement_2d", "PCA"),
            f"UMAP (2-D, k = {NN_MAIN})": (fs.PALETTE["teal"], "o", "outcome_agreement_2d",
                                            f"UMAP-{NN_MAIN}")}
    offs = np.linspace(-0.15, 0.15, len(cols))
    for off, (label, (color, mk, col, m)) in zip(offs, cols.items()):
        g = quality[quality["map"] == m].set_index("stage").reindex(STAGES)
        b.plot(x + off, g[col], ls="", marker=mk, ms=10, color=color, label=label)
    ch = quality["outcome_chance"].iloc[0]
    b.axhline(ch, color="#4D4D4D", ls=":", lw=1.4)
    b.text(-0.4, ch + 0.008, "chance", va="bottom", fontsize=12, color="#4D4D4D")
    b.set_xticks(x, STAGES)
    b.set_xlim(-0.45, len(STAGES) - 0.55)
    b.set_ylim(ch - 0.04, 1.0)
    b.set_ylabel("Neighbours with the same outcome")
    b.set_xlabel("Stage")
    b.legend(frameon=False, fontsize=12, loc="upper left")
    fs.panel_label(b, "b")
    return fig


def fig_clusters(maps, labels, sel, prof, base_rate):
    ks = {s: int(sel[(sel.stage == s) & sel.chosen].k.iloc[0]) for s in STAGES}
    fig, axes = plt.subplots(2, 3, figsize=(fs.SLIDE_WIDTH, 8.4), layout="constrained",
                             height_ratios=[1.25, 1])
    rng = np.random.default_rng(C.SEED)
    for j, stage in enumerate(STAGES):
        ax = axes[0, j]
        Y = maps[stage][f"UMAP-{NN_MAIN}"]
        lab = labels[stage]
        order = rng.permutation(len(lab))
        cols = np.array(fs.CLUSTER_COLORS)[lab]
        ax.scatter(Y[order, 0], Y[order, 1], c=cols[order], s=4, alpha=0.7, linewidths=0,
                   rasterized=True)
        _map_axes(ax, "UMAP")
        ax.set_title(STAGE_NAMES[stage], fontsize=15)
        s = sel[(sel.stage == stage) & sel.chosen].iloc[0]
        note = f"k = {ks[stage]}, silhouette {s.silhouette:.2f}"
        if np.isfinite(s.ari_across_80pct_subsamples):
            note += f"\nstability (ARI) {s.ari_across_80pct_subsamples:.2f}"
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=12, va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
        fs.panel_label(ax, "abc"[j])
        bx = axes[1, j]
        p = prof[prof.stage == stage]
        xs = np.arange(len(p))
        for xi, (_, r) in zip(xs, p.iterrows()):
            c = fs.CLUSTER_COLORS[int(r.cluster) - 1]
            bx.plot([xi, xi], [r.ci_lo, r.ci_hi], color=c, lw=3)
            bx.plot(xi, r.dropout_rate, marker="o", ms=11, color=c, clip_on=False)
        bx.axhline(base_rate, color="#4D4D4D", ls="--", lw=1.4)
        bx.set_xticks(xs, [f"C{int(r.cluster)}\n{int(r.n_resolved):,}" for _, r in p.iterrows()])
        bx.set_xlim(-0.6, len(p) - 0.4)
        bx.set_ylim(0, 1.04)
        if j == 0:
            bx.set_ylabel("Dropout rate (95% CI)")
        else:
            bx.tick_params(labelleft=False)
        bx.set_xlabel("Cluster and n (known outcome)")
        fs.panel_label(bx, "def"[j])
    axes[1, 0].text(len(prof[prof.stage == "T0"]) - 0.45, base_rate + 0.02, "all students",
                    ha="right", va="bottom", fontsize=12, color="#4D4D4D")
    return fig


def _dot_rows(ax, labels, obs, lo, hi, preds, y=None):
    y = np.arange(len(labels)) if y is None else y
    offs = {"T0": -0.22, "T1": 0.0, "T2": 0.22}
    for s, vals in preds.items():
        ax.plot(vals, y + offs[s], ls="", marker="o", ms=8, color=fs.STAGE_RISK_COLORS[s],
                markeredgecolor="#4D4D4D", markeredgewidth=0.6)
    ax.hlines(y, lo, hi, color="black", lw=2)
    ax.plot(obs, y, ls="", marker="D", ms=8, color="black")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)


def _dot_legend():
    return [Line2D([], [], color="black", marker="D", ms=8, lw=2,
                   label="Observed dropout rate (95% CI)")] + [
        Line2D([], [], ls="", marker="o", ms=8, color=fs.STAGE_RISK_COLORS[s],
               markeredgecolor="#4D4D4D", markeredgewidth=0.6, label=f"Mean predicted, {s}")
        for s in STAGES]


def fig_programmes(prog, t1):
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.SLIDE_WIDTH, 8.6), layout="constrained",
                               sharey=True, width_ratios=[1.15, 1])
    labels = [f"{r.programme} ({r.n_resolved})" for _, r in prog.iterrows()]
    _dot_rows(a, labels, prog.observed_dropout_rate, prog.ci_lo, prog.ci_hi,
              {s: prog[f"mean_risk_{s}"] for s in STAGES})
    a.set_xlabel("Dropout rate, students with a known outcome")
    a.tick_params(axis="y", labelsize=12)
    fs.panel_label(a, "a")
    data = [t1[c] for c in prog.course]
    bp = b.boxplot(data, positions=np.arange(len(data)), vert=False, whis=(10, 90),
                   showfliers=False, widths=0.6, patch_artist=True, manage_ticks=False,
                   medianprops={"color": "black", "lw": 1.8})
    for box in bp["boxes"]:
        box.set(facecolor=fs.PALETTE["red_1"], edgecolor="#4D4D4D", lw=1)
    b.set_xlim(0, 1)
    b.set_xlabel("Out-of-fold P(dropout) at T1")
    b.grid(axis="x", color="#E6E6E6", lw=0.8)
    b.set_axisbelow(True)
    fs.panel_label(b, "b")
    fig.legend(handles=_dot_legend(), loc="outside lower center", ncol=4, frameon=False,
               fontsize=12)
    return fig


def fig_groups(groups):
    g = groups.iloc[::-1].reset_index(drop=True)          # first attribute at the top
    y, labels, gap, ypos = [], [], 0.0, 0.0
    prev = None
    for _, r in g.iterrows():
        if prev is not None and r.attribute != prev:
            ypos += 0.6
        y.append(ypos)
        labels.append(f"{r.group} ({r.n_resolved:,})")
        ypos += 1
        prev = r.attribute
    fig, ax = plt.subplots(figsize=(fs.SLIDE_WIDTH, 8.2), layout="constrained")
    _dot_rows(ax, labels, g.observed_dropout_rate, g.ci_lo, g.ci_hi,
              {s: g[f"mean_risk_{s}"] for s in STAGES}, y=np.array(y))
    ax.set_xlabel("Dropout rate, students with a known outcome")
    ax.tick_params(axis="y", labelsize=12)
    fig.legend(handles=_dot_legend(), loc="outside lower center", ncol=4, frameon=False,
               fontsize=12)
    return fig


def fig_transitions(tr):
    rate = np.full((6, 6), np.nan)
    n = np.zeros((6, 6), int)
    for _, r in tr.iterrows():
        n[r.i, r.j] = r.n
        if r.n >= 10:
            rate[r.i, r.j] = r.dropout_rate
    fig, ax = plt.subplots(figsize=(fs.SLIDE_WIDTH, 7.4), layout="constrained")
    cmap = fs.RISK_CMAP.with_extremes(bad="#FFFFFF")
    im = ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    for i in range(6):
        for j in range(6):
            if n[i, j] == 0:
                continue
            if n[i, j] < 10:
                ax.text(j, i, f"n = {n[i, j]}", ha="center", va="center", fontsize=11,
                        color="#767676")
                continue
            dark = rate[i, j] > 0.55
            k = int(tr[(tr.i == i) & (tr.j == j)].dropouts.iloc[0])
            pct = "100%" if k == n[i, j] else (">99%" if rate[i, j] >= 0.995 else f"{rate[i, j]:.0%}")
            ax.text(j, i, f"{pct}\nn = {n[i, j]:,}", ha="center", va="center",
                    fontsize=12, color="white" if dark else "black")
    ax.set_xticks(range(6), BANDS, fontsize=12)
    ax.set_yticks(range(6), BANDS, fontsize=12)
    ax.set_xlabel("Semester 2 approval band")
    ax.set_ylabel("Semester 1 approval band")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("Dropout rate (Dropout vs Graduate)")
    return fig


def fig_risk_flow(flow, outcome, risk):
    fig, axes = plt.subplots(1, 3, figsize=(fs.SLIDE_WIDTH, 5.0), layout="constrained",
                             width_ratios=[1.35, 1, 1])
    a = axes[0]
    pos, ticks, tlabels = 0.0, [], []
    for out in ("Dropout", "Graduate"):
        for s in STAGES:
            g = flow[(flow.outcome == out) & (flow.stage == s)]
            bottom = 0.0
            for k, name in enumerate(TIER_NAMES):
                v = float(g[g.tier == name].share.iloc[0])
                a.bar(pos, v, bottom=bottom, width=0.8, color=TIER_COLORS[k],
                      edgecolor="white", lw=1)
                if v >= 0.1:
                    a.text(pos, bottom + v / 2, f"{v:.0%}", ha="center", va="center",
                           fontsize=11, color="white" if k == 2 else "black")
                bottom += v
            ticks.append(pos)
            tlabels.append(s)
            pos += 1
        pos += 0.6
    a.set_xticks(ticks, tlabels)
    a.set_ylim(0, 1)
    a.set_ylabel("Share of students in each risk tier")
    n_d = int(((outcome == "Dropout")).sum())
    n_g = int(((outcome == "Graduate")).sum())
    for xc, text in ((np.mean(ticks[:3]), f"Dropouts\n(n = {n_d:,})"),
                     (np.mean(ticks[3:]), f"Graduates\n(n = {n_g:,})")):
        a.text(xc, -0.11, text, transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=13)
    fs.panel_label(a, "a")
    res = outcome != "Enrolled"
    for ax, out, base, letter in ((axes[1], "Dropout", fs.PALETTE["red_strong"], "b"),
                                  (axes[2], "Graduate", fs.PALETTE["blue_main"], "c")):
        m = res & (outcome == out)
        t0, t1 = tier(risk.loc[m, "T0"].to_numpy()), tier(risk.loc[m, "T1"].to_numpy())
        cnt = np.array([[np.sum((t0 == i) & (t1 == j)) for j in range(3)] for i in range(3)])
        rowp = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
        from matplotlib.colors import LinearSegmentedColormap
        cm = LinearSegmentedColormap.from_list(out, ["#FFFFFF", base])
        ax.imshow(rowp, cmap=cm, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{rowp[i, j]:.0%}\n{cnt[i, j]:,}", ha="center", va="center",
                        fontsize=11, color="white" if rowp[i, j] > 0.55 else "black")
        short_t = ["Low", "Med.", "High"]
        ax.set_xticks(range(3), short_t)
        ax.set_yticks(range(3), short_t)
        ax.set_xlabel("Tier at T1")
        ax.set_ylabel("Tier at T0")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{out}s (n = {int(m.sum()):,})", fontsize=14)
        fs.panel_label(ax, letter)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(TIER_COLORS, TIER_NAMES)],
               loc="outside lower center", ncol=3, frameon=False, fontsize=12,
               title="Out-of-fold P(dropout) tier", title_fontsize=12)
    return fig


def fig_umap_sensitivity(maps, outcome, stage="T1"):
    names = [m for m in (f"UMAP-15", f"UMAP-{NN_MAIN}", "UMAP-50",
                         f"UMAP-{NN_MAIN}-seed{C.SEED + 1}") if m in maps[stage]]
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(len(outcome))
    cols = np.array([fs.OUTCOME_COLORS[o] for o in outcome])
    fig, axes = plt.subplots(1, len(names), figsize=(fs.SLIDE_WIDTH, 3.9), layout="constrained")
    for ax, name, letter in zip(np.atleast_1d(axes), names, "abcd"):
        Y = maps[stage][name]
        ax.scatter(Y[order, 0], Y[order, 1], c=cols[order], s=3, alpha=0.6, linewidths=0,
                   rasterized=True)
        _map_axes(ax, "UMAP")
        nn = name.split("-")[1]
        title = f"k = {nn}" + (f", seed {C.SEED + 1}" if "seed" in name else f", seed {C.SEED}")
        ax.set_title(title, fontsize=14)
        fs.panel_label(ax, letter)
    return fig


# ---- main -----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one UMAP setting, no stability runs")
    a = ap.parse_args()
    t_all = time.time()
    fs.apply_publication_style(fs.SLIDE_STYLE)
    OUT.mkdir(parents=True, exist_ok=True)

    e, outcome, risk = load_inputs()
    n_counts = pd.Series(outcome).value_counts().to_dict()
    resolved = outcome != "Enrolled"
    base_rate = float((outcome[resolved] == "Dropout").mean())
    print(f"[visual] {len(e):,} students; {resolved.sum():,} with a known outcome "
          f"(dropout rate {base_rate:.3f}); out-of-fold risk for {risk['T1'].notna().sum():,}",
          flush=True)

    quality, maps = run_maps(e, outcome, a.quick)
    decision = vae_decision(quality)
    sel, prof, labels = run_clusters(e, outcome, risk, maps, a.quick)
    prog, t1 = programme_table(e, outcome, risk)
    groups = group_table(e, outcome, risk)
    tr = transition_table(e, outcome)
    flow = risk_flow_table(outcome, risk)

    quality.to_csv(OUT / "embedding_quality.csv", index=False, float_format="%.4f")
    (OUT / "vae_decision.txt").write_text(decision + "\n")
    sel.to_csv(OUT / "cluster_selection.csv", index=False, float_format="%.4f")
    prof.to_csv(OUT / "cluster_profiles.csv", index=False, float_format="%.4f")
    prog.to_csv(OUT / "programme_risk.csv", index=False, float_format="%.4f")
    groups.to_csv(OUT / "group_risk.csv", index=False, float_format="%.4f")
    tr.to_csv(OUT / "semester_transitions.csv", index=False, float_format="%.4f")
    flow.to_csv(OUT / "risk_tiers.csv", index=False, float_format="%.4f")
    coords = pd.DataFrame({"row": e.index, "outcome": outcome})
    for s in STAGES:
        coords[f"pca1_{s}"], coords[f"pca2_{s}"] = maps[s]["PCA"].T
        coords[f"umap1_{s}"], coords[f"umap2_{s}"] = maps[s][f"UMAP-{NN_MAIN}"].T
        coords[f"cluster_{s}"] = labels[s] + 1
        coords[f"oof_risk_{s}"] = risk[s].to_numpy()
    coords.to_csv(OUT / "student_coordinates.csv", index=False, float_format="%.5f")

    figs = {
        "va_maps_outcome": fig_maps_outcome(maps, outcome, n_counts),
        "va_maps_risk": fig_maps_risk(maps, outcome, risk),
        "va_map_quality": fig_quality(quality),
        "va_clusters": fig_clusters(maps, labels, sel, prof, base_rate),
        "va_programme_risk": fig_programmes(prog, t1),
        "va_group_risk": fig_groups(groups),
        "va_semester_transitions": fig_transitions(tr),
        "va_risk_tiers": fig_risk_flow(flow, outcome, risk),
    }
    if not a.quick:
        figs["va_umap_sensitivity_T1"] = fig_umap_sensitivity(maps, outcome)
    for name, fig in figs.items():
        fs.save_figure(fig, C.FIGURES_DIR / name)

    pd.set_option("display.width", 200)
    print("\n=== map quality ===")
    print(quality[["stage", "map", "trustworthiness_k10", "outcome_agreement_2d",
                   "outcome_agreement_full", "outcome_chance", "cohort_agreement_2d",
                   "cohort_chance", "pca_variance_2", "umap_seed_knn_jaccard"]]
          .round(3).to_string(index=False))
    print("\n=== VAE decision ===\n" + decision)
    print("\n=== clusters ===")
    print(prof[["stage", "cluster", "n", "dropout_rate", "ci_lo", "ci_hi", "mean_oof_risk",
                "distinguishing_features"]].round(3).to_string(index=False))
    print("\n=== programmes (sorted by observed dropout) ===")
    print(prog.round(3).to_string(index=False))
    print(f"\n[visual] finished in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
