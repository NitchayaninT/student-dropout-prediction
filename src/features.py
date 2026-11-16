"""Data loading and leakage-safe chronological feature construction."""
import numpy as np
import pandas as pd

from . import config as C


def load_data(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]      # one header has a stray tab
    missing = set(C.T2 + [C.TARGET]) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    return df


def _ratio(num, den):
    return np.where(den > 0, num / den.replace(0, np.nan), 0.0)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise ratios only (no statistics fitted on the data -> no CV leakage)."""
    df = df.copy()
    for sem, name in (("1st", "Approval ratio 1st sem"), ("2nd", "Approval ratio 2nd sem")):
        df[name] = _ratio(df[f"Curricular units {sem} sem (approved)"],
                          df[f"Curricular units {sem} sem (enrolled)"])
    df["Grade change"] = (df["Curricular units 2nd sem (grade)"]
                          - df["Curricular units 1st sem (grade)"])
    return df


def stage_features(stage: str) -> list:
    return C.RAW_STAGE_FEATURES[stage] + C.ENGINEERED[stage]


def get_xy(df: pd.DataFrame, stage: str, task: str = "multiclass"):
    """Return X, y, row index and class names for one stage.

    task = "multiclass": Dropout / Enrolled / Graduate (labels 0/1/2)
    task = "binary":     Dropout (1) vs Graduate (0); Enrolled rows are dropped
    """
    df = add_domain_features(df)
    if task == "binary":
        df = df[df[C.TARGET] != "Enrolled"]
        y = (df[C.TARGET] == "Dropout").astype(int)
        classes = ["Graduate", "Dropout"]
    else:
        y = df[C.TARGET].map({c: i for i, c in enumerate(C.CLASSES)})
        classes = C.CLASSES
    X = df[stage_features(stage)]
    # Guard: a stage may never see columns from a later semester
    if stage in ("T0", "T1"):
        assert not any("2nd sem" in c or c == "Grade change" for c in X.columns)
    if stage == "T0":
        assert not any("1st sem" in c or c in ("Debtor", "Tuition fees up to date")
                       for c in X.columns)
    return X, y.to_numpy(), df.index.to_numpy(), classes
