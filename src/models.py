"""Model pipelines and hyperparameter search spaces."""
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config as C

MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost"]


def build(name: str, columns: list, n_classes: int):
    """Return (pipeline, param_distributions). All preprocessing lives inside
    the pipeline so it is re-fitted on the training part of every fold."""
    if name == "LogisticRegression":
        nominal = [c for c in columns if c in C.NOMINAL]
        numeric = [c for c in columns if c not in C.NOMINAL]
        pre = ColumnTransformer([
            ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=10, sparse_output=False), nominal),
            ("num", StandardScaler(), numeric),
        ])
        pipe = Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=3000, random_state=C.SEED))])
        space = {"clf__C": loguniform(1e-3, 1e2),
                 "clf__class_weight": [None, "balanced"]}
    elif name == "RandomForest":
        pipe = Pipeline([("clf", RandomForestClassifier(random_state=C.SEED, n_jobs=1))])
        space = {"clf__n_estimators": [200, 300, 500],
                 "clf__max_depth": [None, 8, 12, 16],
                 "clf__min_samples_leaf": [1, 2, 5, 10],
                 "clf__max_features": ["sqrt", 0.3, 0.5],
                 "clf__class_weight": [None, "balanced"]}
    elif name == "XGBoost":
        pipe = Pipeline([("clf", XGBClassifier(
            tree_method="hist", random_state=C.SEED, n_jobs=1,
            eval_metric="mlogloss" if n_classes > 2 else "logloss"))])
        space = {"clf__n_estimators": randint(150, 600),
                 "clf__learning_rate": loguniform(0.01, 0.2),
                 "clf__max_depth": randint(3, 8),
                 "clf__min_child_weight": randint(1, 8),
                 "clf__subsample": uniform(0.6, 0.4),
                 "clf__colsample_bytree": uniform(0.5, 0.5),
                 "clf__reg_lambda": loguniform(0.1, 10)}
    else:
        raise ValueError(name)
    return pipe, space
