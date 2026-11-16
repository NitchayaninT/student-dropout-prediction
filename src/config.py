"""Central configuration: paths, chronological stages, column roles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

SEED = 42
N_OUTER = 5      # Stratified 5-fold CV (evaluation)
N_INNER = 3      # inner folds used only for hyperparameter tuning
N_ITER = 10      # random-search draws per model

TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]   # label index = position

# ---- T0: known at enrollment (22 features) --------------------------------
T0 = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs", "Gender",
    "Scholarship holder", "Age at enrollment", "International",
    "Unemployment rate", "Inflation rate", "GDP",
]
SEM_FIELDS = ["credited", "enrolled", "evaluations", "approved", "grade",
              "without evaluations"]
SEM1 = [f"Curricular units 1st sem ({f})" for f in SEM_FIELDS]
SEM2 = [f"Curricular units 2nd sem ({f})" for f in SEM_FIELDS]

# ---- T1: + financial flags + 1st-semester records (30 features) -----------
T1 = T0 + ["Debtor", "Tuition fees up to date"] + SEM1
# ---- T2: + 2nd-semester records (36 features) ------------------------------
T2 = T1 + SEM2

RAW_STAGE_FEATURES = {"T0": T0, "T1": T1, "T2": T2}

# Engineered domain ratios, added on top of the raw features of each stage
ENGINEERED = {
    "T0": [],
    "T1": ["Approval ratio 1st sem"],
    "T2": ["Approval ratio 1st sem", "Approval ratio 2nd sem", "Grade change"],
}

# Integer-coded nominal variables (one-hot encoded for Logistic Regression)
NOMINAL = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
