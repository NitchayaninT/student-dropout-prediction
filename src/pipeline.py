"""Run the whole analysis in dependency order, from data/data.csv to every table and figure.

Each step is the documented command (`python -m src.<module> --task <task>`), started as a
subprocess so a step behaves exactly as when it is run by hand.

    python -m src.pipeline                      # all steps, both tasks (binary first)
    python -m src.pipeline --tasks binary       # one task
    python -m src.pipeline --steps train plot   # a subset, in pipeline order
    python -m src.pipeline --steps visual       # Part 3 only, from the results already there
    python -m src.pipeline --fresh --verify     # what run.sh does (see below)

--fresh   moves the existing results/, figures/ and models/ to reference/ before running,
          so every output is rebuilt from scratch and a step that silently writes nothing
          cannot pass by leaving the old file in place.
--verify  afterwards compares the new outputs with reference/ (src/verify_reproduction.py).

Step order matters: plot, explain, audit, fairness and ablation read what train wrote, and
the binary fairness step also loads the final binary models. The visual step runs once,
after both tasks, and reads the binary out-of-fold predictions.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata

from . import config as C

STEPS = {  # name -> module; order is the run order within a task
    "train": "src.train",
    "plot": "src.plot_results",
    "explain": "src.explain",
    "audit": "src.audit_features",
    "fairness": "src.fairness",
    "ablation": "src.ablation",
}
GLOBAL_STEPS = {"visual": "src.visual_analytics",   # run once, after the per-task steps
                "calibration": "src.calibration"}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib",
            "joblib", "shap", "umap-learn"]
OUTPUT_DIRS = [C.RESULTS_DIR, C.FIGURES_DIR, C.MODELS_DIR]
REFERENCE_DIR = C.ROOT / "reference"


def environment_report():
    print("=== environment ===")
    print(f"python      {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"platform    {platform.platform()} / {platform.machine()}")
    print(f"cpu count   {os.cpu_count()}")
    for p in PACKAGES:
        try:
            print(f"{p:<11} {metadata.version(p)}")
        except metadata.PackageNotFoundError:
            print(f"{p:<11} not installed")
    commit = os.environ.get("ORX_COMMIT")
    if commit:
        print(f"commit      {commit}")
    print(flush=True)


def move_outputs_to_reference():
    if REFERENCE_DIR.exists():
        shutil.rmtree(REFERENCE_DIR)
    REFERENCE_DIR.mkdir()
    for d in OUTPUT_DIRS:
        if d.exists():
            shutil.move(str(d), str(REFERENCE_DIR / d.name))
            print(f"[pipeline] moved {d.name}/ to reference/{d.name}/")
    print(flush=True)


def run_step(step, task=None):
    if task is None:
        cmd, name = [sys.executable, "-m", GLOBAL_STEPS[step]], step
    else:
        cmd, name = [sys.executable, "-m", STEPS[step], "--task", task], f"{step} --task {task}"
    print(f"=== {name} ===", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=C.ROOT)
    dt = time.time() - t0
    print(f"[pipeline] {name} finished in {dt / 60:.1f} min\n", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"],
                    choices=["binary", "multiclass"])
    ap.add_argument("--steps", nargs="+", default=list(STEPS) + list(GLOBAL_STEPS),
                    choices=list(STEPS) + list(GLOBAL_STEPS))
    ap.add_argument("--fresh", action="store_true",
                    help="move existing outputs to reference/ and rebuild everything")
    ap.add_argument("--verify", action="store_true",
                    help="compare the new outputs with reference/ when done")
    a = ap.parse_args()

    environment_report()
    if a.fresh:
        move_outputs_to_reference()

    timings = []
    t_all = time.time()
    for task in a.tasks:
        for step in STEPS:                      # keep pipeline order whatever the CLI order
            if step in a.steps:
                timings.append((task, step, run_step(step, task)))
    for step in GLOBAL_STEPS:
        if step in a.steps:
            timings.append(("all", step, run_step(step)))

    print("=== timings ===")
    for task, step, dt in timings:
        print(f"{task:<10} {step:<9} {dt / 60:6.1f} min")
    print(f"{'total':<20} {(time.time() - t_all) / 60:6.1f} min\n", flush=True)

    if a.verify:
        from .verify_reproduction import verify
        verify(REFERENCE_DIR, C.ROOT, tasks=a.tasks)


if __name__ == "__main__":
    main()
