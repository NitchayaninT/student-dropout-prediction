#!/usr/bin/env bash
# Rebuild every result table and figure from data/data.csv in the locked environment,
# then compare the new outputs with the committed ones.
#
#   bash run.sh                      full pipeline + check against committed outputs
#   bash run.sh --tasks binary       one task only (any src.pipeline option works)
#
# A committed run_args.txt replaces the default arguments (--fresh --verify); an experiment
# branch uses it to run a subset, e.g. "--steps visual".
#
# Needs uv (https://docs.astral.sh/uv/). uv installs Python 3.10 and the exact
# package versions in uv.lock on first use.
set -euo pipefail
cd "$(dirname "$0")"
uv sync --locked
ARGS=(--fresh --verify)
if [ -f run_args.txt ]; then
    read -r -a ARGS < run_args.txt
fi
uv run --locked python -m src.pipeline "${ARGS[@]}" "$@"
