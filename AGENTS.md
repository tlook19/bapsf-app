# AGENTS.md

This file provides guidance to Codex and other coding agents when working in
this repository.

## Project Overview

`bapsf-app` is a local Streamlit web application for configuring, launching,
resuming, and analyzing sweeps of the BAPSF/LAPD plasma transport simulation.
The simulation engine lives in the sibling repository at:

```text
..\bapsf-transport\cablp
```

The app uses that sibling package in editable/development mode through Poetry.

## Local Environment

This repository is currently being developed on a Windows PC under
`D:\bapsf\bapsf-app`.  Use the `bapsf-app` mamba environment:

```powershell
mamba activate bapsf-app
# or, if this environment is managed through conda:
conda activate bapsf-app
```

`mamba`/`conda` may not be visible in every shell.  If activation fails, use an
initialized PowerShell session or the full path to the environment's Python
executable.

`pyproject.toml` currently requires Python `>=3.14`.  If the environment has an
older Python, update or recreate the environment before installing dependencies.

## Commands

```powershell
cd D:\bapsf\bapsf-app
poetry install
poetry run lapd-app
```

Alternative direct launch after installation:

```powershell
python -m streamlit run bapsf_app/app.py
```

No formal test suite or linter is currently configured.  For quick validation,
prefer targeted imports, `python -m compileall bapsf_app`, and a small manual app
launch.

## Architecture

The app has three main tabs: Configure, Run, and Explore.

Data flow:

1. User sets parameter ranges and flags in the Configure tab.
2. `_build_sweep_config()` in `app.py` builds parameter/flag ranges, fixed
   values, and transforms.
3. `param_combinations()` in `sweep.py` expands the sweep.
4. `grid_sweep_parallel()` or `grid_sweep()` executes runs with `LAPDSim`.
5. Results are written to HDF5 through `database.py`.
6. The Explore tab reads HDF5 data and renders plots through `plot.py`.

Key modules:

- `bapsf_app/app.py` - Streamlit UI, session state, config persistence, and
  background sweep coordination.
- `bapsf_app/sweep.py` - sweep generation, neutral equilibration caching,
  multiprocessing, and failure handling.
- `bapsf_app/database.py` - HDF5 storage and index maintenance.
- `bapsf_app/plot.py` - Matplotlib visualizations using the non-interactive
  `"Agg"` backend.
- `bapsf_app/stats.py` - windowed statistics over simulation output.

## Coupling to cablp

The app imports the current simulator from:

```python
from cablp.solvers._sim3 import LAPDSim, input_dict_template, input_flags_template
```

When simulator parameters or flags change, update app metadata and transforms in
`bapsf_app/app.py`, especially `PARAM_META`, `FLAG_META`, and `TWIN_META`.
Treat `_sim3.py` and its templates as the source of truth when app metadata
drifts from simulator behavior.

## Development Notes

- Keep app changes compatible with Windows paths and PowerShell workflows.
- Do not assume old Mac-specific environments such as `fenicsx-env` exist.
- HDF5 writes are intentionally serialized even when simulations run in worker
  processes.
- `_run_single_worker()` in `sweep.py` must stay module-level so
  `ProcessPoolExecutor` can pickle it on Windows.
- Preserve existing sweep manifests and databases unless explicitly asked to
  clean them up.
