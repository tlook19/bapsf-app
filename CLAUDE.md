# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**bapsf-app** is a Streamlit-based web application for running and analyzing LAPD (Large Plasma Device) plasma transport simulations at UCLA's BAPSF (Beam-Accelerated Plasma Source Facility). It provides a UI for configuring parameter sweeps, executing simulations, and exploring results stored in HDF5 databases.

## Commands

```bash
# Activate environment
conda activate fenicsx-env

# Install dependencies (requires cablp sibling package at ../bapsf-transport/cablp)
poetry install

# Run the app
poetry run lapd-app
# or directly:
streamlit run bapsf_app/app.py
```

No test suite or linter is currently configured.

## Architecture

The app has three main tabs: **Configure** → **Run** → **Explore**.

### Data Flow

1. User sets parameter ranges/flags in the Configure tab
2. `_build_sweep_config()` (app.py) constructs a 5-tuple: `(param_ranges, flag_ranges, fixed_params, fixed_flags, param_transforms)`
3. `param_combinations()` (sweep.py) generates all `(params, flags)` tuples via Cartesian product
4. `grid_sweep_parallel()` or `grid_sweep()` (sweep.py) executes simulations via `ProcessPoolExecutor`
5. Each run instantiates `LAPDSim` (from `cablp`) and calls `sim.start_simulation()`
6. Results are written to HDF5 via `save_run()` / `update_index()` (database.py)
7. The Explore tab reads back via `load_index()` / `load_run()` and renders plots via plot.py

### Key Modules

- **[bapsf_app/app.py](bapsf_app/app.py)** — Main Streamlit UI (~1600 lines). Manages session state, tab rendering, config persistence to `~/.lapd_app_config.json`, and spawns a worker thread for background sweep execution.
- **[bapsf_app/sweep.py](bapsf_app/sweep.py)** — Parameter sweep engine. Generates combinations, handles neutral equilibration caching (100-cycle pre-runs cached by neutral-dynamics signature), manages parallel execution and graceful failure handling.
- **[bapsf_app/database.py](bapsf_app/database.py)** — HDF5 database layer. All simulation writes go through here; the index layer (`runs/index/`) enables fast metadata queries without loading run arrays.
- **[bapsf_app/plot.py](bapsf_app/plot.py)** — Matplotlib visualizations (line plots, scatter, heatmap, comparison overlays). Uses `"Agg"` backend (non-interactive, required for Streamlit).
- **[bapsf_app/stats.py](bapsf_app/stats.py)** — Window-based statistics (variance, mean, min/max) over simulation time windows.

### Session State Conventions

Widget state in app.py follows consistent key patterns:

- `pmode_{key}` / `pfixed_{key}` / `pmin_{key}` / `pmax_{key}` / `pstep_{key}` — per-parameter UI controls
- `flagcfg_{key}` — flag setting: `"True"` | `"False"` | `"Both"`
- `dc_on_off` / `dc_type` — dual-cathode mode controls
- `sweep_state` — `SweepState` dataclass instance (total, completed, failed, log, running, done, error, db_path, planned_run_ids)
- `sweep_queue` / `sweep_thread` / `sweep_stop_event` — background thread coordination

Config persistence uses `_PARAM_CFG_KEYS`, `_FLAG_CFG_KEYS`, `_MISC_CFG_KEYS` constants to determine which session keys are written to `~/.lapd_app_config.json`.

### Parallel Execution Model

Sweeps run in a background daemon thread that calls `grid_sweep_parallel()`. Key non-obvious ordering constraints:

- **Equilibration is serial** (main thread, before dispatch): All `equilibrate_neutrals()` pre-runs are done first, then workers are submitted to the `ProcessPoolExecutor`
- **HDF5 writes are serial**: Each worker returns results to the main thread; writes happen sequentially via `with open_db()`
- **Progress** flows through a `queue.Queue` from the worker thread to Streamlit; `_drain_queue()` polls it each render cycle
- `_run_single_worker()` must be a module-level function (not a method) for pickle serialization across processes

### Neutral Equilibration Caching

Before each set of runs, sweep.py performs a 100-cycle plasma-off pre-run to establish neutral gas equilibrium. Results are cached in `_nn_cache` by a signature computed from neutral-dynamics parameters: `S_gp`, `S_pump_L/R`, `Source_nn0`, `cells`, `gas_type`, `Lm`, `Lp`, `TwinCathode`, `Twin_S_gp`. Runs that share the same neutral signature skip re-equilibration. The cache key is deterministic — changing mesh parameters (`cells`, `Lm`, `Lp`) invalidates the cache.

### Parameter Transforms (Non-obvious)

Before reaching the simulator, several transforms are applied:

- **Power → Current**: `P_in` (W) ÷ `Vd` → `Id` (A). The UI exposes power, not current.
- **Adaptive stepping**: When `adaptive=True` flag is set, the single `dt_max` UI field replaces **both** `dt_main` and `dt_after` in the params dict. Range inputs for `dt_main`/`dt_after` are suppressed.
- **Twin symmetric mode**: When `TwinCathode=True` AND `dc_type="Twin (symmetric)"`, `S_gp` and `Source_nn0` are divided by 2 and those halved values are also written to `Twin_S_gp` and `Twin_nn0`. A live preview is shown in the Configure tab.
- **`Twin_Vd` alias**: `Twin_Vd` mirrors `Vd` automatically (handled at sweep layer, not in UI).

### Resume / Abort via Manifest

A `sweep.h5.progress.json` file is written alongside the HDF5 database at sweep start and updated after each run. On re-opening the same database path, the app detects the manifest and offers to resume — only the missing run IDs are re-executed. On abort, `_mark_incomplete_as_failed()` marks unstarted runs as failed in the database.

### HDF5 Schema

```
sweep.h5
├── attrs: {created, description}
├── runs/
│   ├── run_NNNN/
│   │   ├── attrs: {param_*, flag_*, timestamp, status, error?}
│   │   ├── <result arrays> (ne, nn, Te, Ti, v_plasma, isat, ...) — gzip level 4
│   │   ├── cells_at_time (n_timesteps,)
│   │   ├── refinement_events (N, 3): [t_ms, old_cells, new_cells]
│   │   ├── cathode/ — one dataset per field (n_timesteps,): phi_c_plus, phi_c_minus, phi_c, phi_a, V_p, V_b, R_p, I_i, I_e, I_eth, I_eth_star, I_tot, P_wall, P_load, P_comp, P_prim, P_ohmic, P_cathode_e, P_cathode_i, P_anode_e, P_anode_i, P_net, P_net2, P_loss
│   │   ├── cathode_twin/ — same structure; all-NaN when TwinCathode=False
│   │   └── stats_10_20ms/ attrs: {ne_var, ne_tvar, ne_tcov, ne_total_var, ne_min, ne_max, ne_mean, Te_*}
└── index/
    ├── run_ids, status, n_cells   (resizable)
    ├── params/{name}              (resizable float/str)
    ├── flags/{name}               (resizable int 0/1)
    └── stats_10_20ms/{name}       (resizable float)
```

The index uses a union-of-keys approach — new parameter names are added as new datasets with NaN/empty-string padding for earlier runs. `rebuild_index()` reconstructs the full index from the `runs/` group, useful after partial failures.

### External Dependency: cablp

The simulation engine lives in a sibling repo at `../bapsf-transport/cablp`. It provides:
- `LAPDSim` — main simulation class (instantiated with params dict + flags dict)
- `input_dict_template` / `input_flags_template` — default parameter and flag dicts

Parameter metadata (units, display ranges, labels) is defined in `PARAM_META`, `FLAG_META`, and `TWIN_META` dicts in app.py — not in cablp. Three dual-cathode modes: single, twin (symmetric), and asymmetric.
