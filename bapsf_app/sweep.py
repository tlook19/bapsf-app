"""
Grid parameter sweep runner for LAPDSim.

Usage
-----
::

    from bapsf_app.sweep import grid_sweep

    grid_sweep(
        param_ranges={"V_bank": [80, 100], "S_gp": [300, 500]},
        flag_ranges={"cx": [True, False]},
        fixed_params={"gas_type": "He", "cells": 3},
        fixed_flags={"Plasma": True, "icool": True},
        db_path="sweep.h5",
        t_window=(10.0, 20.0),
    )
"""
import itertools
import multiprocessing
import os
import queue as _stdlib_queue
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

from cablp.solvers._sim3 import LAPDSim, input_dict_template, input_flags_template
from cablp.vars._nn_table import lookup_nn0
from .database import open_db, save_run, update_index, list_runs
from .stats import compute_window_stats


_EQUIL_DT_MAX = 1e-2  # fixed max step for neutral equilibration (never inherits main sim value)


def _equil_cache_key(params, flags):
    """
    Return a hashable key summarising the parameters that drive neutral-density
    equilibration.  Runs that share this key can reuse the same ``nn_eq`` result.

    The key covers: S_gp, S_pump_L/R, cells, gas_type, Lm, Lp,
    TwinCathode flag, and Twin_S_gp (when TwinCathode is active).
    tau_discharge is excluded because equilibrate_neutrals() hardcodes it to 20 ms
    regardless of the run's tau_discharge value.
    Parameters that only affect plasma dynamics (V_bank, T_s, …) are excluded.
    """
    twin_active = bool(flags.get("TwinCathode", False))
    return (
        float(params.get("S_gp", 0.0)),
        float(params.get("S_pump_L", 0.0)),
        float(params.get("S_pump_R", 0.0)),
        int(params.get("cells", 3)),
        str(params.get("gas_type", "He")),
        float(params.get("Lm", 1800.0)),
        float(params.get("Lp", 1800.0)),
        twin_active,
        float(params.get("Twin_S_gp", 0.0)) if twin_active else 0.0,
    )


# Fixed conditions under which the pre-baked nn_table.csv is valid.
# If a run's params deviate from these, a full equilibration is required.
_TABLE_CONDITIONS = {
    "S_pump_L": 4000.0,
    "S_pump_R": 4000.0,
    "Lm": 1800.0,
    "Lp": 1800.0,
    "Rm": 50.0,
    "gas_type": "He",
}


def _try_nn_table(params, flags):
    """
    Return the pre-tabulated equilibrium nn0 scalar if the run's neutral
    dynamics match the table's fixed conditions, otherwise return None.

    Twin-cathode runs require symmetric puffing (S_gp == Twin_S_gp) since
    the table was generated under that assumption.
    """
    for key, expected in _TABLE_CONDITIONS.items():
        val = params.get(key, input_dict_template.get(key))
        if isinstance(expected, float):
            if val != expected:
                return None
        else:
            if str(val) != str(expected):
                return None

    twin_active = bool(flags.get("TwinCathode", False))
    s_gp = float(params.get("S_gp", 0.0))
    if twin_active:
        twin_s_gp = float(params.get("Twin_S_gp", 0.0))
        if twin_s_gp != s_gp:
            return None

    try:
        return lookup_nn0(s_gp, twin=twin_active)
    except ValueError:
        return None


def equilibrate_neutrals(
    params,
    flags=None,
    cycles=100,
    t_per_cycle=3.0,
    nn0_init=1e8,
    verbose=True,
):
    """
    Run a plasma-off simulation to find the equilibrium background neutral density.

    Starts all cells at ``nn0_init`` and advances ``cycles`` cycles of
    ``t_per_cycle`` seconds each with plasma off and adaptive time stepping
    (maximum step 1e-2 s).  The neutral density at the end of the final
    cycle is returned and can be used as the initial condition for a
    plasma-on simulation.

    Gas puff (``S_gp``) and pumping (``S_pump_L/R``) rates are taken from
    ``params`` so the result is consistent with the cathode configuration.

    Parameters
    ----------
    params : dict
        Simulation parameters.  ``S_gp``, ``S_pump_L``, ``S_pump_R``,
        ``cells``, ``gas_type``, and cathode geometry keys are used.
    flags : dict or None
        Simulation flags.  ``Plasma`` is forced to ``False``;
        ``Velocity`` is forced to ``False``; ``adaptive`` is forced to
        ``True``.
    cycles : int
        Number of gas-fill cycles to run.  Default 100.
    t_per_cycle : float
        Duration of each cycle in seconds.  Default 3.0.
    nn0_init : float
        Seed neutral density (cm⁻³) for all cells.  Default 1e8.
    verbose : bool
        Print start/end summary lines.

    Returns
    -------
    nn_eq : np.ndarray, shape (cells,)
        Equilibrium neutral density in each cell at the end of the final cycle.
    """
    eq_params = {
        **input_dict_template,
        **params,
        "nn0": nn0_init,
        # S_gp is active only during tau_discharge of each tau_cycle
        "cycles": cycles,
        "tau_cycle": t_per_cycle,
        "tau_discharge": 20e-3,  # gas puff on for first 20 ms, then neutral diffusion/pumping
        # Always use the fixed equilibration step size, never the main sim's step sizes
        "h_max_discharge": _EQUIL_DT_MAX,
        "h_max_afterglow": _EQUIL_DT_MAX,
    }

    eq_flags = {
        **input_flags_template,
        **(flags or {}),
        "Plasma": False,    # no ionisation / recombination
        "Velocity": False,  # velocity meaningless without plasma
    }

    if verbose:
        print(
            f"  [nn equil] {cycles} cycles × {t_per_cycle} s  "
            f"(dt_max={_EQUIL_DT_MAX} s, nn0_init={nn0_init:.1e})"
        )

    sim = LAPDSim(eq_params, eq_flags)
    sim.start_simulation()

    nn_eq = sim.get_results().nn[-1]  # shape (cells,)

    if verbose:
        print(f"  [nn equil] nn_eq = {nn_eq} cm⁻³")

    return nn_eq


def _apply_equilibrated_nn(params, flags, nn_eq):
    """
    Return a copy of ``params`` with ``nn0`` set from the equilibrated neutral
    density array.

    ``nn0`` is set to the mean of the interior cells so it remains a scalar.
    """
    n_cells = int(params.get("cells", 3))
    patched = dict(params)
    # Slice to the configured cell count — nn_eq may be padded to max_cells with NaN
    # when adaptive mesh is enabled, even though equilibration runs with Plasma=False
    # and the mesh never actually refines.
    active_nn = nn_eq[:n_cells]
    if n_cells > 2:
        nn0_eq = float(active_nn[1:-1].mean())
    else:
        nn0_eq = float(active_nn.mean())
    patched["nn0"] = nn0_eq
    return patched


def param_combinations(param_ranges, flag_ranges):
    """
    Generate all (params_dict, flags_dict) combinations.

    Parameters
    ----------
    param_ranges : dict
        ``{param_name: [val1, val2, ...]}``.
    flag_ranges : dict
        ``{flag_name: [True, False, ...]}``.

    Returns
    -------
    list of (params_patch, flags_patch)
        Each element is a pair of dicts containing only the varied keys.
        Keys are sorted for reproducibility.
    """
    param_keys = sorted(param_ranges.keys())
    flag_keys = sorted(flag_ranges.keys())

    param_vals = [param_ranges[k] for k in param_keys]
    flag_vals = [flag_ranges[k] for k in flag_keys]

    all_vals = param_vals + flag_vals
    all_keys = param_keys + flag_keys

    if not all_keys:
        return [({}, {})]

    combos = []
    for combo in itertools.product(*all_vals):
        combined = dict(zip(all_keys, combo))
        p_patch = {k: combined[k] for k in param_keys}
        f_patch = {k: combined[k] for k in flag_keys}
        combos.append((p_patch, f_patch))

    return combos


def grid_sweep(
    param_ranges,
    flag_ranges=None,
    fixed_params=None,
    fixed_flags=None,
    db_path="sweep.h5",
    t_window=(10.0, 20.0),
    param_aliases=None,
    param_transforms=None,
    equilibrate_nn=False,
    verbose=True,
    verbose_equil=None,
):
    """
    Run all combinations of ``param_ranges × flag_ranges`` and save to an HDF5 database.

    Parameters
    ----------
    param_ranges : dict
        ``{param_name: [val1, val2, ...]}``.  Keys must match ``input_dict_template``.
    flag_ranges : dict or None
        ``{flag_name: [True, False, ...]}``.  Keys must match ``input_flags_template``.
    fixed_params : dict or None
        Parameters held constant (merged over ``input_dict_template``).
    fixed_flags : dict or None
        Flags held constant (merged over ``input_flags_template``).
    db_path : str or path-like
        Path to the HDF5 database.  Created if it does not exist.
    t_window : tuple of float
        (t_start, t_end) in ms for window statistics.
    param_aliases : dict or None
        ``{alias_key: source_key}`` pairs applied **after** building each run's
        ``params`` dict.  E.g. ``{"Twin_S_gp": "S_gp"}`` ensures ``Twin_S_gp`` always
        equals the current ``S_gp`` value, even when ``S_gp`` is swept.
    param_transforms : callable or None
        ``(params, flags) -> params`` applied after ``param_aliases``.  Used to
        derive computed parameters, e.g. ``Id = P_in / Vd``.  The callable
        may modify ``params`` in-place and must return the updated dict.
    equilibrate_nn : bool
        If ``True``, run a 100-cycle plasma-off pre-simulation before each
        run to find the equilibrium neutral density.  Results are cached by
        neutral-dynamics key (S_gp, pumping, cells, gas_type, TwinCathode,
        Twin_S_gp) and reused for runs that share the same neutral equilibrium.
    verbose : bool
        Print progress messages including per-run timing.
    verbose_equil : bool or None
        Print equilibration detail messages.  ``None`` (default) inherits
        from ``verbose``.  Set to ``False`` to suppress equilibration inner
        prints while keeping sweep progress prints.

    Returns
    -------
    list of str
        Run IDs that completed successfully.

    Notes
    -----
    - Run IDs are assigned sequentially as ``run_0000``, ``run_0001``, …
    - Runs already present in the database are skipped, so an interrupted sweep
      can be resumed by calling this function again with the same arguments.
    - If a simulation raises an exception the run is marked ``'failed'`` in the
      index and execution continues with the next combination.
    """
    if flag_ranges is None:
        flag_ranges = {}

    _verbose_equil = verbose if verbose_equil is None else verbose_equil

    combos = param_combinations(param_ranges, flag_ranges)
    n_total = len(combos)

    if verbose:
        print(f"Grid sweep: {n_total} combinations → '{db_path}'")

    _nn_cache = {}  # cache_key → (nn_eq, equil_time_s)
    t_sweep_start = time.time()

    with open_db(db_path, mode="a") as db:
        existing = {
            run_id for run_id in list_runs(db)
            if db["runs"][run_id].attrs.get("status") == "ok"
        }
        successful = []

        for i, (p_patch, f_patch) in enumerate(combos):
            run_id = f"run_{i:04d}"

            if run_id in existing:
                if verbose:
                    print(f"  [{i+1}/{n_total}] {run_id} already in database — skipping.")
                successful.append(run_id)
                continue

            # Build full params and flags dicts
            params = {**input_dict_template, **(fixed_params or {}), **p_patch}
            flags = {**input_flags_template, **(fixed_flags or {}), **f_patch}

            # Apply param aliases (e.g. symmetric twin mirroring: Twin_Vd = Vd)
            if param_aliases:
                for alias, source in param_aliases.items():
                    if source in params:
                        params[alias] = params[source]

            # Apply param transforms (derive computed params, e.g. Id = P_in / Vd)
            if param_transforms is not None:
                params = param_transforms(params, flags)

            if verbose:
                varied = {**p_patch, **f_patch}
                print(f"  [{i+1}/{n_total}] {run_id}  {varied}")

            equil_time = 0.0
            cache_hit = False

            # Pre-equilibrate neutral density if requested
            if equilibrate_nn:
                twin_active = bool(flags.get("TwinCathode", False))
                twin_str = (
                    f"  Twin_S_gp={params.get('Twin_S_gp', 0):.0f}" if twin_active else ""
                )
                cache_key = _equil_cache_key(params, flags)
                if cache_key in _nn_cache:
                    nn_eq, equil_time = _nn_cache[cache_key]
                    cache_hit = True
                    if verbose:
                        print(
                            f"    [nn equil] cache hit"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                        )
                else:
                    table_val = _try_nn_table(params, flags)
                    if table_val is not None:
                        n_cells_eq = int(params.get("cells", 3))
                        nn_eq = np.ones(n_cells_eq) * table_val
                        equil_time = 0.0
                        _nn_cache[cache_key] = (nn_eq, equil_time)
                        if verbose:
                            print(
                                f"    [nn equil] table lookup"
                                f"  S_gp={params.get('S_gp', 0):.0f}"
                                f"  twin={'on' if twin_active else 'off'}{twin_str}"
                                f"  nn0={table_val:.3e}"
                            )
                    else:
                        t0_equil = time.time()
                        nn_eq = equilibrate_neutrals(params, flags, verbose=_verbose_equil)
                        equil_time = time.time() - t0_equil
                        _nn_cache[cache_key] = (nn_eq, equil_time)
                        if verbose:
                            print(
                                f"    [nn equil] computed"
                                f"  S_gp={params.get('S_gp', 0):.0f}"
                                f"  twin={'on' if twin_active else 'off'}{twin_str}"
                                f"  time={equil_time:.1f}s"
                            )
                params = _apply_equilibrated_nn(params, flags, nn_eq)

            t0_run = time.time()
            try:
                sim = LAPDSim(params, flags)
                sim.start_simulation()
                results = sim.get_results()
                run_time = time.time() - t0_run
                stats = compute_window_stats(results, t_window)
                cat = getattr(results, "cells_at_time", None)
                n_cells = int(cat.max()) if cat is not None and len(cat) > 0 else int(params.get("cells", 3))

                save_run(db, run_id, params, flags, results, stats)
                update_index(db, run_id, params, flags, stats, n_cells, status="ok")
                successful.append(run_id)

                if verbose:
                    twin_active = bool(flags.get("TwinCathode", False))
                    equil_str = ""
                    if equilibrate_nn:
                        twin_str = (
                            f"  Twin_S_gp={params.get('Twin_S_gp', 0):.0f}" if twin_active else ""
                        )
                        cache_str = "cached" if cache_hit else f"{equil_time:.1f}s"
                        equil_str = (
                            f"  equil={cache_str}"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                        )
                    print(
                        f"    ne_var={stats['ne_var']:.3e}  Te_var={stats['Te_var']:.3e}"
                        f"  ne_mean={stats['ne_mean']:.3e}  Te_mean={stats['Te_mean']:.3f} eV"
                        f"  run={run_time:.1f}s{equil_str}"
                    )

            except Exception:
                run_time = time.time() - t0_run
                tb = traceback.format_exc()
                print(f"  [{i+1}/{n_total}] {run_id} FAILED (run={run_time:.1f}s):\n{tb}")

                # Save a minimal failure record so the index stays consistent
                with open_db(db_path, mode="a") as db2:
                    grp = db2.require_group("runs").require_group(run_id)
                    grp.attrs["status"] = "failed"
                    grp.attrs["error"] = tb[:2000]  # truncate for storage
                    for k, v in params.items():
                        try:
                            grp.attrs[f"param_{k}"] = v
                        except TypeError:
                            grp.attrs[f"param_{k}"] = str(v)
                    for k, v in flags.items():
                        grp.attrs[f"flag_{k}"] = bool(v)
                    update_index(db2, run_id, params, flags, {}, 0, status="failed")

    total_time = time.time() - t_sweep_start
    if verbose:
        print(f"Sweep complete: {len(successful)}/{n_total} runs succeeded.  Total: {total_time:.1f}s")

    return successful


# ── Parallel sweep ─────────────────────────────────────────────────────────────

_WORKER_PROGRESS_Q = None   # set in each worker by _worker_init; never pickled as a task arg
_WORKER_LOG_FILE = None     # holds open file ref so GC doesn't close it prematurely


def _set_worker_thread_limits():
    """
    Keep native math libraries from oversubscribing CPU cores in each worker.

    This must run in the parent before spawning workers because spawned Python
    processes import NumPy/cablp before ``_worker_init`` is called.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = os.path.join(tempfile.gettempdir(), "bapsf_app_matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache_dir = os.path.join(tempfile.gettempdir(), "bapsf_app_cache")
        os.makedirs(xdg_cache_dir, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = xdg_cache_dir


def _worker_init(progress_q, db_path=None):
    """
    Called once per worker process at spawn time.
    Stores the progress queue in a module-level global (avoids pickling it per task),
    pins BLAS/OpenMP to one thread to prevent oversubscription, and redirects the
    worker's stdout/stderr to a per-worker log file so that simulation print() calls
    cannot block on a full pipe (which would freeze the worker when stdout is not a
    TTY), while still preserving the diagnostic output (step sizes, h= values) for
    post-hoc inspection.
    """
    global _WORKER_PROGRESS_Q, _WORKER_LOG_FILE
    _WORKER_PROGRESS_Q = progress_q
    _set_worker_thread_limits()

    import sys as _sys, pathlib, time as _time

    # Workers inherit the parent process's nice value.  When the app is launched
    # from a low-priority shell (e.g. `conda run ... &` from a VSCode terminal)
    # this can be nice=5+, which cuts CPU frequency significantly on Apple Silicon
    # even when running on P-cores.  Try to reset to 0; fails silently if the OS
    # denies it (non-root users cannot lower their own niceness on macOS/Linux).
    _nice_before = 0
    _nice_after = 0
    try:
        _nice_before = os.getpriority(os.PRIO_PROCESS, 0)
        if _nice_before > 0:
            os.setpriority(os.PRIO_PROCESS, 0, 0)
        _nice_after = os.getpriority(os.PRIO_PROCESS, 0)
    except Exception:
        _nice_after = _nice_before

    # Promote to highest-priority CPU cores available on the current platform.
    # Spawned worker processes often inherit a low scheduling class that puts them
    # on efficiency cores or at reduced frequency.
    if _sys.platform == "darwin":
        # Apple Silicon: USER_INTERACTIVE (0x21) targets maximum P-core frequency.
        try:
            import ctypes as _ctypes
            _libsys = _ctypes.CDLL(None)  # libSystem.B.dylib — includes pthread symbols
            _QOS_CLASS_USER_INTERACTIVE = 0x21
            _libsys.pthread_set_qos_class_self_np(_QOS_CLASS_USER_INTERACTIVE, 0)
            _actual_qos = _libsys.qos_class_self()
            _qos_promoted = (_actual_qos == _QOS_CLASS_USER_INTERACTIVE)
        except Exception:
            _qos_promoted = False
    elif _sys.platform == "win32":
        # Windows (Intel/AMD hybrid P/E cores): THREAD_PRIORITY_HIGHEST steers the
        # scheduler toward P-cores on Thread Director-aware systems (12th gen Intel+).
        try:
            import ctypes as _ctypes
            _kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)
            _THREAD_PRIORITY_HIGHEST = 2
            _qos_promoted = bool(_kernel32.SetThreadPriority(
                _kernel32.GetCurrentThread(), _THREAD_PRIORITY_HIGHEST
            ))
        except Exception:
            _qos_promoted = False
    else:
        _qos_promoted = None  # Linux / other: rely on nice reset above

    if db_path is not None:
        db_p = pathlib.Path(db_path).expanduser()
        log_path = db_p.parent / f"{db_p.stem}.worker_{os.getpid()}.log"
    else:
        import tempfile as _tempfile
        log_path = pathlib.Path(_tempfile.gettempdir()) / f"lapd_worker_{os.getpid()}.log"

    # line-buffered so output appears in the file immediately after each newline
    log_file = open(log_path, "w", buffering=1)
    log_file.write(
        f"# bapsf worker pid={os.getpid()} started {_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# db={db_path}\n"
        f"# log={log_path}\n"
        f"# nice: {_nice_before} → {_nice_after}  qos_promoted={_qos_promoted}\n\n"
    )
    log_file.flush()

    # Redirect OS-level fd 1/2 so that C-level printf in native extensions also goes
    # to the log file rather than the parent's pipe (which would block once full on
    # POSIX).  Wrapped in try/except because Windows dup2 semantics differ slightly.
    try:
        log_fd = log_file.fileno()
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
    except Exception:
        pass
    _sys.stdout = log_file
    _sys.stderr = log_file
    _WORKER_LOG_FILE = log_file  # keep alive; sys.stdout ref alone is enough but belt-and-suspenders


_PHASE_CODE = {"pre_breakdown": 0.0, "main_discharge": 1.0, "afterglow": 2.0}


def _run_single_worker(args):
    """
    Module-level worker function for ProcessPoolExecutor (must be picklable).

    Task args contain only plain Python objects (str, dict of scalars) so that
    ForkingPickler can serialize the _CallItem without issues.  The progress
    queue is injected via _worker_init / initargs at spawn time and accessed
    through the module-level _WORKER_PROGRESS_Q global.

    Parameters
    ----------
    args : tuple of (run_id, params, flags)

    Returns
    -------
    tuple of (run_id, params, flags, results, run_time_s)
    """
    run_id, params, flags = args
    progress_q = _WORKER_PROGRESS_Q
    t0 = time.time()
    if progress_q is not None:
        t_total_ms = (
            params.get("tau_prebreakdown", 0.05)
            + params.get("tau_discharge", 20e-3)
            + params.get("tau_afterglow", 5e-3)
        ) * 1e3
        try:
            progress_q.put_nowait({
                "type": "starting",
                "run_id": run_id,
                "_t_total_ms": t_total_ms,
                "_V_bank": params.get("V_bank"),
                "_T_s_K": params.get("T_s"),
                "_S_gp": params.get("S_gp"),
                "_cells": params.get("cells"),
                "_TwinCathode": bool(flags.get("TwinCathode", False)),
                "_Twin_S_gp": params.get("Twin_S_gp"),
            })
        except Exception:
            pass

        def _progress_cb(frac, phase, seg_wall, rate_ema):
            try:
                progress_q.put_nowait({
                    "type": "progress",
                    "run_id": run_id,
                    "frac": frac,
                    "phase_code": _PHASE_CODE.get(phase, 0.0),
                    "seg_wall": seg_wall,
                    "rate_ema": rate_ema,
                })
            except Exception:
                pass
    else:
        _progress_cb = None
    sim = LAPDSim(params, flags, progress_callback=_progress_cb)
    sim.start_simulation()
    results = sim.get_results()
    run_time = time.time() - t0
    return run_id, params, flags, results, run_time


def grid_sweep_parallel(
    param_ranges,
    flag_ranges=None,
    fixed_params=None,
    fixed_flags=None,
    db_path="sweep.h5",
    t_window=(10.0, 20.0),
    n_workers=1,
    progress_callback=None,
    param_aliases=None,
    param_transforms=None,
    equilibrate_nn=False,
    verbose=True,
    verbose_equil=None,
    stop_event=None,
):
    """
    Run all combinations of ``param_ranges × flag_ranges`` in parallel and save to HDF5.

    Simulations are executed in a ``ProcessPoolExecutor`` with ``n_workers`` workers.
    HDF5 writes are performed on the calling thread to avoid concurrent write conflicts.

    Parameters
    ----------
    param_ranges : dict
        ``{param_name: [val1, val2, ...]}``.
    flag_ranges : dict or None
        ``{flag_name: [True, False, ...]}``.
    fixed_params : dict or None
        Parameters held constant.
    fixed_flags : dict or None
        Flags held constant.
    db_path : str or path-like
        Path to the HDF5 database.
    t_window : tuple of float
        (t_start, t_end) in ms for window statistics.
    n_workers : int
        Number of parallel worker processes.
    progress_callback : callable or None
        Called after each run completes:
        ``progress_callback(i, total, run_id, status, stats)``
        where ``stats`` is the window-stats dict (empty on failure) augmented
        with internal timing keys (``_run_time_s``, ``_equil_time_s``,
        ``_equil_cache_hit``, ``_equil_S_gp``, ``_equil_twin``,
        ``_equil_Twin_S_gp``, ``_equilibrate_nn``).  These ``_``-prefixed
        keys are **not** stored in the HDF5 database.
    param_aliases : dict or None
        ``{alias_key: source_key}`` pairs applied after building each run's
        params dict.  E.g. ``{"Twin_S_gp": "S_gp"}`` ensures symmetric twin mode
        tracks the primary even when ``S_gp`` is swept.
    param_transforms : callable or None
        ``(params, flags) -> params`` applied after ``param_aliases``.  Used to
        derive computed parameters, e.g. ``Id = P_in / Vd``.  Applied before
        dispatching to workers, so workers always receive fully-resolved params.
    equilibrate_nn : bool
        If ``True``, run a 100-cycle plasma-off pre-simulation for each
        combination (serially, before dispatch) to find the equilibrium
        neutral density.  Results are cached by neutral-dynamics key so that
        runs sharing the same S_gp / pumping / cells / TwinCathode config
        only equilibrate once.
    verbose : bool
        Print progress messages.
    verbose_equil : bool or None
        Print equilibration detail messages.  ``None`` (default) inherits
        from ``verbose``.  Set to ``False`` to suppress equilibration inner
        prints while keeping sweep progress prints.

    Returns
    -------
    list of str
        Run IDs that completed successfully.
    """
    if flag_ranges is None:
        flag_ranges = {}

    _verbose_equil = verbose if verbose_equil is None else verbose_equil

    combos = param_combinations(param_ranges, flag_ranges)
    n_total = len(combos)

    if verbose:
        print(f"Parallel sweep ({n_workers} workers): {n_total} combinations → '{db_path}'")

    with open_db(db_path, mode="a") as db:
        existing = {
            run_id for run_id in list_runs(db)
            if db["runs"][run_id].attrs.get("status") == "ok"
        }

    _nn_cache = {}  # cache_key → (nn_eq, equil_time_s)

    # Build pending list (skip already-done runs; pre-equilibrate if requested)
    pending = []
    pending_equil_info = []  # parallel list of equil metadata per pending run
    successful = []
    for i, (p_patch, f_patch) in enumerate(combos):
        run_id = f"run_{i:04d}"
        if run_id in existing:
            if verbose:
                print(f"  {run_id} already in database — skipping.")
            successful.append(run_id)
            continue
        params = {**input_dict_template, **(fixed_params or {}), **p_patch}
        flags = {**input_flags_template, **(fixed_flags or {}), **f_patch}

        # Apply param aliases (e.g. symmetric twin mirroring: Twin_Vd = Vd)
        if param_aliases:
            for alias, source in param_aliases.items():
                if source in params:
                    params[alias] = params[source]

        # Apply param transforms (derive computed params, e.g. Id = P_in / Vd)
        if param_transforms is not None:
            params = param_transforms(params, flags)

        equil_time = 0.0
        cache_hit = False
        twin_active = bool(flags.get("TwinCathode", False))

        if equilibrate_nn:
            twin_str = (
                f"  Twin_S_gp={params.get('Twin_S_gp', 0):.0f}" if twin_active else ""
            )
            cache_key = _equil_cache_key(params, flags)
            if cache_key in _nn_cache:
                nn_eq, equil_time = _nn_cache[cache_key]
                cache_hit = True
                if verbose:
                    print(
                        f"  [{i+1}/{n_total}] {run_id}: nn equil cache hit"
                        f"  S_gp={params.get('S_gp', 0):.0f}"
                        f"  twin={'on' if twin_active else 'off'}{twin_str}"
                    )
            else:
                table_val = _try_nn_table(params, flags)
                if table_val is not None:
                    n_cells_eq = int(params.get("cells", 3))
                    nn_eq = np.ones(n_cells_eq) * table_val
                    equil_time = 0.0
                    _nn_cache[cache_key] = (nn_eq, equil_time)
                    if verbose:
                        print(
                            f"  [{i+1}/{n_total}] {run_id}: nn equil table lookup"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                            f"  nn0={table_val:.3e}"
                        )
                else:
                    if verbose:
                        print(f"  [{i+1}/{n_total}] {run_id}: equilibrating nn0 …")
                    t0_equil = time.time()
                    nn_eq = equilibrate_neutrals(params, flags, verbose=_verbose_equil)
                    equil_time = time.time() - t0_equil
                    _nn_cache[cache_key] = (nn_eq, equil_time)
                    if verbose:
                        print(
                            f"    [nn equil] done"
                            f"  S_gp={params.get('S_gp', 0):.0f}"
                            f"  twin={'on' if twin_active else 'off'}{twin_str}"
                            f"  time={equil_time:.1f}s"
                        )
            params = _apply_equilibrated_nn(params, flags, nn_eq)

        pending.append((run_id, params, flags))
        pending_equil_info.append({
            "equil_time_s": equil_time,
            "cache_hit": cache_hit,
            "S_gp": float(params.get("S_gp", 0.0)),
            "twin": twin_active,
            "Twin_S_gp": float(params.get("Twin_S_gp", 0.0)) if twin_active else 0.0,
        })

    if not pending:
        if verbose:
            print("All runs already complete.")
        return successful

    completed_count = len(successful)

    if int(n_workers) <= 1:
        if verbose:
            print("Running sweep in-process (1 worker): no subprocess spawn/pickling overhead.")

        for (run_id, params, flags), equil_info in zip(pending, pending_equil_info):
            if stop_event is not None and stop_event.is_set():
                if verbose:
                    print(f"  Sweep aborted by user after {completed_count}/{n_total} runs.")
                break

            t_total_ms = (
                params.get("tau_prebreakdown", 0.05)
                + params.get("tau_discharge", 20e-3)
                + params.get("tau_afterglow", 5e-3)
            ) * 1e3
            if progress_callback is not None:
                progress_callback(None, n_total, run_id, "starting", {
                    "_start_time": time.time(),
                    "_timestamp": time.strftime("%H:%M:%S"),
                    "_t_total_ms": t_total_ms,
                    "_V_bank": params.get("V_bank"),
                    "_T_s_K": params.get("T_s"),
                    "_S_gp": params.get("S_gp"),
                    "_cells": params.get("cells"),
                    "_TwinCathode": bool(flags.get("TwinCathode", False)),
                    "_Twin_S_gp": params.get("Twin_S_gp"),
                })

                def _progress_cb(frac, phase, seg_wall, rate_ema, _run_id=run_id):
                    progress_callback(None, n_total, _run_id, "progress", {
                        "type": "progress",
                        "run_id": _run_id,
                        "frac": frac,
                        "phase_code": _PHASE_CODE.get(phase, 0.0),
                        "seg_wall": seg_wall,
                        "rate_ema": rate_ema,
                    })
            else:
                _progress_cb = None

            completed_count += 1
            t0_run = time.time()
            try:
                sim = LAPDSim(params, flags, progress_callback=_progress_cb)
                sim.start_simulation()
                results = sim.get_results()
                run_time = time.time() - t0_run
                stats = compute_window_stats(results, t_window)
                cat = getattr(results, "cells_at_time", None)
                n_cells = int(cat.max()) if cat is not None and len(cat) > 0 else int(params.get("cells", 3))
                with open_db(db_path, mode="a") as db:
                    save_run(db, run_id, params, flags, results, stats)
                    update_index(db, run_id, params, flags, stats, n_cells, status="ok")
                successful.append(run_id)
                if verbose:
                    print(f"  [{completed_count}/{n_total}] {run_id} ok — "
                          f"ne_var={stats['ne_var']:.3e}  run={run_time:.1f}s")
                if progress_callback is not None:
                    progress_callback(completed_count, n_total, run_id, "ok", {
                        **stats,
                        "_run_time_s": run_time,
                        "_equil_time_s": equil_info["equil_time_s"],
                        "_equil_cache_hit": equil_info["cache_hit"],
                        "_equil_S_gp": equil_info["S_gp"],
                        "_equil_twin": equil_info["twin"],
                        "_equil_Twin_S_gp": equil_info["Twin_S_gp"],
                        "_equilibrate_nn": equilibrate_nn,
                        "_V_bank": params.get("V_bank"),
                        "_T_s_K": params.get("T_s"),
                        "_TwinCathode": bool(flags.get("TwinCathode", False)),
                    })
            except Exception:
                tb = traceback.format_exc()
                print(f"  [{completed_count}/{n_total}] {run_id} FAILED:\n{tb}")
                with open_db(db_path, mode="a") as db2:
                    grp = db2.require_group("runs").require_group(run_id)
                    grp.attrs["status"] = "failed"
                    grp.attrs["error"] = tb[:2000]
                    for k, v in params.items():
                        try:
                            grp.attrs[f"param_{k}"] = v
                        except TypeError:
                            grp.attrs[f"param_{k}"] = str(v)
                    for k, v in flags.items():
                        grp.attrs[f"flag_{k}"] = bool(v)
                    update_index(db2, run_id, params, flags, {}, 0, status="failed")
                if progress_callback is not None:
                    progress_callback(completed_count, n_total, run_id, "failed", {
                        "_run_time_s": time.time() - t0_run,
                        "_equil_time_s": equil_info["equil_time_s"],
                        "_equil_cache_hit": equil_info["cache_hit"],
                        "_equil_S_gp": equil_info["S_gp"],
                        "_equil_twin": equil_info["twin"],
                        "_equil_Twin_S_gp": equil_info["Twin_S_gp"],
                        "_equilibrate_nn": equilibrate_nn,
                        "_V_bank": params.get("V_bank"),
                        "_T_s_K": params.get("T_s"),
                        "_TwinCathode": bool(flags.get("TwinCathode", False)),
                        "_error": tb.strip().splitlines()[-1],
                    })

        if verbose:
            print(f"Sweep complete: {len(successful)}/{n_total} runs succeeded.")
        return successful

    # Queue is passed to workers via initargs (pickled once during spawn, not per task).
    # Task args contain only plain Python objects to avoid _CallItem serialization errors.
    progress_mp_q = multiprocessing.Queue()

    import pathlib, logging
    _log_path = pathlib.Path(db_path).expanduser().with_suffix(".sweep.log")
    _sweep_log = logging.getLogger("bapsf_app.sweep")
    if not _sweep_log.handlers:
        _fh = logging.FileHandler(str(_log_path))
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _sweep_log.addHandler(_fh)
        _sweep_log.propagate = False
    _sweep_log.setLevel(logging.INFO)
    _sweep_log.info(
        f"sweep start: {n_total} runs, {n_workers} workers, db={db_path}  "
        f"worker logs: {pathlib.Path(db_path).expanduser().parent / '*.worker_<pid>.log'}"
    )

    _set_worker_thread_limits()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(progress_mp_q, db_path),
    ) as executor:
        future_to_run = {}
        for (run_id, params, flags), equil_info in zip(pending, pending_equil_info):
            fut = executor.submit(_run_single_worker, (run_id, params, flags))
            future_to_run[fut] = (run_id, params, flags, equil_info)

        def _drain_progress_q():
            # Drain at most ~200 messages per call with a short timeout so this
            # function is always bounded.  get_nowait() on macOS can occasionally
            # block on the underlying semaphore; using get(timeout=…) is safer.
            for _ in range(200):
                try:
                    msg = progress_mp_q.get(block=True, timeout=0.002)
                    if progress_callback is not None:
                        msg_type = msg.get("type", "progress")
                        if msg_type == "starting":
                            progress_callback(None, n_total, msg["run_id"], "starting", {
                                **{k: v for k, v in msg.items() if k not in ("type", "run_id")},
                                "_start_time": time.time(),
                                "_timestamp": time.strftime("%H:%M:%S"),
                            })
                        else:
                            progress_callback(None, n_total, msg["run_id"], "progress", msg)
                except Exception:
                    break

        def _handle_future(future):
            nonlocal completed_count
            run_id, params, flags, equil_info = future_to_run[future]
            completed_count += 1
            try:
                _, _, _, results, run_time = future.result()
                stats = compute_window_stats(results, t_window)
                cat = getattr(results, "cells_at_time", None)
                n_cells = int(cat.max()) if cat is not None and len(cat) > 0 else int(params.get("cells", 3))
                with open_db(db_path, mode="a") as db:
                    save_run(db, run_id, params, flags, results, stats)
                    update_index(db, run_id, params, flags, stats, n_cells, status="ok")
                successful.append(run_id)
                _sweep_log.info(f"{run_id} ok  ne_var={stats.get('ne_var', float('nan')):.3e}  run={run_time:.1f}s")
                if verbose:
                    twin_str = f"  Twin_S_gp={equil_info['Twin_S_gp']:.0f}" if equil_info["twin"] else ""
                    equil_str = ""
                    if equilibrate_nn:
                        cache_str = "cached" if equil_info["cache_hit"] else f"{equil_info['equil_time_s']:.1f}s"
                        equil_str = (
                            f"  equil={cache_str}"
                            f"  S_gp={equil_info['S_gp']:.0f}"
                            f"  twin={'on' if equil_info['twin'] else 'off'}{twin_str}"
                        )
                    print(f"  [{completed_count}/{n_total}] {run_id} ok — "
                          f"ne_var={stats['ne_var']:.3e}  run={run_time:.1f}s{equil_str}")
                if progress_callback is not None:
                    progress_callback(completed_count, n_total, run_id, "ok", {
                        **stats,
                        "_run_time_s": run_time,
                        "_equil_time_s": equil_info["equil_time_s"],
                        "_equil_cache_hit": equil_info["cache_hit"],
                        "_equil_S_gp": equil_info["S_gp"],
                        "_equil_twin": equil_info["twin"],
                        "_equil_Twin_S_gp": equil_info["Twin_S_gp"],
                        "_equilibrate_nn": equilibrate_nn,
                        "_V_bank": params.get("V_bank"),
                        "_T_s_K": params.get("T_s"),
                        "_TwinCathode": bool(flags.get("TwinCathode", False)),
                    })
            except Exception:
                tb = traceback.format_exc()
                print(f"  [{completed_count}/{n_total}] {run_id} FAILED:\n{tb}")
                _sweep_log.error(f"{run_id} FAILED:\n{tb}")
                with open_db(db_path, mode="a") as db2:
                    grp = db2.require_group("runs").require_group(run_id)
                    grp.attrs["status"] = "failed"
                    grp.attrs["error"] = tb[:2000]
                    for k, v in params.items():
                        try:
                            grp.attrs[f"param_{k}"] = v
                        except TypeError:
                            grp.attrs[f"param_{k}"] = str(v)
                    for k, v in flags.items():
                        grp.attrs[f"flag_{k}"] = bool(v)
                    update_index(db2, run_id, params, flags, {}, 0, status="failed")
                if progress_callback is not None:
                    progress_callback(completed_count, n_total, run_id, "failed", {
                        "_run_time_s": 0.0,
                        "_equil_time_s": equil_info["equil_time_s"],
                        "_equil_cache_hit": equil_info["cache_hit"],
                        "_equil_S_gp": equil_info["S_gp"],
                        "_equil_twin": equil_info["twin"],
                        "_equil_Twin_S_gp": equil_info["Twin_S_gp"],
                        "_equilibrate_nn": equilibrate_nn,
                        "_V_bank": params.get("V_bank"),
                        "_T_s_K": params.get("T_s"),
                        "_TwinCathode": bool(flags.get("TwinCathode", False)),
                        "_error": tb.strip().splitlines()[-1],
                    })

        # Poll for completions and drain progress queue every 0.1s
        pending_futs = set(future_to_run.keys())
        should_stop = False
        while pending_futs and not should_stop:
            done_futs, pending_futs = wait(pending_futs, timeout=0.1, return_when=FIRST_COMPLETED)
            _drain_progress_q()
            for future in done_futs:
                _handle_future(future)
                if stop_event is not None and stop_event.is_set():
                    for remaining in pending_futs:
                        remaining.cancel()
                    if verbose:
                        print(f"  Sweep aborted by user after {completed_count}/{n_total} runs.")
                    should_stop = True
                    break
        _drain_progress_q()  # flush any remaining progress messages

    if verbose:
        print(f"Parallel sweep complete: {len(successful)}/{n_total} runs succeeded.")

    return successful
