"""
HDF5 database read/write for LAPDSim sweep results.

Schema
------
sweep.h5
├── attrs: {created, description}
├── runs/
│   ├── run_0000/
│   │   ├── attrs: {param_*, flag_*, timestamp, status}
│   │   ├── <result arrays>          # scalar fields from get_results(); cell arrays padded to max_cells with NaN
│   │   ├── cells_at_time            # (n_timesteps,) int — active cell count at each recorded step
│   │   ├── refinement_events        # (N, 3) float — columns: [t_ms, old_cells, new_cells]
│   │   ├── cathode/                 # one dataset per _CATHODE_FIELDS entry, shape (n_timesteps,)
│   │   ├── cathode_twin/            # same structure; all-NaN when TwinCathode=False
│   │   └── stats_10_20ms/
│   │       └── attrs: {ne_var, ne_min, ...}
│   └── run_0001/ ...
└── index/
    ├── run_ids          resizable str dataset
    ├── status           resizable str dataset
    ├── n_cells          resizable int dataset
    ├── params/{name}    resizable float dataset per param
    ├── flags/{name}     resizable int (0/1) dataset per flag
    └── stats_10_20ms/{name}  resizable float dataset per stat
"""
import contextlib
import datetime

import h5py
import numpy as np


@contextlib.contextmanager
def open_db(path, mode="r"):
    """
    Context manager returning an open h5py.File.

    Parameters
    ----------
    path : str or path-like
    mode : str
        'r'  read-only, 'r+' read-write, 'a' append/create, 'w' truncate+create.
    """
    import pathlib
    p = pathlib.Path(path).expanduser()
    if mode in ("w", "a"):
        p.parent.mkdir(parents=True, exist_ok=True)
    db = h5py.File(p, mode)
    try:
        if mode in ("w", "a"):
            db.require_group("runs")
            db.require_group("index")
            if "created" not in db.attrs:
                db.attrs["created"] = datetime.datetime.now().isoformat()
        yield db
    finally:
        db.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _is_string_dataset(ds) -> bool:
    """Return True for HDF5 fixed/vlen string datasets."""
    return h5py.check_string_dtype(ds.dtype) is not None or ds.dtype.kind in ("S", "U")


def _stringify_index_value(val) -> str:
    """Convert an index value to a stable string representation."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float) and np.isnan(val):
        return ""
    return str(val)


def _coerce_index_value(val):
    """Return the scalar value and HDF5 dtype to use for a new index dataset."""
    if isinstance(val, (bool, np.bool_)):
        return np.int8(val), "i1"
    if isinstance(val, (int, np.integer)):
        return np.int32(val), "i4"
    if isinstance(val, str):
        return val, _str_dtype()
    return np.float64(val), "f8"


def _promote_dataset_to_strings(grp, key):
    """Replace an existing numeric index dataset with an equivalent string one."""
    old = grp[key][:]
    old_values = [_stringify_index_value(v) for v in old]
    del grp[key]
    grp.create_dataset(
        key,
        data=np.array(old_values, dtype=object),
        maxshape=(None,),
        dtype=_str_dtype(),
    )
    return grp[key]


def _promote_dataset_to_float(grp, key):
    """Replace an existing integer index dataset with a float64 one."""
    old = grp[key][:].astype("f8")
    del grp[key]
    grp.create_dataset(key, data=old, maxshape=(None,), dtype="f8")
    return grp[key]


def _pad_array(arr, n_rows: int, pad_value):
    """Return an array/list padded or trimmed to match the index row count."""
    if isinstance(arr, list):
        values = arr[:n_rows]
        values.extend([pad_value] * (n_rows - len(values)))
        return values

    arr = np.asarray(arr)
    if arr.dtype.kind in ("i", "u", "b") and isinstance(pad_value, float) and np.isnan(pad_value):
        arr = arr.astype("f8")
    if len(arr) >= n_rows:
        return arr[:n_rows]
    pad = np.full(n_rows - len(arr), pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad])


def _append_dataset(grp, key, val):
    """
    Append a single value to a resizable dataset in `grp`, creating it if needed.

    Strings are stored as variable-length UTF-8.
    Booleans are stored as int8.
    Floats and ints are stored as float64 and int32 respectively.
    """
    np_val, dtype = _coerce_index_value(val)

    if key not in grp:
        if isinstance(np_val, str):
            grp.create_dataset(
                key,
                data=np.array([np_val], dtype=object),
                maxshape=(None,),
                dtype=dtype,
            )
        else:
            grp.create_dataset(
                key,
                data=np.array([np_val]),
                maxshape=(None,),
                dtype=dtype,
            )
    else:
        ds = grp[key]
        if _is_string_dataset(ds):
            np_val = _stringify_index_value(val)
        elif isinstance(np_val, str):
            ds = _promote_dataset_to_strings(grp, key)
            np_val = _stringify_index_value(val)
        elif ds.dtype.kind in ("i", "u", "b") and isinstance(np_val, np.floating):
            ds = _promote_dataset_to_float(grp, key)
        n = ds.shape[0]
        ds.resize((n + 1,))
        ds[n] = np_val


# ── Public API ────────────────────────────────────────────────────────────────

def _create_result_dataset(grp, key, val):
    """Create a result dataset, compressing arrays but not HDF5 scalars."""
    arr = np.asarray(val)
    if arr.shape == ():
        grp.create_dataset(key, data=arr)
    else:
        grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)


def save_run(db, run_id, params, flags, results, stats):
    """
    Write one simulation run to ``db['runs/{run_id}']``.

    If the run_id already exists it is overwritten.

    Parameters
    ----------
    db : h5py.File
        Opened in 'a' or 'w' mode.
    run_id : str
    params : dict
        Input parameter dict (from ``input_dict_template``).
    flags : dict
        Input flags dict (from ``input_flags_template``).
    results : dict
        Output of ``sim.get_results()``.
    stats : dict
        Output of ``compute_window_stats()``.
    """
    runs = db.require_group("runs")
    if run_id in runs:
        del runs[run_id]

    grp = runs.create_group(run_id)
    grp.attrs["timestamp"] = datetime.datetime.now().isoformat()
    grp.attrs["status"] = "ok"

    # Store params as attrs
    for k, v in params.items():
        try:
            grp.attrs[f"param_{k}"] = v
        except TypeError:
            grp.attrs[f"param_{k}"] = str(v)

    # Store flags as attrs
    for k, v in flags.items():
        grp.attrs[f"flag_{k}"] = bool(v)

    # Store result arrays (results is a SimpleNamespace or dict)
    result_items = vars(results).items() if not isinstance(results, dict) else results.items()
    for key, val in result_items:
        if key in ("cathode", "cathode_twin"):
            # Each is a SimpleNamespace of per-field 1-D time-series arrays
            cgrp = grp.create_group(key)
            field_items = vars(val).items() if not isinstance(val, dict) else val.items()
            for fname, farr in field_items:
                _create_result_dataset(cgrp, fname, farr)
        elif key == "refinement_events":
            # list of (t, old_cells, new_cells) tuples — convert to uniform (N, 3) float array
            arr = np.array(val, dtype=float).reshape(-1, 3) if val else np.zeros((0, 3), dtype=float)
            grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        else:
            _create_result_dataset(grp, key, val)

    # Store pre-computed stats as attrs on a subgroup
    sg = grp.create_group("stats_10_20ms")
    for k, v in stats.items():
        sg.attrs[k] = float(v)


def load_run(db, run_id, keys=None):
    """
    Load result arrays for one run.

    Parameters
    ----------
    db : h5py.File
    run_id : str
    keys : list of str or None
        Which result arrays to load.  ``None`` loads every array.

    Returns
    -------
    params : dict
    flags  : dict
    results : dict of {key: np.ndarray}
    """
    grp = db["runs"][run_id]

    params = {}
    flags = {}
    for attr_key, attr_val in grp.attrs.items():
        if attr_key.startswith("param_"):
            params[attr_key[6:]] = attr_val
        elif attr_key.startswith("flag_"):
            flags[attr_key[5:]] = bool(attr_val)

    all_keys = [k for k in grp.keys() if k != "stats_10_20ms"]
    load_keys = keys if keys is not None else all_keys

    results = {}
    for k in load_keys:
        if k in grp:
            item = grp[k]
            if isinstance(item, h5py.Group):
                results[k] = {
                    fname: item[fname][()] if item[fname].shape == () else item[fname][:]
                    for fname in item.keys()
                }
            else:
                results[k] = item[()] if item.shape == () else item[:]

    return params, flags, results


def load_run_stats(db, run_id):
    """Load the pre-computed window stats for a single run."""
    sg = db["runs"][run_id]["stats_10_20ms"]
    return {k: float(v) for k, v in sg.attrs.items()}


def list_runs(db):
    """Return sorted list of run_ids present in the database."""
    return sorted(db.get("runs", {}).keys())


def update_index(db, run_id, params, flags, stats, n_cells, status="ok"):
    """
    Append one row to the index datasets.

    Creates datasets on the first call; resizes and appends on subsequent calls.
    """
    idx = db.require_group("index")

    _append_dataset(idx, "run_ids", run_id)
    _append_dataset(idx, "status", status)
    _append_dataset(idx, "n_cells", int(n_cells))

    p_grp = idx.require_group("params")
    n_rows = idx["run_ids"].shape[0]
    # Union of existing param keys and this run's param keys; pad missing entries with NaN
    # (same pattern as stats) so all numeric param arrays stay aligned with run_ids.
    all_param_keys = set(p_grp.keys()) | set(params.keys())
    for k in all_param_keys:
        if k in params:
            v = params[k]
            if isinstance(v, bool):
                val = int(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
            else:
                val = str(v)
        else:
            # Key exists from a prior run but not this one — pad with NaN for numeric,
            # empty string for string datasets.
            if k in p_grp and p_grp[k].dtype.kind in ("S", "O", "U"):
                val = ""
            else:
                val = float("nan")
        current_len = p_grp[k].shape[0] if k in p_grp else 0
        for _ in range(n_rows - 1 - current_len):
            if k in p_grp and p_grp[k].dtype.kind in ("S", "O", "U"):
                pad = ""
            elif isinstance(val, str):
                # Dataset not yet created; infer pad type from current value to avoid
                # creating a float dataset that later rejects the string value.
                pad = ""
            else:
                pad = float("nan")
            _append_dataset(p_grp, k, pad)
        _append_dataset(p_grp, k, val)

    f_grp = idx.require_group("flags")
    all_flag_keys = set(f_grp.keys()) | set(flags.keys())
    for k in all_flag_keys:
        val = int(bool(flags[k])) if k in flags else 0
        current_len = f_grp[k].shape[0] if k in f_grp else 0
        for _ in range(n_rows - 1 - current_len):
            _append_dataset(f_grp, k, 0)
        _append_dataset(f_grp, k, val)

    s_grp = idx.require_group("stats_10_20ms")
    # n_rows = total runs including this one (run_ids was already appended above)
    n_rows = idx["run_ids"].shape[0]
    # Union of existing stat keys and this run's stat keys
    all_stat_keys = set(s_grp.keys()) | set(stats.keys())
    for k in all_stat_keys:
        v = float(stats[k]) if k in stats else float("nan")
        # Pad any missing entries from prior runs that didn't have this key (e.g. failures
        # before the first success, or a new key introduced mid-sweep).
        current_len = s_grp[k].shape[0] if k in s_grp else 0
        for _ in range(n_rows - 1 - current_len):
            _append_dataset(s_grp, k, float("nan"))
        _append_dataset(s_grp, k, v)


def load_index(db):
    """
    Return a dict-of-arrays summarising all runs.

    Returns
    -------
    dict with keys:
        run_ids       : list of str
        status        : list of str
        n_cells       : np.ndarray int
        params        : dict of {name: np.ndarray}
        flags         : dict of {name: np.ndarray bool}
        stats_10_20ms : dict of {name: np.ndarray float}
    """
    if "index" not in db:
        return {
            "run_ids": [],
            "status": [],
            "n_cells": np.array([], dtype=int),
            "params": {},
            "flags": {},
            "stats_10_20ms": {},
        }

    idx = db["index"]

    if "run_ids" not in idx:
        return {
            "run_ids": [],
            "status": [],
            "n_cells": np.array([], dtype=int),
            "params": {},
            "flags": {},
            "stats_10_20ms": {},
        }

    def _decode(arr):
        return [s.decode() if isinstance(s, bytes) else s for s in arr]

    run_ids = _decode(idx["run_ids"][:])
    n_rows = len(run_ids)

    result = {
        "run_ids": run_ids,
        "status": _pad_array(_decode(idx["status"][:]), n_rows, "unknown"),
        "n_cells": _pad_array(idx["n_cells"][:], n_rows, 0).astype(int),
        "params": {},
        "flags": {},
        "stats_10_20ms": {},
    }

    for k in idx.get("params", {}).keys():
        ds = idx["params"][k][:]
        if _is_string_dataset(idx["params"][k]):
            result["params"][k] = _pad_array(_decode(ds), n_rows, "")
        else:
            result["params"][k] = _pad_array(ds, n_rows, float("nan"))

    for k in idx.get("flags", {}).keys():
        result["flags"][k] = _pad_array(idx["flags"][k][:], n_rows, 0).astype(bool)

    for k in idx.get("stats_10_20ms", {}).keys():
        result["stats_10_20ms"][k] = _pad_array(idx["stats_10_20ms"][k][:], n_rows, float("nan"))

    return result


def rebuild_index(db):
    """
    Rebuild the ``index/`` group from the ``runs/`` group.

    Useful after partial failures or manual edits.
    """
    if "index" in db:
        del db["index"]
    db.require_group("index")

    for run_id in sorted(db.get("runs", {}).keys()):
        grp = db["runs"][run_id]
        status = grp.attrs.get("status", "ok")

        if "cells_at_time" in grp and grp["cells_at_time"].shape[0] > 0:
            n_cells = int(grp["cells_at_time"][:].max())
        elif "ne" in grp:
            n_cells = int(grp["ne"].shape[1])
        else:
            n_cells = 0
        params = {k[6:]: v for k, v in grp.attrs.items() if k.startswith("param_")}
        flags = {k[5:]: bool(v) for k, v in grp.attrs.items() if k.startswith("flag_")}
        stats = {}
        if "stats_10_20ms" in grp:
            stats = {k: float(v) for k, v in grp["stats_10_20ms"].attrs.items()}

        # Recompute stats from raw arrays if any keys are missing (e.g. after schema update)
        _required = {"ne_var", "ne_tvar", "ne_tcov", "ne_total_var", "Te_var", "Te_tvar", "Te_tcov", "Te_total_var", "P_net_mean", "P_eff", "P_net_total", "P_eff_total"}
        if status == "ok" and not _required.issubset(stats.keys()) and "ne" in grp and "time" in grp:
            from bapsf_app.stats import compute_window_stats
            try:
                results = {k: grp[k][:] for k in ("time", "ne", "Te") if k in grp}
                for ckey in ("cathode", "cathode_twin"):
                    if ckey in grp:
                        results[ckey] = {fname: grp[ckey][fname][:] for fname in grp[ckey].keys()}
                stats = compute_window_stats(results)  # stats.py accepts dict via _get helper
                sg = grp.require_group("stats_10_20ms")
                for k, v in stats.items():
                    sg.attrs[k] = float(v)
            except Exception:
                pass

        update_index(db, run_id, params, flags, stats, n_cells, status)
