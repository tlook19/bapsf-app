"""
LAPDSim Parameter Sweep GUI — Streamlit application.

Launch with:
    poetry run lapd-app
or:
    streamlit run cablp/cablp/analysis/app.py
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import threading
import time
from dataclasses import dataclass, field

if "MPLCONFIGDIR" not in os.environ:
    _mpl_config_dir = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "bapsf_app_matplotlib"
    _mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
    _xdg_cache_dir = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "bapsf_app_cache"
    _xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(_xdg_cache_dir)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
import streamlit as st

try:
    import setproctitle

    setproctitle.setproctitle("bapsf-app")
except ImportError:
    pass

matplotlib.use("Agg")  # non-interactive backend required for Streamlit

_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _WORKSPACE_ROOT / "database" / "sweep.h5"

# Absolute imports (relative imports fail when Streamlit runs app.py as __main__)
from bapsf_app.sweep import (
    get_worker_affinity_info,
    grid_sweep_parallel,
    param_combinations,
)
from bapsf_app.database import open_db, load_index, list_runs, load_run, rebuild_index, update_index
from bapsf_app.plot import (
    plot_run,
    plot_run_comparison,
    plot_sweep_variance,
    plot_sweep_heatmap,
    plot_time_slice,
)
from bapsf_app.stats import compute_cathode_peak_stats

# ── Sweep progress manifest (persisted alongside the HDF5 database) ───────────

def _manifest_path(db_path: str) -> pathlib.Path:
    """Return the JSON manifest path for a given database path."""
    p = pathlib.Path(db_path).expanduser()
    return p.parent / (p.stem + ".progress.json")


def _log_dir(db_path: str) -> pathlib.Path:
    """Return the diagnostic log directory for a database path."""
    return pathlib.Path(db_path).expanduser().parent / "logs"


def _save_manifest(db_path: str, data: dict) -> None:
    """Save sweep-progress state to a JSON file next to the database."""
    try:
        mp = _manifest_path(db_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # non-critical; never crash the sweep


def _load_manifest(db_path: str) -> dict | None:
    """Load sweep-progress state.  Returns None if not found or invalid."""
    try:
        mp = _manifest_path(db_path)
        if not mp.exists():
            return None
        with open(mp) as f:
            return json.load(f)
    except Exception:
        return None


# ── Directory / file browser widget ───────────────────────────────────────────

def _render_path_input(key: str, label: str, default: str, file_ext: str = ".h5") -> str:
    """
    Text input with a collapsible directory browser for selecting an HDF5 file path.
    Returns the current path value (unexpanded, as the user typed it).
    """
    txt_key = f"_pathtxt_{key}"
    pending_key = f"_pathtxt_pending_{key}"
    dir_key = f"_pathdir_{key}"
    show_key = f"_pathshow_{key}"

    if txt_key not in st.session_state:
        st.session_state[txt_key] = default
    # Apply pending selection from file browser before the widget renders
    if pending_key in st.session_state:
        st.session_state[txt_key] = st.session_state[pending_key]
        del st.session_state[pending_key]

    col_input, col_btn = st.columns([8, 1])
    with col_input:
        st.text_input(label, key=txt_key)
    if col_btn.button("📂", key=f"_pathbtn_{key}", help="Browse filesystem"):
        cur = pathlib.Path(st.session_state[txt_key]).expanduser()
        # Start browse from parent dir of current path, or best fallback
        candidates = [
            cur.parent if (cur.suffix == file_ext or cur.is_file()) else cur,
            _DEFAULT_DB_PATH.parent,
            pathlib.Path("~/lapd_data").expanduser(),
            pathlib.Path.home(),
        ]
        for c in candidates:
            if c.exists():
                st.session_state[dir_key] = str(c)
                break
        st.session_state[show_key] = not st.session_state.get(show_key, False)
        st.rerun()

    if st.session_state.get(show_key, False):
        cur_dir = pathlib.Path(st.session_state.get(dir_key, str(pathlib.Path.home())))
        st.caption(f"📁 `{cur_dir}`")

        parent = cur_dir.parent
        if parent != cur_dir:
            if st.button("⬆ Parent dir", key=f"_pathup_{key}"):
                st.session_state[dir_key] = str(parent)
                st.rerun()

        try:
            entries = sorted(
                cur_dir.iterdir(),
                key=lambda e: (e.is_file(), e.name.lower()),
            )
            dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
            files = [e for e in entries if e.is_file() and e.suffix == file_ext]

            if dirs:
                n_cols = min(4, len(dirs))
                dir_cols = st.columns(n_cols)
                for ci, d in enumerate(dirs[:12]):
                    if dir_cols[ci % n_cols].button(f"📁 {d.name}", key=f"_pd_{key}_{d.name}"):
                        st.session_state[dir_key] = str(d)
                        st.rerun()

            for f_entry in files:
                if st.button(f"📄 {f_entry.name}", key=f"_pf_{key}_{f_entry.name}"):
                    st.session_state[pending_key] = str(f_entry)
                    st.session_state[show_key] = False
                    st.rerun()

            nc1, nc2 = st.columns([5, 1])
            new_name = nc1.text_input(
                "New filename",
                key=f"_pnew_{key}",
                placeholder=f"filename{file_ext}",
                label_visibility="collapsed",
            )
            if nc2.button("Use", key=f"_puse_{key}") and new_name:
                fn = new_name if new_name.endswith(file_ext) else new_name + file_ext
                st.session_state[pending_key] = str(cur_dir / fn)
                st.session_state[show_key] = False
                st.rerun()

        except PermissionError:
            st.warning("Permission denied")

    return st.session_state.get(txt_key, default)


# ── Abort-helper: mark planned-but-not-started runs as failed in the DB ────────

def _mark_incomplete_as_failed(db_path: str, manifest: dict) -> None:
    """Write a 'failed' record for every planned run_id not yet in the database."""
    planned_ids = manifest.get("planned_run_ids", [])
    if not planned_ids:
        st.warning("Manifest has no planned run IDs; nothing to mark.")
        return
    try:
        with open_db(db_path, mode="a") as db:
            existing = set(list_runs(db))
            to_mark = [rid for rid in planned_ids if rid not in existing]
            for run_id in to_mark:
                grp = db.require_group("runs").require_group(run_id)
                grp.attrs["status"] = "failed"
                grp.attrs["error"] = "Sweep was aborted before this run could execute."
                update_index(db, run_id, {}, {}, {}, 0, status="failed")
        if to_mark:
            st.success(f"Marked {len(to_mark)} incomplete run(s) as failed.")
        else:
            st.info("All planned runs are already recorded in the database.")
    except Exception as exc:
        st.error(f"Failed to mark incomplete runs: {exc}")


# ── Parameter / flag metadata ─────────────────────────────────────────────────
# Each entry: key → {label, unit, default, type, group}
# type: "float" | "int" | "str" | "bool"
# str entries also have "choices": list

PARAM_META: dict[str, dict] = {
    # ── Gas & Initial Conditions ──────────────────────────────────────────────
    "gas_type": {
        "label": "Gas type", "unit": "", "default": "He",
        "type": "str", "group": "Gas & Initial Conditions",
        "choices": ["He", "H"],
    },
    "ne0": {
        "label": "Initial electron density (ne0)", "unit": "cm⁻³", "default": 1e9,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "nn0": {
        "label": "Initial neutral density (nn0)", "unit": "cm⁻³", "default": 5e12,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "Te0": {
        "label": "Initial electron temperature (Te0)", "unit": "eV", "default": 0.1,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "Ti0": {
        "label": "Initial ion temperature (Ti0)", "unit": "eV", "default": 0.1,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    "Tn_fit": {
        "label": "Neutral temp for rate fits (Tn_fit)", "unit": "eV", "default": 0.1,
        "type": "float", "group": "Gas & Initial Conditions",
    },
    # ── Machine Geometry ──────────────────────────────────────────────────────
    "Lm": {
        "label": "Machine length (Lm)", "unit": "cm", "default": 1800.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Rm": {
        "label": "Machine radius (Rm)", "unit": "cm", "default": 50.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Lp": {
        "label": "Plasma length (Lp)", "unit": "cm", "default": 1800.0,
        "type": "float", "group": "Machine Geometry",
    },
    "Rp": {
        "label": "Plasma radius (Rp)", "unit": "cm", "default": 18.0,
        "type": "float", "group": "Machine Geometry",
    },
    # ── Discharge (Primary Cathode) ───────────────────────────────────────────
    "V_bank": {
        "label": "Power supply voltage (V_bank)", "unit": "V", "default": 100.0,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "gas_puff_mode": {
        "label": "Gas puff mode (gas_puff_mode)", "unit": "",
        "default": "decay_after_breakdown", "type": "str", "group": "Discharge (Primary Cathode)",
        "choices": ["decay_after_breakdown", "pulse_decay_to_level"],
    },
    "S_gp": {
        "label": "Gas puff source rate (S_gp)", "unit": "", "default": 500.0,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "S_gp_decay_target": {
        "label": "Gas puff target rate (S_gp_decay_target)", "unit": "", "default": 0.0,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "T_s": {
        "label": "Cathode surface temperature (T_s)", "unit": "°C", "default": 1626.85,
        "type": "float", "group": "Discharge (Primary Cathode)",
    },
    # ── Cathode Hardware ──────────────────────────────────────────────────────
    "phi_wf": {
        "label": "Work function (phi_wf)", "unit": "eV", "default": 3.0,
        "type": "float", "group": "Cathode Hardware",
    },
    "C_R": {
        "label": "Richardson constant (C_R)", "unit": "A cm⁻² K⁻²", "default": 29.0,
        "type": "float", "group": "Cathode Hardware",
    },
    "R_comp": {
        "label": "Compliance resistor (R_comp)", "unit": "Ω", "default": 0.004,
        "type": "float", "group": "Cathode Hardware",
    },
    "eta": {
        "label": "Anode/cathode area ratio (eta)", "unit": "", "default": 0.358,
        "type": "float", "group": "Cathode Hardware",
    },
    "L_cath": {
        "label": "Cathode-to-anode distance (L_cath)", "unit": "cm", "default": 50.0,
        "type": "float", "group": "Cathode Hardware",
    },
    "R_cath": {
        "label": "Cathode radius (R_cath)", "unit": "cm", "default": 18.0,
        "type": "float", "group": "Cathode Hardware",
    },
    # ── Source / Sinks ────────────────────────────────────────────────────────
    "S_pump_L": {
        "label": "Pump sink rate left (S_pump_L)", "unit": "s⁻¹", "default": 4000.0,
        "type": "float", "group": "Source / Sinks",
    },
    "S_pump_R": {
        "label": "Pump sink rate right (S_pump_R)", "unit": "s⁻¹", "default": 4000.0,
        "type": "float", "group": "Source / Sinks",
    },
    # ── Time & Solver ─────────────────────────────────────────────────────────
    "cells": {
        "label": "Number of cells", "unit": "", "default": 3,
        "type": "int", "group": "Time & Solver",
    },
    "max_cells": {
        "label": "Max cells (adaptive mesh)", "unit": "", "default": 18,
        "type": "int", "group": "Time & Solver",
    },
    "min_cells": {
        "label": "Min cells (adaptive mesh)", "unit": "", "default": 3,
        "type": "int", "group": "Time & Solver",
    },
    "mfp_refine_threshold": {
        "label": "MFP refine threshold", "unit": "", "default": 0.5,
        "type": "float", "group": "Time & Solver",
    },
    "mfp_coarsen_threshold": {
        "label": "MFP coarsen threshold", "unit": "", "default": 2.0,
        "type": "float", "group": "Time & Solver",
    },
    "tau_prebreakdown": {
        "label": "Pre-breakdown timeout (tau_prebreakdown)", "unit": "s", "default": 0.05,
        "type": "float", "group": "Time & Solver",
    },
    "tau_discharge": {
        "label": "Discharge duration (tau_discharge)", "unit": "s", "default": 20e-3,
        "type": "float", "group": "Time & Solver",
    },
    "tau_gp_after_breakdown": {
        "label": "Gas puff decay start after breakdown (tau_gp_after_breakdown)", "unit": "ms",
        "default": None, "type": "float_or_none", "group": "Discharge (Primary Cathode)",
    },
    "tau_gp_decay_factor": {
        "label": "Gas puff decay time factor (tau_gp_decay_factor)", "unit": "",
        "default": 1.0, "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "tau_gp_pulse_duration": {
        "label": "Gas puff pulse duration (tau_gp_pulse_duration)", "unit": "s",
        "default": 0.0, "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "tau_gp_decay_duration": {
        "label": "Gas puff decay duration (tau_gp_decay_duration)", "unit": "s",
        "default": 1e-3, "type": "float", "group": "Discharge (Primary Cathode)",
    },
    "tau_afterglow": {
        "label": "Afterglow duration (tau_afterglow)", "unit": "s", "default": 5e-3,
        "type": "float", "group": "Time & Solver",
    },
    "tau_cycle": {
        "label": "Cycle length, Plasma=False (tau_cycle)", "unit": "s", "default": 3.0,
        "type": "float", "group": "Time & Solver",
    },
    "I_prebreakdown": {
        "label": "Pre-breakdown exit current (I_prebreakdown)", "unit": "A", "default": 0.0,
        "type": "float", "group": "Time & Solver",
    },
    "I_breakdown": {
        "label": "Breakdown current threshold (I_breakdown)", "unit": "A", "default": 1000.0,
        "type": "float", "group": "Time & Solver",
    },
    "h0": {
        "label": "Initial step size (h0)", "unit": "s", "default": 1e-6,
        "type": "float", "group": "Time & Solver",
    },
    "h_max_discharge": {
        "label": "Max step, discharge (h_max_discharge)", "unit": "s", "default": 1e-4,
        "type": "float", "group": "Time & Solver",
    },
    "h_max_afterglow": {
        "label": "Max step, afterglow (h_max_afterglow)", "unit": "s", "default": 1e-4,
        "type": "float", "group": "Time & Solver",
    },
    "max_step_rejections": {
        "label": "Max consecutive step rejections (max_step_rejections)", "unit": "",
        "default": 200, "type": "int", "group": "Time & Solver",
    },
    "cycles": {
        "label": "Equilibration cycles (Plasma=False)", "unit": "", "default": 1,
        "type": "int", "group": "Time & Solver",
    },
    "rtol": {
        "label": "Relative tolerance (rtol)", "unit": "", "default": 1e-3,
        "type": "float", "group": "Time & Solver",
    },
    "h_min": {
        "label": "Min step size (h_min)", "unit": "s", "default": 1e-12,
        "type": "float", "group": "Time & Solver",
    },
    "h_min_prebreakdown": {
        "label": "Min step, pre-breakdown (h_min_prebreakdown)", "unit": "s", "default": 1e-6,
        "type": "float", "group": "Time & Solver",
    },
    "prebreakdown_cfl_factor": {
        "label": "Pre-breakdown CFL factor (prebreakdown_cfl_factor)", "unit": "",
        "default": 10.0, "type": "float", "group": "Time & Solver",
    },
    # ── Transport Scaling ─────────────────────────────────────────────────────
    "b_epara": {"label": "e⁻ parallel scale (b_epara)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_ipara": {"label": "Ion parallel scale (b_ipara)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_ioniz": {"label": "Ionization scale (b_ioniz)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_rec_rad": {"label": "Rad recombination scale (b_rec_rad)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_rec_3b": {"label": "3-body recombination scale (b_rec_3b)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qcx": {"label": "Charge exchange scale (b_Qcx)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_source": {"label": "Source heating scale (b_source)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qie": {"label": "Q_ie scale (b_Qie)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qei": {"label": "Q_ei scale (b_Qei)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_Qen": {"label": "Q_en scale (b_Qen)", "unit": "", "default": 1.0, "type": "float", "group": "Transport Scaling"},
    "b_div_v_elec": {"label": "Electron compression scale (b_div_v_elec)", "unit": "", "default": 0.0, "type": "float", "group": "Transport Scaling"},
    "b_div_v_ions": {"label": "Ion compression scale (b_div_v_ions)", "unit": "", "default": 0.0, "type": "float", "group": "Transport Scaling"},
    "b_Te_conv": {"label": "Electron convection scale (b_Te_conv)", "unit": "", "default": 0.0, "type": "float", "group": "Transport Scaling"},
    "b_Ti_conv": {"label": "Ion convection scale (b_Ti_conv)", "unit": "", "default": 0.0, "type": "float", "group": "Transport Scaling"},
}

# Twin cathode params rendered separately under Dual Cathode section.
# Both cathodes share the same device hardware (V_bank, T_s, etc.); only
# the neutral-side initial conditions differ per cathode.
TWIN_META: dict[str, dict] = {
    "Twin_S_gp": {"label": "Twin gas puff rate (Twin_S_gp)", "unit": "", "default": 500.0, "type": "float"},
    "Twin_S_gp_decay_target": {
        "label": "Twin gas puff target rate (Twin_S_gp_decay_target)",
        "unit": "",
        "default": 0.0,
        "type": "float",
    },
}

_GAS_PUFF_DECAY_AFTER_BREAKDOWN_PARAMS = {
    "tau_gp_after_breakdown",
    "tau_gp_decay_factor",
}
_GAS_PUFF_PULSE_DECAY_PARAMS = {
    "S_gp_decay_target",
    "tau_gp_pulse_duration",
    "tau_gp_decay_duration",
    "Twin_S_gp_decay_target",
}

FLAG_META: dict[str, dict] = {
    "icool": {"label": "Ion cooling", "default": True, "group": "Physics"},
    "ncool": {"label": "Neutral cooling", "default": True, "group": "Physics"},
    "cx": {"label": "Charge exchange", "default": True, "group": "Physics"},
    "icool_recomb": {"label": "Ion cooling from recombination", "default": False, "group": "Physics"},
    "Plasma": {"label": "Plasma physics", "default": True, "group": "Simulation"},
    "Velocity": {"label": "Plasma velocity", "default": True, "group": "Simulation"},
    "advection": {"label": "Convective v·∇v acceleration", "default": True, "group": "Simulation"},
    "hybrid_ne": {"label": "Hybrid density flux (sonic correction)", "default": True, "group": "Simulation"},
    "adaptive_mesh": {"label": "Adaptive spatial mesh", "default": False, "group": "Simulation"},
}

PARAM_GROUP_ORDER = [
    "Gas & Initial Conditions",
    "Machine Geometry",
    "Discharge (Primary Cathode)",
    "Cathode Hardware",
    "Source / Sinks",
    "Time & Solver",
    "Transport Scaling",
]

FLAG_GROUP_ORDER = ["Physics", "Simulation"]

_ADAPTIVE_MESH_PARAMS = {"max_cells", "min_cells", "mfp_refine_threshold", "mfp_coarsen_threshold"}


# ── Config persistence ─────────────────────────────────────────────────────────

_CONFIG_PATH = pathlib.Path.home() / ".lapd_app_config.json"

# All session-state keys that represent widget settings to save/restore
_PARAM_CFG_KEYS: list[str] = []
for _k in list(PARAM_META.keys()) + list(TWIN_META.keys()):
    _PARAM_CFG_KEYS += [
        f"pmode_{_k}", f"pfixed_{_k}",
        f"pmin_{_k}", f"pmax_{_k}", f"pstep_{_k}", f"pvary_{_k}",
        f"penable_{_k}",
    ]
_FLAG_CFG_KEYS: list[str] = [f"flag_{k}" for k in FLAG_META]
_MISC_CFG_KEYS: list[str] = ["dc_on_off", "dc_type"]
_ALL_CFG_KEYS: list[str] = _PARAM_CFG_KEYS + _FLAG_CFG_KEYS + _MISC_CFG_KEYS

_SLICE_META_LABELS: dict[str, str] = {
    "ne":           "Electron density",
    "nn":           "Neutral density",
    "Te":           "Electron temp",
    "Ti":           "Ion temp",
    "v_plasma":     "Plasma velocity",
    "isat":         "Ion sat. current",
    "ion_fraction": "Ionization fraction",
    "density_flux": "Density flux",
    "electron_heat_terms": "Electron heat terms",
    "ion_heat_terms":       "Ion heat terms",
    "density_source_terms": "Density source/sink terms",
    "Ne_face_flux":         "Electron face flux",
    "Nn_face_flux":         "Neutral face flux",
    "e_par_face_flux":      "e⁻ parallel heat face flux",
    "i_par_face_flux":      "Ion parallel heat face flux",
}


def _get_serializable_state() -> dict:
    """Extract serialisable widget state from session_state."""
    data: dict = {}
    for key in _ALL_CFG_KEYS:
        val = st.session_state.get(key)
        if val is None:
            continue
        if isinstance(val, np.integer):
            val = int(val)
        elif isinstance(val, np.floating):
            val = float(val)
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        elif isinstance(val, list):
            val = [
                int(v) if isinstance(v, np.integer)
                else float(v) if isinstance(v, np.floating)
                else v
                for v in val
            ]
        data[key] = val
    return data


def _apply_state(data: dict) -> None:
    """Write config data to session_state so widgets pick it up on next render."""
    valid = set(_ALL_CFG_KEYS)
    for key, val in data.items():
        if key in valid:
            st.session_state[key] = val


def _save_config(path: pathlib.Path = _CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_get_serializable_state(), fh, indent=2)


def _load_config(path: pathlib.Path = _CONFIG_PATH) -> bool:
    """Load config from *path*.  Returns True on success."""
    try:
        with open(path) as fh:
            data = json.load(fh)
        _apply_state(data)
        return True
    except Exception:
        return False


def _load_defaults() -> None:
    """Reset all widget state to PARAM_META / FLAG_META built-in defaults."""
    for key, meta in PARAM_META.items():
        st.session_state[f"pmode_{key}"] = "Fixed"
        default = meta["default"]
        if meta["type"] == "str":
            st.session_state[f"pfixed_{key}"] = default
        elif meta["type"] == "float_or_none":
            st.session_state[f"penable_{key}"] = False
        else:
            st.session_state[f"pfixed_{key}"] = float(default)
    for key, meta in TWIN_META.items():
        st.session_state[f"pmode_{key}"] = "Fixed"
        st.session_state[f"pfixed_{key}"] = float(meta["default"])
    for key, meta in FLAG_META.items():
        st.session_state[f"flag_{key}"] = "True" if meta["default"] else "False"
    st.session_state["dc_on_off"] = "Off"
    st.session_state["dc_type"] = "Twin (symmetric)"


# ── Sweep state ────────────────────────────────────────────────────────────────

@dataclass
class SweepState:
    total: int = 0
    completed: int = 0
    failed: int = 0
    log: list = field(default_factory=list)
    failed_log: list = field(default_factory=list)
    running: bool = True
    done: bool = False
    error: str = ""
    total_time_s: float = 0.0
    db_path: str = ""
    planned_run_ids: list = field(default_factory=list)
    active_runs: dict = field(default_factory=dict)   # run_id -> {start_time, label}
    finished_run_ids: set = field(default_factory=set)
    completed_run_times: list = field(default_factory=list)


# ── Session state helpers ──────────────────────────────────────────────────────

def _ss(key, default=None):
    """Get session state value with a default."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _set_ss(key, value):
    st.session_state[key] = value


# ── Range computation ─────────────────────────────────────────────────────────

def _process_tree_metrics():
    """Return RAM and CPU for the Streamlit process plus live worker children."""
    parent = psutil.Process()
    try:
        procs = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        procs = [parent]

    ram_gb = 0.0
    now = time.time()
    last = st.session_state.get("_proc_tree_cpu_sample")
    current_times = {}

    for proc in procs:
        try:
            times = proc.cpu_times()
            current_times[proc.pid] = times.user + times.system
            ram_gb += proc.memory_info().rss / 1e9
        except psutil.Error:
            continue

    cpu_pct = 0.0
    if last is not None:
        last_wall, last_times = last
        wall_dt = max(now - last_wall, 1e-6)
        cpu_dt = sum(
            max(total_time - last_times[pid], 0.0)
            for pid, total_time in current_times.items()
            if pid in last_times
        )
        n_cpu = max(psutil.cpu_count(logical=True) or 1, 1)
        cpu_pct = min(100.0, 100.0 * cpu_dt / (wall_dt * n_cpu))

    st.session_state["_proc_tree_cpu_sample"] = (now, current_times)
    return ram_gb, cpu_pct, max(len(current_times) - 1, 0)


def _arange_inclusive(min_val, max_val, step):
    """np.arange with inclusive upper bound."""
    if step <= 0:
        return np.array([min_val])
    vals = np.arange(min_val, max_val + step * 1e-9, step)
    return vals[vals <= max_val + step * 1e-9]


def _format_vals(vals):
    if len(vals) == 0:
        return "[]"
    def _fv(v):
        if isinstance(v, str):
            return v
        try:
            return f"{v:g}"
        except (TypeError, ValueError):
            return str(v)
    if len(vals) <= 6:
        return "[" + ", ".join(_fv(v) for v in vals) + "]"
    return f"[{_fv(vals[0])}, {_fv(vals[1])}, … {_fv(vals[-1])}]  ({len(vals)} values)"


# ── Widget renderers ───────────────────────────────────────────────────────────

def _widget_value(key: str, default):
    """Avoid Streamlit's warning when restored session state owns a widget value."""
    return None if key in st.session_state else default


def _widget_index(key: str, default: int) -> int | None:
    """Avoid mixing widget index defaults with values restored into session state."""
    return None if key in st.session_state else default


def _widget_default(key: str, default):
    """Avoid mixing multiselect defaults with values restored into session state."""
    return None if key in st.session_state else default


def _num_format(value) -> str:
    """Return a printf format string for a number input.

    Uses scientific notation for values with magnitude >= 1e5 or < 1e-3.
    """
    if value is None or value == 0:
        return "%g"
    v = abs(float(value))
    if v >= 1e5 or v < 1e-3:
        return "%.3e"
    return "%g"


def _selected_gas_puff_modes() -> set[str]:
    """Return gas puff modes currently fixed or included in the sweep."""
    meta = PARAM_META["gas_puff_mode"]
    choices = set(meta.get("choices", []))
    if st.session_state.get("pmode_gas_puff_mode", "Fixed") == "Vary":
        selected = st.session_state.get("pvary_gas_puff_mode", list(choices))
        return set(selected) or choices
    return {st.session_state.get("pfixed_gas_puff_mode", meta["default"])}


def _is_gas_puff_param_visible(key: str) -> bool:
    modes = _selected_gas_puff_modes()
    if key in _GAS_PUFF_DECAY_AFTER_BREAKDOWN_PARAMS:
        return "decay_after_breakdown" in modes
    if key in _GAS_PUFF_PULSE_DECAY_PARAMS:
        return "pulse_decay_to_level" in modes
    return True


def _render_param_row(key: str, meta: dict) -> None:
    """Render a fixed/range selector for one numeric or string parameter."""
    param_type = meta["type"]
    label = meta["label"]
    unit = meta.get("unit", "")
    default = meta["default"]

    label_disp = f"**{label}**" + (f"  [{unit}]" if unit else "")
    st.markdown(label_disp)

    if param_type == "str":
        choices = meta.get("choices", [])
        mode_key = f"pmode_{key}"
        mode = st.radio(
            f"##mode_{key}", ["Fixed", "Vary"], horizontal=True,
            index=_widget_index(mode_key, 0),
            label_visibility="collapsed", key=mode_key,
        )
        if mode == "Fixed":
            idx_default = choices.index(default) if default in choices else 0
            fixed_key = f"pfixed_{key}"
            val = st.selectbox(
                f"##fix_{key}", choices,
                index=_widget_index(fixed_key, idx_default),
                label_visibility="collapsed", key=fixed_key,
            )
            _set_ss(f"param_{key}", {"mode": "fixed", "value": val})
        else:
            vary_key = f"pvary_{key}"
            selected = st.multiselect(
                f"##vary_{key}", choices,
                default=_widget_default(vary_key, choices),
                label_visibility="collapsed", key=vary_key,
            )
            _set_ss(f"param_{key}", {"mode": "range", "values": selected or choices})
        return

    # Optional float (None = disabled / use solver default)
    if param_type == "float_or_none":
        mode_key = f"pmode_{key}"
        mode = st.radio(
            f"##mode_{key}", ["Fixed", "Range"], horizontal=True,
            index=_widget_index(mode_key, 0),
            label_visibility="collapsed", key=mode_key,
        )
        enable_key = f"penable_{key}"
        enabled = st.checkbox(
            "Enable (blank = use solver default / None)",
            value=_widget_value(enable_key, False),
            key=enable_key,
        )
        if not enabled:
            _set_ss(f"param_{key}", {"mode": "fixed", "value": None})
            st.divider()
            return
        fmt = "%g"
        if mode == "Fixed":
            fixed_key = f"pfixed_{key}"
            val_ms = st.number_input(
                f"##fix_{key}", value=_widget_value(fixed_key, 2.0),
                format=fmt, label_visibility="collapsed", key=fixed_key,
                min_value=0.0,
            )
            _set_ss(f"param_{key}", {"mode": "fixed", "value": val_ms / 1e3})
        else:
            c1, c2, c3 = st.columns(3)
            min_key = f"pmin_{key}"
            max_key = f"pmax_{key}"
            step_key = f"pstep_{key}"
            min_v = c1.number_input("Min", value=_widget_value(min_key, 0.5), format=fmt, key=min_key, min_value=0.0)
            max_v = c2.number_input("Max", value=_widget_value(max_key, 5.0), format=fmt, key=max_key, min_value=0.0)
            step_v = c3.number_input("Step", value=_widget_value(step_key, 0.5), format=fmt, key=step_key, min_value=1e-30)
            vals = _arange_inclusive(min_v, max_v, step_v)
            st.caption(f"→ {_format_vals(vals)} ms")
            _set_ss(f"param_{key}", {"mode": "range", "values": (vals / 1e3).tolist()})
        st.divider()
        return

    # Numeric (float / int)
    mode_key = f"pmode_{key}"
    mode = st.radio(
        f"##mode_{key}", ["Fixed", "Range"], horizontal=True,
        index=_widget_index(mode_key, 0),
        label_visibility="collapsed", key=mode_key,
    )
    fmt = _num_format(default)
    if mode == "Fixed":
        fixed_key = f"pfixed_{key}"
        val = st.number_input(
            f"##fix_{key}", value=_widget_value(fixed_key, float(default)),
            format=fmt, label_visibility="collapsed", key=fixed_key,
        )
        _set_ss(f"param_{key}", {"mode": "fixed", "value": val})
    else:
        c1, c2, c3 = st.columns(3)
        min_key = f"pmin_{key}"
        max_key = f"pmax_{key}"
        step_key = f"pstep_{key}"
        min_v = c1.number_input(
            "Min", value=_widget_value(min_key, float(default)),
            format=fmt, key=min_key,
        )
        max_v = c2.number_input(
            "Max", value=_widget_value(max_key, float(default) * 2),
            format=fmt, key=max_key,
        )
        step_v = c3.number_input(
            "Step", value=_widget_value(step_key, abs(float(default)) or 1.0),
            format=fmt, key=step_key, min_value=1e-30,
        )
        vals = _arange_inclusive(min_v, max_v, step_v)
        if param_type == "int":
            vals = np.unique(vals.astype(int))
        st.caption(f"→ {_format_vals(vals)}")
        _set_ss(f"param_{key}", {"mode": "range", "values": vals.tolist()})
    st.divider()


def _render_flag_row(key: str, meta: dict) -> None:
    """Render a True / False / Both radio for one flag."""
    label = meta["label"]
    default = meta["default"]
    default_sel = "True" if default else "False"
    flag_key = f"flag_{key}"
    choice = st.radio(
        label, ["False", "True", "Both"],
        index=_widget_index(flag_key, ["False", "True", "Both"].index(default_sel)),
        horizontal=True, key=flag_key,
    )
    _set_ss(f"flagcfg_{key}", choice)


# ── Sweep config builder ──────────────────────────────────────────────────────

def _build_sweep_config():
    """
    Read session state widgets and build
    (param_ranges, flag_ranges, fixed_params, fixed_flags, param_transforms).

    param_transforms is a callable ``(params, flags) -> params`` applied by the
    sweep engine after building each run's full params dict.  In symmetric twin
    mode it splits S_gp equally between cathodes.
    """
    param_ranges = {}
    fixed_params = {}
    flag_ranges = {}
    fixed_flags = {}

    for key in PARAM_META:
        cfg = st.session_state.get(f"param_{key}")
        ptype = PARAM_META[key]["type"]
        if cfg is None:
            # Widget not yet rendered; use default
            fixed_params[key] = PARAM_META[key]["default"]
            continue
        if cfg["mode"] == "fixed":
            val = cfg["value"]
            fixed_params[key] = int(val) if ptype == "int" else val
        else:
            vals = cfg.get("values", [])
            if len(vals) == 1:
                fixed_params[key] = int(vals[0]) if ptype == "int" else vals[0]
            elif len(vals) > 1:
                param_ranges[key] = [int(v) for v in vals] if ptype == "int" else list(vals)
            else:
                fixed_params[key] = PARAM_META[key]["default"]

    # Dual Cathode
    dc_on_off = st.session_state.get("dc_on_off", "Off")
    dc_type = st.session_state.get("dc_type", "Twin (symmetric)")

    if dc_on_off == "Off":
        fixed_flags["TwinCathode"] = False
    elif dc_on_off == "On":
        fixed_flags["TwinCathode"] = True
    else:  # Both
        flag_ranges["TwinCathode"] = [True, False]

    # Whether twin symmetric splitting applies (captured for the transform closure)
    _is_symmetric = (dc_on_off != "Off") and (dc_type == "Twin (symmetric)")

    if dc_on_off in ("On", "Both"):
        if dc_type != "Twin (symmetric)":  # Asymmetric — independent twin controls
            for key in TWIN_META:
                cfg = st.session_state.get(f"param_{key}")
                if cfg is None:
                    fixed_params[key] = TWIN_META[key]["default"]
                elif cfg["mode"] == "fixed":
                    fixed_params[key] = cfg["value"]
                else:
                    vals = cfg.get("values", [])
                    if len(vals) == 1:
                        fixed_params[key] = vals[0]
                    elif len(vals) > 1:
                        param_ranges[key] = list(vals)
                    else:
                        fixed_params[key] = TWIN_META[key]["default"]

    # Regular flags
    for key in FLAG_META:
        choice = st.session_state.get(f"flagcfg_{key}", "False" if not FLAG_META[key]["default"] else "True")
        if choice == "True":
            fixed_flags[key] = True
        elif choice == "False":
            fixed_flags[key] = False
        else:  # Both
            flag_ranges[key] = [True, False]

    # Build param_transforms closure.
    # In symmetric twin mode, splits gas puff equally between cathodes.
    # Both cathodes share the same device hardware (V_bank, T_s, etc.) so no power
    # splitting is needed — the cathode solver self-consistently determines currents.
    def _param_transform(params, flags, _sym=_is_symmetric):
        # T_s is entered in °C; simulator expects K
        params["T_s"] = params.get("T_s", PARAM_META["T_s"]["default"]) + 273.15
        twin_active = flags.get("TwinCathode", False)
        if twin_active and _sym:
            params["S_gp"] = params.get("S_gp", PARAM_META["S_gp"]["default"]) / 2.0
            params["Twin_S_gp"] = params["S_gp"]
            params["S_gp_decay_target"] = (
                params.get(
                    "S_gp_decay_target",
                    PARAM_META["S_gp_decay_target"]["default"],
                )
                / 2.0
            )
            params["Twin_S_gp_decay_target"] = params["S_gp_decay_target"]
        return params

    return param_ranges, flag_ranges, fixed_params, fixed_flags, _param_transform


def _count_combos(param_ranges, flag_ranges):
    n = 1
    for vals in param_ranges.values():
        n *= len(vals)
    for vals in flag_ranges.values():
        n *= len(vals)
    return n


def _describe_sweep(param_ranges, flag_ranges):
    parts = []
    for k, vals in sorted(param_ranges.items()):
        parts.append(f"{k} ∈ {_format_vals(vals)}")
    for k, vals in sorted(flag_ranges.items()):
        parts.append(f"{k} ∈ {vals}")
    return ";  ".join(parts) if parts else "No varied parameters"


def _fmt_val(v) -> str:
    """Format a parameter value, using scientific notation for large/small floats."""
    if isinstance(v, str):
        return v
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    fv = float(v)
    av = abs(fv)
    if av == 0:
        return "0"
    if av >= 1e5 or (0 < av < 1e-2):
        return f"{fv:.3e}"
    return f"{fv:g}"


def _decode_param_value(v):
    """Return a display-friendly scalar from HDF5/string-like values."""
    if isinstance(v, bytes):
        return v.decode()
    return v


def _gas_puff_summary(params, flags=None):
    """Return compact display fields for the active gas-puff schedule."""
    flags = flags or {}
    mode = str(_decode_param_value(params.get("gas_puff_mode", "decay_after_breakdown")))
    twin = bool(flags.get("TwinCathode", False))
    s_gp = float(params.get("S_gp", 0.0) or 0.0)
    twin_s_gp = float(params.get("Twin_S_gp", 0.0) or 0.0) if twin else 0.0
    total = s_gp + twin_s_gp

    if mode == "pulse_decay_to_level":
        target = float(params.get("S_gp_decay_target", 0.0) or 0.0)
        twin_target = float(params.get("Twin_S_gp_decay_target", 0.0) or 0.0) if twin else 0.0
        target_total = target + twin_target
        hold_ms = float(params.get("tau_gp_pulse_duration", 0.0) or 0.0) * 1e3
        tau_ms = float(params.get("tau_gp_decay_duration", 0.0) or 0.0) * 1e3
        return {
            "mode": mode,
            "initial_total": total,
            "target_total": target_total,
            "hold_ms": hold_ms,
            "tau_ms": tau_ms,
            "summary": f"pulse {total:.0f}->{target_total:.0f}, hold={hold_ms:g}ms, tau={tau_ms:g}ms",
        }

    start = params.get("tau_gp_after_breakdown")
    factor = float(params.get("tau_gp_decay_factor", 1.0) or 1.0)
    try:
        start_ms = float(start) * 1e3
    except (TypeError, ValueError):
        start_ms = np.nan
    if start is None or np.isnan(start_ms):
        summary = f"decay-after-breakdown S_gp={total:.0f}, steady"
    else:
        summary = f"decay S_gp={total:.0f}, start={start_ms:g}ms, factor={factor:g}"
    return {
        "mode": mode,
        "initial_total": total,
        "target_total": None,
        "hold_ms": None,
        "tau_ms": None,
        "summary": summary,
    }


# ── Sweep thread ──────────────────────────────────────────────────────────────

def _drain_queue():
    """Pull all pending messages from the sweep queue into sweep_state."""
    q: queue.Queue = st.session_state.get("sweep_queue")
    state: SweepState = st.session_state.get("sweep_state")
    if q is None or state is None:
        return
    changed = False
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        changed = True
        if "done" in msg:
            state.running = False
            state.done = True
            state.active_runs.clear()
            state.total_time_s = msg.get("total_time_s", 0.0)
            st.session_state["sweep_running"] = False
            if state.db_path:
                _save_manifest(state.db_path, {
                    "running": False,
                    "db_path": state.db_path,
                    "total": state.total,
                    "completed": state.completed,
                    "failed": state.failed,
                    "planned_run_ids": state.planned_run_ids,
                    "log": state.log[-50:],
                    "failed_log": state.failed_log[-50:],
                })
        elif "error" in msg:
            state.error = msg["error"]
            state.running = False
            state.done = True
            state.active_runs.clear()
            st.session_state["sweep_running"] = False
            if state.db_path:
                _save_manifest(state.db_path, {
                    "running": False,
                    "db_path": state.db_path,
                    "total": state.total,
                    "completed": state.completed,
                    "failed": state.failed,
                    "planned_run_ids": state.planned_run_ids,
                    "error": state.error,
                    "log": state.log[-50:],
                    "failed_log": state.failed_log[-50:],
                })
        else:
            state.total = msg.get("total", state.total)
            run_id = msg.get("run_id", "")
            status = msg.get("status", "ok")
            stats = msg.get("stats", {})

            if run_id in state.finished_run_ids and status in {"starting", "progress"}:
                continue

            if status == "starting":
                start_time = stats.get("_start_time", time.time())
                v_bank = stats.get("_V_bank")
                t_s_k = stats.get("_T_s_K")
                s_gp = stats.get("_S_gp")
                cells = stats.get("_cells")
                twin = stats.get("_TwinCathode", False)
                twin_s_gp = stats.get("_Twin_S_gp")
                parts = []
                if v_bank is not None:
                    parts.append(f"V_bank={v_bank:.0f}V")
                if s_gp is not None:
                    parts.append(f"S_gp={s_gp:.0f}")
                if cells is not None:
                    parts.append(f"cells={cells}")
                if t_s_k is not None:
                    parts.append(f"T_s={t_s_k - 273.15:.0f}°C")
                if twin:
                    twin_s_str = f"/{twin_s_gp:.0f}" if twin_s_gp is not None else ""
                    parts.append(f"Twin=on{twin_s_str}")
                label = "  ".join(parts) or run_id
                state.active_runs[run_id] = {
                    "start_time": start_time,
                    "label": label,
                    "t_total_ms": stats.get("_t_total_ms", 25.0),
                    "frac": 0.0,
                    "phase_code": 0.0,
                    "seg_wall": 0.0,
                    "rate_ema": 0.0,
                }
            elif status == "progress":
                if run_id in state.active_runs:
                    info = state.active_runs[run_id]
                    info["frac"] = stats.get("frac", info["frac"])
                    info["phase_code"] = stats.get("phase_code", info["phase_code"])
                    info["seg_wall"] = stats.get("seg_wall", info["seg_wall"])
                    info["rate_ema"] = stats.get("rate_ema", info["rate_ema"])
            else:
                state.completed = msg.get("i", state.completed)
                state.finished_run_ids.add(run_id)
                state.active_runs.pop(run_id, None)
                if status == "failed":
                    state.failed += 1
                ne_var = stats.get("ne_var", float("nan"))
                run_time = stats.get("_run_time_s")
                if run_time is not None and status == "ok":
                    state.completed_run_times.append(run_time)
                equil_time = stats.get("_equil_time_s")
                equil_cache_hit = stats.get("_equil_cache_hit", False)
                equil_S_gp = stats.get("_equil_S_gp")
                equil_twin = stats.get("_equil_twin", False)
                equil_Twin_S_gp = stats.get("_equil_Twin_S_gp")
                do_equil = stats.get("_equilibrate_nn", False)
                v_bank = stats.get("_V_bank")
                t_s_k = stats.get("_T_s_K")
                twin_cathode = stats.get("_TwinCathode", False)

                line = f"[{state.completed}/{state.total}] {run_id} {status}"
                param_parts = []
                if v_bank is not None:
                    param_parts.append(f"V_bank={v_bank:.0f}V")
                if t_s_k is not None:
                    param_parts.append(f"T_s={t_s_k - 273.15:.0f}°C")
                param_parts.append(f"Twin={'on' if twin_cathode else 'off'}")
                line += "  " + "  ".join(param_parts)
                if not np.isnan(ne_var):
                    line += f"  ne_var={ne_var:.3e}"
                if run_time is not None:
                    line += f"  run={run_time:.1f}s"
                if do_equil and equil_S_gp is not None:
                    twin_str = f"/twin={equil_Twin_S_gp:.0f}" if equil_twin else ""
                    if equil_cache_hit:
                        line += f"  equil=cached(S_gp={equil_S_gp:.0f}{twin_str})"
                    else:
                        line += f"  equil={equil_time:.1f}s(fresh,S_gp={equil_S_gp:.0f}{twin_str})"
                if status == "failed":
                    error_msg = stats.get("_error", "")
                    if error_msg:
                        line += f"  → {error_msg}"
                if status == "failed":
                    state.failed_log.append(f"{run_id}  {stats.get('_error', '') or '(no error detail)'}")
                state.log.append(line)

    # Persist progress to manifest every time something changed
    if changed and state.running and state.db_path:
        _save_manifest(state.db_path, {
            "running": True,
            "db_path": state.db_path,
            "total": state.total,
            "completed": state.completed,
            "failed": state.failed,
            "planned_run_ids": state.planned_run_ids,
            "log": state.log[-50:],
            "failed_log": state.failed_log[-50:],
        })


def _start_sweep_thread(db_path, n_workers, t_window, param_ranges, flag_ranges,
                        fixed_params, fixed_flags, param_transforms=None, equilibrate_nn=False):
    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # Compute all planned run IDs (for reconnect / abort marking)
    all_combos = param_combinations(param_ranges, flag_ranges)
    n_total = len(all_combos)
    planned_ids = [f"run_{i:04d}" for i in range(n_total)]

    state = SweepState(
        total=n_total,
        db_path=db_path,
        planned_run_ids=planned_ids,
    )

    # Persist the initial manifest so reconnect works immediately
    _save_manifest(db_path, {
        "running": True,
        "db_path": db_path,
        "total": n_total,
        "completed": 0,
        "failed": 0,
        "planned_run_ids": planned_ids,
        "log": [],
        "failed_log": [],
    })

    def progress_cb(i, total, run_id, status, stats):
        q.put({"i": i, "total": total, "run_id": run_id, "status": status, "stats": stats})

    def target():
        t_start = time.time()
        try:
            grid_sweep_parallel(
                param_ranges=param_ranges,
                flag_ranges=flag_ranges,
                fixed_params=fixed_params,
                fixed_flags=fixed_flags,
                db_path=db_path,
                t_window=t_window,
                n_workers=n_workers,
                progress_callback=progress_cb,
                param_transforms=param_transforms,
                equilibrate_nn=equilibrate_nn,
                verbose=False,
                verbose_equil=False,
                stop_event=stop_event,
            )
        except Exception as exc:
            q.put({"error": str(exc)})
        finally:
            q.put({"done": True, "total_time_s": time.time() - t_start})

    thread = threading.Thread(target=target, daemon=True)
    st.session_state["sweep_thread"] = thread
    st.session_state["sweep_queue"] = q
    st.session_state["sweep_state"] = state
    st.session_state["sweep_running"] = True
    st.session_state["sweep_stop_event"] = stop_event
    st.session_state["_sweep_final_app_rerun_done"] = False
    thread.start()


# ── Index → DataFrame ─────────────────────────────────────────────────────────

def _index_to_df(idx):
    import pandas as pd

    rows = []
    n = len(idx["run_ids"])
    for i in range(n):
        row = {
            "run_id": idx["run_ids"][i],
            "status": idx["status"][i],
            "n_cells": int(idx["n_cells"][i]) if i < len(idx["n_cells"]) else None,
        }
        for k, arr in idx["params"].items():
            row[f"p:{k}"] = arr[i] if i < len(arr) else None
        for k, arr in idx["flags"].items():
            row[f"f:{k}"] = bool(arr[i]) if i < len(arr) else None
        for k, arr in idx["stats_10_20ms"].items():
            row[f"s:{k}"] = float(arr[i]) if i < len(arr) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if "p:T_s" in df.columns:
        df["p:T_s_C"] = df["p:T_s"] - 273.15
    return df


def _add_peak_stats_to_index_df(df, db_path):
    """Populate peak cathode metric columns from raw runs if the index lacks them."""
    peak_cols = {
        "P_peak": "s:P_peak",
        "I_tot_peak": "s:I_tot_peak",
        "V_b_at_I_tot_peak": "s:V_b_at_I_tot_peak",
    }
    if df.empty:
        return df
    missing_or_empty = [
        stat_key for stat_key, col in peak_cols.items()
        if col not in df.columns or df[col].isna().all()
    ]
    if not missing_or_empty:
        return df

    df = df.copy()
    with open_db(db_path) as db:
        for i, row in df.iterrows():
            if row.get("status") != "ok":
                continue
            run_id = row.get("run_id")
            try:
                _, _, results = load_run(db, run_id, keys=["cathode"])
                peak_stats = compute_cathode_peak_stats(results)
            except Exception:
                continue
            for stat_key, col in peak_cols.items():
                if stat_key in missing_or_empty:
                    df.loc[i, col] = peak_stats.get(stat_key, float("nan"))
    return df


def _format_metric_value(value, fmt):
    """Format a metric value, returning an em dash for missing/NaN values."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(value):
        return "—"
    return fmt.format(value)


# ── Tab renderers ─────────────────────────────────────────────────────────────

@st.fragment(run_every=1.0)
def _render_sweep_progress():
    """Render the live sweep status without repainting the whole app."""
    if "sweep_state" not in st.session_state:
        return

    _drain_queue()
    state: SweepState = st.session_state["sweep_state"]

    st.divider()

    _PHASE_NAMES = {0.0: "pre-breakdown", 1.0: "main discharge", 2.0: "afterglow"}
    now = time.time()
    if state.active_runs:
        st.caption(f"Active runs ({len(state.active_runs)})")
        for rid, info in state.active_runs.items():
            elapsed = now - info["start_time"]
            frac = info["frac"]
            phase_str = _PHASE_NAMES.get(info["phase_code"], "pre-breakdown")
            seg_wall = info["seg_wall"]
            rate_ema = info["rate_ema"]
            seg_str = f"  last 1ms: {seg_wall:.2f}s" if seg_wall > 0 else ""
            if rate_ema > 0 and frac > 0:
                remaining_ms = (1.0 - frac) * info["t_total_ms"]
                eta_s = remaining_ms * rate_ema
                m, s = divmod(int(eta_s), 60)
                eta_str = f"  ETA {m}m{s:02d}s" if m else f"  ETA {s}s"
            else:
                eta_str = ""
            text = f"{rid}  {info['label']}  [{phase_str}]  {elapsed:.0f}s elapsed  {frac*100:.0f}%{seg_str}{eta_str}"
            st.progress(frac, text=text)
            if seg_wall > 30:
                _db_dir = _log_dir(state.db_path) if state.db_path else pathlib.Path("~").expanduser()
                _db_stem = pathlib.Path(state.db_path).stem if state.db_path else "sweep"
                st.warning(
                    f"{rid}: very slow - {seg_wall:.0f}s per 1ms of sim time. "
                    f"Check worker logs for step size (h= lines): "
                    f"`{_db_dir}/{_db_stem}.*.worker_*.log`  "
                    "Possible causes: CFL-forced tiny steps, or resource contention."
                )

    progress_frac = state.completed / max(state.total, 1)
    st.progress(
        progress_frac,
        text=f"{state.completed}/{state.total} runs complete"
        + (f"  ({state.failed} failed)" if state.failed else ""),
    )

    proc_ram, cpu_pct, worker_count = _process_tree_metrics()
    sys_vm = psutil.virtual_memory()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("App + workers RAM", f"{proc_ram:.2f} GB")
    m2.metric("System available", f"{sys_vm.available / 1e9:.2f} GB")
    m3.metric("System RAM used", f"{sys_vm.percent:.0f}%")
    m4.metric("App + workers CPU", f"{cpu_pct:.0f}%", help=f"{worker_count} worker process(es)")

    if state.failed_log:
        failed_text = "\n".join(state.failed_log[-20:])
        st.text_area("Failed runs (last 20)", failed_text, height=120)

    log_text = "\n".join(state.log[-30:]) if state.log else "(waiting for first run...)"
    st.text_area("Completed runs (last 30)", log_text, height=200)

    if state.done:
        time_str = f"  Total: {state.total_time_s:.1f}s" if state.total_time_s > 0 else ""
        if state.error:
            st.error(f"Sweep failed: {state.error}")
        elif state.failed == 0:
            st.success(f"Sweep complete: {state.completed}/{state.total} runs succeeded.{time_str}")
        else:
            st.warning(
                f"Sweep done: {state.completed - state.failed} succeeded, "
                f"{state.failed} failed.{time_str}"
            )
        if not st.session_state.get("_sweep_final_app_rerun_done", False):
            st.session_state["_sweep_final_app_rerun_done"] = True
            st.rerun()


def _render_configure_tab():
    st.header("Configure Parameter Sweep")

    # Config toolbar
    c_save, c_load, c_defaults, _ = st.columns([1, 1, 1, 5])
    if c_save.button("💾 Save Config", help=f"Save current configuration to {_CONFIG_PATH}"):
        _save_config()
        st.toast(f"Config saved to {_CONFIG_PATH}")
    if c_load.button("📂 Load Config", help=f"Reload configuration from {_CONFIG_PATH}"):
        if _CONFIG_PATH.exists():
            if _load_config():
                st.rerun()
        else:
            st.warning(f"No saved config found at {_CONFIG_PATH}")
    if c_defaults.button("🔄 Load Defaults", help="Reset all parameters to built-in defaults"):
        _load_defaults()
        st.rerun()

    col_params, col_flags = st.columns([3, 2])

    with col_params:
        for group in PARAM_GROUP_ORDER:
            expanded = group in ("Discharge (Primary Cathode)", "Gas & Initial Conditions")
            with st.expander(group, expanded=expanded):
                for key, meta in PARAM_META.items():
                    if meta["group"] != group:
                        continue
                    if not _is_gas_puff_param_visible(key):
                        continue
                    if key in _ADAPTIVE_MESH_PARAMS and st.session_state.get("flagcfg_adaptive_mesh", "False") == "False":
                        continue
                    _render_param_row(key, meta)

        # Dual Cathode section
        with st.expander("Dual Cathode", expanded=False):
            dc_on_off = st.radio(
                "Dual Cathode",
                ["Off", "On", "Both"],
                horizontal=True,
                index=_widget_index("dc_on_off", 0),
                key="dc_on_off",
                help=(
                    "**Off**: single cathode only.  "
                    "**On**: second cathode active.  "
                    "**Both**: sweep over single and dual cathode configurations."
                ),
            )
            if dc_on_off in ("On", "Both"):
                dc_type = st.radio(
                    "Second cathode type",
                    ["Twin (symmetric)", "Asymmetric"],
                    horizontal=True,
                    index=_widget_index("dc_type", 0),
                    key="dc_type",
                    help=(
                        "**Twin**: S_gp is split equally between cathodes; "
                        "both share V_bank and hardware parameters.  "
                        "**Asymmetric**: second cathode has an independent gas puff rate."
                    ),
                )
                if dc_type == "Twin (symmetric)":
                    # Show live splitting values for S_gp
                    s_gp_mode = st.session_state.get("pmode_S_gp", "Fixed")
                    s_gp = float(st.session_state.get("pfixed_S_gp", PARAM_META["S_gp"]["default"]))

                    if s_gp_mode == "Range":
                        sg_min = st.session_state.get("pmin_S_gp", s_gp)
                        sg_max = st.session_state.get("pmax_S_gp", s_gp)
                        info_line = (
                            f"**S_gp** = **Twin_S_gp** = S_gp/2"
                            f"  *(range {sg_min/2:g} → {sg_max/2:g} per cathode)*"
                        )
                    else:
                        info_line = (
                            f"**S_gp** = **Twin_S_gp** = {_fmt_val(s_gp/2)}"
                            f"  *(= {_fmt_val(s_gp)}/2 per cathode)*"
                        )
                    st.info(
                        "Twin mode — gas puff split equally between cathodes; "
                        "both cathodes share V_bank and hardware parameters:\n"
                        f"- {info_line}"
                    )
                else:
                    st.markdown("**Second cathode parameters:**")
                    for key, meta in TWIN_META.items():
                        if not _is_gas_puff_param_visible(key):
                            continue
                        _render_param_row(key, meta)

    with col_flags:
        for group in FLAG_GROUP_ORDER:
            with st.expander(group + " Flags", expanded=True):
                for key, meta in FLAG_META.items():
                    if meta["group"] == group:
                        _render_flag_row(key, meta)

    # Run count + parameter summary
    param_ranges, flag_ranges, fixed_params, fixed_flags, _transforms = _build_sweep_config()
    n_combos = _count_combos(param_ranges, flag_ranges)
    st.divider()
    col1, col2 = st.columns([1, 3])
    col1.metric("Total runs", n_combos)
    if n_combos > 0:
        col2.markdown(_describe_sweep(param_ranges, flag_ranges))

    with st.expander("Fixed Parameter Summary", expanded=True):
        import pandas as pd

        col_p, col_f = st.columns([3, 2])

        with col_p:
            st.markdown("**Fixed Parameters**")
            rows = []
            for group in PARAM_GROUP_ORDER:
                for k, v in fixed_params.items():
                    if k in PARAM_META and PARAM_META[k]["group"] == group:
                        if not _is_gas_puff_param_visible(k):
                            continue
                        meta = PARAM_META[k]
                        rows.append({
                            "Group": group,
                            "Parameter": meta["label"],
                            "Value": _fmt_val(v),
                            "Unit": meta.get("unit", ""),
                        })
            for k, v in fixed_params.items():
                if k in TWIN_META:
                    if not _is_gas_puff_param_visible(k):
                        continue
                    meta = TWIN_META[k]
                    rows.append({
                        "Group": "Dual Cathode",
                        "Parameter": meta["label"],
                        "Value": _fmt_val(v),
                        "Unit": meta.get("unit", ""),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with col_f:
            st.markdown("**Fixed Flags**")
            flag_rows = []
            for k, v in sorted(fixed_flags.items()):
                label = FLAG_META[k]["label"] if k in FLAG_META else k
                flag_rows.append({"Flag": label, "Value": str(v)})
            if flag_rows:
                st.dataframe(pd.DataFrame(flag_rows), width="stretch", hide_index=True)


def _render_run_tab():
    st.header("Run Parameter Sweep")

    db_path = _render_path_input("run_db", "Database path", str(_DEFAULT_DB_PATH))
    db_path_exp = os.path.expanduser(db_path)

    worker_cpu_info = get_worker_affinity_info()
    worker_cpu_count = max(int(worker_cpu_info.get("count") or 1), 1)
    logical_cpu_count = max(int(worker_cpu_info.get("logical_count") or worker_cpu_count), 1)
    max_workers = worker_cpu_count if worker_cpu_info.get("limited") else logical_cpu_count
    max_workers = max(max_workers, 1)
    current_workers = int(st.session_state.get("run_n_workers", 1))
    if current_workers > max_workers:
        st.session_state["run_n_workers"] = max_workers

    col1, col2, _col3 = st.columns(3)
    n_workers = col1.number_input("Workers", min_value=1, max_value=max_workers,
                                  value=1, key="run_n_workers")
    with col2:
        t_start = st.number_input("t_window start (ms, 0=breakdown)", min_value=0.0, value=10.0,
                                  key="run_t_start")
        t_end = st.number_input("t_window end (ms, 0=breakdown)", min_value=0.0, value=20.0,
                                key="run_t_end")

    if worker_cpu_info.get("limited"):
        cpu_ids = worker_cpu_info.get("cpus") or []
        if worker_cpu_info.get("source") == "BAPSF_WORKER_CPUS":
            label = "Worker CPU override"
        else:
            label = "Detected P-core worker pool"
        st.caption(
            f"{label}: {worker_cpu_count} logical CPU(s) "
            f"out of {logical_cpu_count}. Worker launch is capped to this count. "
            f"CPU IDs: {cpu_ids}"
        )
    else:
        note = (
            "P-core detection unavailable; workers are not affinity-limited "
            f"(logical CPUs: {logical_cpu_count})."
        )
        if worker_cpu_info.get("error"):
            note += f" Detection note: {worker_cpu_info['error']}."
        st.caption(note)

    # ── Reconnect / interrupt banner ──────────────────────────────────────────
    already_running = st.session_state.get("sweep_running", False)
    if not already_running:
        manifest = _load_manifest(db_path_exp)
        if manifest and manifest.get("running"):
            with st.container(border=True):
                st.warning(
                    f"⚠ A previous sweep was interrupted.  "
                    f"{manifest.get('completed', '?')}/{manifest.get('total', '?')} run(s) completed, "
                    f"{manifest.get('failed', 0)} failed."
                )
                c1, c2, c3 = st.columns(3)
                if c1.button("▶ Resume Sweep", help="Re-launch sweep — completed runs are automatically skipped"):
                    # Dismiss the banner then fall through to the normal launch flow below
                    manifest["running"] = False
                    _save_manifest(db_path_exp, manifest)
                    st.session_state["_resume_sweep"] = True
                    st.rerun()
                if c2.button("❌ Mark incomplete as failed",
                             help="Write 'failed' records for planned runs not yet in the database"):
                    _mark_incomplete_as_failed(db_path_exp, manifest)
                    manifest["running"] = False
                    _save_manifest(db_path_exp, manifest)
                if c3.button("✕ Dismiss", help="Hide this banner without taking action"):
                    manifest["running"] = False
                    _save_manifest(db_path_exp, manifest)
                    st.rerun()
                # Show last log lines
                last_log = manifest.get("log", [])[-5:]
                if last_log:
                    st.caption("Last log entries: " + " | ".join(last_log))
                last_failed = manifest.get("failed_log", [])[-5:]
                if last_failed:
                    st.caption("Failed runs: " + " | ".join(last_failed))

    if n_workers > 1:
        st.info(
            f"Parallel mode: {n_workers} workers.  "
            "Simulations run in separate processes; HDF5 writes are serialised on the main thread. "
            "On Windows hybrid CPUs, workers are pinned to the detected P-core CPU set."
        )

    equilibrate_nn = st.checkbox(
        "Auto-equilibrate nn0",
        value=False,
        key="run_equilibrate_nn",
        help=(
            "Before each plasma-on run, automatically determine the equilibrium background "
            "neutral density by running 100 plasma-off cycles (tau_cycle = 3 s each, "
            "S_gp active for tau_discharge = 20 ms per cycle, h_max = 10 ms) "
            "starting from nn0 = 1×10⁸ cm⁻³.  "
            "nn0 is set to the equilibrated interior-cell mean."
        ),
    )

    param_ranges, flag_ranges, fixed_params, fixed_flags, param_transforms = _build_sweep_config()
    n_combos = _count_combos(param_ranges, flag_ranges)
    st.markdown(f"**Ready to run {n_combos} combination(s).**  {_describe_sweep(param_ranges, flag_ranges)}")

    # ── Launch / Abort buttons ─────────────────────────────────────────────────
    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
    launch = btn_col1.button("Launch Sweep", type="primary", disabled=already_running)
    abort_clicked = btn_col2.button(
        "⏹ Abort Sweep",
        type="secondary",
        disabled=not already_running,
        help="Signal the sweep to stop after finishing the current runs. Already-running worker processes will complete naturally.",
    )

    if abort_clicked and already_running:
        stop_ev = st.session_state.get("sweep_stop_event")
        if stop_ev is not None:
            stop_ev.set()
        st.toast("Abort signal sent — sweep will stop after current runs complete.")

    if st.session_state.pop("_resume_sweep", False):
        # User clicked "Resume Sweep" in the reconnect banner — treat like a normal launch
        launch = True

    if launch and not already_running:
        if n_combos == 0:
            st.warning("No parameter combinations — adjust configuration on the Configure tab.")
        else:
            # Clear previous sweep state so the "Sweep complete" banner resets
            st.session_state.pop("sweep_state", None)
            st.session_state.pop("sweep_queue", None)
            st.session_state.pop("sweep_stop_event", None)
            _start_sweep_thread(
                db_path=db_path_exp,
                n_workers=int(n_workers),
                t_window=(float(t_start), float(t_end)),
                param_ranges=param_ranges,
                flag_ranges=flag_ranges,
                fixed_params=fixed_params,
                fixed_flags=fixed_flags,
                param_transforms=param_transforms,
                equilibrate_nn=equilibrate_nn,
            )
            st.rerun()

    # ── Tab-close warning (beforeunload) when sweep is active ─────────────────
    if already_running:
        st.markdown(
            """
            <script>
            (function () {
                window._lapd_sweep_running = true;
                function _lapd_warn(e) {
                    if (window._lapd_sweep_running) {
                        e.preventDefault();
                        e.returnValue = '';
                        return '';
                    }
                }
                if (!window._lapd_warn_attached) {
                    window.addEventListener('beforeunload', _lapd_warn);
                    window._lapd_warn_attached = true;
                }
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Clear the flag so closing the tab no longer triggers the dialog
        st.markdown(
            "<script>window._lapd_sweep_running = false;</script>",
            unsafe_allow_html=True,
        )

    # ── Progress display ───────────────────────────────────────────────────────
    _render_sweep_progress()
    if False and "sweep_state" in st.session_state:
        _drain_queue()
        state: SweepState = st.session_state["sweep_state"]

        st.divider()

        # ── Active runs widget ─────────────────────────────────────────────────
        _PHASE_NAMES = {0.0: "pre-breakdown", 1.0: "main discharge", 2.0: "afterglow"}
        now = time.time()
        if state.active_runs:
            st.caption(f"Active runs ({len(state.active_runs)})")
            for rid, info in state.active_runs.items():
                elapsed = now - info["start_time"]
                frac = info["frac"]
                phase_str = _PHASE_NAMES.get(info["phase_code"], "pre-breakdown")
                seg_wall = info["seg_wall"]
                rate_ema = info["rate_ema"]
                seg_str = f"  last 1ms: {seg_wall:.2f}s" if seg_wall > 0 else ""
                if rate_ema > 0 and frac > 0:
                    remaining_ms = (1.0 - frac) * info["t_total_ms"]
                    eta_s = remaining_ms * rate_ema
                    m, s = divmod(int(eta_s), 60)
                    eta_str = f"  ETA {m}m{s:02d}s" if m else f"  ETA {s}s"
                else:
                    eta_str = ""
                text = f"{rid}  {info['label']}  [{phase_str}]  {elapsed:.0f}s elapsed  {frac*100:.0f}%{seg_str}{eta_str}"
                st.progress(frac, text=text)
                if seg_wall > 30:
                    import pathlib as _pl
                    _db_dir = _log_dir(state.db_path) if state.db_path else _pl.Path("~").expanduser()
                    _db_stem = _pl.Path(state.db_path).stem if state.db_path else "sweep"
                    st.warning(
                        f"{rid}: very slow — {seg_wall:.0f}s per 1ms of sim time. "
                        f"Check worker logs for step size (h= lines): "
                        f"`{_db_dir}/{_db_stem}.*.worker_*.log`  "
                        "Possible causes: CFL-forced tiny steps, or resource contention."
                    )

        progress_frac = state.completed / max(state.total, 1)
        st.progress(
            progress_frac,
            text=f"{state.completed}/{state.total} runs complete"
            + (f"  ({state.failed} failed)" if state.failed else ""),
        )

        # Memory / CPU monitor
        proc = psutil.Process()
        proc_ram = proc.memory_info().rss / 1e9
        sys_vm = psutil.virtual_memory()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Process RAM", f"{proc_ram:.2f} GB")
        m2.metric("System available", f"{sys_vm.available / 1e9:.2f} GB")
        m3.metric("System RAM used", f"{sys_vm.percent:.0f}%")
        m4.metric("CPU", f"{psutil.cpu_percent(interval=None):.0f}%")

        # Log — completions only
        log_text = "\n".join(state.log[-30:]) if state.log else "(waiting for first run…)"
        st.text_area("Completed runs (last 30)", log_text, height=200)

        if state.running:
            time.sleep(0.5)
            st.rerun()
        elif state.done:
            time_str = f"  Total: {state.total_time_s:.1f}s" if state.total_time_s > 0 else ""
            if state.error:
                st.error(f"Sweep failed: {state.error}")
            elif state.failed == 0:
                st.success(f"Sweep complete: {state.completed}/{state.total} runs succeeded.{time_str}")
            else:
                st.warning(
                    f"Sweep done: {state.completed - state.failed} succeeded, "
                    f"{state.failed} failed.{time_str}"
                )


def _render_explore_tab():
    st.header("Explore Database")
    db_path = _render_path_input("explore_db", "Database path", str(_DEFAULT_DB_PATH))
    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        st.warning(f"File not found: `{db_path}`")
        return

    col_rebuild, col_info = st.columns([1, 4])
    if col_rebuild.button("Rebuild Index", help="Recompute the index from raw run data. Use this after partial failures or if variance plots show mismatched array sizes."):
        try:
            with open_db(db_path, mode="r+") as db:
                rebuild_index(db)
            col_info.success("Index rebuilt successfully.")
        except Exception as exc:
            col_info.error(f"Rebuild failed: {exc}")

    with open_db(db_path) as db:
        idx = load_index(db)
        all_run_ids = list_runs(db)

    if not all_run_ids:
        st.warning("Database contains no runs.")
        return

    ok_runs = [r for r, s in zip(idx["run_ids"], idx["status"]) if s == "ok"]

    # Build rich display labels for run-selection dropdowns
    def _run_display_labels(idx):
        labels = {}
        p = idx["params"]
        f = idx["flags"]
        for i, run_id in enumerate(idx["run_ids"]):
            def _p(key, default=0.0, _i=i):
                arr = p.get(key)
                if arr is None or _i >= len(arr):
                    return default
                v = arr[_i]
                try:
                    return default if np.isnan(float(v)) else v
                except (TypeError, ValueError):
                    return v if v else default
            def _f(key, default=False, _i=i):
                arr = f.get(key)
                return bool(arr[_i]) if arr is not None and _i < len(arr) else default

            twin = _f("TwinCathode")
            V_bank = _p("V_bank")
            T_s_k = _p("T_s", default=float("nan"))
            S_gp = _p("S_gp")
            Twin_S_gp = _p("Twin_S_gp") if twin else 0.0
            gas = _p("gas_type", "?")
            gas = _decode_param_value(gas)
            run_params = {
                "gas_puff_mode": _p("gas_puff_mode", "decay_after_breakdown"),
                "S_gp": S_gp,
                "Twin_S_gp": Twin_S_gp,
                "S_gp_decay_target": _p("S_gp_decay_target"),
                "Twin_S_gp_decay_target": _p("Twin_S_gp_decay_target") if twin else 0.0,
                "tau_gp_after_breakdown": _p("tau_gp_after_breakdown", None),
                "tau_gp_decay_factor": _p("tau_gp_decay_factor", 1.0),
                "tau_gp_pulse_duration": _p("tau_gp_pulse_duration"),
                "tau_gp_decay_duration": _p("tau_gp_decay_duration", 1e-3),
            }
            gp_summary = _gas_puff_summary(run_params, {"TwinCathode": twin})["summary"]
            twin_str = "twin" if twin else "single"
            t_s_str = f"  T_s={T_s_k - 273.15:.0f}°C" if not np.isnan(T_s_k) else ""
            labels[run_id] = (
                f"{run_id}  |  {gas}  V_bank={V_bank:.0f}V{t_s_str}  "
                f"{gp_summary}  [{twin_str}]"
            )
        return labels
    run_labels = _run_display_labels(idx)

    sub_tabs = st.tabs(["📋 Table", "📊 Variance", "🔬 Inspector", "⚖️ Comparison"])

    # ── Table ─────────────────────────────────────────────────────────────────
    with sub_tabs[0]:
        st.subheader("Run Index")
        df = _index_to_df(idx)
        df = _add_peak_stats_to_index_df(df, db_path)
        col_conf = {
            col: st.column_config.NumberColumn(format="%.3e")
            for col in df.columns
            if col.startswith(("p:", "s:"))
        }
        col_conf["p:T_s_C"] = st.column_config.NumberColumn("T_s [°C]", format="%.0f")
        col_conf["s:P_peak"] = st.column_config.NumberColumn("Peak power [W]", format="%.3e")
        col_conf["s:I_tot_peak"] = st.column_config.NumberColumn("Peak I_tot [A]", format="%.3e")
        col_conf["s:V_b_at_I_tot_peak"] = st.column_config.NumberColumn(
            "V_b at peak I_tot [V]", format="%.3e"
        )
        # Default visible columns; all columns still available in CSV export
        _DEFAULT_PARAMS = [
            "V_bank", "T_s_C", "gas_type", "nn0",
            "gas_puff_mode", "S_gp", "Twin_S_gp",
            "S_gp_decay_target", "Twin_S_gp_decay_target",
            "tau_gp_pulse_duration", "tau_gp_decay_duration",
            "tau_gp_after_breakdown", "tau_gp_decay_factor",
        ]
        default_cols = ["run_id", "status", "n_cells"]
        default_cols += [f"p:{k}" for k in _DEFAULT_PARAMS if f"p:{k}" in df.columns]
        default_cols += [c for c in df.columns if c.startswith("f:")]
        default_cols += [
            c for c in ("s:P_peak", "s:I_tot_peak", "s:V_b_at_I_tot_peak")
            if c in df.columns
        ]
        st.dataframe(df, width="stretch", height=500, column_config=col_conf,
                     column_order=default_cols)
        csv = df.to_csv(index=False).encode()
        st.download_button("Export CSV", csv, "run_index.csv", mime="text/csv")

    # ── Variance analysis ─────────────────────────────────────────────────────
    with sub_tabs[1]:
        st.subheader("Variance / Uniformity Analysis")

        if not ok_runs:
            st.warning("No successful runs in database.")
        else:
            ok_mask = np.array(idx["status"]) == "ok"
            param_keys = list(idx["params"].keys())
            flag_keys = list(idx["flags"].keys())

            # ── Filter controls (gas_type and TwinCathode) ────────────────────
            # These are used only as filters, not as x/hue axis options.
            var_filter_mask = ok_mask.copy()

            _gas_vals_all = np.asarray(idx["params"].get("gas_type", []))[ok_mask]
            _unique_gases = sorted(set(str(v) for v in _gas_vals_all)) if len(_gas_vals_all) else []
            _tc_vals_all = np.asarray(idx["flags"].get("TwinCathode", []))[ok_mask]
            _unique_tc = sorted(set(bool(v) for v in _tc_vals_all)) if len(_tc_vals_all) else []

            filt_cols = st.columns(2)
            if len(_unique_gases) > 1:
                sel_gas = filt_cols[0].multiselect(
                    "Filter gas type", _unique_gases, default=_unique_gases, key="var_gas_filter"
                )
                gas_arr = np.asarray(idx["params"].get("gas_type", []))
                if len(gas_arr) == len(ok_mask):
                    var_filter_mask &= np.array([str(v) in sel_gas for v in gas_arr])
            if len(_unique_tc) > 1:
                tc_labels = {True: "Twin", False: "Single"}
                sel_tc_label = filt_cols[1].radio(
                    "Filter cathode mode", ["All", "Single", "Twin"],
                    horizontal=True, key="var_tc_filter"
                )
                tc_arr = np.asarray(idx["flags"].get("TwinCathode", []))
                if len(tc_arr) == len(ok_mask) and sel_tc_label != "All":
                    sel_tc_bool = sel_tc_label == "Twin"
                    var_filter_mask &= np.array([bool(v) == sel_tc_bool for v in tc_arr])

            # Only keep keys where the value actually varies across filtered ok runs
            def _is_varied(arr, mask=var_filter_mask):
                vals = np.asarray(arr)[mask]
                if vals.dtype.kind in ("O", "U", "S"):
                    return len(set(str(v) for v in vals)) > 1
                try:
                    fv = vals.astype(float)
                    return len(np.unique(fv[~np.isnan(fv)])) > 1 if vals.size else False
                except (ValueError, TypeError):
                    return False

            # Exclude gas_type and TwinCathode — they are filter-only
            _filter_only = {"gas_type", "TwinCathode"}
            varied_param_keys = [
                k for k in param_keys
                if k not in _filter_only and _is_varied(idx["params"][k])
            ]
            varied_flag_keys = [
                k for k in flag_keys
                if k not in _filter_only and _is_varied(idx["flags"][k])
            ]
            all_keys = varied_param_keys + varied_flag_keys

            # Build filtered index for plot functions
            def _apply_mask(idx_src, mask):
                return {
                    "run_ids": [r for r, m in zip(idx_src["run_ids"], mask) if m],
                    "status": [s for s, m in zip(idx_src["status"], mask) if m],
                    "params": {k: np.asarray(v)[mask] for k, v in idx_src["params"].items()},
                    "flags":  {k: np.asarray(v)[mask] for k, v in idx_src["flags"].items()},
                    "stats_10_20ms": {
                        k: np.asarray(v)[mask] for k, v in idx_src["stats_10_20ms"].items()
                    },
                }

            filtered_idx = _apply_mask(idx, var_filter_mask)

            if not all_keys:
                st.info("No numeric varied parameters in the current selection.")
            else:
                _metric_opts = {
                    "Spatial variance": "var",
                    "Temporal variance": "tvar",
                    "Total variance": "total_var",
                    "Spatial CoV": "cov",
                    "Temporal CoV": "tcov",
                    "Mean": "mean",
                }
                c1, c2, c3, c4, c5 = st.columns(5)
                x_param = c1.selectbox("X axis", all_keys, key="var_x")
                hue_opts = ["None"] + all_keys
                hue_param = c2.selectbox("Color by", hue_opts, key="var_hue")
                quantity = c3.radio("Quantity", ["ne", "Te"], key="var_qty", horizontal=True)
                metric_label = c4.radio(
                    "Metric", list(_metric_opts.keys()), key="var_metric", horizontal=False
                )
                metric_key = _metric_opts[metric_label]
                plot_type = c5.radio("Plot type", ["Scatter", "Heatmap"], key="var_type", horizontal=True)

                try:
                    if plot_type == "Scatter":
                        fig = plot_sweep_variance(
                            filtered_idx, x_param,
                            hue_param=None if hue_param == "None" else hue_param,
                            quantity=quantity,
                            metric=metric_key,
                        )
                        st.pyplot(fig, width="stretch")
                        plt.close(fig)
                    else:
                        y_param_opts = [k for k in all_keys if k != x_param]
                        if not y_param_opts:
                            st.info("Need at least 2 varied parameters for a heatmap.")
                        else:
                            y_param = st.selectbox("Y axis (heatmap)", y_param_opts, key="var_y")
                            qty_key = f"{quantity}_{metric_key}"
                            fig = plot_sweep_heatmap(filtered_idx, x_param, y_param, quantity=qty_key)
                            st.pyplot(fig, width="stretch")
                            plt.close(fig)
                except Exception as exc:
                    st.error(f"Plot error: {exc}")

    # ── Inspector ─────────────────────────────────────────────────────────────
    with sub_tabs[2]:
        st.subheader("Single Run Inspector")

        if not ok_runs:
            st.warning("No successful runs to inspect.")
        else:
            col1, col2 = st.columns([2, 1])
            run_id = col1.selectbox("Select run", ok_runs, key="inspect_run",
                                    format_func=lambda r: run_labels.get(r, r))
            z_conv = col2.radio("Z convention", ["sim", "exp"], horizontal=True, key="inspect_z")

            try:
                with open_db(db_path) as db:
                    params, flags, results = load_run(db, run_id)

                gp = _gas_puff_summary(params, flags)
                st.markdown("**Gas puff schedule**")
                st.dataframe(
                    [
                        {
                            "Mode": gp["mode"],
                            "Initial total": _fmt_val(gp["initial_total"]),
                            "Target total": "" if gp["target_total"] is None else _fmt_val(gp["target_total"]),
                            "Hold [ms]": "" if gp["hold_ms"] is None else _fmt_val(gp["hold_ms"]),
                            "Decay tau [ms]": "" if gp["tau_ms"] is None else _fmt_val(gp["tau_ms"]),
                            "Summary": gp["summary"],
                        }
                    ],
                    width="stretch",
                    hide_index=True,
                )

                figs = plot_run(results, params, flags, z_convention=z_conv)
                fig_name = st.selectbox("Figure", list(figs.keys()), key="inspect_fig")

                st.pyplot(figs[fig_name], width="stretch")
                for f in figs.values():
                    plt.close(f)

                peak_stats = compute_cathode_peak_stats(results)
                st.markdown("**Cathode peak metrics**")
                st.dataframe(
                    [
                        {
                            "Peak power [W]": _format_metric_value(peak_stats.get("P_peak"), "{:.3e}"),
                            "Peak I_tot [A]": _format_metric_value(peak_stats.get("I_tot_peak"), "{:.3e}"),
                            "V_b at peak I_tot [V]": _format_metric_value(
                                peak_stats.get("V_b_at_I_tot_peak"), "{:.3e}"
                            ),
                        }
                    ],
                    width="stretch",
                    hide_index=True,
                )

                st.divider()
                st.markdown("**Time slice**")
                _face_keys = {"Ne_face_flux", "Nn_face_flux", "e_par_face_flux", "i_par_face_flux"}
                _slice_quantities = [
                    q for q in _SLICE_META_LABELS
                    if q not in _face_keys or q in results
                ]
                ts_col1, ts_col2 = st.columns([1, 2])
                _t_arr = results["time"]
                _t_min, _t_max = float(_t_arr[0]), float(_t_arr[-1])
                _t_mid = float(_t_arr[len(_t_arr) // 2])
                slice_qty = ts_col1.selectbox(
                    "Quantity",
                    _slice_quantities,
                    key="slice_qty",
                    format_func=lambda q: _SLICE_META_LABELS[q],
                )
                slice_t = ts_col2.slider(
                    "Time (ms)",
                    min_value=_t_min,
                    max_value=_t_max,
                    value=st.session_state.get("slice_t", _t_mid),
                    step=(_t_max - _t_min) / max(len(_t_arr) - 1, 1),
                    key="slice_t",
                )
                fig_slice = plot_time_slice(results, params, z_conv, slice_t, slice_qty)
                st.pyplot(fig_slice, use_container_width=True)
                plt.close(fig_slice)

                with st.expander("Run parameters"):
                    col_p, col_f = st.columns(2)
                    with col_p:
                        st.markdown("**Parameters**")
                        for k, v in sorted(params.items()):
                            st.write(f"- `{k}` = `{_fmt_val(v)}`")
                    with col_f:
                        st.markdown("**Flags**")
                        for k, v in sorted(flags.items()):
                            st.write(f"- `{k}` = `{v}`")

            except Exception as exc:
                st.error(f"Error loading run: {exc}")

    # ── Comparison ────────────────────────────────────────────────────────────
    with sub_tabs[3]:
        st.subheader("Side-by-Side Run Comparison")

        if len(ok_runs) < 2:
            st.info("Need at least 2 successful runs in the database to compare.")
        else:
            col1, col2, col3 = st.columns(3)
            selected = col1.multiselect(
                "Select runs (2–4)", ok_runs,
                default=ok_runs[:min(2, len(ok_runs))],
                max_selections=4,
                key="compare_runs",
                format_func=lambda r: run_labels.get(r, r),
            )
            _COMPARE_QUANTITIES = ["ne", "nn", "Te", "Ti", "v_plasma", "isat", "ln_lambda",
                                   "primary_mfp", "bulk_mfp"]
            quantity = col2.selectbox("Quantity", _COMPARE_QUANTITIES, key="compare_qty")

            # Derive cell count limit from selected runs
            run_id_to_n_cells = dict(zip(idx["run_ids"], idx["n_cells"].tolist()))
            max_cells = max((int(run_id_to_n_cells.get(r, 1)) for r in selected), default=1) if selected else 1
            cell_idx = col3.number_input(
                "Cell index (−1 = all cells)", min_value=-1, max_value=max_cells - 1, value=-1,
                key="compare_cell",
            )

            if len(selected) >= 2:
                try:
                    fig = plot_run_comparison(db_path, selected, quantity, int(cell_idx))
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Error generating comparison: {exc}")
            else:
                st.info("Select at least 2 runs above.")


# ── App entry ─────────────────────────────────────────────────────────────────

def main():
    try:
        import setproctitle
        setproctitle.setproctitle("lapd-app")
    except ImportError:
        pass

    st.set_page_config(
        page_title="LAPDSim Explorer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Auto-load saved config once per session (before any widgets are rendered)
    if not st.session_state.get("_config_auto_loaded"):
        st.session_state["_config_auto_loaded"] = True
        if _CONFIG_PATH.exists():
            _load_config()

    st.title("⚡ LAPDSim Parameter Explorer")
    st.caption(
        "Configure a parameter sweep, launch it (optionally in parallel), "
        "then explore the results database."
    )

    tabs = st.tabs(["⚙️ Configure", "▶ Run", "🔍 Explore"])
    with tabs[0]:
        _render_configure_tab()
    with tabs[1]:
        _render_run_tab()
    with tabs[2]:
        _render_explore_tab()


if __name__ == "__main__":
    main()
