"""
Per-run and cross-run statistics for LAPDSim results.
"""
import numpy as np


def compute_cathode_peak_stats(results):
    """
    Compute peak cathode power from the primary cathode time series.

    Peak power is defined from the peak primary-cathode ``I_tot`` and the
    ``V_b`` sample at the same timestep.
    """
    _get = (lambda k: results[k]) if isinstance(results, dict) else (lambda k: getattr(results, k))

    stats = {
        "P_peak": float("nan"),
        "I_tot_peak": float("nan"),
        "V_b_at_I_tot_peak": float("nan"),
    }
    try:
        cathode_obj = _get("cathode")
        _cget = (
            (lambda k: cathode_obj[k])
            if isinstance(cathode_obj, dict)
            else (lambda k: getattr(cathode_obj, k))
        )
        i_tot = np.asarray(_cget("I_tot"), dtype=float)
        v_b = np.asarray(_cget("V_b"), dtype=float)
        if i_tot.size == 0 or v_b.size == 0:
            return stats

        n = min(i_tot.size, v_b.size)
        i_tot = i_tot[:n]
        v_b = v_b[:n]
        valid = np.isfinite(i_tot) & np.isfinite(v_b)
        if not np.any(valid):
            return stats

        valid_indices = np.flatnonzero(valid)
        peak_idx = valid_indices[np.argmax(i_tot[valid])]
        peak_i = float(i_tot[peak_idx])
        peak_v = float(v_b[peak_idx])
        stats["I_tot_peak"] = peak_i
        stats["V_b_at_I_tot_peak"] = peak_v
        stats["P_peak"] = peak_i * peak_v
    except (KeyError, AttributeError, TypeError, IndexError):
        pass
    return stats


def cell_centers(n_cells, L_plasma, convention="sim"):
    """
    Compute cell-center axial positions.

    Parameters
    ----------
    n_cells : int
    L_plasma : float
        Plasma column length [cm].
    convention : str
        'sim'  — z=0 at source/left end.
            centers = [(i + 0.5) * L_plasma / n_cells]
            e.g. 3 cells, 1800 cm → [300, 900, 1500] cm
        'exp'  — z=0 at far/right end (experimental/machine convention).
            centers = L_plasma - sim_centers
            e.g. 3 cells, 1800 cm → [1500, 900, 300] cm

    Returns
    -------
    np.ndarray, shape (n_cells,)
    """
    sim_z = np.array([(i + 0.5) * L_plasma / n_cells for i in range(n_cells)])
    if convention == "sim":
        return sim_z
    else:
        return L_plasma - sim_z


def compute_window_stats(results, t_window=(10.0, 20.0)):
    """
    Compute statistics for ne and Te within a time window.

    Parameters
    ----------
    results : SimpleNamespace or dict
        Output of ``sim.get_results()``.  Time array must be in milliseconds.
    t_window : tuple of float
        (t_start, t_end) in ms, relative to breakdown (t=0 at breakdown).

    Returns
    -------
    dict with keys:
        ne_var   float  spatial variance: var of per-cell time-means across cells
        ne_cov   float  spatial CoV: std(cell_means) / overall_mean
        ne_tvar  float  temporal variance: mean of per-cell time-variances
        ne_tcov  float  temporal CoV: mean of (per-cell std / per-cell mean)
        ne_min   float  minimum ne over all cells and all t in window
        ne_max   float  maximum ne over all cells and all t in window
        ne_mean  float  mean ne over all cells and all t in window
        Te_var / Te_cov / Te_tvar / Te_tcov / Te_min / Te_max / Te_mean — same for Te

    Raises
    ------
    ValueError
        If no timesteps fall inside t_window.
    """
    _get = (lambda k: results[k]) if isinstance(results, dict) else (lambda k: getattr(results, k))
    t = _get("time")
    t_start, t_end = t_window
    mask = (t >= t_start) & (t <= t_end)

    if not np.any(mask):
        raise ValueError(
            f"No timesteps found in window [{t_start}, {t_end}] ms. "
            f"Time range: [{t.min():.3f}, {t.max():.3f}] ms."
        )

    stats = {}
    for key in ("ne", "Te"):
        arr = _get(key)[mask]  # (n_t_window, n_cells)
        n_cells = arr.shape[1]

        # Exclude cathode/boundary cells (first and last) from variation analysis.
        # Both ends are boundary-condition dominated and skew spatial stats.
        interior = slice(1, -1) if n_cells > 2 else slice(None)
        arr_i = arr[:, interior]

        cell_means = arr_i.mean(axis=0)
        cell_vars = arr_i.var(axis=0)
        cell_stds = np.sqrt(cell_vars)
        overall_mean = float(arr_i.mean())

        # Spatial: variability of steady-state values across interior cells
        stats[f"{key}_var"] = float(np.var(cell_means))
        stats[f"{key}_cov"] = (
            float(np.std(cell_means) / overall_mean) if overall_mean > 0 else 0.0
        )

        # Temporal: fluctuation amplitude within the window, averaged over interior cells
        stats[f"{key}_tvar"] = float(cell_vars.mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cell_tcov = np.where(cell_means > 0, cell_stds / cell_means, 0.0)
        stats[f"{key}_tcov"] = float(per_cell_tcov.mean())

        # Total: spatial + temporal (law of total variance)
        stats[f"{key}_total_var"] = stats[f"{key}_var"] + stats[f"{key}_tvar"]

        stats[f"{key}_min"] = float(arr_i.min())
        stats[f"{key}_max"] = float(arr_i.max())
        stats[f"{key}_mean"] = overall_mean

    stats.update(compute_cathode_peak_stats(results))

    # Cathode power stats: P_net_mean and P_eff = P_net_mean / P_wall_mean
    try:
        cathode_obj = _get("cathode")
        _cget = (
            (lambda k: cathode_obj[k])
            if isinstance(cathode_obj, dict)
            else (lambda k: getattr(cathode_obj, k))
        )
        p_net_mean = float(_cget("P_net")[mask].mean())
        p_wall_mean = float(_cget("P_wall")[mask].mean())
        stats["P_net_mean"] = p_net_mean
        stats["P_eff"] = p_net_mean / p_wall_mean if p_wall_mean != 0 else float("nan")
    except (KeyError, AttributeError, TypeError, IndexError):
        p_net_mean = float("nan")
        p_wall_mean = float("nan")
        stats["P_net_mean"] = p_net_mean
        stats["P_eff"] = float("nan")

    # Twin cathode power stats (all-NaN arrays when TwinCathode=False)
    try:
        twin_obj = _get("cathode_twin")
        _tget = (
            (lambda k: twin_obj[k])
            if isinstance(twin_obj, dict)
            else (lambda k: getattr(twin_obj, k))
        )
        twin_p_net = _tget("P_net")[mask]
        twin_p_wall = _tget("P_wall")[mask]
        if np.all(np.isnan(twin_p_net)):
            p_net_mean_twin = float("nan")
            p_wall_mean_twin = float("nan")
        else:
            p_net_mean_twin = float(np.nanmean(twin_p_net))
            p_wall_mean_twin = float(np.nanmean(twin_p_wall))
    except (KeyError, AttributeError, TypeError, IndexError):
        p_net_mean_twin = float("nan")
        p_wall_mean_twin = float("nan")

    # Totals across both cathodes
    have_primary = not np.isnan(p_net_mean)
    have_twin = not np.isnan(p_net_mean_twin)
    if have_primary and have_twin:
        p_net_total = p_net_mean + p_net_mean_twin
        p_wall_total = p_wall_mean + p_wall_mean_twin
        stats["P_net_total"] = p_net_total
        stats["P_eff_total"] = p_net_total / p_wall_total if p_wall_total != 0 else float("nan")
    elif have_primary:
        stats["P_net_total"] = p_net_mean
        stats["P_eff_total"] = stats["P_eff"]
    else:
        stats["P_net_total"] = float("nan")
        stats["P_eff_total"] = float("nan")

    return stats
