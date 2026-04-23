"""
Per-run and cross-run statistics for LAPDSim results.
"""
import numpy as np


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
    results : dict
        Output of ``sim.get_results()``.  Time array must be in milliseconds.
    t_window : tuple of float
        (t_start, t_end) in ms.

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
    t = results["time"]
    t_start, t_end = t_window
    mask = (t >= t_start) & (t <= t_end)

    if not np.any(mask):
        raise ValueError(
            f"No timesteps found in window [{t_start}, {t_end}] ms. "
            f"Time range: [{t.min():.3f}, {t.max():.3f}] ms."
        )

    stats = {}
    for key in ("ne", "Te"):
        arr = results[key][mask]  # (n_t_window, n_cells)
        cell_means = arr.mean(axis=0)  # shape (n_cells,) — time-mean per cell
        cell_vars = arr.var(axis=0)    # shape (n_cells,) — time-variance per cell
        cell_stds = np.sqrt(cell_vars)
        overall_mean = float(arr.mean())

        # Spatial: variability of steady-state values across cells
        stats[f"{key}_var"] = float(np.var(cell_means))
        stats[f"{key}_cov"] = (
            float(np.std(cell_means) / overall_mean) if overall_mean > 0 else 0.0
        )

        # Temporal: fluctuation amplitude within the window, averaged over cells
        stats[f"{key}_tvar"] = float(cell_vars.mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cell_tcov = np.where(cell_means > 0, cell_stds / cell_means, 0.0)
        stats[f"{key}_tcov"] = float(per_cell_tcov.mean())

        # Total: spatial + temporal (law of total variance)
        stats[f"{key}_total_var"] = stats[f"{key}_var"] + stats[f"{key}_tvar"]

        stats[f"{key}_min"] = float(arr.min())
        stats[f"{key}_max"] = float(arr.max())
        stats[f"{key}_mean"] = overall_mean

    return stats
