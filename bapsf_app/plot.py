"""
Visualization routines for LAPDSim results.

Single-run plots
----------------
``plot_run(results, params, flags)`` is the main entry point.  It routes to
2-D line plots (≤5 cells) or 2-D contour plots / position-vs-time heatmaps (>5 cells).

Sweep analysis plots
--------------------
``plot_sweep_variance`` and ``plot_sweep_heatmap`` operate on the index dict
returned by ``database.load_index()``.

Run comparison
--------------
``plot_run_comparison(db_path, run_ids, quantity)`` overlays one quantity from
multiple archived runs on a single axes.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .stats import cell_centers

_qe_SI = 1.602176634e-19  # J per eV

# ── Contour plot helpers ──────────────────────────────────────────────────────

_DENSE_MANTISSAS = np.array([
    1., 1.125, 1.25, 1.375, 1.5, 1.75, 2., 2.25,
    2.5, 2.75, 3., 3.5, 4., 4.5, 5., 6.25, 7.5, 8.75,
])


def _log_tick_label(v, pos=None):
    value = 10 ** v
    if not np.isfinite(value) or value <= 0:
        return ""
    exp = int(np.floor(np.log10(value)))
    pref = value / (10 ** exp)
    if np.isclose(pref, 1.0):
        return rf"$10^{{{exp}}}$"
    return rf"${pref:.1f}\times10^{{{exp}}}$"


def _ratio_tick_label(v, pos=None):
    value = 10 ** v
    if not np.isfinite(value):
        return ""
    return f"{value:g}" if value >= 1 else f"{value:.3g}"


def _linear_tick_label(v, pos=None):
    return f"{v:g}"


def _nice_log_levels(vmin_log, vmax_log, mantissas=(1, 2, 5)):
    emin = int(np.floor(vmin_log))
    emax = int(np.ceil(vmax_log))
    vals = []
    for exp in range(emin, emax + 1):
        for m in mantissas:
            v = np.log10(float(m)) + exp
            if vmin_log - 1e-9 <= v <= vmax_log + 1e-9:
                vals.append(v)
    return np.array(vals) if vals else np.linspace(vmin_log, vmax_log, 5)


def _nice_linear_levels(vmin, vmax, target=16):
    if vmax <= vmin:
        return np.array([vmin, vmax])
    raw_step = (vmax - vmin) / max(target - 1, 1)
    exp = np.floor(np.log10(max(abs(raw_step), 1e-30)))
    frac = raw_step / 10 ** exp
    if frac <= 1:
        nf = 1.0
    elif frac <= 2:
        nf = 2.0
    elif frac <= 2.5:
        nf = 2.5
    elif frac <= 5:
        nf = 5.0
    else:
        nf = 10.0
    step = nf * 10 ** exp
    start = np.floor(vmin / step) * step
    stop = np.ceil(vmax / step) * step
    return np.arange(start, stop + 0.5 * step, step)


def _contour_panel(ax, fig, Z_mesh, T_mesh, data, title, cbar_label,
                   is_log=True, is_ratio=False, vmin=None, vmax=None,
                   xlabel="Position [cm]", ylabel="Time [ms]"):
    """Draw one filled-contour panel into *ax* with colorbar and overlaid contour lines."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    vmin_plot = float(np.floor(np.nanmin(finite))) if vmin is None else float(vmin)
    vmax_plot = float(np.ceil(np.nanmax(finite))) if vmax is None else float(vmax)
    if vmin_plot >= vmax_plot:
        vmax_plot = vmin_plot + 1.0

    levels = np.linspace(vmin_plot, vmax_plot, 100)

    if is_ratio:
        line_values = np.array([
            0.01, 0.015, 0.02, 0.03, 0.05, 0.07,
            0.1, 0.15, 0.2, 0.3, 0.5, 0.7,
            1, 1.5, 2, 3, 5, 7,
            10, 15, 20, 30, 50, 70, 100,
        ])
        label_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                                  1, 2, 5, 10, 20, 50, 100])
        line_levels = np.log10(line_values)
        label_levels = np.log10(label_values)
        cbar_ticks = label_levels
        tick_fmt = FuncFormatter(_ratio_tick_label)
    elif is_log:
        line_levels = _nice_log_levels(vmin_plot, vmax_plot, _DENSE_MANTISSAS)
        label_levels = _nice_log_levels(vmin_plot, vmax_plot, (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5))
        cbar_ticks = _nice_log_levels(vmin_plot, vmax_plot, (1, 1.5, 2, 3, 5, 7))
        tick_fmt = FuncFormatter(_log_tick_label)
    else:
        line_levels = _nice_linear_levels(vmin_plot, vmax_plot, target=16)
        label_levels = _nice_linear_levels(vmin_plot, vmax_plot, target=8)
        label_levels = np.array([
            lev for lev in label_levels
            if np.any(np.abs(lev - line_levels) < 1e-10)
        ])
        cbar_ticks = label_levels
        tick_fmt = FuncFormatter(_linear_tick_label)

    line_levels = line_levels[(line_levels >= vmin_plot) & (line_levels <= vmax_plot)]
    label_levels = label_levels[(label_levels >= vmin_plot) & (label_levels <= vmax_plot)]
    cbar_ticks = cbar_ticks[(cbar_ticks >= vmin_plot) & (cbar_ticks <= vmax_plot)]

    cf = ax.contourf(Z_mesh, T_mesh, data, levels=levels, cmap="plasma", extend="both")
    if line_levels.size >= 2:
        cs = ax.contour(Z_mesh, T_mesh, data, levels=line_levels,
                        colors="black", linewidths=0.8, alpha=0.9)
        if label_levels.size:
            ax.clabel(cs, levels=label_levels, inline=True, fontsize=6, fmt=tick_fmt)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)
    if cbar_ticks.size:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(tick_fmt)


# ── Position helpers ──────────────────────────────────────────────────────────


def position_labels(z_positions, convention="sim"):
    """
    Build legend/axis labels from cell-center positions.

    Parameters
    ----------
    z_positions : array-like
        Cell-center positions in cm (from ``cell_centers()``).
    convention : str
        'sim' or 'exp' — used only to annotate the label if desired.

    Returns
    -------
    list of str, e.g. ['z=300 cm', 'z=900 cm', 'z=1500 cm']
    """
    return [f"z={z:.0f} cm" for z in z_positions]


def _z_axis_label(convention):
    if convention == "sim":
        return "Simulation z [cm]  (z=0 at source end)"
    return "Experimental z [cm]  (z=0 at far end)"


# ── Title / subtitle helper ───────────────────────────────────────────────────


def _run_title(params, flags):
    """One-line parameter summary for plot titles."""
    gas = params.get("gas_type", "?")
    V_bank = params.get("V_bank", 0)
    T_s_K = params.get("T_s", 0)
    T_s_C = T_s_K - 273.15 if T_s_K else 0
    cells = params.get("cells", "?")
    twin_active = flags.get("TwinCathode", False)
    S_gp = params.get("S_gp", 0.0)
    Twin_S_gp = params.get("Twin_S_gp", 0.0) if twin_active else 0.0
    S_gp_total = S_gp + Twin_S_gp
    twin = "twin" if twin_active else "single"
    return f"{gas}  V_bank={V_bank:.0f} V  T_s={T_s_C:.0f}°C  S_gp={S_gp_total:.0f}  {cells} cells  [{twin}]"


# ── Main entry point ──────────────────────────────────────────────────────────


def plot_run(results, params, flags, z_convention="sim", save_dir=None):
    """
    Plot one simulation run.

    Routes to :func:`_plot_2d` (≤5 cells) or :func:`_plot_contour` (>5 cells).

    Parameters
    ----------
    results : dict
        Output of ``sim.get_results()``.
    params : dict
        Input parameter dict used for the run.
    flags : dict
        Input flags dict used for the run.
    z_convention : str
        'sim'  — z=0 at source/left end.
        'exp'  — z=0 at far/right end (experimental convention).
    save_dir : str or path-like or None
        If given, PNGs are saved here with auto-generated filenames.

    Returns
    -------
    dict of {figure_name: matplotlib.Figure}
    """
    n_cells = results["ne"].shape[1]
    L_plasma = params.get("Lp", params.get("L_plasma", 1800))
    z_pos = cell_centers(n_cells, L_plasma, convention=z_convention)

    if n_cells <= 5:
        return _plot_2d(results, params, flags, z_pos, z_convention, save_dir)
    else:
        return _plot_contour(results, params, flags, z_pos, z_convention, save_dir)


# ── 2-D plots (≤5 cells) ─────────────────────────────────────────────────────


def _save(fig, name, save_dir):
    if save_dir is not None:
        import pathlib

        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(pathlib.Path(save_dir) / f"{name}.png", dpi=150, bbox_inches="tight")


def _plot_2d(results, params, flags, z_pos, z_convention, save_dir):
    """2-D time-series plots for ≤5 cells."""
    t = results["time"]
    labels = position_labels(z_pos, z_convention)
    title_base = _run_title(params, flags)

    n_cells = results["ne"].shape[1]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_cells)]

    figs = {}

    # Helper: one figure with one axes
    def _fig(title, ylabel, yscale="linear"):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{title_base}", fontsize=10)
        ax.set_yscale(yscale)
        return fig, ax

    _LEGEND_KW = dict(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    def _lines(ax, arr, legend=True):
        for ci in range(n_cells):
            ax.plot(t, arr[:, ci], color=colors[ci], label=labels[ci])
        if legend:
            ax.legend(**_LEGEND_KW)

    def _autolim(ax, *arrs, t_cut=0.5):
        """Set ylim from non-transient data (t >= t_cut ms), ignoring early spikes."""
        mask = t >= t_cut
        if not mask.any():
            return
        vals = []
        for arr in arrs:
            a = np.asarray(arr)
            sub = a[mask, :].ravel() if a.ndim == 2 else a[mask].ravel()
            finite = sub[np.isfinite(sub)]
            if ax.get_yscale() == "log":
                finite = finite[finite > 0]
            if finite.size:
                vals.append(finite)
        if not vals:
            return
        all_v = np.concatenate(vals)
        if not all_v.size:
            return
        if ax.get_yscale() == "log":
            lv = np.log10(all_v)
            span = lv.max() - lv.min()
            margin = max(0.05 * span, 0.3)
            ax.set_ylim(10 ** (lv.min() - margin), 10 ** (lv.max() + margin))
        else:
            span = all_v.max() - all_v.min()
            margin = max(0.05 * span, abs(np.median(all_v)) * 0.05, 1e-30)
            ax.set_ylim(all_v.min() - margin, all_v.max() + margin)

    # ── Electron density ──────────────────────────────────────────────────────
    fig, ax = _fig("Electron Density", r"$n_e$ [cm$^{-3}$]")
    _lines(ax, results["ne"])
    _autolim(ax, results["ne"])
    figs["ne"] = fig
    _save(fig, "ne", save_dir)

    # ── Neutral density ───────────────────────────────────────────────────────
    fig, ax = _fig("Neutral Density", r"$n_n$ [cm$^{-3}$]")
    _lines(ax, results["nn"])
    _autolim(ax, results["nn"])
    figs["nn"] = fig
    _save(fig, "nn", save_dir)

    # ── Ionisation fraction (ne/nn) ───────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ion_frac = np.where(results["nn"] > 0, results["ne"] / results["nn"], np.nan)
    fig, ax = _fig("Ionisation Fraction", r"$n_e / n_n$", yscale="log")
    _lines(ax, ion_frac)
    ax.set_ylim(1e-3, 1e3)           # fixed symmetric bounds (overrides autolim)
    ax.axhline(1.0, color="lightgray", lw=1.0, zorder=0)
    figs["ion_ratio"] = fig
    _save(fig, "ion_ratio", save_dir)

    # ── Electron temperature ──────────────────────────────────────────────────
    fig, ax = _fig("Electron Temperature", r"$T_e$ [eV]")
    _lines(ax, results["Te"])
    _autolim(ax, results["Te"])
    figs["Te"] = fig
    _save(fig, "Te", save_dir)

    # ── Ion temperature ───────────────────────────────────────────────────────
    fig, ax = _fig("Ion Temperature", r"$T_i$ [eV]")
    _lines(ax, results["Ti"])
    _autolim(ax, results["Ti"])
    figs["Ti"] = fig
    _save(fig, "Ti", save_dir)

    # ── Neutral cooling (Qen) — power per cell ────────────────────────────────
    # Convert from temperature-rate units to watts per cell:
    #   P [W] = (3/2) * ne * Qen * cell_vol * qe_SI
    # en_factor = 2/3 is already folded in, so (3/2) * Qen/en_factor = (3/2)*(3/2)*Qen
    # For consistency with existing notebooks we approximate:
    #   P ≈ ne * Qen * cell_vol * qe_SI  (drops the 3/2 pre-factor)
    L_plasma = params.get("Lp", 1800)
    Rp = params.get("Rp", 18)
    cell_vol = math.pi * Rp**2 * (L_plasma / n_cells)  # cm³ per cell

    Qen_W = results["Qen"] * results["ne"] * cell_vol * _qe_SI
    Qen_W = np.where(Qen_W > 0, Qen_W, np.nan)
    Qen_total = np.nansum(Qen_W, axis=1)
    Qen_total = np.where(Qen_total > 0, Qen_total, np.nan)

    fig, ax = _fig("Electron Cooling by Neutral Radiation", "Power [W]", yscale="log")
    _lines(ax, Qen_W, legend=False)
    ax.plot(t, Qen_total, color="black", lw=2.0, ls="--", label="Total")
    ax.legend(**_LEGEND_KW)
    _autolim(ax, Qen_W, Qen_total)
    ax.set_ylim(bottom=1e3)
    figs["Qen_power"] = fig
    _save(fig, "Qen_power", save_dir)

    # ── Ion power loss to charge exchange (Qcx) ───────────────────────────────
    Qcx_W = np.abs(results["Qcx"]) * results["ne"] * cell_vol * _qe_SI
    Qcx_W = np.where(Qcx_W > 0, Qcx_W, np.nan)
    Qcx_total = np.nansum(Qcx_W, axis=1)
    Qcx_total = np.where(Qcx_total > 0, Qcx_total, np.nan)

    fig, ax = _fig("Ion Power Loss to Charge Exchange", "Power [W]", yscale="log")
    _lines(ax, Qcx_W, legend=False)
    ax.plot(t, Qcx_total, color="black", lw=2.0, ls="--", label="Total")
    ax.legend(**_LEGEND_KW)
    _autolim(ax, Qcx_W, Qcx_total)
    figs["Qcx_power"] = fig
    _save(fig, "Qcx_power", save_dir)

    # ── Power balance ─────────────────────────────────────────────────────────
    # Normalise each term (summed over cells) to mean cathode input power during discharge.
    tau_discharge_ms = params.get("tau_discharge", 20e-3) * 1e3
    discharge_mask = (t > 1.0) & (t <= tau_discharge_ms)
    cathode = results.get("cathode", {})
    p_wall = cathode.get("P_wall") if isinstance(cathode, dict) else getattr(cathode, "P_wall", None)
    if p_wall is not None and discharge_mask.any():
        input_power = float(np.nanmean(p_wall[discharge_mask]))
        if flags.get("TwinCathode", False):
            cathode_twin = results.get("cathode_twin", {})
            p_wall_twin = cathode_twin.get("P_wall") if isinstance(cathode_twin, dict) else getattr(cathode_twin, "P_wall", None)
            if p_wall_twin is not None:
                input_power += float(np.nanmean(p_wall_twin[discharge_mask]))
    else:
        input_power = 1.0
    if input_power == 0 or not np.isfinite(input_power):
        input_power = 1.0  # avoid divide-by-zero

    def _frac(key):
        total = np.abs(results[key]).sum(axis=1)
        return np.where(total > 0, total / input_power, np.nan)

    _pb_fracs = [_frac(k) for k in ("e_par_flux", "Qie", "Qei", "Qen", "Qeb", "Qcx")]
    fig, ax = _fig("Power Balance (fraction of input)", "Fraction of Input Power", yscale="log")
    ax.plot(t, _pb_fracs[0], label=r"$e$-par flux", lw=1.5)
    ax.plot(t, _pb_fracs[1], label=r"$Q_{ie}$", lw=1.5)
    ax.plot(t, _pb_fracs[2], label=r"$Q_{ei}$", lw=1.5)
    ax.plot(t, _pb_fracs[3], label=r"$Q_{en}$", lw=1.5)
    ax.plot(t, _pb_fracs[4], label=r"$Q_{eb}$", lw=1.5)
    ax.plot(t, _pb_fracs[5], label=r"$Q_{cx}$", lw=1.5)
    ax.legend(**_LEGEND_KW)
    _autolim(ax, *_pb_fracs)
    ax.set_ylim(bottom=1e-2)
    figs["power_balance"] = fig
    _save(fig, "power_balance", save_dir)

    # ── Ion power balance ─────────────────────────────────────────────────────
    _ion_fracs = [_frac(k) for k in ("Qie", "Qcx", "i_par_flux")]
    fig, ax = _fig("Ion Power Balance (fraction of input)", "Fraction of Input Power", yscale="log")
    ax.plot(t, _ion_fracs[0], label=r"$Q_{io}$ (e-i exchange)", lw=1.5)
    ax.plot(t, _ion_fracs[1], label=r"$Q_{cx}$ (charge exchange)", lw=1.5)
    ax.plot(t, _ion_fracs[2], label=r"$i$-par flux (conductive)", lw=1.5)
    ax.legend(**_LEGEND_KW)
    _autolim(ax, *_ion_fracs)
    figs["ion_power_balance"] = fig
    _save(fig, "ion_power_balance", save_dir)

    # ── Isat synthetic diagnostic (normalised to t=tau_discharge) ────────────
    if "isat" in results:
        isat_raw = results["isat"]
        if isat_raw.ndim == 1:          # old run: only first cell was stored
            isat_raw = isat_raw[:, np.newaxis]

        # Full-run plot: normalise to value at end of main discharge (tau_discharge)
        t_norm_ms = params.get("tau_discharge", 20e-3) * 1e3  # tau_discharge seconds → ms
        norm_idx = int(np.argmin(np.abs(t - t_norm_ms)))
        norm_vals = isat_raw[norm_idx, :]  # (n_cells,)
        with np.errstate(invalid="ignore", divide="ignore"):
            isat_norm = np.where(norm_vals != 0,
                                 isat_raw / norm_vals[np.newaxis, :],
                                 np.nan)
        fig, ax = _fig(
            rf"Normalised $I_{{sat}}$  ($n_e\sqrt{{T_e}}$, norm. at t={t_norm_ms:.0f} ms)",
            r"$I_{sat}$ [norm.]",
        )
        _lines(ax, isat_norm)
        _autolim(ax, isat_norm)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        figs["isat"] = fig
        _save(fig, "isat", save_dir)

        # Afterglow-only plot: t >= 20 ms, normalise to first point in window
        ag_mask = t >= 20.0
        if ag_mask.any():
            t_ag = t[ag_mask]
            isat_ag = isat_raw[ag_mask, :]
            norm_vals_ag = isat_ag[0, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                isat_ag_norm = np.where(norm_vals_ag != 0,
                                        isat_ag / norm_vals_ag[np.newaxis, :],
                                        np.nan)
            fig, ax = _fig(
                r"Afterglow $I_{sat}$ (t$\geq$20 ms, norm. at t=20 ms)",
                r"$I_{sat}$ [norm.]",
            )
            for ci in range(n_cells):
                ax.plot(t_ag, isat_ag_norm[:, ci], color=colors[ci], label=labels[ci])
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.axhline(1, color="k", lw=0.8, ls="--")
            figs["isat_afterglow"] = fig
            _save(fig, "isat_afterglow", save_dir)

    # ── Parallel velocity ─────────────────────────────────────────────────────
    if "v_plasma" in results:
        _v_plot = results["v_plasma"] / 100.0
        fig, ax = _fig("Parallel Plasma Velocity", r"$v_\parallel$ [m/s]")
        _lines(ax, _v_plot)
        _autolim(ax, _v_plot)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        figs["v_plasma"] = fig
        _save(fig, "v_plasma", save_dir)

    # ── Parallel Mach number ──────────────────────────────────────────────────
    if "v_plasma" in results and "Te" in results:
        _gas = str(params.get("gas_type", "He")).strip().lower()
        _mu = 4.0 if _gas in ("he", "helium") else 1.0  # He=4, H=1
        c_s = 9.79e5 * np.sqrt(results["Te"] / _mu)  # cm/s, same units as v_plasma
        with np.errstate(invalid="ignore", divide="ignore"):
            mach = np.where(c_s > 0, results["v_plasma"] / c_s, np.nan)
        fig, ax = _fig("Parallel Mach Number", r"$M_\parallel = v_\parallel / c_s(T_e)$")
        _lines(ax, mach)
        _autolim(ax, mach)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        figs["mach"] = fig
        _save(fig, "mach", save_dir)

    # ── Mean free paths ───────────────────────────────────────────────────────
    if "primary_mfp" in results and "bulk_mfp" in results:
        fig, ax = _fig("Electron Mean Free Paths / Cell Length", "MFP / cell length", yscale="log")
        for ci in range(n_cells):
            ax.plot(
                t,
                results["primary_mfp"][:, ci],
                color=colors[ci],
                ls="-",
                label=f"primary {labels[ci]}",
            )
            ax.plot(
                t,
                results["bulk_mfp"][:, ci],
                color=colors[ci],
                ls="--",
                label=f"bulk {labels[ci]}",
            )
        ax.axhline(1, color="k", lw=0.8, ls=":", label="MFP = cell length")
        _autolim(ax, results["primary_mfp"], results["bulk_mfp"])
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
        figs["mfp"] = fig
        _save(fig, "mfp", save_dir)

    # ── Coulomb logarithm ─────────────────────────────────────────────────────
    if "ln_lambda" in results:
        fig, ax = _fig("Coulomb Logarithm", r"$\ln \Lambda$")
        _lines(ax, results["ln_lambda"])
        _autolim(ax, results["ln_lambda"])
        figs["ln_lambda"] = fig
        _save(fig, "ln_lambda", save_dir)

    return figs


# ── Contour plots (>5 cells) ──────────────────────────────────────────────────


def _plot_contour(results, params, flags, z_pos, z_convention, save_dir):
    """Position-vs-time contour plots for >5 cells."""
    t = results["time"]
    title_base = _run_title(params, flags)
    z_label = _z_axis_label(z_convention)
    figs = {}

    Z_mesh, T_mesh = np.meshgrid(z_pos, t)  # both (n_t, n_cells)
    _kw = dict(xlabel=z_label, ylabel="Time [ms]")

    # ── Densities (ne, nn, ionization fraction) ────────────────────────────────
    ne = np.where(results["ne"] > 0, results["ne"], np.nan)
    nn_arr = np.where(results["nn"] > 0, results["nn"], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((ne > 0) & (nn_arr > 0), ne / nn_arr, np.nan)

    ne_log = np.where(ne > 0, np.log10(ne), np.nan)
    nn_log = np.where(nn_arr > 0, np.log10(nn_arr), np.nan)
    ratio_log = np.where(ratio > 0, np.log10(ratio), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    _contour_panel(axes[0], fig, Z_mesh, T_mesh, ne_log,
                   "Electron Density", r"$n_e$ [cm$^{-3}$]", is_log=True, **_kw)
    _contour_panel(axes[1], fig, Z_mesh, T_mesh, nn_log,
                   "Neutral Density", r"$n_n$ [cm$^{-3}$]", is_log=True, **_kw)
    _contour_panel(axes[2], fig, Z_mesh, T_mesh, ratio_log,
                   r"Ionisation Fraction $n_e/n_n$", r"$n_e/n_n$",
                   is_log=True, is_ratio=True, vmin=-2, vmax=2, **_kw)
    fig.suptitle(f"Densities\n{title_base}", fontsize=10)
    figs["densities"] = fig
    _save(fig, "densities", save_dir)

    # ── Temperatures (Te, Ti) ──────────────────────────────────────────────────
    Te = np.where(np.isfinite(results["Te"]), results["Te"], np.nan)
    Ti = np.where(np.isfinite(results["Ti"]), results["Ti"], np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    _contour_panel(axes[0], fig, Z_mesh, T_mesh, Te,
                   "Electron Temperature", r"$T_e$ [eV]", is_log=False, **_kw)
    _contour_panel(axes[1], fig, Z_mesh, T_mesh, Ti,
                   "Ion Temperature", r"$T_i$ [eV]", is_log=False, **_kw)
    fig.suptitle(f"Temperatures\n{title_base}", fontsize=10)
    figs["temperatures"] = fig
    _save(fig, "temperatures", save_dir)

    # ── Parallel velocity ──────────────────────────────────────────────────────
    if "v_plasma" in results:
        v = np.where(np.isfinite(results["v_plasma"]),
                     results["v_plasma"] / 100.0, np.nan)  # cm/s → m/s
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        _contour_panel(ax, fig, Z_mesh, T_mesh, v,
                       "Parallel Velocity", r"$v_\parallel$ [m/s]", is_log=False, **_kw)
        fig.suptitle(f"Parallel Velocity\n{title_base}", fontsize=10)
        figs["v_plasma"] = fig
        _save(fig, "v_plasma", save_dir)

    return figs


# ── Sweep analysis plots ──────────────────────────────────────────────────────


def plot_sweep_variance(
    index,
    x_param,
    hue_param=None,
    quantity="ne",
    metric="var",
    t_label="10–20 ms",
    save_dir=None,
):
    """
    Scatter plot: varied parameter vs. cell-to-cell variance (or other metric).

    Parameters
    ----------
    index : dict
        Output of ``load_index()``.
    x_param : str
        Parameter name to plot on the x-axis.  Must be in ``index['params']``.
    hue_param : str or None
        If given, color points by this second parameter or flag.
    quantity : str
        'ne' or 'Te'.
    metric : str
        'var'  — variance of per-cell time-means (spatial variance).
        'cov'  — coefficient of variation (std/mean).
        'min' / 'max' / 'mean'.
    t_label : str
        Time window description shown in title.
    save_dir : str or None

    Returns
    -------
    matplotlib.Figure
    """
    stats = index["stats_10_20ms"]
    params = index["params"]
    flags = index["flags"]

    y_key = f"{quantity}_{metric}"
    if y_key not in stats:
        raise KeyError(
            f"'{y_key}' not found in index stats.  "
            f"Available: {list(stats.keys())}"
        )

    # Retrieve x values
    def _to_float_array(arr, name):
        try:
            return np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Parameter '{name}' contains non-numeric values and cannot be "
                f"used as an axis.  Values: {list(arr)[:5]}"
            ) from exc

    if x_param in params:
        x = _to_float_array(params[x_param], x_param)
    elif x_param in flags:
        x = _to_float_array(flags[x_param], x_param)
    else:
        raise KeyError(f"'{x_param}' not found in params or flags.")

    y = np.asarray(stats[y_key], dtype=float)

    # Filter to successful runs
    ok = np.array(index["status"]) == "ok"
    x, y = x[ok], y[ok]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    if hue_param is not None:
        if hue_param in params:
            hue = _to_float_array(params[hue_param], hue_param)[ok]
        elif hue_param in flags:
            hue = _to_float_array(flags[hue_param], hue_param)[ok]
        else:
            raise KeyError(f"hue_param '{hue_param}' not found in params or flags.")

        sc = ax.scatter(x, y, c=hue, cmap="viridis", s=60, zorder=3)
        fig.colorbar(sc, ax=ax, label=hue_param)
    else:
        ax.scatter(x, y, s=60, zorder=3)

    ax.set_xlabel(x_param)
    metric_labels = {
        "var": "spatial variance (var of cell time-means)",
        "cov": "spatial CoV (std/mean across cells)",
        "tvar": "temporal variance (mean of cell time-variances)",
        "tcov": "temporal CoV (mean of cell std/mean)",
        "total_var": "total variance (spatial + temporal)",
        "min": "minimum",
        "max": "maximum",
        "mean": "mean",
    }
    title_labels = {
        "var": f"{quantity} spatial uniformity",
        "cov": f"{quantity} spatial uniformity",
        "tvar": f"{quantity} temporal stability",
        "tcov": f"{quantity} temporal stability",
        "total_var": f"{quantity} total variance",
        "min": f"{quantity}",
        "max": f"{quantity}",
        "mean": f"{quantity}",
    }
    ax.set_ylabel(
        f"{quantity} {metric_labels.get(metric, metric)}"
        f" [{t_label}]"
    )
    ax.set_title(
        f"Effect of {x_param} on {title_labels.get(metric, quantity)}  [{t_label}]"
    )
    ax.grid(True, alpha=0.3)

    if save_dir is not None:
        _save(fig, f"sweep_var_{quantity}_{x_param}", save_dir)

    return fig


def plot_sweep_heatmap(
    index,
    x_param,
    y_param,
    quantity="ne_var",
    t_label="10–20 ms",
    save_dir=None,
):
    """
    2-D heatmap of a sweep statistic as a function of two varied parameters.

    Parameters
    ----------
    index : dict
        Output of ``load_index()``.
    x_param, y_param : str
        Parameter names for the two axes.
    quantity : str
        Key in ``index['stats_10_20ms']``, e.g. ``'ne_var'``, ``'Te_mean'``.
    t_label : str
    save_dir : str or None

    Returns
    -------
    matplotlib.Figure
    """
    stats = index["stats_10_20ms"]
    params = index["params"]
    flags = index["flags"]

    if quantity not in stats:
        raise KeyError(f"'{quantity}' not found in index stats.")

    def _get(name):
        try:
            if name in params:
                return np.asarray(params[name], dtype=float)
            if name in flags:
                return np.asarray(flags[name], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Parameter '{name}' contains non-numeric values and cannot be used as an axis."
            ) from exc
        raise KeyError(f"'{name}' not found in params or flags.")

    ok = np.array(index["status"]) == "ok"
    x_all = _get(x_param)[ok]
    y_all = _get(y_param)[ok]
    z_all = np.asarray(stats[quantity], dtype=float)[ok]

    x_vals = np.unique(x_all)
    y_vals = np.unique(y_all)

    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for xi, xv in enumerate(x_vals):
        for yi, yv in enumerate(y_vals):
            mask = (x_all == xv) & (y_all == yv)
            if mask.any():
                grid[yi, xi] = z_all[mask].mean()

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    im = ax.pcolormesh(x_vals, y_vals, grid, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label=quantity)
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{quantity}  [{t_label}]")

    if save_dir is not None:
        _save(fig, f"sweep_heatmap_{quantity}_{x_param}_{y_param}", save_dir)

    return fig


# ── Run comparison ────────────────────────────────────────────────────────────

_COMPARISON_YLABELS = {
    "ne": r"$n_e$ [cm$^{-3}$]",
    "nn": r"$n_n$ [cm$^{-3}$]",
    "Te": r"$T_e$ [eV]",
    "Ti": r"$T_i$ [eV]",
    "v_plasma": r"$v_\parallel$ [m/s]",
    "isat": r"$I_{sat}$ [norm.]",
    "ln_lambda": r"$\ln\Lambda$",
    "primary_mfp": "Primary MFP / cell length",
    "bulk_mfp": "Bulk MFP / cell length",
}

_RUN_LINESTYLES = ["-", "--", ":", "-."]


def plot_run_comparison(db_path, run_ids, quantity, cell_idx=-1):
    """
    Load multiple archived runs and overlay one quantity on a single axes.

    Parameters
    ----------
    db_path : str or path-like
        Path to the HDF5 database.
    run_ids : list of str
        Run IDs to compare (typically 2–4).
    quantity : str
        Key in the results dict, e.g. ``'ne'``, ``'Te'``, ``'v_plasma'``.
    cell_idx : int
        Which cell to plot.  ``-1`` (default) plots every cell individually,
        using color to distinguish cells and linestyle to distinguish runs.

    Returns
    -------
    matplotlib.Figure
    """
    from matplotlib.lines import Line2D
    from .database import open_db, load_run

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    run_labels = []   # (run_i, label_str) for legend
    cells_seen = set()

    with open_db(db_path) as db:
        for run_i, run_id in enumerate(run_ids):
            params, flags, results = load_run(db, run_id, keys=["time", quantity])
            t = results["time"]
            data = results.get(quantity)
            if data is None:
                continue

            gas = params.get("gas_type", "?")
            V_bank = params.get("V_bank", 0)
            T_s_K = params.get("T_s", 0)
            T_s_C = T_s_K - 273.15 if T_s_K else 0
            twin_active = flags.get("TwinCathode", False)
            S_gp = params.get("S_gp", 0.0)
            Twin_S_gp = params.get("Twin_S_gp", 0.0) if twin_active else 0.0
            S_gp_total = S_gp + Twin_S_gp
            twin_str = "twin" if twin_active else "single"
            run_label = f"{run_id}  {gas}  V_bank={V_bank:.0f}V  T_s={T_s_C:.0f}°C  S_gp={S_gp_total:.0f}  [{twin_str}]"
            run_labels.append((run_i, run_label))

            ls = _RUN_LINESTYLES[run_i % len(_RUN_LINESTYLES)]

            # Scale velocity from cm/s to m/s
            if quantity == "v_plasma":
                data = data / 100.0

            # Normalise isat per-cell to value at t=20 ms
            if quantity == "isat":
                if data.ndim == 1:      # old run: only first cell was stored
                    data = data[:, np.newaxis]
                norm_idx = int(np.argmin(np.abs(t - 20.0)))
                norm_vals = data[norm_idx, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    data = np.where(norm_vals != 0,
                                    data / norm_vals[np.newaxis, :],
                                    np.nan)

            if cell_idx == -1 and data.ndim == 2:
                # One line per cell: color = cell, linestyle = run
                for c in range(data.shape[1]):
                    ax.plot(t, data[:, c], color=cmap(c % 10), ls=ls, lw=1.5)
                    cells_seen.add(c)
            elif data.ndim == 2:
                ax.plot(t, data[:, int(cell_idx)], color=cmap(run_i % 10), ls="-", lw=1.5)
            else:
                ax.plot(t, data, color=cmap(run_i % 10), ls="-", lw=1.5)

    ylabel = _COMPARISON_YLABELS.get(quantity, quantity)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(ylabel)

    cell_desc = "all cells" if cell_idx == -1 else f"cell {cell_idx}"
    if quantity == "isat":
        ax.set_title(rf"Normalised $I_{{sat}}$ comparison  ({cell_desc}, norm. at t=20 ms)")
        ax.axhline(1, color="k", lw=0.8, ls="--")
    else:
        ax.set_title(f"{quantity} comparison  ({cell_desc})")

    if cell_idx == -1 and cells_seen:
        # Two-part legend: linestyle = run, color = cell
        run_handles = [
            Line2D([0], [0], color="dimgray",
                   ls=_RUN_LINESTYLES[i % len(_RUN_LINESTYLES)], lw=1.5, label=lbl)
            for i, lbl in run_labels
        ]
        cell_handles = [
            Line2D([0], [0], color=cmap(c % 10), ls="-", lw=2.5, label=f"cell {c}")
            for c in sorted(cells_seen)
        ]
        ax.legend(handles=run_handles + cell_handles, fontsize=7,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    else:
        handles = [
            Line2D([0], [0], color=cmap(i % 10), ls="-", lw=1.5, label=lbl)
            for i, lbl in run_labels
        ]
        ax.legend(handles=handles, fontsize=8,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.grid(True, alpha=0.3)
    return fig
