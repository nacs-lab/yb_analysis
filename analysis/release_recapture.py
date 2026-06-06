"""release_recapture.py -- release-and-recapture atom-temperature analysis.

Faithful, reusable port of the "Release and recapture" cell in
``analysis/GeneralAnalysis.ipynb``: a per-axis Monte-Carlo release-recapture model
swept over temperature and weighted-chi2 fit to the measured survival-vs-release-time
curve (best-fit T + 68 % CI from delta-chi2 = 1).

Model (unchanged from the notebook): each atom starts in a thermal state of the 3 trap
modes (geometric n-bar sampling), the trap is switched OFF for the release time t_rr, the
atom flies ballistically, then the trap is switched back ON; it is *recaptured* iff its
total energy in the re-activated Gaussian trap is below the trap depth (with a small flat
imaging-loss term). Survival vs t_rr is a thermometer: colder atoms move less and are
recaptured more often.

Two ways to use it
------------------
* library::

      from yb_analysis.analysis.release_recapture import (
          fit_release_recapture_temperature, analyze_scan_temperature)
      res = fit_release_recapture_temperature(times_s, survival, sem)   # arrays
      res = analyze_scan_temperature("D:/.../data_YYYYMMDD_HHMMSS")      # load+fit+plot

* CLI (run in the ``yb_analysis`` conda env)::

      python -m yb_analysis.analysis.release_recapture                 # latest scan
      python -m yb_analysis.analysis.release_recapture --path <dir>
      python -m yb_analysis.analysis.release_recapture --path <dir> --t-uK 1 80 1 --reps 4000

``res`` is a dict: ``T_best`` / ``T_low`` / ``T_high`` / ``T_sigma`` (Kelvin),
``temp_sweep`` / ``rmse`` / ``chi2`` / ``curves`` (the MC grid), ``times`` / ``survival`` /
``sem`` (the data used), ``fig_path`` (PNG, if plotted).

Default trap parameters mirror the notebook (174Yb, trap_freqs=[15,82,82] kHz,
trap_depth=390 uK, lambda=532 nm). Override per run if the trap changes.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# --- Physical constants (notebook values) ---------------------------------- #
KB = 1.381e-23        # Boltzmann constant [J/K]
HBAR = 1.055e-34      # reduced Planck constant [J*s]
M_YB174 = 2.872e-25   # 174Yb mass [kg]

# --- Default trap / MC parameters (GeneralAnalysis.ipynb cell "Release and recapture") --
DEFAULT_TRAP_FREQS = (15e3, 82e3, 82e3)   # (axial, radial, radial) [Hz]
DEFAULT_TRAP_DEPTH = 0.390e-3             # [K]
DEFAULT_LAMBDA = 532e-9                   # tweezer wavelength [m]
DEFAULT_REPS = 2000                       # MC atoms per (T, time) point
DEFAULT_LOSS = 0.01                       # flat imaging-loss prob (notebook's `> 0.01`)


# =========================================================================== #
# Monte-Carlo release-recapture model (vectorised over the `reps` atoms; the
# physics is identical to the notebook's per-atom loop -- thermal n-bar geometric
# sampling, ballistic flight, energy-vs-depth recapture in a Gaussian trap).
# =========================================================================== #
def montecarlo(trap_freqs, temps, t_rr, *, mass=M_YB174, trap_depth=DEFAULT_TRAP_DEPTH,
               lambda_tw=DEFAULT_LAMBDA, reps=DEFAULT_REPS, loss=DEFAULT_LOSS, rng=None):
    """Recapture probability for one release time ``t_rr`` (s).

    ``trap_freqs`` = (f_axial, f_radial1, f_radial2) [Hz]; ``temps`` = matching 3-vector of
    temperatures [K] (release-recapture usually fits one isotropic T -> pass (T,T,T)).
    """
    if rng is None:
        rng = np.random.default_rng()
    trap_freqs = np.asarray(trap_freqs, float)
    temps = np.asarray(temps, float)
    w = 2.0 * np.pi * trap_freqs                       # angular trap freqs (3,)

    # thermal mean occupation per mode
    nbar = 1.0 / (np.exp(HBAR * w / (KB * temps)) - 1.0)   # (3,)
    # waist from the radial trap frequency + depth (notebook: trap_freqs[1])
    w0 = np.sqrt(KB * trap_depth / (trap_freqs[1] ** 2 * mass * np.pi ** 2))

    # geometric thermal-occupation sampling, vectorised: n = floor(ln(1-u)/ln(1-1/(1+nbar)))
    u = rng.random((reps, 3))
    n = np.floor(np.log1p(-u) / np.log1p(-1.0 / (1.0 + nbar)))   # (reps,3)
    theta = 2.0 * np.pi * rng.random((reps, 3))

    etot = HBAR * w * (n + 0.5)                        # (reps,3) energy per mode
    v_amp = np.sqrt(2.0 * etot / mass)
    v_init = v_amp * np.cos(theta)                     # velocity at release
    x_init = (v_amp * np.sin(theta)) / w               # position at release
    x_fin = x_init + v_init * t_rr                     # ballistic flight

    k_final = 0.5 * mass * np.sum(v_init ** 2, axis=1)         # kinetic energy (reps,)
    z = x_fin[:, 0]                                            # axial coord
    r2sq = x_fin[:, 1] ** 2 + x_fin[:, 2] ** 2                 # radial^2
    w_z = w0 * np.sqrt(1.0 + (lambda_tw * z / (np.pi * w0 ** 2)) ** 2)
    u_final = KB * trap_depth * (1.0 - (w0 ** 2 / w_z ** 2) * np.exp(-2.0 * r2sq / w_z ** 2))
    e_final = k_final + u_final

    bound = e_final <= KB * trap_depth                 # recaptured if below depth
    kept = rng.random(reps) > loss                     # small flat imaging loss
    return float(np.count_nonzero(bound & kept)) / reps


def simulate_curve(times, T, **kw):
    """MC survival curve at one isotropic temperature ``T`` (K) over release ``times`` (s)."""
    trap_freqs = kw.pop("trap_freqs", DEFAULT_TRAP_FREQS)
    temps = (T, T, T)
    return np.array([montecarlo(trap_freqs, temps, t, **kw) for t in times])


# =========================================================================== #
# Temperature fit (weighted RMSE / chi2 sweep, 68% CI from delta-chi2 = 1)
# =========================================================================== #
def fit_release_recapture_temperature(
        times, survival, sem, *,
        temp_sweep=None, trap_freqs=DEFAULT_TRAP_FREQS, mass=M_YB174,
        trap_depth=DEFAULT_TRAP_DEPTH, lambda_tw=DEFAULT_LAMBDA,
        reps=DEFAULT_REPS, loss=DEFAULT_LOSS, seed=42):
    """Sweep isotropic temperature, weighted-chi2 fit to (times, survival +/- sem).

    Returns a result dict (see module docstring). Drops points with non-finite / zero-SEM
    data, exactly like the notebook.
    """
    times = np.asarray(times, float).ravel()
    survival = np.asarray(survival, float).ravel()
    sem = np.asarray(sem, float).ravel()

    good = (np.isfinite(times) & np.isfinite(survival) & np.isfinite(sem) & (sem > 0))
    n_dropped = int((~good).sum())
    if good.sum() < 2:
        raise RuntimeError("Only %d valid scan point(s) after filtering -- cannot fit."
                           % int(good.sum()))
    times, survival, sem = times[good], survival[good], sem[good]

    if temp_sweep is None:
        temp_sweep = np.arange(1.0, 80.0, 1.0) * 1e-6     # 1..79 uK @ 1 uK
    temp_sweep = np.asarray(temp_sweep, float)
    rng = np.random.default_rng(seed)

    curves = np.zeros((len(temp_sweep), len(times)))
    rmse = np.zeros(len(temp_sweep))
    weights = 1.0 / sem ** 2
    wsum = float(np.sum(weights))
    for k, T in enumerate(temp_sweep):
        c = simulate_curve(times, T, trap_freqs=trap_freqs, mass=mass,
                           trap_depth=trap_depth, lambda_tw=lambda_tw, reps=reps,
                           loss=loss, rng=rng)
        curves[k] = c
        rmse[k] = np.sqrt(np.sum(weights * (c - survival) ** 2) / wsum)

    # chi2 normalised so its minimum ~ 1 (notebook convention), then delta-chi2 = 1 CI.
    chi2_raw = rmse ** 2 * wsum
    chi2_min_raw = float(np.min(chi2_raw))
    chi2_red = chi2_min_raw if chi2_min_raw > 0 else 1.0
    chi2 = chi2_raw / chi2_red

    idx_best = int(np.argmin(chi2))
    T_best = float(temp_sweep[idx_best])

    T_low = T_high = T_sigma = float("nan")
    try:
        from scipy.interpolate import PchipInterpolator
        finite = np.isfinite(chi2)
        if finite.sum() >= 2:
            T_grid = np.linspace(temp_sweep.min(), temp_sweep.max(), 2000)
            chi2_grid = PchipInterpolator(temp_sweep[finite], chi2[finite])(T_grid)
            in_ci = chi2_grid <= (chi2.min() + 1.0)
            if in_ci.any():
                T_low = float(T_grid[in_ci].min())
                T_high = float(T_grid[in_ci].max())
                T_sigma = (T_high - T_low) / 2.0
    except Exception:
        pass

    return {
        "T_best": T_best, "T_low": T_low, "T_high": T_high, "T_sigma": T_sigma,
        "chi2_red": chi2_red, "temp_sweep": temp_sweep, "rmse": rmse, "chi2": chi2,
        "curves": curves, "idx_best": idx_best,
        "times": times, "survival": survival, "sem": sem, "n_dropped": n_dropped,
        "trap_freqs": np.asarray(trap_freqs, float), "trap_depth": trap_depth,
        "mass": mass, "lambda_tw": lambda_tw, "reps": reps,
    }


# =========================================================================== #
# Plot (headless matplotlib -> PNG)
# =========================================================================== #
def plot_fit(res, save_path=None, title=None, show=False):
    """Two-panel figure: survival data + nearby MC curves; chi2 vs T with 68% CI."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = res["temp_sweep"]; chi2 = res["chi2"]; curves = res["curves"]
    times = res["times"]; surv = res["survival"]; sem = res["sem"]
    i0 = res["idx_best"]
    lo, hi = max(i0 - 2, 0), min(i0 + 2, len(ts) - 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    ax0.errorbar(1e6 * times, surv, yerr=sem, fmt="o", ms=6, color="#2e8b57",
                 capsize=2, label="data", zorder=5)
    for k in range(lo, hi + 1):
        is_best = (k == i0)
        ax0.plot(1e6 * times, curves[k], lw=2.4 if is_best else 1.3,
                 ls="-" if is_best else "--",
                 label="T=%.1f uK (RMSE=%.3f)%s"
                       % (ts[k] * 1e6, res["rmse"][k], "  *best" if is_best else ""))
    ax0.set_xlabel("Release time (us)"); ax0.set_ylabel("Survival probability")
    ax0.set_title(title or "Release-recapture: MC temperature sweep vs data")
    ax0.legend(fontsize=9); ax0.grid(alpha=0.3)

    ax1.plot(ts * 1e6, chi2, "o", color="C0", ms=5, label="chi2 (sweep)")
    try:
        from scipy.interpolate import PchipInterpolator
        finite = np.isfinite(chi2)
        Tg = np.linspace(ts.min(), ts.max(), 2000)
        ax1.plot(Tg * 1e6, PchipInterpolator(ts[finite], chi2[finite])(Tg), "-",
                 color="C3", lw=1.5, label="interp")
    except Exception:
        pass
    ax1.axhline(chi2.min() + 1.0, ls="--", color="k", lw=1, label="68% CI (dchi2=1)")
    if np.isfinite(res["T_low"]):
        ax1.axvline(res["T_low"] * 1e6, ls=":", color="g")
        ax1.axvline(res["T_high"] * 1e6, ls=":", color="g")
    ax1.axvline(res["T_best"] * 1e6, ls="-", color="g", lw=1.2)
    sig = res["T_sigma"]
    ax1.set_title("chi2 vs T   ->   T = %.1f %s uK"
                  % (res["T_best"] * 1e6, ("+/- %.1f" % (sig * 1e6)) if np.isfinite(sig) else ""))
    ax1.set_xlabel("Temperature (uK)"); ax1.set_ylabel("chi2 (normalised)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


# =========================================================================== #
# Scan loading -> survival curve -> fit -> plot
# =========================================================================== #
def survival_from_scan(path=None):
    """Load a scan (latest if ``path`` is None) and return (times_s, survival, sem, meta)."""
    from yb_analysis.analysis import probabilities as prob
    from yb_analysis.analysis.load_data import load_latest_scan, load_scan_from_path
    from yb_analysis.analysis.unpack import unpack_scan_logicals

    data = load_scan_from_path(path) if path else load_latest_scan()
    Scan = data["Scan"]
    scan_params, logic1, logic2, reps_per_param = unpack_scan_logicals(
        Scan, data["logicals"], data["seq_ids"], mat_path=data.get("mat_path"))
    if logic2 is None:
        raise RuntimeError("Scan has NumImages != 2 (no survival pair) -- not a "
                           "release-recapture survival scan.")
    p11_mean, p11_sem = prob.prob11(logic1, logic2)

    sp = np.asarray(scan_params, float)
    times = sp.ravel() if (sp.ndim == 1 or sp.shape[1] == 1) else sp[:, 0]

    meta = {
        "path": data.get("path"), "scan_dir": os.path.dirname(data.get("path", "") or ""),
        "n_sites": int(logic1.shape[0]), "n_params": len(times),
        "reps_per_param": reps_per_param,
    }
    return times, np.asarray(p11_mean, float), np.asarray(p11_sem, float), meta


def analyze_scan_temperature(path=None, *, save_dir=None, plot=True, **fit_kw):
    """Load a release-recapture scan, fit T, and (optionally) save a PNG into its data folder.

    Extra keyword args are forwarded to :func:`fit_release_recapture_temperature`
    (``trap_freqs``, ``trap_depth``, ``temp_sweep``, ``reps``, ...).
    """
    times, surv, sem, meta = survival_from_scan(path)
    res = fit_release_recapture_temperature(times, surv, sem, **fit_kw)
    res["meta"] = meta

    # report per-point SEM (the <5% acceptance check)
    sem_pct = 100.0 * res["sem"]
    res["sem_max_pct"] = float(np.max(sem_pct))
    res["sem_med_pct"] = float(np.median(sem_pct))

    if plot:
        sd = save_dir or meta.get("scan_dir") or os.getcwd()
        os.makedirs(sd, exist_ok=True)
        fig_path = os.path.join(sd, "release_recapture_temperature.png")
        base = os.path.basename(meta.get("scan_dir") or "")
        plot_fit(res, save_path=fig_path,
                 title="Release-recapture temperature  (%s)" % base)
        res["fig_path"] = fig_path
    return res


def _format(res):
    sig = res["T_sigma"]
    lines = [
        "Release-recapture temperature fit",
        "  data points used : %d  (dropped %d)" % (len(res["times"]), res["n_dropped"]),
        "  per-point SEM    : median %.2f%%, max %.2f%%"
        % (res.get("sem_med_pct", float("nan")), res.get("sem_max_pct", float("nan"))),
        "  trap_freqs (kHz) : %s" % (np.asarray(res["trap_freqs"]) / 1e3),
        "  trap_depth (uK)  : %.1f    lambda (nm): %.0f    MC reps: %d"
        % (res["trap_depth"] * 1e6, res["lambda_tw"] * 1e9, res["reps"]),
        "  reduced chi2     : %.3f" % res["chi2_red"],
        "  >>> T_best = %.1f uK" % (res["T_best"] * 1e6)
        + ("   68%% CI [%.1f, %.1f] uK   (+/- %.1f uK)"
           % (res["T_low"] * 1e6, res["T_high"] * 1e6, sig * 1e6) if np.isfinite(sig) else ""),
    ]
    if "fig_path" in res:
        lines.append("  figure           : %s" % res["fig_path"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Release-recapture atom-temperature fit (port of GeneralAnalysis.ipynb).")
    ap.add_argument("--path", default=None,
                    help="scan data dir (default: latest scan via load_latest_scan)")
    ap.add_argument("--save-dir", default=None,
                    help="where to write the PNG (default: the scan's data folder)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    ap.add_argument("--t-uK", type=float, nargs=3, metavar=("LO", "HI", "STEP"),
                    default=(1.0, 80.0, 1.0), help="temperature sweep in uK (default 1 80 1)")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS, help="MC atoms per point")
    ap.add_argument("--trap-freqs", type=float, nargs=3, metavar=("FAX", "FR1", "FR2"),
                    default=DEFAULT_TRAP_FREQS, help="trap freqs [Hz] (default 15e3 82e3 82e3)")
    ap.add_argument("--trap-depth-uK", type=float, default=DEFAULT_TRAP_DEPTH * 1e6,
                    help="trap depth in uK (default 390)")
    ap.add_argument("--mass-amu", type=float, default=M_YB174 / 1.66054e-27,
                    help="atomic mass in amu (default 174Yb)")
    args = ap.parse_args(argv)

    lo, hi, step = args.t_uK
    temp_sweep = np.arange(lo, hi, step) * 1e-6
    res = analyze_scan_temperature(
        path=args.path, save_dir=args.save_dir, plot=not args.no_plot,
        temp_sweep=temp_sweep, reps=args.reps, trap_freqs=tuple(args.trap_freqs),
        trap_depth=args.trap_depth_uK * 1e-6, mass=args.mass_amu * 1.66054e-27)
    print(_format(res))
    return res


if __name__ == "__main__":
    main()
