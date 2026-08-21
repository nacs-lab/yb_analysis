"""Interactive spatial-subset selector for STIRAP verify-conditioned excitation (+ its variance).

This is HOW we check STIRAP efficiency AND its spatial variance across a rearranged array. The
forward-STIRAP push-out drives atoms to Rydberg (then field-ionized -> lost), so excitation =
1 - survival. On a 3-image rearrange-STIRAP scan the honest, drift-clean metric is the
VERIFY-conditioned survival ``P(final=1 | mid=1)`` over sites OCCUPIED IN THE MID (verify) frame
(``logicals_mid`` = post-rearrange verify, ``logicals_img2`` = post-pushout final). This tool maps
that per-site, RESTRICTED TO THE REARRANGEMENT-TARGET SITES (the sites atoms were moved INTO, from
``slm_diag.h5`` ``target_paired``) so untargeted leftovers -- stragglers that also light up in
mid/img2 and excite worse -- do not contaminate the number or the map.

Adapted from ``select_subset_survival`` (which is img1-conditioned + swept-spectrum); this one is
mid-conditioned and built for a FIXED-POINT verify (no swept axis), so the right panel is a
bar of the SELECTION vs the all-target reference, not a spectrum.

  * Left panel  : the array; TARGET sites colored by per-site excitation 1-P(final|mid) (high=good),
                  non-target sites faint grey (geometry context, not selectable/counted).
  * Right panel : selection vs all-target pooled excitation +/- per-shot SEM (+ mid-event count).
    A cold pocket (a region the STIRAP beam under-addresses) shows up as a low-excitation cluster;
    that spatial variance is the remaining lever once the pulse is optimal (beam uniformity/pointing).

Select : lasso-drag (default) | p=polygon | l=lasso | r=reset(all targets) | s/Save | q=quit
         (auto-saves the last non-trivial selection on window close).
Saves  : <SCAN_DIR>/subset_stirap_target_<scan_id>_{selection.png,selected_idx.npy,mask.npy}.

Usage
  # interactive (needs a display on the exp-control machine):
  python -m yb_analysis.scripts.select_subset_stirap <SCAN_DIR>
  python -m yb_analysis.scripts.select_subset_stirap <SCAN_DIR> --all-sites   # skip target restriction
  # headless (no GUI) -- print all-target / non-target / whole-array excitation + save the map PNG:
  python -m yb_analysis.scripts.select_subset_stirap <SCAN_DIR> --dump
"""
import os
import argparse
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from yb_analysis.analysis.load_data import load_scan_from_path


def _read_three(scan_dir):
    """(scan_json, seq_ids, mid, final) -- logicals_mid (verify) + logicals_img2 (final)."""
    import time, h5py
    b = load_scan_from_path(scan_dir)
    scan = b.get("Scan") or {}
    h5 = next((os.path.join(scan_dir, f) for f in os.listdir(scan_dir)
               if f.endswith(".h5") and not f.startswith("slm_diag")), None)
    if h5 is None:
        raise SystemExit("no data .h5 in %s" % scan_dir)
    last = None
    for _ in range(40):
        try:
            with h5py.File(h5, "r", swmr=True) as f:
                if "logicals_mid" not in f or "logicals_img2" not in f:
                    raise SystemExit("not a 3-image verify scan (need logicals_mid + logicals_img2)")
                n = min(f["logicals_mid"].shape[0], f["logicals_img2"].shape[0], f["seq_ids"].shape[0])
                return (scan, f["seq_ids"][:n],
                        f["logicals_mid"][:n].astype(bool), f["logicals_img2"][:n].astype(bool))
        except SystemExit:
            raise
        except Exception as e:
            last = e
            time.sleep(0.25)
    raise RuntimeError("could not read h5 after retries: %r" % last)


def target_mask(scan_dir, seq_ids, nS):
    """Per-site 'was EVER a rearrangement target' bool (union of slm_diag target_paired over the
    scan's shots), or None if no diag. Only the sites atoms were moved INTO -- excludes untargeted
    leftovers that also show up in mid/img2."""
    import h5py
    diag = os.path.join(scan_dir, "slm_diag.h5")
    if not os.path.exists(diag):
        print("no slm_diag.h5 -> cannot restrict to target sites"); return None
    with h5py.File(diag, "r", swmr=True) as f:
        g = f["/diag"]
        d_sid = np.asarray(g["seq_id"][:], np.int64)
        tp = g["target_paired"]
        dmap = {int(s): np.asarray(tp[i]).ravel() for i, s in enumerate(d_sid)}
    m = np.zeros(nS, bool)
    matched = 0
    for s in np.asarray(seq_ids, np.int64):
        t = dmap.get(int(s))
        if t is None:
            continue
        matched += 1
        t = t[(t >= 0) & (t < nS)]
        m[t] = True
    print("target mask: %d target sites (union over %d/%d shots with diag)"
          % (int(m.sum()), matched, len(seq_ids)))
    return m


def target_mask_union(diag_dir, nS):
    """Per-site 'ever a target' bool = union of ALL slm_diag target_paired rows in diag_dir,
    NOT seq-matched. Use when THIS run has no slm_diag but a SIBLING run of the SAME target
    array does (the target-site set is identical across runs of the same rearrangement)."""
    import h5py
    diag = os.path.join(diag_dir, "slm_diag.h5")
    if not os.path.exists(diag):
        print("no slm_diag.h5 in %s" % diag_dir); return None
    with h5py.File(diag, "r", swmr=True) as f:
        tp = f["/diag"]["target_paired"]
        m = np.zeros(nS, bool)
        for i in range(tp.shape[0]):
            t = np.asarray(tp[i]).ravel()
            t = t[(t >= 0) & (t < nS)]
            m[t] = True
    print("target mask (union from sibling %s): %d target sites" % (os.path.basename(diag_dir), int(m.sum())))
    return m


def load(scan_dir, target_only=True, target_from=None):
    scan, seq_ids, mid, fin = _read_three(scan_dir)
    gx = np.asarray(scan.get("initGridLocationsX"), float).reshape(-1)
    gy = np.asarray(scan.get("initGridLocationsY"), float).reshape(-1)
    nS = mid.shape[1]
    if gx.size != nS or gy.size != nS:
        raise SystemExit("grid X/Y (%d/%d) != site count %d" % (gx.size, gy.size, nS))
    name = str(scan.get("scan_id") or os.path.basename(scan_dir))
    if not target_only:
        tmask = None
    elif target_from:                          # borrow the target union from a sibling run's diag
        tmask = target_mask_union(target_from, nS)
    else:
        tmask = target_mask(scan_dir, seq_ids, nS)
    return mid, fin, gx, gy, name, tmask, scan, seq_ids


def sweep_axis(scan_dir, scan, seq_ids):
    """(p_of_shot 0-based, x values, axis name) for a SWEPT scan, or (None, None, None) for a
    fixed-point run. Params maps seq_id (1-indexed) -> flat param index (1-indexed); axis values
    come from the cached analysis payload (flat-param order) when present, else the param index."""
    import json as _json
    _P = scan.get("Params")
    P = np.asarray([] if _P is None else _P, int)
    if P.size == 0 or int(P.max()) <= 1:
        return None, None, None
    n_p = int(P.max())
    li = np.asarray(seq_ids, np.int64) - 1
    p_of_shot = np.where((li >= 0) & (li < P.size), P[np.clip(li, 0, P.size - 1)] - 1, -1)
    x, aname = np.arange(n_p, dtype=float), "scan point #"
    pj = os.path.join(scan_dir, "analysis_payload.json")
    try:
        with open(pj) as f:
            pl = _json.load(f).get("payload", {})
        sw = pl.get("sweep") or {}
        v = np.asarray((sw.get("values") or [[]])[0], float)
        if v.size == n_p:
            x = v
            aname = str((sw.get("cols") or ["swept param"])[0])
    except Exception:
        pass
    return p_of_shot, x, aname


def subset_curve(mid, fin, mask, p_of_shot, n_p):
    """Pooled P(final|mid) +/- binomial SEM per scan point, over the subset sites."""
    m = mid[:, mask]; fq = fin[:, mask]
    num = np.zeros(n_p); den = np.zeros(n_p)
    for p in range(n_p):
        sh = p_of_shot == p
        if not sh.any():
            continue
        num[p] = (m[sh] & fq[sh]).sum(); den[p] = m[sh].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(den > 0, num / den, np.nan)
        e = np.where(den > 0, np.sqrt(np.maximum(y * (1 - y), 0) / np.maximum(den, 1)), np.nan)
    return y, e


def subset_stats(mid, fin, mask):
    """Pooled + per-shot verify-conditioned excitation over subset sites occupied in mid."""
    mid_s = mid[:, mask]; fin_s = fin[:, mask]
    events = int(mid_s.sum())
    if events == 0:
        return dict(exc=np.nan, sem=np.nan, events=0, shots=0, pooled=np.nan)
    num = (mid_s & fin_s).sum(1); den = mid_s.sum(1).astype(float)
    ok = den > 0
    ps = 1.0 - num[ok] / den[ok]           # per-shot excitation
    exc = float(ps.mean())
    sem = float(ps.std(ddof=1) / np.sqrt(ps.size)) if ps.size > 1 else float("nan")
    pooled = 1.0 - (mid_s & fin_s).sum() / mid_s.sum()
    return dict(exc=exc, sem=sem, events=events, shots=int(ok.sum()), pooled=float(pooled))


def per_site_excitation(mid, fin):
    """Per-site excitation 1-P(final|mid); NaN where a site was never occupied in mid."""
    loaded = mid.sum(0).astype(float)
    joint = (mid & fin).sum(0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        surv = np.where(loaded > 0, joint / loaded, np.nan)
    return 1.0 - surv


def dump(scan_dir, target_only=True, target_from=None):
    """Headless: print all-target / non-target / whole-array excitation, save a static map PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mid, fin, gx, gy, name, tmask, _scan, _seq = load(scan_dir, target_only=target_only,
                                                      target_from=target_from)
    nS = mid.shape[1]
    tmask = tmask if tmask is not None else np.ones(nS, bool)
    exc = per_site_excitation(mid, fin) * 100
    for lbl, m in (("ALL TARGET", tmask),
                   ("NON-target (occupied)", (~tmask) & (mid.sum(0) > 0)),
                   ("WHOLE array", np.ones(nS, bool))):
        st = subset_stats(mid, fin, m)
        print("%-22s excitation %.2f%% pooled, %.2f +/- %.2f%% per-shot (%d mid events)"
              % (lbl, st["pooled"] * 100, st["exc"] * 100, st["sem"] * 100, st["events"]))
    show = tmask & np.isfinite(exc)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.scatter(gx[~tmask], gy[~tmask], c="#dddddd", s=6, marker=".")
    sc = ax.scatter(gx[show], gy[show], c=exc[show], s=22, cmap="viridis",
                    vmin=np.nanpercentile(exc[show], 3), vmax=np.nanpercentile(exc[show], 97))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("grid X (px)"); ax.set_ylabel("grid Y (px)")
    st = subset_stats(mid, fin, tmask)
    ax.set_title("STIRAP excitation per TARGET site (mid->img3)  %s\n"
                 "%d target sites, all-target %.2f%%" % (name, int(tmask.sum()), st["pooled"] * 100))
    plt.colorbar(sc, ax=ax, label="excitation 1-P(final|mid) (%)")
    out = os.path.join(scan_dir, "fig_v_stirap_target_map.png")
    fig.text(0.01, 0.005, out, fontsize=6, color="0.4")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("saved map ->", out)


class StirapSelector:
    def __init__(self, mid, fin, gx, gy, name, prefix, tmask=None, sweep=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import LassoSelector, Button  # noqa: F401 (import check)
        self.plt = plt
        self.mid, self.fin = mid, fin           # (shots, nS) bool
        self.gx, self.gy = gx, gy
        self.name = name; self.prefix = prefix
        self.nS = mid.shape[1]
        self.pts = np.column_stack([gx, gy])
        # target sites atoms were rearranged INTO (union of diag target_paired). When set, ONLY these
        # sites are shown/selectable/counted -- excludes untargeted leftovers seen in mid/img2.
        self.target = tmask if tmask is not None else np.ones(self.nS, bool)
        self.mask = self.target.copy()          # start = all target sites
        self._saved_once = False
        self.exc_site = per_site_excitation(mid, fin)
        # Swept-scan mode: sweep = (p_of_shot, x_values, axis_name). The right panel then shows
        # the subset's 1-D scan curve (e.g. the Rabi oscillation) instead of the fixed-point bar.
        self.sweep = sweep if (sweep and sweep[0] is not None) else None

        self.fig = plt.figure(figsize=(13.5, 7.0))
        self.fig.canvas.manager.set_window_title("STIRAP subset (mid->img3) -- %s" % name)
        self.axL = self.fig.add_axes([0.05, 0.13, 0.44, 0.78])
        self.axR = self.fig.add_axes([0.60, 0.16, 0.36, 0.72])
        ax_btn = self.fig.add_axes([0.22, 0.03, 0.10, 0.05])
        self.btn_save = Button(ax_btn, "Save"); self.btn_save.on_clicked(lambda _e: self.save())
        self._draw_array(); self._draw_panel(); self._connect_lasso()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.suptitle("LASSO a loop = select TARGET sites | metric = STIRAP excitation "
                          "1-P(final|mid), mid-occupied | p/l r s/Save q  [auto-saves on close]",
                          fontsize=9)

    def _draw_array(self):
        self.axL.clear()
        v = np.where(self.target, self.exc_site, np.nan)   # color TARGET sites only
        show = self.target & np.isfinite(v)
        nontgt = ~self.target
        if nontgt.any():
            self.axL.scatter(self.gx[nontgt], self.gy[nontgt], c="#dddddd", s=6, marker=".")
        vv = v[show]
        self.sc = self.axL.scatter(self.gx[show], self.gy[show], c=vv, s=20, cmap="viridis",
                                   vmin=np.nanpercentile(vv, 2) if vv.size else 0,
                                   vmax=np.nanpercentile(vv, 98) if vv.size else 1)
        self.sel_marker = self.axL.scatter(self.gx[self.mask], self.gy[self.mask],
                                           s=46, facecolors="none", edgecolors="red", linewidths=1.1)
        if not hasattr(self, "_cb"):
            self._cb = self.fig.colorbar(self.sc, ax=self.axL, fraction=0.046)
            self._cb.set_label("per-site STIRAP excitation  1-P(final|mid)")
        self.axL.set_aspect("equal"); self.axL.invert_yaxis()
        self.axL.set_xlabel("grid X (px)"); self.axL.set_ylabel("grid Y (px)")
        self._set_title()

    def _set_title(self):
        self.axL.set_title("TARGET sites (%d of %d) -- %d selected"
                           % (int(self.target.sum()), self.nS, self.mask.sum()))

    def _draw_panel(self):
        if self.sweep is not None:
            return self._draw_curve()
        self.axR.clear()
        ref = subset_stats(self.mid, self.fin, self.target)   # reference = ALL target sites
        sel = subset_stats(self.mid, self.fin, self.mask)
        labels = ["all targets\n(%d sites)" % int(self.target.sum()),
                  "selection\n(%d sites)" % self.mask.sum()]
        vals = [ref["exc"] * 100, sel["exc"] * 100]
        errs = [ref["sem"] * 100, sel["sem"] * 100]
        bars = self.axR.bar(labels, vals, yerr=errs, capsize=6, color=["#bbbbbb", "#d62728"])
        for b, v, e, st in zip(bars, vals, errs, (ref, sel)):
            txt = "%.2f%%" % v if np.isfinite(v) else "n/a"
            if np.isfinite(e):
                txt += "\n+/-%.2f (%d ev)" % (e, st["events"])
            self.axR.text(b.get_x() + b.get_width() / 2, (v if np.isfinite(v) else 0) + 0.4,
                          txt, ha="center", va="bottom", fontsize=10)
        self.axR.set_ylabel("STIRAP excitation  1 - P(final|mid)  (%)")
        self.axR.set_title("selection vs all-target excitation")
        lo = min([v for v in vals if np.isfinite(v)] + [90.0])
        self.axR.set_ylim(max(0, lo - 6), 100)
        self.axR.grid(alpha=0.25, axis="y")

    def _draw_curve(self):
        """Swept-scan right panel: subset 1-D scan curve (P(final|mid) vs the swept axis),
        selection (red) overlaid on the all-target reference (grey)."""
        p_of_shot, x, aname = self.sweep
        n_p = x.size
        # time-like axes stored in seconds -> plot in us
        xs, unit = (1e6, " [us]") if (np.nanmax(np.abs(x)) < 1e-2 and np.nanmax(np.abs(x)) > 0) \
            else (1.0, "")
        self.axR.clear()
        yr, er = subset_curve(self.mid, self.fin, self.target, p_of_shot, n_p)
        ys, es = subset_curve(self.mid, self.fin, self.mask, p_of_shot, n_p)
        self.axR.errorbar(x * xs, yr, yerr=er, fmt="o-", ms=3, lw=0.7, color="0.65", capsize=2,
                          label="all targets (%d sites)" % int(self.target.sum()))
        self.axR.errorbar(x * xs, ys, yerr=es, fmt="o-", ms=3.5, lw=0.9, color="#d62728", capsize=2,
                          label="selection (%d sites)" % int(self.mask.sum()))
        self.axR.set_xlabel(aname + unit)
        self.axR.set_ylabel("survival  P(final|mid)")
        self.axR.set_title("subset 1-D scan curve")
        self.axR.legend(fontsize=8)
        self.axR.grid(alpha=0.25)

    def refresh(self):
        self.sel_marker.set_offsets(np.column_stack([self.gx[self.mask], self.gy[self.mask]]))
        self._set_title(); self._draw_panel(); self.fig.canvas.draw_idle()

    def _on_select(self, verts):
        from matplotlib.path import Path
        if len(verts) < 3:
            return
        m = Path(verts).contains_points(self.pts) & self.target   # restrict to TARGET sites
        print("lasso/poly: %d target sites in loop" % int(m.sum()), flush=True)
        if m.sum() < 1:
            print("  (ignored: no target sites in loop)", flush=True); return
        self.mask = m
        st = subset_stats(self.mid, self.fin, m)
        print("  -> excitation %.2f +/- %.2f %% (%d mid events, %d shots)"
              % (st["exc"] * 100, st["sem"] * 100, st["events"], st["shots"]), flush=True)
        self.refresh()

    def _connect_lasso(self):
        from matplotlib.widgets import LassoSelector
        self._disconnect()
        self.selector = LassoSelector(self.axL, onselect=self._on_select, useblit=False)

    def _connect_polygon(self):
        from matplotlib.widgets import PolygonSelector
        self._disconnect()
        self.selector = PolygonSelector(self.axL, onselect=self._on_select, useblit=False)

    def _disconnect(self):
        sel = getattr(self, "selector", None)
        if sel is not None:
            for fn in ("disconnect_events", "set_active"):
                try:
                    getattr(sel, fn)(False) if fn == "set_active" else getattr(sel, fn)()
                except Exception:
                    pass
            self.selector = None

    def on_key(self, ev):
        if ev.key == "r":
            self.mask = self.target.copy(); self.refresh()   # reset = all target sites
        elif ev.key == "p":
            self._connect_polygon(); print("polygon mode")
        elif ev.key == "l":
            self._connect_lasso(); print("lasso mode")
        elif ev.key == "s":
            self.save()
        elif ev.key == "q":
            self.plt.close(self.fig)

    def save(self):
        idx = np.where(self.mask)[0]
        np.save(self.prefix + "_selected_idx.npy", idx)
        np.save(self.prefix + "_mask.npy", self.mask)
        self.fig.savefig(self.prefix + "_selection.png", dpi=140)
        st = subset_stats(self.mid, self.fin, self.mask)
        print("saved: %d sites -> %s_{selection.png,selected_idx.npy,mask.npy} | "
              "excitation %.2f +/- %.2f %% (%d ev)"
              % (idx.size, self.prefix, st["exc"] * 100, st["sem"] * 100, st["events"]), flush=True)
        self._saved_once = True

    def on_close(self, _ev):
        # auto-save the last non-trivial selection (not the full target set, not already saved)
        if not self._saved_once and 0 < self.mask.sum() < int(self.target.sum()):
            print("window closed -- auto-saving last selection", flush=True); self.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir")
    ap.add_argument("--all-sites", action="store_true",
                    help="show ALL sites (skip the target-only restriction)")
    ap.add_argument("--dump", action="store_true",
                    help="headless: print excitation numbers + save the target map PNG, no GUI")
    ap.add_argument("--target-from", default=None,
                    help="borrow the target-site union from a SIBLING run's slm_diag (same target array) "
                         "when THIS run has no slm_diag.h5. Pass the sibling scan_dir.")
    ap.add_argument("--no-curve", action="store_true",
                    help="force the fixed-point BAR right panel even on a swept scan "
                         "(default: swept scans show the subset 1-D scan curve)")
    a = ap.parse_args()
    if a.dump:
        dump(a.scan_dir, target_only=not a.all_sites, target_from=a.target_from)
        return
    import matplotlib
    matplotlib.use("QtAgg")
    matplotlib.rcParams["toolbar"] = "None"   # left-drag is ALWAYS a lasso, never pan/zoom
    for _k in ("keymap.save", "keymap.fullscreen", "keymap.pan", "keymap.yscale",
               "keymap.xscale", "keymap.all_axes", "keymap.grid", "keymap.home",
               "keymap.quit_all"):
        try:
            matplotlib.rcParams[_k] = []
        except Exception:
            pass
    import matplotlib.pyplot as plt
    mid, fin, gx, gy, name, tmask, scan, seq_ids = load(a.scan_dir, target_only=not a.all_sites,
                                                        target_from=a.target_from)
    sweep = None if a.no_curve else sweep_axis(a.scan_dir, scan, seq_ids)
    if sweep and sweep[0] is not None:
        print("swept scan detected (%s, %d pts) -> right panel = subset 1-D scan curve"
              % (sweep[2], sweep[1].size))
    prefix = os.path.join(a.scan_dir, "subset_stirap_target_%s" % name)
    sel = StirapSelector(mid, fin, gx, gy, name, prefix, tmask=tmask, sweep=sweep)
    st = subset_stats(mid, fin, sel.target)
    print("loaded %d sites (%d target), %d shots. all-target excitation %.2f +/- %.2f %% "
          "(%d mid events). prefix=%s"
          % (mid.shape[1], int(sel.target.sum()), mid.shape[0], st["exc"] * 100, st["sem"] * 100,
             st["events"], prefix))
    plt.show()


if __name__ == "__main__":
    main()
