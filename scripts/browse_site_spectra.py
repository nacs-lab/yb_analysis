"""Interactive per-site spectrum browser for a tweezer-array survival scan.

Inspect EACH site's spectrum line one at a time:

  Left panel  : the array (grid X/Y), sites colored by per-site peak survival.
                Click any site to inspect it (nearest site to the click is picked).
  Right panel : that site's survival P(img2=1|img1=1) vs the swept parameter, with its
                Lorentzian fit, the array grand-mean (faint) for reference, and the
                fitted center / FWHM / R2.

Step through sites
  right / left arrow  : next / previous site (by index)
  up / down arrow     : jump +/- 10 sites
  n / N               : next / prev site that HAS a usable fit
  j                   : JUMP to a site index (typed in the terminal) -- use this
                        if clicking does nothing (Qt toolbar zoom/pan can eat clicks)
Other keys
  g : toggle showing the array grand-mean on the right panel
  s : save the current site's panel -> <prefix>_site<idx>.png
  q : quit

Works on a finished scan OR a still-running one (reads the growing h5 with a retry loop).

Usage
  python -m yb_analysis.scripts.browse_site_spectra <SCAN_DIR>
"""
import os
import time
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
# keep the nav toolbar: with it removed, the Qt canvas did not receive mouse/key events
# (no focus) so site selection silently did nothing. zoom/pan don't conflict here.
for _k in ("keymap.save", "keymap.fullscreen", "keymap.pan", "keymap.yscale",
           "keymap.xscale", "keymap.all_axes", "keymap.grid", "keymap.home",
           "keymap.back", "keymap.forward"):
    try:
        matplotlib.rcParams[_k] = []
    except Exception:
        pass
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from yb_analysis.analysis.load_data import load_scan_from_path
from yb_analysis.analysis.unpack import unpack_scan_logicals
from yb_analysis.analysis.probabilities import prob11_site_resolved


def _read_two_array(scan_dir):
    """Return (scan_json, seq_ids, logic_img1, logic_img2) for a finished OR running scan.

    Tries the normal loader first; if the logicals come back unpopulated (live scan), reads
    the growing data_*.h5 directly with file-locking off + a retry loop, snapshotting counts."""
    bundle = load_scan_from_path(scan_dir)
    scan = bundle.get("Scan") or {}
    l1, l2 = bundle.get("logicals_img1"), bundle.get("logicals_img2")
    sid = bundle.get("seq_ids")
    if l1 is not None and l2 is not None and sid is not None and len(sid):
        return scan, np.asarray(sid), np.asarray(l1), np.asarray(l2)

    # live fallback: read the h5 by hand
    import h5py
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    h5 = None
    for fn in os.listdir(scan_dir):
        if fn.endswith(".h5"):
            h5 = os.path.join(scan_dir, fn); break
    if h5 is None:
        raise SystemExit("no .h5 in %s" % scan_dir)
    last = None
    for _ in range(40):
        try:
            with h5py.File(h5, "r", swmr=True) as f:
                if not ("logicals_img1" in f and "logicals_img2" in f):
                    raise SystemExit("h5 is not a two-array (img1/img2) survival scan")
                n = min(f["seq_ids"].shape[0],
                        f["logicals_img1"].shape[0], f["logicals_img2"].shape[0])
                return scan, f["seq_ids"][:n], f["logicals_img1"][:n], f["logicals_img2"][:n]
        except SystemExit:
            raise
        except Exception as e:
            last = e
            time.sleep(0.25)
    raise RuntimeError("could not read live h5 after retries: %r" % last)


def load_cube(scan_dir):
    scan, sid, l1, l2 = _read_two_array(scan_dir)
    sp, logic1, logic2, reps = unpack_scan_logicals(
        scan, seq_ids=sid, logicals_img1=l1, logicals_img2=l2)
    sp = np.asarray(sp, float)
    axis = sp.reshape(sp.shape[0], -1)[:, 0]
    scale = 1e6 if np.nanmax(np.abs(axis)) > 1e4 else 1.0
    x = axis / scale
    order = np.argsort(x)
    x = x[order]
    mean_sr, sem_sr = prob11_site_resolved(logic1, logic2)
    mean_sr = mean_sr[:, order]
    sem_sr = sem_sr[:, order]
    gx = np.asarray(scan.get("initGridLocationsX"), float).reshape(-1)
    gy = np.asarray(scan.get("initGridLocationsY"), float).reshape(-1)
    nS = mean_sr.shape[0]
    if gx.size != nS or gy.size != nS:
        gx = np.arange(nS, dtype=float); gy = np.zeros(nS)   # fall back to index layout
    name = str(scan.get("scan_id") or os.path.basename(scan_dir))
    xlabel = "swept param" + (" (MHz)" if scale != 1.0 else "")
    return x, mean_sr, sem_sr, gx, gy, name, xlabel, int(reps.max()), len(sid)


def lorentzian(x, y0, A, x0, w):
    return y0 + A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def fit_site(x, y):
    m = np.isfinite(y)
    if m.sum() < 6:
        return None
    xx, yy = x[m], y[m]
    y0g = np.percentile(yy, 25); i = int(np.argmax(yy)); Ag = yy[i] - y0g
    try:
        p, _ = curve_fit(lorentzian, xx, yy, p0=[y0g, max(Ag, 0.05), xx[i], 8.0],
                         bounds=([0, 0, x.min(), 1e-6], [1, 1.5, x.max(), np.ptp(x)]), maxfev=10000)
    except Exception:
        return None
    sst = np.sum((yy - yy.mean()) ** 2)
    r2 = 1 - np.sum((yy - lorentzian(xx, *p)) ** 2) / sst if sst > 0 else 0
    return dict(p=p, r2=r2)


class SiteBrowser:
    def __init__(self, x, M, S, gx, gy, name, xlabel, prefix, maxreps, nshot):
        self.x, self.M, self.S = x, M, S
        self.gx, self.gy = gx, gy
        self.name, self.xlabel, self.prefix = name, xlabel, prefix
        self.maxreps, self.nshot = maxreps, nshot
        self.nS = M.shape[0]
        self.pts = np.column_stack([gx, gy])
        self.show_grand = True
        # per-site fits (precomputed for navigation + the array colormap)
        self.fits = [fit_site(x, M[s]) for s in range(self.nS)]
        self.peak = np.array([(f["p"][0] + f["p"][1]) if f else np.nan for f in self.fits])
        self.has_fit = np.array([f is not None and f["r2"] >= 0.3 for f in self.fits])
        self.grand = np.nanmean(M, axis=0)
        self.cur = int(np.argmax(np.where(self.has_fit, self.peak, -1)))  # start on a bright fit

        # Use plt.subplots (matches the working lasso selector tool) so the Qt canvas wires
        # its event loop the same proven way; bare figure()+add_axes left events undelivered.
        self.fig, (self.axL, self.axR) = plt.subplots(1, 2, figsize=(14.5, 6.8))
        try:
            self.fig.canvas.manager.set_window_title("site spectrum browser -- %s" % name)
        except Exception:
            pass
        self._draw_array()
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        print("connected events: click cid=%s key cid=%s" % (self.cid_click, self.cid_key), flush=True)
        # Qt canvas needs keyboard focus to deliver key_press_event. The focus-policy enum
        # is scoped in Qt6 (Qt.FocusPolicy.StrongFocus) but flat in Qt5 (Qt.StrongFocus) --
        # the flat path raised 'Qt has no attribute StrongFocus' on this PyQt6 build, which
        # left the canvas focusless so clicks+keys were silently dropped. Try both.
        try:
            from matplotlib.backends.qt_compat import QtCore
            try:
                pol = QtCore.Qt.FocusPolicy.StrongFocus      # Qt6
            except AttributeError:
                pol = QtCore.Qt.StrongFocus                  # Qt5
            self.fig.canvas.setFocusPolicy(pol)
            self.fig.canvas.setFocus()
            win = self.fig.canvas.window() if hasattr(self.fig.canvas, "window") else None
            if win is not None:
                win.activateWindow(); win.raise_()
            print("focus set OK", flush=True)
        except Exception as e:
            print("focus set failed:", e, flush=True)
        self.fig.suptitle("click a site  |  <-/-> step  up/down +-10  n/N next-prev fitted  "
                          "g=grand-mean  s=save  q=quit", fontsize=9)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._draw_site()

    def _draw_array(self):
        self.axL.clear()
        finite = np.isfinite(self.peak)
        self.axL.scatter(self.gx[~finite], self.gy[~finite], c="#ddd", s=12)
        sc = self.axL.scatter(self.gx[finite], self.gy[finite], c=self.peak[finite], s=16,
                              cmap="viridis")
        if not hasattr(self, "_cb"):
            self._cb = self.fig.colorbar(sc, ax=self.axL, fraction=0.046)
            self._cb.set_label("per-site peak survival")
        self.cur_marker, = self.axL.plot([], [], "o", ms=14, mfc="none", mec="red", mew=2.0)
        self.axL.set_aspect("equal"); self.axL.invert_yaxis()
        self.axL.set_xlabel("grid X (px)"); self.axL.set_ylabel("grid Y (px)")
        self.axL.set_title("array (%d sites)  --  click to inspect" % self.nS)

    def _draw_site(self):
        s = self.cur
        self.cur_marker.set_data([self.gx[s]], [self.gy[s]])
        self.axR.clear()
        if self.show_grand:
            self.axR.plot(self.x, self.grand, color="#bbb", lw=1.2, label="array grand mean")
        y, e = self.M[s], self.S[s]
        self.axR.errorbar(self.x, y, yerr=e, fmt="o", ms=4, color="#1f77b4",
                          ecolor="#9ec5e8", elinewidth=0.8, capsize=2,
                          label="site %d" % s, zorder=4)
        f = self.fits[s]
        if f is not None:
            xf = np.linspace(self.x.min(), self.x.max(), 800)
            p, r2 = f["p"], f["r2"]
            self.axR.plot(xf, lorentzian(xf, *p), "r--", lw=1.8,
                          label="fit x0=%.2f FWHM=%.2f R2=%.2f" % (p[2], abs(p[3]), r2))
        self.axR.set_xlabel(self.xlabel)
        self.axR.set_ylabel("survival  P(img2=1|img1=1)")
        ff = "fit ok" if self.has_fit[s] else "no usable fit"
        self.axR.set_title("site %d / %d   [%s]   (%s, %d shots, <=%d reps)"
                           % (s, self.nS - 1, ff, self.name, self.nshot, self.maxreps))
        self.axR.set_ylim(-0.05, 1.05)
        self.axR.grid(alpha=0.25); self.axR.legend(fontsize=8, loc="upper left")
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        # accept the click if it lands in the array axes OR carries data coords for it
        if ev.xdata is None or ev.ydata is None:
            print("click: no data coords (inaxes=%r)" % (ev.inaxes,), flush=True)
            return
        if ev.inaxes is not None and ev.inaxes is not self.axL:
            return                      # clicked the right panel -> ignore
        d = (self.gx - ev.xdata) ** 2 + (self.gy - ev.ydata) ** 2
        self.cur = int(np.argmin(d))
        print("click -> site %d (x=%.0f y=%.0f)" % (self.cur, ev.xdata, ev.ydata), flush=True)
        self._draw_site()

    def _step_fitted(self, direction):
        idxs = np.where(self.has_fit)[0]
        if not idxs.size:
            return
        nxt = idxs[idxs > self.cur] if direction > 0 else idxs[idxs < self.cur]
        self.cur = int(nxt[0] if direction > 0 else nxt[-1]) if nxt.size else int(
            idxs[0] if direction > 0 else idxs[-1])
        self._draw_site()

    def on_key(self, ev):
        if ev.key in ("right", "left", "up", "down"):
            step = {"right": 1, "left": -1, "up": 10, "down": -10}[ev.key]
            self.cur = int(np.clip(self.cur + step, 0, self.nS - 1))
            self._draw_site()
        elif ev.key == "n":
            self._step_fitted(+1)
        elif ev.key == "N":
            self._step_fitted(-1)
        elif ev.key == "j":
            # Jump to a site by index (typed in the terminal) -- click-independent,
            # works even when the Qt nav-toolbar zoom/pan mode swallows clicks.
            try:
                raw = input("jump to site index (0-%d): " % (self.nS - 1))
                idx = int(raw.strip())
                if 0 <= idx < self.nS:
                    self.cur = idx
                    self._draw_site()
                else:
                    print("out of range", flush=True)
            except (ValueError, EOFError):
                print("bad index", flush=True)
        elif ev.key == "g":
            self.show_grand = not self.show_grand; self._draw_site()
        elif ev.key == "s":
            out = "%s_site%d.png" % (self.prefix, self.cur)
            self.fig.savefig(out, dpi=140)
            print("saved:", out, flush=True)
        elif ev.key == "q":
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", help="two-array survival scan folder (finished or running)")
    args = ap.parse_args()
    x, M, S, gx, gy, name, xlabel, maxreps, nshot = load_cube(args.scan_dir)
    prefix = os.path.join(args.scan_dir, "sitespec_%s" % name)
    print("loaded %d sites, %d params (%.3f..%.3f), %d shots <=%d reps. prefix=%s"
          % (M.shape[0], M.shape[1], x.min(), x.max(), nshot, maxreps, prefix))
    SiteBrowser(x, M, S, gx, gy, name, xlabel, prefix, maxreps, nshot)
    plt.show()


if __name__ == "__main__":
    main()
