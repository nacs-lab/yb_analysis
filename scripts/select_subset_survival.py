"""Interactive spatial-subset survival selector for a tweezer-array scan.

Pick any spatial subset of the tweezer array (e.g. the sites a tilted beam hits) and see the
average survival spectrum (P(img2=1|img1=1) vs the swept parameter) of just those sites, live.

Left panel  : the array (grid X/Y), sites colored by per-site peak survival; red rings = selection.
Right panel : selection-averaged survival spectrum + whole-array reference + optional Lorentzian fit.

Select
  - Lasso (default): left-click-drag a freehand loop around the sites you want.
  - p : polygon mode (click vertices, close on the first point)   l : back to lasso
Keys
  r : reset (all sites)   f : toggle fit   a : toggle complement-mean overlay
  s / Save button : save -> <prefix>_selection.png, _selected_idx.npy, _mask.npy
  q : quit            (the last selection also auto-saves on window close)
  (prefix = SCAN_DIR/subset_<scan_id>)

Usage
  python -m yb_analysis.scripts.select_subset_survival <SCAN_DIR>
  (SCAN_DIR = any two-array img1/img2 survival scan folder)
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams["toolbar"] = "None"   # no nav toolbar -> a left-drag is ALWAYS a lasso, never pan/zoom
# Free up single-letter keys that matplotlib reserves by default (s=save, l=log,
# p=pan, f=fullscreen, a=all-axes, ...) so our own on_key handler receives them.
for _k in ("keymap.save", "keymap.fullscreen", "keymap.pan", "keymap.yscale",
           "keymap.xscale", "keymap.all_axes", "keymap.grid", "keymap.home",
           "keymap.quit_all"):
    try:
        matplotlib.rcParams[_k] = []
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, PolygonSelector, Button
from matplotlib.path import Path
from scipy.optimize import curve_fit

from yb_analysis.analysis.load_data import load_scan_from_path
from yb_analysis.analysis.unpack import unpack_scan_logicals
from yb_analysis.analysis.probabilities import prob11_site_resolved


# ---------- load + compute the per-site survival cube ----------
def _read_two_array(scan_dir):
    """(scan_json, seq_ids, l1, l2) for a finished OR running scan.

    Tries the normal loader; if logicals are unpopulated (live scan), reads the growing
    data_*.h5 directly with file-locking off + a retry loop (the file is mid-write -> the
    bare loader hits PermissionError, hence the retry)."""
    import os, time
    b = load_scan_from_path(scan_dir)
    scan = b.get("Scan") or {}
    l1, l2, sid = b.get("logicals_img1"), b.get("logicals_img2"), b.get("seq_ids")
    if l1 is not None and l2 is not None and sid is not None and len(sid):
        return scan, np.asarray(sid), np.asarray(l1), np.asarray(l2)
    import h5py
    from yb_analysis.io.scan_files import resolve_scan_files
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    # the DATA file specifically -- a split scan's sibling image_<stamp>.h5 also ends
    # in .h5 but carries no logicals (a first-.h5 pick would SystemExit misleadingly).
    h5 = resolve_scan_files(scan_dir, probe_attrs=False).data_path
    if h5 is None:
        raise SystemExit("no .h5 in %s" % scan_dir)
    last = None
    for _ in range(40):
        try:
            with h5py.File(h5, "r", swmr=True) as f:
                if "logicals_img1" not in f or "logicals_img2" not in f:
                    raise SystemExit("not a two-array (img1/img2) survival scan")
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
    b = {"Scan": scan}
    sp, l1, l2, reps = unpack_scan_logicals(
        scan, seq_ids=sid, logicals_img1=l1, logicals_img2=l2)
    sp = np.asarray(sp, float)
    # 1-D sweeps only for the spectrum axis; if 2-D, collapse to the first column
    axis = sp.reshape(sp.shape[0], -1)[:, 0]
    scale = 1e6 if np.nanmax(np.abs(axis)) > 1e4 else 1.0   # Hz -> MHz when it looks like Hz
    x = axis / scale
    order = np.argsort(x)
    x = x[order]
    mean_sr, sem_sr = prob11_site_resolved(l1, l2)  # (nSites, nParams)
    mean_sr = mean_sr[:, order]
    sem_sr = sem_sr[:, order]
    gx = np.asarray(b["Scan"].get("initGridLocationsX"), float).reshape(-1)
    gy = np.asarray(b["Scan"].get("initGridLocationsY"), float).reshape(-1)
    nS = mean_sr.shape[0]
    if gx.size != nS or gy.size != nS:
        raise SystemExit("grid X/Y (%d/%d) do not match site count %d" % (gx.size, gy.size, nS))
    name = str(b["Scan"].get("scan_id") or os.path.basename(scan_dir))
    xlabel = "swept param" + (" (MHz)" if scale != 1.0 else "")
    return x, mean_sr, sem_sr, gx, gy, name, xlabel


def lorentzian(x, y0, A, x0, w):
    return y0 + A * (w / 2) ** 2 / ((x - x0) ** 2 + (w / 2) ** 2)


def smooth(y, k=5):
    yy = np.where(np.isfinite(y), y, 0.0)
    w = np.isfinite(y).astype(float)
    num = np.convolve(yy, np.ones(k), "same")
    den = np.convolve(w, np.ones(k), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


class SubsetSelector:
    def __init__(self, x, mean_sr, sem_sr, gx, gy, name, xlabel, prefix):
        self.x = x
        self.M = mean_sr            # (nS, nP)
        self.S = sem_sr
        self.gx, self.gy = gx, gy
        self.name = name
        self.xlabel = xlabel
        self.prefix = prefix
        self.nS = mean_sr.shape[0]
        self.pts = np.column_stack([gx, gy])
        self.mask = np.ones(self.nS, bool)
        self.show_fit = True
        self.show_complement = False
        self._last_fit = None
        # per-site peak survival for the array colormap
        self.peak = np.nanmax(np.vstack([smooth(mean_sr[s], 5) for s in range(self.nS)]), axis=1)

        self.fig = plt.figure(figsize=(14.5, 7.0))
        self.fig.canvas.manager.set_window_title("subset survival selector -- %s" % name)
        self.axL = self.fig.add_axes([0.05, 0.14, 0.42, 0.76])
        self.axR = self.fig.add_axes([0.56, 0.14, 0.40, 0.76])
        # explicit Save button (in case single-letter keys don't reach the canvas)
        ax_btn = self.fig.add_axes([0.21, 0.03, 0.10, 0.05])
        self.btn_save = Button(ax_btn, "Save")
        self.btn_save.on_clicked(lambda _ev: self.save())
        self._saved_once = False

        self._draw_array()
        self._draw_spectrum()
        self._connect_lasso()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.suptitle("LASSO-drag a loop to select sites  |  p=polygon  l=lasso  r=reset  "
                          "f=fit  a=complement  s/Save=save  q=quit  [auto-saves on close]",
                          fontsize=9)

    # ---- array panel ----
    def _draw_array(self):
        self.axL.clear()
        self.sc = self.axL.scatter(self.gx, self.gy, c=self.peak, s=18, cmap="viridis",
                                   vmin=np.nanpercentile(self.peak, 2),
                                   vmax=np.nanpercentile(self.peak, 98))
        self.sel_marker = self.axL.scatter(self.gx[self.mask], self.gy[self.mask],
                                           s=46, facecolors="none", edgecolors="red", linewidths=1.1)
        if not hasattr(self, "_cb"):
            self._cb = self.fig.colorbar(self.sc, ax=self.axL, fraction=0.046)
            self._cb.set_label("per-site peak survival")
        self.axL.set_aspect("equal"); self.axL.invert_yaxis()
        self.axL.set_xlabel("grid X (px)"); self.axL.set_ylabel("grid Y (px)")
        self._set_array_title()

    def _set_array_title(self):
        self.axL.set_title("array (%d sites) -- %d selected" % (self.nS, self.mask.sum()))

    def refresh(self):
        self.sel_marker.set_offsets(np.column_stack([self.gx[self.mask], self.gy[self.mask]]))
        self._set_array_title()
        self._draw_spectrum()
        self.fig.canvas.draw_idle()

    # ---- spectrum panel ----
    def _mean_over(self, mask):
        if mask.sum() == 0:
            return np.full_like(self.x, np.nan), np.full_like(self.x, np.nan)
        sub = self.M[mask]
        m = np.nanmean(sub, axis=0)
        n = np.sum(np.isfinite(sub), axis=0)
        sd = np.nanstd(sub, axis=0)
        with np.errstate(invalid="ignore"):
            sem = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), np.nan)
        return m, sem

    def _draw_spectrum(self):
        self.axR.clear()
        ref, _ = self._mean_over(np.ones(self.nS, bool))
        self.axR.plot(self.x, ref, color="#bbb", lw=1.2, label="whole array")
        m, sem = self._mean_over(self.mask)
        self.axR.errorbar(self.x, m, yerr=sem, fmt="o", ms=3.5, color="#d62728",
                          ecolor="#f3b0b0", elinewidth=0.7, capsize=1.5,
                          label="selection (%d sites)" % self.mask.sum(), zorder=4)
        if self.show_complement:
            cm, _ = self._mean_over(~self.mask)
            self.axR.plot(self.x, cm, "-", color="#1f77b4", lw=1.3,
                          label="complement (%d)" % (~self.mask).sum())
        self._last_fit = None
        if self.show_fit and self.mask.sum() >= 5:
            ok = np.isfinite(m)
            if ok.sum() >= 6:
                try:
                    y0g = np.percentile(m[ok], 25)
                    i = int(np.argmax(m[ok]))
                    p, _ = curve_fit(lorentzian, self.x[ok], m[ok],
                                     p0=[y0g, m[ok][i] - y0g, self.x[ok][i], 10.0],
                                     bounds=([0, 0, self.x.min(), 1e-6],
                                             [1, 1.5, self.x.max(), np.ptp(self.x)]), maxfev=10000)
                    xf = np.linspace(self.x.min(), self.x.max(), 1000)
                    sst = np.sum((m[ok] - m[ok].mean()) ** 2)
                    r2 = 1 - np.sum((m[ok] - lorentzian(self.x[ok], *p)) ** 2) / sst if sst > 0 else 0
                    self.axR.plot(xf, lorentzian(xf, *p), "k--", lw=1.6,
                                  label="fit: x0=%.2f, FWHM=%.2f, R2=%.2f" % (p[2], abs(p[3]), r2))
                    self._last_fit = (p, r2)
                except Exception:
                    pass
        self.axR.set_xlabel(self.xlabel)
        self.axR.set_ylabel("survival  P(img2=1|img1=1)")
        self.axR.set_title("selection-averaged survival spectrum")
        self.axR.grid(alpha=0.25)
        self.axR.legend(fontsize=8, loc="upper left")

    # ---- lasso / polygon ----
    def _on_select(self, verts):
        if len(verts) < 3:
            return
        m = Path(verts).contains_points(self.pts)
        print("lasso/poly: %d sites in loop" % int(m.sum()), flush=True)
        # ignore tiny accidental loops (a stray click) so they can't clobber a real
        # selection; <3 sites is almost never intentional for an array spectrum.
        if m.sum() < 3:
            print("  (ignored: too small -- keeping previous selection)", flush=True)
            return
        self.mask = m
        self.refresh()

    def _connect_lasso(self):
        self._disconnect()
        # useblit=False: blitting can drop the onselect redraw on some Qt builds
        self.selector = LassoSelector(self.axL, onselect=self._on_select, useblit=False)

    def _connect_polygon(self):
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

    # ---- keys ----
    def on_key(self, ev):
        if ev.key == "r":
            self.mask = np.ones(self.nS, bool); self.refresh()
        elif ev.key == "f":
            self.show_fit = not self.show_fit; self.refresh()
        elif ev.key == "a":
            self.show_complement = not self.show_complement; self.refresh()
        elif ev.key == "p":
            self._connect_polygon(); print("polygon mode")
        elif ev.key == "l":
            self._connect_lasso(); print("lasso mode")
        elif ev.key == "s":
            self.save()
        elif ev.key == "q":
            plt.close(self.fig)

    def save(self):
        idx = np.where(self.mask)[0]
        np.save(self.prefix + "_selected_idx.npy", idx)
        np.save(self.prefix + "_mask.npy", self.mask)
        self.fig.savefig(self.prefix + "_selection.png", dpi=140)
        msg = "saved: %d sites -> %s_{selection.png,selected_idx.npy,mask.npy}" % (idx.size, self.prefix)
        if self._last_fit:
            p, r2 = self._last_fit
            msg += "  | fit x0=%.3f FWHM=%.3f R2=%.2f" % (p[2], abs(p[3]), r2)
        print(msg, flush=True)
        self._saved_once = True

    def on_close(self, _ev):
        # auto-save the last selection on window close, unless it's the full array
        # or already saved by hand -- so you never lose a pick by closing the window.
        if not self._saved_once and 0 < self.mask.sum() < self.nS:
            print("window closed -- auto-saving last selection", flush=True)
            self.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", help="two-array survival scan folder")
    args = ap.parse_args()
    x, M, S, gx, gy, name, xlabel = load_cube(args.scan_dir)
    prefix = os.path.join(args.scan_dir, "subset_%s" % name)
    print("loaded %d sites, %d params (%.3f..%.3f). prefix=%s"
          % (M.shape[0], M.shape[1], x.min(), x.max(), prefix))
    SubsetSelector(x, M, S, gx, gy, name, xlabel, prefix)
    plt.show()


if __name__ == "__main__":
    main()
