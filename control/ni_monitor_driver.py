#!/usr/bin/env python3
"""ni_monitor_driver.py -- read the NI PCIe-6738 analog-out channels' ACTUAL output
voltages, for the dashboard's "NI DAC channels" monitor in the Hardware/Molecube tab.

NOT via molecube. The molecube2 daemon fronts FPGA1 (DDS/TTL/clock) only -- it knows
nothing about the NI DAQ. The PCIe-6738 has no external analog input, but exposes a
per-AO-channel INTERNAL monitor (``Dev1/_aoN_vs_aognd``), so each output voltage can be
read back electronically with no external meter. This is exactly the read path of
``pyctrl/devices/nidaq/nidaq_io_handler.read_channel``; we batch all channels into one
AI task here.

WHY A SEPARATE SUBPROCESS (run by the ENGINE python, e.g. ``.venv-engine-py312``):
``nidaqmx`` lives only in the engine venv, not in the dashboard's ``yb_analysis`` conda
env. So the dashboard spawns this standalone script with the engine interpreter -- the
same proven pattern as ``tools/reconstruct_scan.py`` / ``tools/provenance_scan.py``. This
file does NOT need ``yb_analysis`` importable; it only needs ``nidaqmx`` plus pyctrl's
``expConfig`` (for the authoritative alias -> ``Dev1/N`` map + default values), reached
via ``--pyctrl-root``.

READ-ONLY: opens an AI task on the internal monitor channels and reads. It drives no
output. The DAC holds its last value between sequences, so when the experiment is idle
this reports the parked voltage on each channel. The caller MUST only invoke this when
no scan is running -- a scan reserves Dev1's AO subsystem and the read would fail (and an
ill-timed task open could perturb the live run); the dashboard route defers while running.

Output: a single line ``NI_MONITOR_RESULT:{json}`` on stdout (engine chatter, if any,
is irrelevant -- the caller scans for this prefix). Shape::

    {"ok": true, "device": "Dev1",
     "channels": [{"alias": "VElectrode1", "backend": "Dev1/12", "chn": 12,
                   "monitor": "Dev1/_ao12_vs_aognd", "voltage": 0.0021,
                   "default": 0.0, "error": null}, ...],
     "error": null}
"""

import argparse
import json
import os
import sys
import traceback

RESULT_PREFIX = "NI_MONITOR_RESULT:"


def _ao_monitor_name(backend):
    """``"Dev1/12"`` -> ``"Dev1/_ao12_vs_aognd"`` (the card's internal AO-monitor AI).

    Mirrors ``nidaq_io_handler._ao_monitor_name``. ``backend`` is the codebase
    ``dev/number`` convention (no ``ao``); an explicit ``Dev1/ao12`` is tolerated.
    """
    dev, _, chn = backend.rpartition("/")
    num = chn[2:] if chn.startswith("ao") else chn
    return "%s/_ao%s_vs_aognd" % (dev, num)


def _ni_channels_from_expconfig(pyctrl_root):
    """Return the ordered list of NI AO channels from pyctrl ``expConfig``.

    Each item: ``{alias, backend ("Dev1/N"), chn (int), default (float|None)}``. The
    alias->backend map and per-channel defaults are the SAME source the live runner uses,
    so this monitor never drifts from the experiment config.
    """
    # expConfig lives at the pyctrl root; expConfig_helper at <root>/lib. Put both on the
    # path (this is exactly the live runner's import environment).
    for p in (pyctrl_root, os.path.join(pyctrl_root, "lib")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import expConfig  # noqa: E402  (path is set up just above)
    cfg = expConfig.build_config()
    alias_keys = cfg.get("channel_alias_keys", [])
    alias_vals = cfg.get("channel_alias_vals", [])
    defaults = dict(zip(cfg.get("default_vals_keys", []),
                        cfg.get("default_vals_vals", [])))
    out = []
    for alias, backend in zip(alias_keys, alias_vals):
        # NI analog-out channels are aliased to "Dev1/<int>" (the bare "Dev1" device
        # alias and the PFI clock/trigger lines are not AO outputs -> skip).
        if not isinstance(backend, str):
            continue
        dev, _, chn = backend.rpartition("/")
        if not dev or not chn.isdigit():
            continue
        d = defaults.get(alias)
        out.append({
            "alias": alias,
            "backend": backend,
            "chn": int(chn),
            "default": (float(d) if isinstance(d, (int, float)) else None),
        })
    # Stable, human-friendly order: by channel number.
    out.sort(key=lambda c: c["chn"])
    return out


def _read_voltages(channels, avg):
    """Read each channel's monitor voltage. Returns a dict ``backend -> (voltage, error)``.

    Tries ONE multi-channel AI task first (one open, ``avg`` software-timed reads,
    averaged). If that fails (e.g. one bad monitor channel aborts the whole task), falls
    back to a per-channel task so a single failure can't blank every reading.
    """
    import nidaqmx
    monitors = [(c["backend"], _ao_monitor_name(c["backend"])) for c in channels]
    result = {}

    def _avg_reads(task, nchan):
        acc = None
        n = max(1, int(avg))
        for _ in range(n):
            vals = task.read()                 # float (1 chan) or list (>1 chan)
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            if acc is None:
                acc = [0.0] * len(vals)
            for i, v in enumerate(vals):
                acc[i] += float(v)
        return [a / n for a in acc]

    # --- fast path: all monitors in one task ---
    try:
        with nidaqmx.Task() as task:
            for _backend, mon in monitors:
                task.ai_channels.add_ai_voltage_chan(mon)
            avgvals = _avg_reads(task, len(monitors))
        for (backend, _mon), v in zip(monitors, avgvals):
            result[backend] = (v, None)
        return result
    except Exception:  # noqa: BLE001 - drop to the per-channel fallback below
        pass

    # --- fallback: one task per channel (isolates a bad monitor) ---
    for backend, mon in monitors:
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(mon)
                v = _avg_reads(task, 1)[0]
            result[backend] = (v, None)
        except Exception as ex:  # noqa: BLE001
            result[backend] = (None, str(ex))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read NI PCIe-6738 AO monitor voltages.")
    ap.add_argument("--pyctrl-root", required=True,
                    help="pyctrl repo root (so expConfig is importable)")
    ap.add_argument("--avg", type=int, default=4,
                    help="samples averaged per channel (default 4)")
    ap.add_argument("--dry", action="store_true",
                    help="resolve channels from expConfig but DON'T touch hardware")
    args = ap.parse_args(argv)

    try:
        channels = _ni_channels_from_expconfig(args.pyctrl_root)
    except Exception as ex:  # noqa: BLE001
        print(RESULT_PREFIX + json.dumps(
            {"ok": False, "error": "expConfig load failed: %s" % ex,
             "trace": traceback.format_exc()[-1500:]}))
        return 1

    device = channels[0]["backend"].split("/")[0] if channels else None

    if args.dry:
        rows = [dict(c, monitor=_ao_monitor_name(c["backend"]),
                     voltage=None, error="dry-run") for c in channels]
        print(RESULT_PREFIX + json.dumps(
            {"ok": True, "device": device, "channels": rows, "error": None,
             "dry": True}))
        return 0

    try:
        volts = _read_voltages(channels, args.avg)
    except Exception as ex:  # noqa: BLE001
        print(RESULT_PREFIX + json.dumps(
            {"ok": False, "device": device,
             "error": "NI read failed: %s" % ex,
             "trace": traceback.format_exc()[-1500:]}))
        return 1

    rows = []
    for c in channels:
        v, err = volts.get(c["backend"], (None, "not read"))
        rows.append(dict(c, monitor=_ao_monitor_name(c["backend"]),
                         voltage=v, error=err))
    print(RESULT_PREFIX + json.dumps(
        {"ok": True, "device": device, "channels": rows, "error": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
