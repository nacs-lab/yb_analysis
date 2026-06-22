#!/usr/bin/env python3
"""ni_set_driver.py -- set ONE NI PCIe-6738 analog-out channel to a DC voltage, for the
dashboard's NI DAC write capability. NOT via molecube (the molecube2 daemon fronts FPGA1
DDS/TTL/clock only and knows nothing about the NI DAQ).

PROPER CONVENTION: this uses ``pyctrl/devices/nidaq/nidaq_io_handler.set_channel`` -- the
on-demand single-channel DC write that mirrors MATLAB ``NIUSBDAQ.setV`` (immediate, on the
card's own timing, NO external FPGA clock; the DAC holds the value after the task closes).
It is the lighter-weight one-off set, and is DELIBERATELY NOT ``set_chns()`` (which runs a
full ExpSeq through the engine and resets EVERY channel to its expConfig default).

Spawned by the ENGINE python (``.venv-engine-py312``) exactly like the read-side
``ni_monitor_driver.py`` -- ``nidaqmx`` lives only in the engine venv, not the dashboard's
``yb_analysis`` conda env. Channel resolution (alias -> ``Dev1/N``) reuses
``ni_monitor_driver`` so it never drifts from expConfig.

SAFETY: the caller MUST only invoke this when no scan is running -- a scan reserves Dev1's
AO subsystem and an out-of-band write would fight/perturb it. The dashboard route defers
while running and serializes card access with the monitor read.

Output: one line ``NI_WRITE_RESULT:{json}`` on stdout, e.g.::

    {"ok": true, "channel": "VElectrode1", "backend": "Dev1/12", "chn": 12,
     "voltage": 1.5, "readback": 1.498, "error": null}
"""

import argparse
import json
import os
import sys
import traceback

RESULT_PREFIX = "NI_WRITE_RESULT:"
AO_LIMIT_V = 10.0          # PCIe-6738 analog-out range is +-10 V


def _emit(obj):
    print(RESULT_PREFIX + json.dumps(obj))


def _resolve(pyctrl_root, channel):
    """Resolve an alias ("VElectrode1") or backend name ("Dev1/12") to a channel dict
    {alias, backend, chn, default}, using the SAME expConfig source as the monitor read.
    A raw "Dev1/N" not present in expConfig is accepted (advanced/manual use)."""
    # The driver dir is on sys.path[0] when run as a script -> reuse the monitor resolver.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ni_monitor_driver as nm
    channels = nm._ni_channels_from_expconfig(pyctrl_root)  # also puts pyctrl on sys.path
    req = str(channel).strip()
    by_alias = {c["alias"]: c for c in channels}
    by_backend = {c["backend"]: c for c in channels}
    info = by_alias.get(req) or by_backend.get(req)
    if info is None:
        dev, _, chn = req.rpartition("/")
        if dev and chn.isdigit():
            info = {"alias": req, "backend": req, "chn": int(chn), "default": None}
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Set one NI PCIe-6738 AO channel (DC volts).")
    ap.add_argument("--pyctrl-root", required=True,
                    help="pyctrl repo root (so expConfig + devices are importable)")
    ap.add_argument("--channel", required=True,
                    help="alias (e.g. VElectrode1) or backend name (e.g. Dev1/12)")
    ap.add_argument("--voltage", required=True, type=float, help="DC volts (+-10 V)")
    ap.add_argument("--readback", action="store_true",
                    help="read the AO monitor back after setting (set->read verify)")
    ap.add_argument("--dry", action="store_true",
                    help="resolve + range-check only; DON'T touch hardware")
    args = ap.parse_args(argv)

    try:
        info = _resolve(args.pyctrl_root, args.channel)
    except Exception as ex:  # noqa: BLE001
        _emit({"ok": False, "error": "expConfig load failed: %s" % ex,
               "trace": traceback.format_exc()[-1500:]})
        return 1

    if info is None:
        _emit({"ok": False, "error": "unknown NI channel: %r" % args.channel})
        return 1

    v = float(args.voltage)
    if not (-AO_LIMIT_V <= v <= AO_LIMIT_V):
        _emit({"ok": False, "channel": info["alias"], "backend": info["backend"],
               "error": "voltage %g out of allowed +-%g V" % (v, AO_LIMIT_V)})
        return 1

    if args.dry:
        _emit({"ok": True, "dry": True, "channel": info["alias"], "backend": info["backend"],
               "chn": info["chn"], "voltage": v, "readback": None, "error": None})
        return 0

    try:
        from devices.nidaq.nidaq_io_handler import set_channel
        set_channel(info["backend"], v)            # proper one-off DC write (setV-style)
        readback = None
        if args.readback:
            try:
                from devices.nidaq.nidaq_io_handler import read_channel
                readback = float(read_channel(info["backend"]))
            except Exception:  # noqa: BLE001 - readback is best-effort
                readback = None
        _emit({"ok": True, "channel": info["alias"], "backend": info["backend"],
               "chn": info["chn"], "voltage": v, "readback": readback, "error": None})
        return 0
    except Exception as ex:  # noqa: BLE001
        _emit({"ok": False, "channel": info["alias"], "backend": info["backend"],
               "voltage": v, "error": "NI write failed: %s" % ex,
               "trace": traceback.format_exc()[-1500:]})
        return 1


if __name__ == "__main__":
    sys.exit(main())
