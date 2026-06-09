"""Sequence-plotter support for the yb dashboard.

Reads the flattened ``.seq`` files (SeqPlotter format) that the experiment
runtime writes per scan, and shapes them for the dashboard's Sequence tab.

The ``.seq`` byte layout is the spec kept canonical in pyctrl's
``compare_seq_bytes.py`` and the MATLAB framework source ``lib/ExpSeq.m``.
See ``seq_parse`` for the reader.
"""

from yb_analysis.sequence.seq_parse import (
    Channel,
    Frame,
    PULSE_ID_DEFAULT,
    SeqDump,
    Sequence,
    decode,
    load,
    parse,
)

__all__ = [
    "Channel",
    "Frame",
    "PULSE_ID_DEFAULT",
    "SeqDump",
    "Sequence",
    "decode",
    "load",
    "parse",
]
