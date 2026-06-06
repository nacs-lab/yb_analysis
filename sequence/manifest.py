"""Discover and read a scan's flattened sequences from a data folder.

The experiment runtime (when the "save sequence dumps" toggle is on) writes,
into ``<scan_folder>/sequence/``:

  * one ``*.seq`` per *unique* compiled sequence (deduplicated by serialize hash);
  * a ``manifest.json`` mapping each scan point -> the ``.seq`` file that point
    ran, plus the scanned-axis names/values.

``manifest.json`` schema (all optional fields tolerated)::

    {
      "scan_id": "20250619_142657",
      "seq": "RydDetSeq",
      "scanned_axes": [{"dim": 1, "path": "Pushout.Time", "values": [1.0e-3, ...]}],
      "points": [{"n": 1, "seqid": 1, "file": "point_00001__seqid_0001.seq",
                  "scanned": {"Pushout.Time": 1.0e-3}}, ...],
      "unique_seqs": {"1": "point_00001__seqid_0001.seq"}
    }

This reader is robust to a **manifest-free** folder too (e.g. a raw drop of
``.seq`` files): it then exposes one synthetic point per file. So the dashboard
viewer works whether or not the auto-dump writer ran.
"""

import json
import os

from yb_analysis.sequence import seq_parse

MANIFEST_NAME = "manifest.json"


def _has_seq(directory):
    try:
        return any(f.endswith(".seq") for f in os.listdir(directory))
    except OSError:
        return False


def find_sequence_dir(folder):
    """Return the directory holding ``.seq`` files for ``folder``, or ``None``.

    Accepts a scan folder (prefers its ``sequence/`` subdir), the ``sequence/``
    dir itself, or any directory that directly contains ``.seq`` files.
    """
    if not folder:
        return None
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return None
    sub = os.path.join(folder, "sequence")
    if os.path.isdir(sub) and _has_seq(sub):
        return sub
    if _has_seq(folder):
        return folder
    return None


def load_manifest(seq_dir):
    """Return the parsed ``manifest.json`` in ``seq_dir`` or ``None``."""
    path = os.path.join(seq_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class SequenceFolder:
    """A scan's ``.seq`` files plus its optional manifest."""

    def __init__(self, seq_dir, manifest=None):
        self.dir = seq_dir
        self.manifest = manifest

    @classmethod
    def open(cls, folder):
        """Open the sequence dir for ``folder``; ``None`` if there is none."""
        seq_dir = find_sequence_dir(folder)
        if seq_dir is None:
            return None
        return cls(seq_dir, load_manifest(seq_dir))

    # -- files ---------------------------------------------------------------
    def seq_files(self):
        return sorted(f for f in os.listdir(self.dir) if f.endswith(".seq"))

    def file_path(self, fname):
        """Resolve ``fname`` within the dir, refusing path traversal."""
        base = os.path.basename(str(fname))
        if not base.endswith(".seq"):
            raise ValueError("not a .seq file: %r" % (fname,))
        path = os.path.join(self.dir, base)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    def load(self, fname):
        """Parse one ``.seq`` file into a :class:`seq_parse.SeqDump`."""
        return seq_parse.load(self.file_path(fname))

    # -- scan metadata -------------------------------------------------------
    def scanned_axes(self):
        if self.manifest:
            return self.manifest.get("scanned_axes", []) or []
        return []

    def points(self):
        """Ordered scan points. Manifest-free -> one synthetic point per file."""
        if self.manifest and self.manifest.get("points"):
            return self.manifest["points"]
        return [{"n": i + 1, "file": f, "scanned": {}}
                for i, f in enumerate(self.seq_files())]

    def index(self):
        """Compact listing for the API: per-file sequences/channels + points."""
        files = []
        for f in self.seq_files():
            try:
                dump = self.load(f)
                seqs = [{"name": s.name, "seq_idx": s.seq_idx,
                         "nchns": len(s.channels),
                         "channels": s.channel_names,
                         "has_params": s.params is not None}
                        for s in dump.sequences]
                err = None
            except Exception as ex:  # corrupt file -> report, keep going
                seqs, err = [], str(ex)
            entry = {"file": f, "sequences": seqs}
            if err:
                entry["error"] = err
            files.append(entry)
        return {
            "sequence_dir": self.dir,
            "has_manifest": self.manifest is not None,
            "scan_id": (self.manifest or {}).get("scan_id"),
            "seq": (self.manifest or {}).get("seq"),
            "scanned_axes": self.scanned_axes(),
            "points": self.points(),
            "files": files,
        }
