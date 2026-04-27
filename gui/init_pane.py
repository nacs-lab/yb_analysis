"""InitPane — shows and lets the user change the initialization folder.

The init folder is a Data/YYYYMMDD directory that contains
gridLocations.txt and threshold.mat, used to seed tweezer positions
and detection thresholds at startup.
"""

import os
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from yb_analysis.config import DATA_DIR

_FONT = ('Segoe UI', 10)
_FONT_SM = ('Segoe UI', 9)

_COLOR_OK = '#006600'
_COLOR_ERR = '#aa0000'
_COLOR_NEUTRAL = '#555555'


def _display_path(path):
    """Return a compact representation: relative to DATA_DIR if possible."""
    if not path:
        return ''
    try:
        rel = os.path.relpath(path, DATA_DIR)
        # Only use relative form when it doesn't climb too far up
        if not rel.startswith('..'):
            return rel
    except ValueError:
        pass
    return path


class InitPane(ttk.LabelFrame):

    def __init__(self, parent, on_change, init_dir=None, init_status=''):
        super().__init__(parent, text='Init folder')
        self._on_change = on_change

        self._current_dir = init_dir
        self._current_status = init_status or ('No data loaded' if init_dir is None else '')
        self._is_init_scan = False

        self._path_var = tk.StringVar(value=_display_path(init_dir))
        self._status_var = tk.StringVar(value=self._current_status)
        self._status_color = _COLOR_OK if init_dir else _COLOR_NEUTRAL

        self._build()

    def _build(self):
        row = ttk.Frame(self)
        row.pack(fill='x', padx=6, pady=4)

        ttk.Label(row, text='Folder:', font=_FONT).pack(side='left')
        ttk.Label(row, textvariable=self._path_var, font=_FONT,
                  width=32, anchor='w').pack(side='left', padx=(4, 8))
        ttk.Button(row, text='Browse…', command=self._on_browse,
                   width=9).pack(side='left', padx=(0, 4))
        ttk.Button(row, text='Today', command=self._on_today,
                   width=7).pack(side='left')
        self._status_lbl = ttk.Label(row, textvariable=self._status_var,
                                     font=_FONT_SM,
                                     foreground=self._status_color)
        self._status_lbl.pack(side='left', padx=(12, 0))

    # ---------------------------------------------------------------- public

    def set_is_init_scan(self, is_init: bool):
        """Called from ControlPanel when a scan is processed."""
        if is_init == self._is_init_scan:
            return
        self._is_init_scan = is_init
        if is_init:
            self._path_var.set('')
            self._status_var.set('initialization scan — only saving images')
            self._status_lbl.config(foreground=_COLOR_NEUTRAL)
        else:
            self._path_var.set(_display_path(self._current_dir))
            self._status_var.set(self._current_status)
            self._status_lbl.config(foreground=self._status_color)

    # --------------------------------------------------------------- actions

    def _on_browse(self):
        initial = self._current_dir or DATA_DIR
        chosen = filedialog.askdirectory(
            initialdir=initial, parent=self,
            title='Select initialization folder',
            mustexist=True,
        )
        if chosen:
            threading.Thread(
                target=self._load_in_background, args=(chosen,), daemon=True
            ).start()

    def _on_today(self):
        threading.Thread(
            target=self._load_in_background, args=(None,), daemon=True
        ).start()

    # ------------------------------------------------------- background load

    def _load_in_background(self, path_or_none):
        """Load from a specific dir (or auto-select if None). Runs on daemon thread."""
        if path_or_none is None:
            from yb_analysis.io.preload import load_background_data
            data, src = load_background_data()
            path = src
            status = f'{data["num_sites"]} sites loaded' if data else 'No data found'
        else:
            from yb_analysis.io.preload import load_from_dir
            data, status = load_from_dir(path_or_none)
            path = path_or_none if data is not None else self._current_dir

        self.after(0, self._apply_result, data, path, status)

    def _apply_result(self, data, path, status_msg):
        self._current_dir = path
        self._current_status = status_msg
        self._status_color = _COLOR_OK if data is not None else _COLOR_ERR

        if not self._is_init_scan:
            self._path_var.set(_display_path(path))
            self._status_var.set(status_msg)
            self._status_lbl.config(foreground=self._status_color)

        if data is not None:
            self._on_change(data)
