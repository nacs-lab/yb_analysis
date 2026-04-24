"""Tk pane for the SequenceRunner job queue.

Treeview layout:
    marker · id · seq · scan-axis summary · reps · imgs · time · status

A selection-driven detail pane below the tree shows the explicitly-set scan
parameters plus the whitelisted defaults (suffixed "*") so the user sees
the operating point at a glance. The 1 Hz ZMQ poll runs on a worker thread
so the UI stays responsive when the runner is offline.
"""

import logging
import threading
import time
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)


# --- Formatting helpers ---------------------------------------------------

def _fmt_time(ts):
    if not ts:
        return ''
    return time.strftime('%H:%M:%S', time.localtime(ts))


# SI prefix table — index = power-of-1000, offset by 8 so 0 is "y" / 8 is unity.
_SI_PREFIXES = {
    -24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p',
    -9:  'n',  -6:  'u',  -3:  'm',   0:  '',    3: 'k',
    6:   'M',   9:  'G',  12:  'T',  15:  'P',  18: 'E',
}


def _pretty_value(val, units=''):
    """Format a value for display. When `units` is non-empty we render with
    an SI prefix (e.g. `50k Hz`, `1 ms`). When `units` is empty (detail-pane
    case where we don't know the unit) we just print `%.4g` so dimensionless
    values like 0.8 don't turn into `800m`."""
    if val is None:
        return '—'
    if isinstance(val, bool):
        return 'yes' if val else 'no'
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        if not val:
            return '[]'
        if all(isinstance(x, (int, float)) for x in val):
            return '[' + ', '.join(_pretty_value(x, '') for x in val) + ']'
        return repr(val)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return repr(val)
    if not units:
        # Dimensionless / unknown-unit → plain %.4g, no SI prefix
        return f'{f:.4g}'
    if f == 0:
        return f'0 {units}'
    import math
    exp = int(math.floor(math.log10(abs(f)) / 3) * 3)
    exp = max(-24, min(18, exp))
    mant = f / (10 ** exp)
    pfx = _SI_PREFIXES.get(exp, f'e{exp:+d}')
    mant_s = f'{mant:.4g}'
    return f'{mant_s} {pfx}{units}'.strip()


def _format_axes(axes):
    """axes: list of dicts with keys {dim, name, scale, units, min, max, npts}.
    Returns a single compact string suitable for the scan-axis column."""
    if not axes:
        return '—'

    def _range(ax):
        u = ax.get('units', '') or ''
        lo = _pretty_value(ax.get('min', 0), u)
        hi = _pretty_value(ax.get('max', 0), u)
        n  = ax.get('npts', 0)
        return f'{lo}..{hi} ({n} pt)'

    dims = sorted({ax.get('dim', 1) for ax in axes})
    if len(dims) == 1:
        # all axes on the same dim → parallel-1D or single
        if len(axes) == 1:
            ax = axes[0]
            return f"{ax.get('name', '?')} = {_range(ax)}"
        names = ' + '.join(ax.get('name', '?') for ax in axes)
        npts = axes[0].get('npts', 0)
        return f'{names} ({npts} pt)'
    # 2D: one axis per dim
    parts = []
    for ax in axes:
        nm = ax.get('name', '?')
        parts.append(f'{nm} ({ax.get("npts", 0)})')
    return ' × '.join(parts)


def _format_detail(summary):
    """Build the detail-pane text from a summary dict. Groups set_params and
    default_params by top-level namespace (SLM, Imag399, Pushout, …).
    Whitelist defaults are suffixed with "*"."""
    if not summary:
        return ''
    groups = {}

    def _add(key_flat, val, is_default):
        parts = key_flat.split('_', 1)
        head = parts[0]
        tail = parts[1] if len(parts) > 1 else ''
        groups.setdefault(head, []).append((tail, val, is_default))

    for k, v in (summary.get('set_params') or {}).items():
        _add(k, v, False)
    for k, v in (summary.get('default_params') or {}).items():
        _add(k, v, True)

    lines = []
    for head in sorted(groups.keys()):
        rendered = []
        for tail, val, is_default in groups[head]:
            name = tail if tail else head
            star = '*' if is_default else ''
            rendered.append(f'{name}={_pretty_value(val)}{star}')
        lines.append(f'{head}: {"  ".join(rendered)}')

    # flags line
    flags = []
    for k in ('num_per_group', 'num_images', 'scramble', 'is_init', 'is_hc', 'rearrangement'):
        if k in summary:
            flags.append(f'{k}={_pretty_value(summary[k])}')
    if flags:
        lines.append('flags: ' + '  '.join(flags))
    return '\n'.join(lines)


# --- Widget ---------------------------------------------------------------

_COLUMNS = [
    ('marker', '', 24, 'center'),
    ('id',     'ID', 44, 'center'),
    ('seq',    'Seq', 160, 'w'),
    ('axis',   'Scan axis', 240, 'w'),
    ('reps',   'Reps', 56, 'center'),
    ('imgs',   'Imgs', 44, 'center'),
    ('time',   'Time', 80, 'center'),
    ('status', 'Status', 68, 'center'),
]


class QueuePane(ttk.LabelFrame):
    def __init__(self, parent, zmq_client, refresh_ms=1000):
        super().__init__(parent, text='Scan queue')
        self._client = zmq_client
        self._refresh_ms = refresh_ms
        self._poll_lock = threading.Lock()
        self._poll_busy = False
        self._offline = False
        self._entry_by_iid = {}   # tree iid → summary dict, for detail pane

        self._build()
        self._schedule_refresh()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill='both', expand=True, padx=6, pady=(6, 3))

        self._tree = ttk.Treeview(
            top, columns=[c[0] for c in _COLUMNS], show='headings',
            height=8, selectmode='browse')
        for cid, title, width, anchor in _COLUMNS:
            self._tree.heading(cid, text=title)
            self._tree.column(cid, width=width, anchor=anchor, stretch=(cid == 'axis'))
        self._tree.pack(side='left', fill='both', expand=True)

        sb = ttk.Scrollbar(top, orient='vertical', command=self._tree.yview)
        sb.pack(side='right', fill='y')
        self._tree.configure(yscrollcommand=sb.set)

        self._tree.tag_configure('running', foreground='#006600')
        self._tree.tag_configure('error',   foreground='#aa0000')
        self._tree.tag_configure('done',    foreground='#444444')
        self._tree.tag_configure('sep',     foreground='#888888')

        self._tree.bind('<<TreeviewSelect>>', self._on_select)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill='x', padx=6, pady=(0, 3))
        ttk.Button(btn_row, text='↑ Up',     command=lambda: self._move('up')).pack(side='left', padx=2)
        ttk.Button(btn_row, text='↓ Down',   command=lambda: self._move('down')).pack(side='left', padx=2)
        ttk.Button(btn_row, text='🗑 Remove', command=self._remove).pack(side='left', padx=2)
        self._status = ttk.Label(btn_row, text='')
        self._status.pack(side='right')

        self._detail = tk.Label(
            self, text='Select a scan to see parameter details',
            anchor='nw', justify='left', wraplength=540,
            font=('Menlo', 11), fg='#333333', bg='#f5f5f5',
            relief='solid', bd=1, padx=6, pady=4)
        self._detail.pack(fill='both', expand=False, padx=6, pady=(0, 6))

    # --- Refresh cycle ----------------------------------------------------

    def _schedule_refresh(self):
        self.after(self._refresh_ms, self._refresh)

    def _refresh(self):
        with self._poll_lock:
            if self._poll_busy:
                self._schedule_refresh()
                return
            self._poll_busy = True
        threading.Thread(target=self._poll_worker, daemon=True).start()
        self._schedule_refresh()

    def _poll_worker(self):
        try:
            q = self._client.queue_list()
        except Exception as e:
            self.after(0, self._on_poll_error, str(e))
        else:
            self.after(0, self._on_poll_ok, q)
        finally:
            with self._poll_lock:
                self._poll_busy = False

    def _on_poll_ok(self, q):
        if self._offline:
            logger.info('Runner back online')
            self._offline = False
        self._render(q)

    def _on_poll_error(self, msg):
        if not self._offline:
            logger.debug('queue_list failed: %s', msg)
            self._offline = True
        self._status.config(text='runner offline')

    # --- Rendering --------------------------------------------------------

    @staticmethod
    def _row(entry, is_running=False, is_history=False):
        summary = entry.get('summary') or {}
        if is_running:
            marker, tag = '▶', 'running'
            time_cell = _fmt_time(entry.get('start_ts')) or 'pending'
        elif is_history:
            status = entry.get('status') or entry.get('state') or ''
            if status == 'ok':
                marker, tag = '✓', 'done'
            elif status == 'error':
                marker, tag = '✗', 'error'
            else:
                marker, tag = '·', 'done'
            time_cell = _fmt_time(entry.get('finish_ts') or entry.get('start_ts'))
        else:
            marker, tag = '·', ''
            time_cell = 'pending'
        return (
            (
                marker,
                str(entry.get('id', '')),
                entry.get('seqName') or '—',
                _format_axes(summary.get('axes')),
                _pretty_value(summary.get('num_per_group')) if summary.get('num_per_group') else '—',
                _pretty_value(summary.get('num_images'))    if summary.get('num_images')    else '—',
                time_cell,
                entry.get('status') or entry.get('state') or '',
            ),
            tag,
        )

    def _render(self, q):
        saved_id = self._selected_job_id()

        # rebuild tree
        for iid in self._tree.get_children():
            self._tree.delete(iid)
        self._entry_by_iid.clear()

        running = q.get('running')
        if running:
            values, tag = self._row(running, is_running=True)
            iid = self._tree.insert('', 'end', values=values,
                                    tags=(tag,) if tag else ())
            self._entry_by_iid[iid] = running

        for e in q.get('queued', []):
            values, tag = self._row(e)
            iid = self._tree.insert('', 'end', values=values,
                                    tags=(tag,) if tag else ())
            self._entry_by_iid[iid] = e

        hist = q.get('history') or []
        if hist:
            sep_iid = self._tree.insert('', 'end',
                                        values=('', '', '── history ──', '', '', '', '', ''),
                                        tags=('sep',))
            self._entry_by_iid[sep_iid] = None
            for e in hist[:10]:
                values, tag = self._row(e, is_history=True)
                iid = self._tree.insert('', 'end', values=values,
                                        tags=(tag,) if tag else ())
                self._entry_by_iid[iid] = e

        # restore selection on the same job id if present
        if saved_id is not None:
            for iid, entry in self._entry_by_iid.items():
                if entry and entry.get('id') == saved_id:
                    self._tree.selection_set(iid)
                    self._tree.see(iid)
                    break

        total = len(q.get('queued', []))
        if running:
            total += 1
        self._status.config(text=f'{total} in queue')

    # --- Selection / actions ---------------------------------------------

    def _selected_entry(self):
        sel = self._tree.selection()
        if not sel:
            return None
        iid = sel[0]
        return self._entry_by_iid.get(iid)

    def _selected_job_id(self):
        e = self._selected_entry()
        return e.get('id') if e else None

    def _selected_queued(self):
        e = self._selected_entry()
        if e and e.get('state') == 'queued':
            return e.get('id'), e
        return None, None

    def _on_select(self, _event=None):
        e = self._selected_entry()
        if not e or not e.get('summary'):
            self._detail.config(text='Select a scan to see parameter details')
            return
        text = _format_detail(e['summary']) or '(no explicit parameters)'
        self._detail.config(text=text)

    def _move(self, direction):
        jid, _ = self._selected_queued()
        if jid is None:
            self._status.config(text='select a queued job to move')
            return
        try:
            self._client.queue_move(jid, direction)
        except Exception as e:
            self._status.config(text=f'move failed: {e}')
            return
        self._refresh()

    def _remove(self):
        jid, _ = self._selected_queued()
        if jid is None:
            self._status.config(text='select a queued job to remove')
            return
        try:
            self._client.queue_remove(jid)
        except Exception as e:
            self._status.config(text=f'remove failed: {e}')
            return
        self._refresh()
