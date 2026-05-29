"""Tk pane for the SequenceRunner job queue.

Treeview with hover tooltips for full scan details.
"""

import logging
import threading
import time
import tkinter as tk
from tkinter import ttk

from yb_analysis import config

logger = logging.getLogger(__name__)


# --- Formatting helpers ---------------------------------------------------

def _fmt_time(ts):
    if not ts:
        return ''
    return time.strftime('%H:%M:%S', time.localtime(ts))


def _fmt_duration(start_ts, end_ts=None):
    if not start_ts:
        return ''
    end = end_ts or time.time()
    elapsed = max(0, int(end - start_ts))
    mins, secs = divmod(elapsed, 60)
    if mins >= 60:
        hours, mins = divmod(mins, 60)
        return f'{hours}:{mins:02d}:{secs:02d}'
    return f'{mins}:{secs:02d}'


_SI_PREFIXES = {
    -24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p',
    -9:  'n',  -6:  'u',  -3:  'm',   0:  '',    3: 'k',
    6:   'M',   9:  'G',  12:  'T',  15:  'P',  18: 'E',
}


def _pretty_value(val, units=''):
    if val is None:
        return '--'
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
    import math
    if not math.isfinite(f):
        return f'{f:.4g}' + (f' {units}' if units else '')
    if not units:
        return f'{f:.4g}'
    if f == 0:
        return f'0 {units}'
    exp = int(math.floor(math.log10(abs(f)) / 3) * 3)
    exp = max(-24, min(18, exp))
    mant = f / (10 ** exp)
    pfx = _SI_PREFIXES.get(exp, f'e{exp:+d}')
    return f'{mant:.4g} {pfx}{units}'.strip()


def _axis_name(ax):
    n = ax.get('name') or ''
    if not n:
        return f"dim{ax.get('dim', '?')}"
    parts = n.split('.')
    return '.'.join(parts[-2:]) if len(parts) > 2 else n


def _axis_range(ax):
    u = ax.get('units', '') or ''
    lo = _pretty_value(ax.get('min', 0), u)
    hi = _pretty_value(ax.get('max', 0), u)
    n = ax.get('npts', 0)
    return f'{lo}..{hi} ({n} pt)'


def _format_axes_short(axes):
    """Column cell format with ranges."""
    if not axes:
        return '--'
    dims = sorted({ax.get('dim', 1) for ax in axes})
    if len(dims) == 1:
        if len(axes) == 1:
            ax = axes[0]
            return f"{_axis_name(ax)}: {_axis_range(ax)}"
        names = ' + '.join(_axis_name(ax) for ax in axes)
        return f'{names} ({axes[0].get("npts", 0)} pt)'
    parts = [f'{_axis_name(ax)}: {_axis_range(ax)}' for ax in axes]
    return ' x '.join(parts)


def _format_axes_full(axes):
    """Detailed format for the tooltip."""
    if not axes:
        return 'No scan axes'
    lines = []
    for ax in axes:
        lines.append(f"  dim{ax.get('dim','?')}: {ax.get('name','')} = {_axis_range(ax)}")
    return '\n'.join(lines)


def _format_detail(summary):
    """Build tooltip text from a summary dict."""
    if not summary:
        return ''
    parts = []

    # Axes
    axes = summary.get('axes')
    if axes:
        parts.append('Scan axes:\n' + _format_axes_full(axes))

    # Parameters
    groups = {}

    def _add(key_flat, val, is_default):
        segs = key_flat.split('_', 1)
        head = segs[0]
        tail = segs[1] if len(segs) > 1 else ''
        groups.setdefault(head, []).append((tail, val, is_default))

    for k, v in (summary.get('set_params') or {}).items():
        _add(k, v, False)
    for k, v in (summary.get('default_params') or {}).items():
        _add(k, v, True)

    if groups:
        lines = []
        for head in sorted(groups.keys()):
            rendered = []
            for tail, val, is_default in groups[head]:
                name = tail if tail else head
                star = '*' if is_default else ''
                rendered.append(f'{name}={_pretty_value(val)}{star}')
            lines.append(f'  {head}: {"  ".join(rendered)}')
        parts.append('Parameters:\n' + '\n'.join(lines))

    # Flags
    flags = []
    for k in ('num_per_group', 'num_images', 'scramble', 'is_init',
              'is_hc', 'rearrangement'):
        if k in summary:
            flags.append(f'{k}={_pretty_value(summary[k])}')
    if flags:
        parts.append('Flags: ' + '  '.join(flags))

    return '\n'.join(parts)


# --- Tooltip --------------------------------------------------------------

class _HoverTooltip:
    """Column-specific tooltip for Treeview rows."""

    def __init__(self, tree, delay_ms=350):
        self._tree = tree
        self._delay_ms = delay_ms
        self._tip = None
        self._after_id = None
        self._current_key = None   # (iid, col)
        self._get_text = None      # callback(iid, col_id) -> str

        tree.bind('<Motion>', self._on_motion)
        tree.bind('<Leave>', self._hide)

    def set_text_callback(self, cb):
        self._get_text = cb

    def _on_motion(self, event):
        iid = self._tree.identify_row(event.y)
        col = self._tree.identify_column(event.x)  # '#1', '#2', ...
        key = (iid, col)
        if key == self._current_key:
            return
        self._hide()
        self._current_key = key
        if iid and col:
            self._after_id = self._tree.after(self._delay_ms,
                                               lambda: self._show(event))

    def _show(self, event):
        if not self._current_key or not self._get_text:
            return
        iid, col = self._current_key
        # Map '#N' to column id
        cols = self._tree['columns']
        try:
            col_idx = int(col.replace('#', '')) - 1
            col_id = cols[col_idx] if 0 <= col_idx < len(cols) else ''
        except (ValueError, IndexError):
            col_id = ''
        text = self._get_text(iid, col_id)
        if not text:
            return
        self._tip = tw = tk.Toplevel(self._tree)
        tw.wm_overrideredirect(True)
        tw.wm_attributes('-topmost', True)
        lbl = tk.Label(tw, text=text, justify='left',
                       background='#ffffdd', foreground='#333333',
                       relief='solid', borderwidth=1,
                       font=('Consolas', 9), padx=6, pady=4)
        lbl.pack()
        x = self._tree.winfo_rootx() + event.x + 16
        y = self._tree.winfo_rooty() + event.y + 16
        tw.wm_geometry(f'+{x}+{y}')

    def _hide(self, _event=None):
        if self._after_id:
            self._tree.after_cancel(self._after_id)
            self._after_id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None
        self._current_key = None


# --- Widget ---------------------------------------------------------------

# (cid, title, width, anchor, stretch)
_COLUMNS = [
    ('marker', '',          22,  'center', False),
    ('id',     'ID',        28,  'center', False),
    ('scan',   'Scan',      86,  'w',      True),
    ('seq',    'Seq',       86,  'w',      True),
    ('axis',   'Scan axis', 110, 'w',      True),
    ('reps',   'Reps',      34,  'center', False),
    ('fileid', 'Data ID',   100, 'w',      False),
    ('added',  'Added',     56,  'center', False),
    ('status', 'Status',    68,  'center', False),
]


class QueuePane(ttk.LabelFrame):
    def __init__(self, parent, zmq_client, dashboard=None, refresh_ms=1000):
        super().__init__(parent, text='Scan queue')
        self._client = zmq_client
        # Reference (or None) to the DashboardRenderer; after each successful
        # poll we push the queue snapshot into its shared pickle so the
        # in-browser Queue panel and /api/queue stay in sync without opening
        # a second ZMQ socket.
        self._dashboard = dashboard
        self._refresh_ms = refresh_ms
        self._poll_lock = threading.Lock()
        self._poll_busy = False
        self._offline = False
        self._entry_by_iid = {}
        # Stable-key -> Treeview iid. Lets _render diff against the existing
        # tree (insert/move/update/delete) instead of clearing and rebuilding
        # every poll, which would drop selection, scroll position, and the
        # hover tooltip's iid reference.
        self._iid_by_key = {}

        self._build()
        self._schedule_refresh()

    def _build(self):
        # Style scoped to this pane: taller rows, padded headings.
        style = ttk.Style(self)
        style.configure('Queue.Treeview', rowheight=22)
        style.configure('Queue.Treeview.Heading', padding=(4, 4))

        # Vertical split: queue list (top) and selection detail (bottom).
        paned = ttk.PanedWindow(self, orient='vertical')
        paned.pack(fill='both', expand=True, padx=6, pady=(4, 6))

        # ---- Top: tree + buttons ----
        tree_section = ttk.Frame(paned)
        paned.add(tree_section, weight=3)

        tree_wrap = ttk.Frame(tree_section)
        tree_wrap.pack(fill='both', expand=True, padx=2, pady=(2, 4))

        self._tree = ttk.Treeview(
            tree_wrap, columns=[c[0] for c in _COLUMNS], show='headings',
            height=8, selectmode='browse', style='Queue.Treeview')
        for cid, title, width, anchor, stretch in _COLUMNS:
            self._tree.heading(cid, text=title)
            self._tree.column(cid, width=width, minwidth=width,
                              anchor=anchor, stretch=stretch)
        self._tree.pack(side='left', fill='both', expand=True)

        sb = ttk.Scrollbar(tree_wrap, orient='vertical',
                           command=self._tree.yview)
        sb.pack(side='right', fill='y')
        self._tree.configure(yscrollcommand=sb.set)

        self._tree.tag_configure('running', foreground='#006600')
        self._tree.tag_configure('error',   foreground='#aa0000')
        self._tree.tag_configure('done',    foreground='#444444')
        self._tree.tag_configure('sep',     foreground='#888888')

        # Hover tooltip
        self._tooltip = _HoverTooltip(self._tree, delay_ms=350)
        self._tooltip.set_text_callback(self._tooltip_text)

        # Click to show detail in bottom pane
        self._tree.bind('<<TreeviewSelect>>', self._on_select)

        btn_row = ttk.Frame(tree_section)
        btn_row.pack(fill='x', padx=2, pady=(2, 2))
        ttk.Button(btn_row, text='Up', command=lambda: self._move('up')).pack(
            side='left', padx=2)
        ttk.Button(btn_row, text='Down', command=lambda: self._move('down')).pack(
            side='left', padx=2)
        ttk.Button(btn_row, text='Remove', command=self._remove).pack(
            side='left', padx=2)
        self._status = ttk.Label(btn_row, text='')
        self._status.pack(side='right')

        # ---- Bottom: scrollable detail ----
        detail_section = ttk.Frame(paned, relief='solid', borderwidth=1)
        paned.add(detail_section, weight=1)

        self._detail = tk.Text(
            detail_section, height=6, wrap='word',
            font=('Consolas', 9), fg='#333333', bg='#fafafa',
            relief='flat', borderwidth=0, padx=8, pady=6,
            state='disabled', cursor='arrow',
            highlightthickness=0)
        self._detail.pack(side='left', fill='both', expand=True)

        detail_sb = ttk.Scrollbar(detail_section, orient='vertical',
                                  command=self._detail.yview)
        detail_sb.pack(side='right', fill='y')
        self._detail.configure(yscrollcommand=detail_sb.set)

    # --- Tooltip ---

    def _tooltip_text(self, iid, col_id):
        entry = self._entry_by_iid.get(iid)
        if not entry:
            return ''
        summary = entry.get('summary') or {}

        if col_id == 'scan':
            name = (summary.get('scan_name') or
                    summary.get('scan_filename') or '')
            return name if name else ''

        if col_id == 'seq':
            return entry.get('seqName') or ''

        if col_id == 'axis':
            axes = summary.get('axes')
            if not axes:
                return ''
            lines = []
            for ax in axes:
                full_name = ax.get('name') or f"dim{ax.get('dim','?')}"
                lines.append(f"{full_name} = {_axis_range(ax)}")
            return '\n'.join(lines)

        if col_id == 'fileid':
            fid = entry.get('file_id') or ''
            if not fid:
                return ''
            return f"Data ID: {fid}\nFolder: ...\\{fid[:8]}\\*_{fid[9:]}"

        if col_id == 'status':
            status = entry.get('status') or ''
            dur = _fmt_duration(entry.get('start_ts'), entry.get('finish_ts'))
            added = _fmt_time(entry.get('enqueued_ts'))
            started = _fmt_time(entry.get('start_ts'))
            parts = []
            if status:
                parts.append(f'Status: {status}')
            if added:
                parts.append(f'Added: {added}')
            if started:
                parts.append(f'Started: {started}')
            if dur:
                parts.append(f'Duration: {dur}')
            return '\n'.join(parts)

        if col_id == 'reps':
            lines = []
            total = summary.get('total_per_group')
            nseqs = summary.get('nseqs')
            if total and nseqs:
                rep = total // nseqs if nseqs else total
                lines.append(f'total = {total}  ({rep} rep x {nseqs} pt)')
            for k in ('num_per_group', 'num_images', 'scramble',
                      'is_init', 'is_hc', 'rearrangement'):
                if k in summary:
                    lines.append(f'{k}={_pretty_value(summary[k])}')
            return '\n'.join(lines) if lines else ''

        return ''

    # --- Refresh cycle ---

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
        if self._dashboard is not None:
            try:
                self._dashboard.update_queue(q)
            except Exception as e:
                logger.debug('dashboard.update_queue failed: %s', e)

    def _on_poll_error(self, msg):
        if not self._offline:
            logger.debug('queue_list failed: %s', msg)
            self._offline = True
        self._status.config(text='runner offline', foreground='red')

    # --- Rendering ---

    @staticmethod
    def _row(entry, is_running=False, is_history=False):
        summary = entry.get('summary') or {}
        scan_name = (summary.get('scan_name') or
                     summary.get('scan_filename') or '--')

        added_cell = _fmt_time(entry.get('enqueued_ts'))

        if is_running:
            marker, tag = '>', 'running'
            dur = _fmt_duration(entry.get('start_ts'))
            status_cell = f'run {dur}' if dur else 'running'
        elif is_history:
            status = entry.get('status') or entry.get('state') or ''
            dur = _fmt_duration(entry.get('start_ts'), entry.get('finish_ts'))
            if status == 'ok':
                marker, tag = '+', 'done'
                status_cell = f'ok {dur}' if dur else 'ok'
            else:
                # Any non-ok status is an error — show the specific message
                marker, tag = 'x', 'error'
                status_cell = status if status != 'error' else 'error'
        else:
            marker, tag = '.', ''
            status_cell = 'queued'

        file_id = entry.get('file_id') or ''

        # Reps is the actual sequence count after ybBuildScanJob's StackNum
        # stacking (each scan point repeats >=2x). Fall back to the user's
        # NumPerGroup if total_per_group isn't in the summary (older runner).
        reps = summary.get('total_per_group') or summary.get('num_per_group')

        return (
            (
                marker,
                str(entry.get('id', '')),
                scan_name,
                entry.get('seqName') or '--',
                _format_axes_short(summary.get('axes')),
                _pretty_value(reps) if reps else '--',
                file_id,
                added_cell,
                status_cell,
            ),
            tag,
        )

    def _render(self, q):
        desired = self._compute_desired(q)
        desired_keys = {key for key, _, _, _ in desired}

        # Drop rows that are no longer present.
        for key in list(self._iid_by_key):
            if key not in desired_keys:
                iid = self._iid_by_key.pop(key)
                self._entry_by_iid.pop(iid, None)
                self._tree.delete(iid)

        # Insert / move / update each row in the desired position.
        for index, (key, values, tag, entry) in enumerate(desired):
            tags = (tag,) if tag else ()
            iid = self._iid_by_key.get(key)
            if iid is None:
                iid = self._tree.insert('', index, values=values, tags=tags)
                self._iid_by_key[key] = iid
            else:
                if self._tree.index(iid) != index:
                    self._tree.move(iid, '', index)
                if tuple(self._tree.item(iid, 'values')) != tuple(values):
                    self._tree.item(iid, values=values)
                if tuple(self._tree.item(iid, 'tags')) != tags:
                    self._tree.item(iid, tags=tags)
            self._entry_by_iid[iid] = entry

        total = len(q.get('queued', []))
        if q.get('running'):
            total += 1
        self._status.config(text=f'{total} in queue', foreground='black')

    def _compute_desired(self, q):
        """Build the (key, values, tag, entry) list for the next render, in
        display order. Stable keys: ('job', id) for jobs, ('sep',) for the
        history separator. Entries without an id are skipped (shouldn't
        happen — server always assigns ids)."""
        out = []
        running = q.get('running')
        if running and running.get('id') is not None:
            values, tag = self._row(running, is_running=True)
            out.append((('job', running['id']), values, tag, running))
        for e in q.get('queued', []):
            if e.get('id') is None:
                continue
            values, tag = self._row(e)
            out.append((('job', e['id']), values, tag, e))
        hist = q.get('history') or []
        if hist:
            sep_values = ('', '', '-- history --', '', '', '', '', '', '')
            out.append((('sep',), sep_values, 'sep', None))
            for e in hist[:config.QUEUE_HISTORY_DISPLAY]:
                if e.get('id') is None:
                    continue
                values, tag = self._row(e, is_history=True)
                out.append((('job', e['id']), values, tag, e))
        return out

    # --- Selection / actions ---

    def _selected_entry(self):
        sel = self._tree.selection()
        if not sel:
            return None
        return self._entry_by_iid.get(sel[0])

    def _selected_queued(self):
        e = self._selected_entry()
        if e and e.get('state') == 'queued':
            return e.get('id'), e
        return None, None

    def _on_select(self, _event=None):
        e = self._selected_entry()
        text = ''
        if e and e.get('summary'):
            text = _format_detail(e['summary']) or ''
        self._detail.configure(state='normal')
        self._detail.delete('1.0', 'end')
        if text:
            self._detail.insert('1.0', text)
        self._detail.configure(state='disabled')

    def _move(self, direction):
        jid, _ = self._selected_queued()
        if jid is None:
            self._status.config(text='select a queued job')
            return
        try:
            self._client.queue_move(jid, direction)
        except Exception as e:
            self._status.config(text=f'move failed: {e}')

    def _remove(self):
        jid, _ = self._selected_queued()
        if jid is None:
            self._status.config(text='select a queued job')
            return
        try:
            self._client.queue_remove(jid)
        except Exception as e:
            self._status.config(text=f'remove failed: {e}')
