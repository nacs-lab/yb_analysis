"""Tk pane for the SequenceRunner job queue.

Shows the running entry + the queued tail + a small history list.
Buttons: Up / Down / Remove act on the selection. The running entry is
never selectable for reorder/remove.
"""

import logging
import threading
import time
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)


def _ts(t):
    if not t:
        return '—'
    return time.strftime('%H:%M:%S', time.localtime(t))


class QueuePane(ttk.LabelFrame):
    def __init__(self, parent, zmq_client, refresh_ms=1000):
        super().__init__(parent, text='Scan queue')
        self._client = zmq_client
        self._refresh_ms = refresh_ms
        self._entries = []      # flat list mirroring listbox rows
        self._running_id = None
        self._poll_lock = threading.Lock()
        self._poll_busy = False
        self._offline = False

        self._build()
        self._schedule_refresh()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill='both', expand=True, padx=6, pady=(6, 3))

        self._listbox = tk.Listbox(
            top, font=('Menlo', 11), height=8, activestyle='dotbox',
            selectmode='browse')
        self._listbox.pack(side='left', fill='both', expand=True)

        sb = ttk.Scrollbar(top, orient='vertical', command=self._listbox.yview)
        sb.pack(side='right', fill='y')
        self._listbox.configure(yscrollcommand=sb.set)

        btn_row = ttk.Frame(self)
        btn_row.pack(fill='x', padx=6, pady=(0, 6))
        ttk.Button(btn_row, text='↑ Up', command=lambda: self._move('up')).pack(side='left', padx=2)
        ttk.Button(btn_row, text='↓ Down', command=lambda: self._move('down')).pack(side='left', padx=2)
        ttk.Button(btn_row, text='🗑 Remove', command=self._remove).pack(side='left', padx=2)
        self._status = ttk.Label(btn_row, text='')
        self._status.pack(side='right')

    # ---- state → UI

    def _schedule_refresh(self):
        self.after(self._refresh_ms, self._refresh)

    def _refresh(self):
        """Kick off a background poll. UI stays responsive; results land via
        self.after(0, ...) on the main thread."""
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

    def _render(self, q):
        saved_sel = self._selected_id()

        self._listbox.delete(0, tk.END)
        self._entries = []

        running = q.get('running')
        if running:
            self._running_id = running['id']
            line = (f"▶ #{running['id']:>4}  {running['seqName'] or '<unknown>':<28} "
                    f"running since {_ts(running.get('start_ts'))}")
            self._listbox.insert(tk.END, line)
            self._entries.append(('running', running))
            self._listbox.itemconfig(tk.END, fg='#006600')
        else:
            self._running_id = None

        for e in q.get('queued', []):
            line = (f"  #{e['id']:>4}  {e['seqName'] or '<unknown>':<28} "
                    f"queued {_ts(e.get('enqueued_ts'))}")
            self._listbox.insert(tk.END, line)
            self._entries.append(('queued', e))

        hist = q.get('history', [])
        if hist:
            self._listbox.insert(tk.END, '─' * 40)
            self._entries.append(('sep', None))
            self._listbox.itemconfig(tk.END, fg='#888888')
            for e in hist[:10]:
                status = e.get('status') or e.get('state')
                marker = '✓' if status == 'ok' else '✗'
                line = (f"{marker} #{e['id']:>4}  {e['seqName'] or '<unknown>':<28} "
                        f"finished {_ts(e.get('finish_ts'))}")
                self._listbox.insert(tk.END, line)
                self._entries.append(('history', e))
                color = '#444444' if status == 'ok' else '#aa0000'
                self._listbox.itemconfig(tk.END, fg=color)

        # restore selection on the same job id if still present
        if saved_sel is not None:
            for i, (kind, e) in enumerate(self._entries):
                if kind in ('queued', 'running') and e and e['id'] == saved_sel:
                    self._listbox.selection_set(i)
                    break

        total = len(q.get('queued', []))
        if self._running_id:
            total += 1
        self._status.config(text=f'{total} in queue')

    def _selected_id(self):
        sel = self._listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx >= len(self._entries):
            return None
        kind, entry = self._entries[idx]
        if kind in ('queued', 'running') and entry:
            return entry['id']
        return None

    def _selected_queued(self):
        """Return (id, entry) for selection only if it's reorderable."""
        sel = self._listbox.curselection()
        if not sel:
            return None, None
        kind, entry = self._entries[sel[0]]
        if kind != 'queued' or not entry:
            return None, None
        return entry['id'], entry

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
