/* Yb Tweezer Dashboard — vanilla-JS client.
   Polls the lab-PC /api/* endpoints and renders 4 tabs (Live, SLM
   Hardware, Analysis, Queue) in SLM-server-style cards. */

(function() {
  "use strict";

  // ---- FEATURE FLAGS ----
  // Set FLOATING_ANALYSIS_CARDS to false to fully revert to the legacy
  // in-grid Analysis tab layout. When true, the Runs picker and the
  // Filter card are reparented into a fixed-position container at the
  // top-right of the viewport, stacked vertically, half-width when
  // expanded, yellow-tinted. Both work as collapsible pills otherwise.
  // (Everything else on the tab keeps its place; this only moves these
  // two specific cards into a floating overlay.)
  const FLOATING_ANALYSIS_CARDS = true;

  // ---- Config (poll intervals in ms) ----
  // Hardware tab is intentionally slow: every dashboard hit on
  // /api/slm/* either reads from the proxy's cached pickle (Phase 0
  // endpoints) or passes through to the SLM PC. Slow polling keeps
  // load off the SLM server -- the operator can hit "Refresh" for
  // an immediate update.
  const POLL = {
    live:     3000,
    hardware: 10000,
    queue:    3000,
    // analysis is on-demand only (no auto-poll)
  };

  // ---- DOM helpers ----
  const $  = (id) => document.getElementById(id);
  const $$ = (sel, ctx) => Array.from((ctx || document).querySelectorAll(sel));
  const fmtTs = (epoch) => {
    if (!epoch) return "—";
    const d = new Date(epoch * 1000);
    return d.toLocaleTimeString();
  };
  const fmtAge = (epoch) => {
    if (!epoch) return "—";
    const s = (Date.now() / 1000 - epoch);
    if (s < 60)  return s.toFixed(0) + "s ago";
    if (s < 3600) return (s / 60).toFixed(0) + "m ago";
    return (s / 3600).toFixed(1) + "h ago";
  };
  const fmtPct = (v) => (v == null || Number.isNaN(v))
    ? "—" : (100 * v).toFixed(1) + "%";
  const fmtNum = (v, n) => (v == null || Number.isNaN(v))
    ? "—" : v.toFixed(n || 2);
  const setText = (id, t) => { const el = $(id); if (el) el.textContent = t; };

  // ---- Discrimination-infidelity formatting (matches the LIVE view:
  // infidelity as a raw fraction in scientific notation, color-coded
  // green<1% / yellow<5% / red). See dashboard.py:_fig_infid. ----
  const fmtInfid = (v) => (v == null || Number.isNaN(v))
    ? "—" : Number(v).toExponential(1);
  const infidColor = (v) => (v == null || Number.isNaN(v)) ? "var(--text-dim)"
    : (v < 0.01 ? "#4cc762" : v < 0.05 ? "#d29922" : "#f85149");

  // 14-digit scan_id (YYYYMMDDHHMMSS) -> "2026-05-29 · 02:50:15".
  function scanIdToDate(scanId) {
    const s = String(scanId || "");
    if (!/^\d{14}$/.test(s)) return null;
    return `${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)} · `
         + `${s.slice(8,10)}:${s.slice(10,12)}:${s.slice(12,14)}`;
  }
  const hasFinite = (arr) => Array.isArray(arr)
    && arr.some((v) => v != null && !Number.isNaN(v) && isFinite(v));

  // ---- Sweep-visualization prefs (mirrors the SLM dashboard's
  // sweepPrefs): error metric + 2D view/swap/square cells, persisted to
  // localStorage so they survive reloads + scan switches. ----
  const SWEEP_PREFS_KEY = "yb_dash_analysis_sweep_prefs";
  const sweepPrefs = (() => {
    const defaults = { errMode: "sem_pershot", view: "both",
                       axisSwap: false, square: false };
    try {
      const saved = JSON.parse(localStorage.getItem(SWEEP_PREFS_KEY) || "{}");
      return Object.assign(defaults, saved || {});
    } catch { return defaults; }
  })();
  function saveSweepPrefs() {
    try { localStorage.setItem(SWEEP_PREFS_KEY, JSON.stringify(sweepPrefs)); }
    catch { /* private mode */ }
  }
  let recomputeInfid = false;       // refit discrimination from this run's data
  let svdMode = "total";            // survival-vs-distance x-axis: total|per_step

  // Shrink the font of each top-row status value until it fits its tile
  // (no ellipsis). Resets to the stylesheet base first so values that got
  // shorter grow back. Skips hidden/unlaid-out tiles (clientWidth 0).
  function fitStatValues() {
    $$(".status-strip .stat-value").forEach((el) => {
      el.style.fontSize = "";                      // back to stylesheet base
      if (!el.clientWidth) return;                 // hidden / not laid out
      const base = parseFloat(getComputedStyle(el).fontSize) || 20;
      let size = base, guard = 0;
      while (el.scrollWidth > el.clientWidth && size > 8 && guard++ < 48) {
        size -= 1;
        el.style.fontSize = size + "px";
      }
    });
  }

  function toast(msg, kind) {
    const t = $("toast");
    if (!t) return;
    t.textContent = msg;
    t.className = "toast show" + (kind ? " " + kind : "");
    clearTimeout(t._h);
    t._h = setTimeout(() => { t.className = "toast"; }, 3500);
  }

  async function api(path, opts) {
    opts = opts || {};
    const r = await fetch(path, opts);
    let body = null;
    try { body = await r.json(); } catch {}
    if (!r.ok) {
      const msg = (body && body.error) || r.statusText;
      const err = new Error(msg);
      err.status = r.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  // ---- Tab switching ----
  const TABS = ["live", "hardware", "analysis", "queue", "diag", "sequence"];
  let activeTab = "live";

  function setTab(tab) {
    if (!TABS.includes(tab)) return;
    activeTab = tab;
    TABS.forEach((t) => {
      $("tab-btn-" + t).classList.toggle("active", t === tab);
      $("tab-btn-" + t).setAttribute("aria-selected", String(t === tab));
      $("tab-" + t).hidden = (t !== tab);
    });
    // Floating analysis overlay (Runs + Filter cards) tracks the
    // active tab -- only visible when Analysis is selected. No-op if
    // FLOATING_ANALYSIS_CARDS=false: the host stays hidden anyway.
    const floatHost = document.getElementById("floating-analysis-host");
    if (floatHost) floatHost.hidden = (tab !== "analysis");
    // The Sequence tab's floating pickers live in TWO hosts now: Channels
    // docked right (#floating-sequence-host), Scans docked left
    // (#floating-seqscan-host). Both track the Sequence tab.
    const seqFloatHost = document.getElementById("floating-sequence-host");
    if (seqFloatHost) seqFloatHost.hidden = (tab !== "sequence");
    const seqScanHost = document.getElementById("floating-seqscan-host");
    if (seqScanHost) seqScanHost.hidden = (tab !== "sequence");
    if (location.hash !== "#" + tab) location.hash = "#" + tab;
    // Render once on tab switch in case the poll interval hasn't fired.
    pollOnceForTab(tab);
    if (tab === "analysis") { try { loadCalibration(); } catch (e) {} }
    // Plotly re-sizes layouts when the container becomes visible.
    setTimeout(() => {
      if (window.Plotly) {
        $$(".plot-container", $("tab-" + tab)).forEach((el) => {
          try { Plotly.Plots.resize(el); } catch {}
        });
      }
    }, 50);
  }

  function initTabs() {
    TABS.forEach((t) => {
      $("tab-btn-" + t).addEventListener("click", () => setTab(t));
    });
    const hash = location.hash.replace("#", "");
    if (TABS.includes(hash)) setTab(hash);
  }

  // ---- Calibration card: global SLM->camera affine + loading patterns ----
  function _calFmt(v, d) {
    return (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toFixed(d);
  }
  async function loadCalibration() {
    const body = document.getElementById("calib-body");
    const statusEl = document.getElementById("calib-status");
    if (!body) return;
    try {
      const [aff, pats] = await Promise.all([
        api("/api/affine/current"), api("/api/patterns"),
      ]);
      let html = "";
      if (aff && aff.A) {
        const A = aff.A;
        html += '<div class="mono" style="font-size:12px;line-height:1.6;">';
        html += '<b>SLM&rarr;camera affine</b> (knm[x,y] &rarr; absolute camera[Y,X]; crop applied separately)<br>';
        html += `rotation ${_calFmt(aff.rotation_deg, 2)}&deg; &nbsp; scale ${_calFmt(aff.scale_x, 3)} / ${_calFmt(aff.scale_y, 3)} &nbsp; rms ${_calFmt(aff.rms_px, 2)} px &nbsp; coverage ${_calFmt((aff.coverage || 0) * 100, 1)}%<br>`;
        html += `last scan ${aff.last_scan_id || "—"} &nbsp; updated ${aff.updated_iso || "—"}${aff.rolled_back ? " (rolled back)" : ""}<br>`;
        html += `A = [[${_calFmt(A[0][0], 4)}, ${_calFmt(A[0][1], 4)}, ${_calFmt(A[0][2], 2)}], [${_calFmt(A[1][0], 4)}, ${_calFmt(A[1][1], 4)}, ${_calFmt(A[1][2], 2)}]]`;
        html += "</div>";
        if (statusEl) statusEl.textContent = `affine rot ${_calFmt(aff.rotation_deg, 1)}° rms ${_calFmt(aff.rms_px, 2)}px`;
      } else {
        html += '<div class="hint">No affine committed yet — bootstrap with <span class="mono">yb_analysis.scripts.bootstrap_affine</span>.</div>';
        if (statusEl) statusEl.textContent = "no affine";
      }
      const names = (pats && pats.patterns) ? Object.keys(pats.patterns) : [];
      html += '<h3 style="margin:12px 0 4px;">Loading patterns</h3>';
      if (names.length) {
        html += '<table class="mono" style="font-size:12px;border-collapse:collapse;">';
        html += '<tr><th style="text-align:left;padding:2px 12px;">pattern</th><th>sites</th><th>order</th><th>defocus z</th><th>updated</th></tr>';
        for (const n of names) {
          const m = pats.patterns[n] || {};
          const z = m.default_loading_zernike ? JSON.stringify(m.default_loading_zernike) : "—";
          html += `<tr><td style="padding:2px 12px;">${n}</td><td style="text-align:center;">${m.n_sites || "—"}</td><td style="text-align:center;">${m.order || "—"}</td><td style="text-align:center;">${z}</td><td style="text-align:center;">${m.updated_iso || "—"}</td></tr>`;
        }
        html += "</table>";
      } else {
        html += '<div class="hint">No loading patterns registered yet.</div>';
      }
      body.innerHTML = html;
    } catch (e) {
      body.innerHTML = `<div class="hint">Calibration unavailable: ${e.message || e}</div>`;
    }
  }

  // ---- Polling orchestrator ----
  let autoRefresh = true;
  const timers = {};

  function pollOnceForTab(tab) {
    if (tab === "live")     return pollLive();
    // Hardware tab is just an iframe wrapping the SLM dashboard;
    // no lab-side polling needed -- the iframe drives its own.
    if (tab === "queue")    return pollQueue();
    if (tab === "diag")     return pollDiag();
    if (tab === "sequence") return loadSequence();
    // Analysis: the top-right Refresh re-fetches the runs list and reloads
    // the current analysis (otherwise it's a no-op on this tab).
    if (tab === "analysis") {
      loadRunsList();
      if (selectedScanId) loadAnalysis(selectedScanId, {keepFilters: true});
      return;
    }
  }

  // Hook the iframe reload button now that the DOM is ready (wired
  // at bootstrap, not here).

  function startPolling() {
    function loop(tab, fn, interval) {
      const tick = async () => {
        if (autoRefresh && activeTab === tab) {
          try { await fn(); } catch (e) { console.warn(tab, e); }
        }
        timers[tab] = setTimeout(tick, interval);
      };
      tick();
    }
    loop("live",     pollLive,     POLL.live);
    // Hardware tab is a self-contained iframe to the SLM dashboard --
    // it owns its own polling. We DON'T poll /api/slm/* here, or we
    // get null-querySelector crashes against UI elements that only
    // existed in the old manual-port version of this tab.
    loop("queue",    pollQueue,    POLL.queue);
    // Connection-status pill polls every 5s regardless of active tab.
    setInterval(updateConnStatus, 5000);
    updateConnStatus();
  }

  // On viewport resize, re-fit the top-row status fonts and re-size the
  // per-site map dots (pixels-per-data-unit changes with the container).
  // Debounced so a drag-resize doesn't thrash restyles.
  let _resizeFitTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(_resizeFitTimer);
    _resizeFitTimer = setTimeout(() => {
      fitStatValues();
      if (window.Plotly) LIVE_AUTOSIZE_PANELS.forEach(autoFitDotSize);
    }, 200);
  });

  $("auto-refresh").addEventListener("change", (e) => {
    autoRefresh = e.target.checked;
  });
  $("manual-refresh").addEventListener("click", () => {
    pollOnceForTab(activeTab);
    toast("Refreshed", "warn");
  });

  // Calibration card buttons (Analysis tab).
  (function wireCalibration() {
    const rb = document.getElementById("calib-refresh");
    if (rb) rb.addEventListener("click", () => loadCalibration());
    const back = document.getElementById("calib-rollback");
    if (back) back.addEventListener("click", async () => {
      try {
        const r = await api("/api/affine/rollback", {method: "POST"});
        toast(r && r.ok ? "Affine rolled back" : "Nothing to roll back",
              r && r.ok ? "warn" : "err");
      } catch (e) { toast("Rollback failed: " + (e.message || e), "err"); }
      loadCalibration();
    });
  })();

  // Survival-vs-distance x-axis mode (total | per-step). Re-render from the
  // already-loaded analysis (both curves are in the payload — no refetch).
  (function wireSvdMode() {
    const sel = document.getElementById("svd-mode");
    if (sel) sel.addEventListener("change", () => {
      svdMode = sel.value;
      if (activeAnalysis) renderSurvivalVsDistance(activeAnalysis);
    });
  })();

  // Site picker: changing the number reruns the site histogram against
  // the new index. Debounced via natural Plotly poll cadence.
  const sitePick = document.getElementById("site-pick");
  if (sitePick) {
    sitePick.addEventListener("input", () => {
      const v = Math.max(1, Math.floor(Number(sitePick.value) || 1));
      selectedSiteIdx = v;
      pollSiteHist();
    });
  }

  // ---- Connection status (top-bar pills) ----
  // Labels stay short and stable ("lab" / "SLM" / "runner") -- the dot
  // color carries the state. No more "ok" suffix, no label churn.
  async function updateConnStatus() {
    const pill = $("conn-status");
    try {
      const r = await fetch("/api/status", {cache: "no-store"});
      if (!r.ok) throw new Error(r.statusText);
      pill.className = "status-pill";
      pill.title = "Lab dashboard: connected";
    } catch {
      pill.className = "status-pill bad";
      pill.title = "Lab dashboard: offline";
    }
    try {
      const q = await api("/api/queue");
      const p = $("runner-status");
      p.className = (q !== null) ? "status-pill" : "status-pill dim";
      p.title = (q !== null) ? "MATLAB runner: up" : "MATLAB runner: idle";
    } catch {
      const p = $("runner-status");
      p.className = "status-pill bad";
      p.title = "MATLAB runner: offline";
    }
    try {
      await api("/api/slm/health");
      const p = $("slm-status");
      p.className = "status-pill";
      p.title = "SLM PC proxy: up";
    } catch (e) {
      const p = $("slm-status");
      p.className = e.status === 503 ? "status-pill warn" : "status-pill bad";
      p.title = e.status === 503 ? "SLM proxy: busy" : "SLM proxy: error";
    }
  }

  // =====================================================================
  // LIVE TAB — server-rendered Plotly figures.
  //
  // The Dash app builds Plotly Figure objects via `_fig_*` functions.
  // Rather than re-port that logic to JS (and drift from production),
  // we expose `/api/live/figures` which returns each figure as JSON
  // (the same JSON Plotly.py would have shipped to Dash's callback
  // output). The client just hands `figure.data` + `figure.layout`
  // to `Plotly.react()`. Byte-identical to the old Live page.
  // =====================================================================
  const LIVE_FIG_PANELS = [
    ["array",     "plot-array1"],
    ["array_mid", "plot-array-mid"],
    ["array2",    "plot-array2"],
    ["scan",      "plot-scan"],
    ["intens",    "plot-intensities"],
    ["loadlive",  "plot-loading-live"],
    ["load",      "plot-load-map"],
    ["infid",     "plot-infid-map"],
    ["shift",     "plot-shift"],
    ["avghist",   "plot-hist-avg"],
    ["rep0",      "plot-hist-rep0"],
    ["rep1",      "plot-hist-rep1"],
    ["rep2",      "plot-hist-rep2"],
    ["rep3",      "plot-hist-rep3"],
  ];

  // Currently-selected site for the per-site histogram panel.
  let selectedSiteIdx = 1;

  // Live top-row state: last-seen NumImages and the operator's "Middle
  // frame" switch preference (persisted; default ON). applyMidVisibility()
  // reconciles the two to show/hide the middle card and swap col widths.
  let lastNumImages = 0;
  let showMidFrame = (() => {
    try { return localStorage.getItem("yb-dash-show-mid") !== "0"; }
    catch { return true; }
  })();

  // Scan-curve colorbar autoscale switch (2-D heatmaps only): OFF pins the
  // Survival/Loading colorbar to 0–1, ON lets it autoscale to the data range
  // (sent as ?cbar_scale=auto to the figures endpoint). Persisted; default
  // OFF — 0–1 is the right reference frame for survival/loading.
  let scanAutoscale = (() => {
    try { return localStorage.getItem("yb-dash-scan-autoscale") === "1"; }
    catch { return false; }
  })();

  function applyMidVisibility() {
    const cardMid  = $("card-array-mid");
    const card1    = $("card-array1");
    const card2    = $("card-array2");
    const cardScan = $("card-scan-curve");
    if (!cardMid || !card1 || !card2) return;
    const hasMid = lastNumImages >= 3 && showMidFrame;
    const _swapCol = (el, from, to) => {
      if (!el) return;
      el.classList.remove(from);
      el.classList.add(to);
    };
    if (hasMid) {
      cardMid.hidden = false;
      _swapCol(card1,    "col-4", "col-3");
      _swapCol(cardMid,  "col-4", "col-3");
      _swapCol(card2,    "col-4", "col-3");
      _swapCol(cardScan, "col-4", "col-3");
    } else {
      cardMid.hidden = true;
      _swapCol(card1,    "col-3", "col-4");
      _swapCol(card2,    "col-3", "col-4");
      _swapCol(cardScan, "col-3", "col-4");
    }
    // The "Middle frame" switch is meaningless when the scan has < 3
    // images; dim it so the operator knows it has no effect right now.
    const midSwitch = $("show-mid-switch");
    if (midSwitch) midSwitch.style.opacity = lastNumImages >= 3 ? "" : "0.45";
  }

  async function pollLive() {
    // Status card uses /api/snapshot (small fields).
    let snap = null;
    try { snap = await api("/api/snapshot"); } catch (e) {
      console.warn("snapshot failed", e);
    }
    if (snap) {
      setText("kv-scan-id",    snap.scan_id != null ? String(snap.scan_id) : "—");
      setText("kv-scan-name",  snap.scan_name || snap.scan_filename || "—");
      setText("kv-shot",       snap.n_accum_shots ?? "—");
      setText("kv-sites",      snap.num_sites ?? "—");
      const lr = avg(snap.loading_rates);
      setText("kv-loading-rate", lr != null ? fmtPct(lr) : "—");
      if ($("scan-progress")) {
        const cur = snap.n_accum_shots || 0;
        $("scan-progress").textContent = `${cur} shots`;
      }
      // Show/hide the middle-image card. It appears only when the scan
      // produces a middle frame (NumImages >= 3) AND the operator hasn't
      // turned it off via the "Middle frame" switch. applyMidVisibility()
      // owns the actual DOM/col-swap so the switch handler can reuse it.
      lastNumImages = snap.num_images != null ? Number(snap.num_images) : 0;
      applyMidVisibility();
      // Site picker bounds + info block.
      const ns = Math.max(1, Number(snap.num_sites || 1));
      const picker = $("site-pick");
      if (picker) {
        picker.max = String(ns);
        if (Number(picker.value) > ns) picker.value = String(ns);
      }
      renderSiteInfo(snap, selectedSiteIdx);
      renderControlSidebar(snap);
    }
    // Queue preview in the Yb Control sidebar (running + next-up + recent
    // history). Same /api/queue source as the Queue tab. Sets
    // queueActiveRunning, which pollLiveDiag relies on below.
    try {
      const q = await api("/api/queue");
      renderSidebarQueue(q);
    } catch (e) { /* keep last-known state on transient error */ }
    // Live diag pull-poll (after the queue render: needs queueActiveRunning).
    await pollLiveDiag(snap);
    // Auto-shrink the top-row status values to fit their tiles (the diag
    // readout is set inside pollLiveDiag above, so fit after it).
    fitStatValues();
    // Control-sidebar state (dummy mode, last seq, runner state, exposure
    // gate) from the main process.
    await pollControlStatus();
    // Camera card (status + ROI + exposure) from the main process.
    await pollCameraStatus();
    // Plots come pre-built from the server's /api/live/figures.
    await pollLiveFigures();
    // Per-site hist refreshes too (separate endpoint with site index).
    await pollSiteHist();
  }

  // Compact queue preview in the Yb Control sidebar — mirrors the Tkinter
  // queue pane (running + next-up + recent history). The full table with
  // move/cancel lives in the Queue tab. Fed by the same /api/queue source.
  const SIDEBAR_QUEUE_MAX = 5;     // queued rows shown
  const SIDEBAR_HISTORY_MAX = 5;   // recent-history rows shown

  function renderSidebarQueue(q) {
    if (!q) return;
    const running = q.running;
    const queued  = q.queued || [];
    const history = q.history || [];

    // --- Active line ---
    const actEl = $("ctrl-queue-active");
    const actText = $("ctrl-queue-active-text");
    if (actEl && actText) {
      if (running) {
        const label = running.label || running.seqName || `job #${running.id}`;
        let state = "running";
        if ((running.state || "").toLowerCase() === "building" ||
            (running.kind === "descriptor" && running.state !== "running")) {
          state = "loading";
        }
        actEl.dataset.state = state;
        actText.textContent =
          `${state === "loading" ? "loading… " : ""}${label}`;
        queueActiveRunning = (state === "running");
      } else {
        actEl.dataset.state = "idle";
        actText.textContent = "(idle)";
        queueActiveRunning = false;
      }
    }

    // --- Preview list: next-up queued + recent history ---
    const listEl = $("ctrl-queue-list");
    if (!listEl) return;
    const rows = [];
    queued.slice(0, SIDEBAR_QUEUE_MAX).forEach((e) => {
      const label = e.label || e.seqName || `#${e.id}`;
      rows.push(
        `<li class="q-queued"><span class="q-mark">·</span>` +
        `<span class="q-name">${escHtml(label)}</span>` +
        `<span class="q-status">#${e.id}</span></li>`);
    });
    if (queued.length > SIDEBAR_QUEUE_MAX) {
      rows.push(`<li class="q-sep">+${queued.length - SIDEBAR_QUEUE_MAX} more queued</li>`);
    }
    if (history.length) {
      rows.push(`<li class="q-sep">recent</li>`);
      history.slice(0, SIDEBAR_HISTORY_MAX).forEach((e) => {
        const label = e.label || e.seqName || `#${e.id}`;
        const ok = (e.status || e.state) === "ok";
        const mark = ok ? "+" : "×";
        const statusTxt = ok ? "ok" : (e.status || e.state || "err");
        rows.push(
          `<li class="q-history ${ok ? "q-ok" : "q-err"}">` +
          `<span class="q-mark">${mark}</span>` +
          `<span class="q-name">${escHtml(label)}</span>` +
          `<span class="q-status">${escHtml(statusTxt)}</span></li>`);
      });
    }
    listEl.innerHTML = rows.length
      ? rows.join("")
      : `<li class="ctrl-queue-empty">queue empty</li>`;
  }

  function renderSiteInfo(snap, idx1) {
    const info = $("site-info");
    if (!info || !snap) return;
    const i = idx1 - 1;
    const t = (snap.thresholds || [])[i];
    const inf = (snap.infidelities || [])[i];
    const rate = (snap.loading_rates || [])[i];
    const lines = [
      `site = ${idx1}`,
      `threshold = ${t != null ? fmtNum(t) : "—"}`,
      `loading   = ${rate != null ? fmtPct(rate) : "—"}`,
      `infidelity = ${inf != null ? inf.toExponential(2) : "—"}`,
    ];
    info.innerHTML = lines.map(escHtml).join("<br>");
  }

  // --- Control sidebar (Phase 5.5 Track A) ---------------------------
  // Timestamp of the operator's last dummy-radio click, so a status poll
  // arriving mid-flight doesn't yank the radio back before the change has
  // round-tripped through the main process.
  let lastDummyUserChangeMs = 0;
  let controlsAllowed = true;
  // Active sequence backend ('matlab' | 'pyctrl'), published by the main
  // process. Drives the backend-toggle highlight and blocks switching to the
  // already-active backend.
  let currentBackend = null;
  // True while a job is actively running (not just building/loading). Set by
  // renderSidebarQueue; read by pollLiveDiag (which used to read the now-
  // removed mini-queue tile's CSS class).
  let queueActiveRunning = false;
  // Timestamp of the operator's last camera-field edit, so a status poll
  // doesn't overwrite a value they're about to apply (4 s grace).
  let camFieldsTouched = 0;

  function renderControlSidebar(snap) {
    if (!snap) return;
    setText("ctrl-scan-id",
            snap.scan_id != null ? String(snap.scan_id) : "—");
    setText("ctrl-scan-file",
            snap.scan_filename || snap.scan_name || "—");
  }

  async function pollControlStatus() {
    let st = null;
    try { st = await api("/api/control/status"); }
    catch (e) { return; }   // keep last-known on transient error
    if (!st) return;

    // Remote-exposure gate: hide/disable controls when the server says so.
    controlsAllowed = st.controls_allowed !== false;
    const banner = $("ctrl-disabled-banner");
    if (banner) banner.hidden = controlsAllowed;
    document.querySelectorAll(".live-sidebar .ctrl-btn").forEach((b) => {
      b.classList.toggle("is-disabled", !controlsAllowed);
    });

    if (st.seq_id != null) setText("ctrl-seq-id", String(st.seq_id));
    if (st.state) setText("ctrl-runner-state", String(st.state));
    renderRunnerStatusBanner(st.state);

    // Backend toggle: highlight the active backend and disable its button
    // (switching to the already-active backend is a no-op).
    if (st.backend) {
      currentBackend = st.backend;
      const labels = {matlab: "MATLAB", pyctrl: "pyctrl"};
      setText("ctrl-backend-active", labels[st.backend] || st.backend);
      document.querySelectorAll(
        ".backend-btn[data-backend-target]").forEach((b) => {
        const isActive = b.dataset.backendTarget === st.backend;
        b.classList.toggle("is-active", isActive);
      });
    }

    // Dummy-mode radio sync (unless the operator just clicked one).
    if (st.dummy_mode && Date.now() - lastDummyUserChangeMs > 4000) {
      const r = document.querySelector(
        `input[name="dummy-mode-ph5"][value="${st.dummy_mode}"]`);
      if (r) r.checked = true;
    }
    // Last-seq label, mirroring the Tkinter window's wording.
    const ls = st.last_seq || {};
    const lastEl = $("ctrl-dummy-last");
    if (lastEl) {
      const name = ls.name || "(unnamed)";
      const fid = ls.file_id ? ` (${ls.file_id})` : "";
      if (st.dummy_mode === "last") {
        lastEl.textContent = ls.available
          ? `Replaying: ${name}${fid}`
          : "Running default (no last seq cached)";
      } else {
        lastEl.textContent = ls.available ? `Cached: ${name}${fid}` : "Cached: —";
      }
    }
  }

  // Map a runner-status string to a data-state class. Mirrors
  // control_panel.py's _STATUS_COLORS (Idle (last seq) → blue,
  // Idle (last fallback) → amber, plain Idle → gray, Running → green,
  // Paused/Pausing… → amber, Stopped → red).
  function statusToState(s) {
    const t = (s || "").toLowerCase();
    if (t.includes("running")) return "running";
    if (t.includes("paus")) return "paused";          // Paused / Pausing...
    if (t.includes("stopped")) return "stopped";
    if (t.includes("last seq")) return "idleblue";
    if (t.includes("fallback")) return "idlewarn";
    if (t.includes("idle")) return "idle";
    return "unknown";
  }

  function renderRunnerStatusBanner(state) {
    const banner = $("ctrl-status-banner");
    const text = $("ctrl-status-text");
    if (!banner || !text) return;
    text.textContent = state || "—";
    banner.dataset.state = statusToState(state);
  }

  // --- Camera card (Phase 5.5) — mirrors the Tkinter CameraPane --------
  async function pollCameraStatus() {
    let st = null;
    try { st = await api("/api/control/camera/status"); }
    catch (e) { return; }   // keep last-known on transient error
    if (st) renderCameraStatus(st);
  }

  function setCamField(id, val) {
    const el = $(id);
    if (el && String(el.value) !== String(val)) el.value = String(val);
  }

  function renderCameraStatus(st) {
    const box = $("cam-status");
    const txt = $("cam-status-text");
    if (box && txt) {
      txt.textContent = st.status_text ||
        (st.connected ? "Connected" : "Disconnected");
      let state = "disconnected";
      if (st.busy) state = "busy";
      else if (st.connected) state = "connected";
      else if (st.error) state = "error";
      box.dataset.state = state;
    }
    const errEl = $("cam-error");
    if (errEl) {
      if (st.error) { errEl.textContent = st.error; errEl.hidden = false; }
      else { errEl.textContent = ""; errEl.hidden = true; }
    }
    // Extended Orca telemetry (pyctrl backend only). The MATLAB backend never
    // sends these keys, so the block stays hidden there.
    renderCameraTelemetry(st);
    // Sync ROI/exposure fields from the server unless the operator is
    // editing them or edited them in the last 4 s (about to Apply).
    if (Date.now() - camFieldsTouched < 4000) return;
    const active = document.activeElement;
    if (active && active.classList &&
        active.classList.contains("cam-num")) return;
    if (Array.isArray(st.roi) && st.roi.length === 4) {
      setCamField("cam-roi-x", st.roi[0]);
      setCamField("cam-roi-y", st.roi[1]);
      setCamField("cam-roi-w", st.roi[2]);
      setCamField("cam-roi-h", st.roi[3]);
    }
    if (st.exposure_time != null) setCamField("cam-exposure", st.exposure_time);
  }

  function renderCameraTelemetry(st) {
    const box = $("cam-telemetry");
    if (!box) return;
    const hasTrigger = st.trigger != null && st.trigger !== "";
    const hasCooler = st.cooler != null && st.cooler !== "";
    const hasTemp = st.temperature != null && st.temperature !== "";
    // Only the pyctrl backend reports these; hide the block entirely otherwise.
    if (!hasTrigger && !hasCooler && !hasTemp) { box.hidden = true; return; }
    box.hidden = false;
    const trig = $("cam-trigger");
    if (trig) trig.textContent = hasTrigger ? st.trigger : "—";
    const cool = $("cam-cooler");
    if (cool) {
      let txt = hasCooler ? String(st.cooler) : "—";
      if (st.cooler_status) txt += ` (${st.cooler_status})`;
      cool.textContent = txt;
    }
    const temp = $("cam-temperature");
    if (temp) temp.textContent = hasTemp ? `${Number(st.temperature).toFixed(1)} °C` : "—";
  }

  // --- Live diag pull-poll (Phase 5.5 Track D) -----------------------
  // While a scan is running, incrementally pull SLM per-shot diag rows
  // (~2 s cadence) and show a glanceable count + last seq. Stops when the
  // scan is no longer the active job; post-scan sync writes the sidecar.
  const liveDiag = {scanId: null, lastSeqId: null, count: 0, lastFetchMs: 0};

  async function pollLiveDiag(snap) {
    const tile = $("live-diag-tile");
    if (!tile) return;
    const scanId = snap && snap.scan_id != null ? String(snap.scan_id) : null;
    const isLive = !!scanId && queueActiveRunning;
    if (!isLive) return;     // not live: leave last readout, stop polling
    if (scanId !== liveDiag.scanId) {     // new scan -> reset the buffer
      liveDiag.scanId = scanId;
      liveDiag.lastSeqId = null;
      liveDiag.count = 0;
    }
    const now = Date.now();
    if (now - liveDiag.lastFetchMs < 1800) return;   // throttle to ~2 s
    liveDiag.lastFetchMs = now;
    let url = `/api/runs/${scanId}/diag_live`;
    if (liveDiag.lastSeqId != null) url += `?since_seq_id=${liveDiag.lastSeqId}`;
    let d = null;
    try { d = await api(url); }
    catch (e) { return; }     // SLM offline -> retry next tick
    if (!d) return;
    const entries = d.entries || [];
    if (entries.length) {
      const seqs = entries.map((e) => e.seq_id)
        .filter((s) => typeof s === "number");
      if (seqs.length) liveDiag.lastSeqId = Math.max(...seqs);
    }
    if (d.count != null) liveDiag.count = d.count;
    else liveDiag.count += entries.length;
    tile.hidden = false;
    setText("live-diag-readout",
            `${liveDiag.count} rows · seq ${liveDiag.lastSeqId ?? "—"}`);
  }

  async function pollSiteHist() {
    if (!window.Plotly) return;
    const el = $("plot-hist-site");
    if (!el) return;
    try {
      const url = `/api/live/figures?which=site&site=${selectedSiteIdx}`;
      const f = await api(url);
      if (!f || !Array.isArray(f.data)) return;
      Plotly.react(el, f.data, f.layout || {},
                   {displayModeBar: false, responsive: true});
    } catch (e) {
      console.warn("site hist fetch failed", e);
    }
  }

  // Visible banner if Plotly.js failed to load -- replaces the old
  // silent failure mode where pollLiveFigures bailed early and every
  // live-tab panel stayed empty (rendered as card-bg dark blue, looking
  // "completely black"). main.html sets window.__plotlyLoadFailed=true
  // from the script-tag onerror handler if both local + CDN fail.
  function showPlotlyMissingBanner() {
    if (document.getElementById("plotly-missing-banner")) return;
    const bar = document.createElement("div");
    bar.id = "plotly-missing-banner";
    bar.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:9999;" +
      "background:#f85149;color:#fff;padding:8px 16px;text-align:center;" +
      "font-family:ui-monospace,monospace;font-size:13px;";
    bar.textContent =
      "Plotly.js failed to load (local /vendor/plotly.min.js + CDN both 4xx/5xx). " +
      "Live + Analysis charts cannot render. Check the run_monitor log + browser DevTools Network tab.";
    document.body.appendChild(bar);
  }

  // ---- Auto-fit dot size for the Live per-site maps ------------------
  // The Loading-rate + Discrimination-infidelity panels are server-rendered
  // Scattergl traces (one marker per tweezer site) on a square (scaleanchor)
  // axis. Plotly marker sizes are in SCREEN PIXELS, so the right
  // non-overlapping size depends on both the site pitch (data units) and
  // the current zoom (pixels per data unit). We size each dot to ~the
  // nearest-neighbor pitch so dots nearly touch without overlapping, and
  // re-fit on zoom / pan / resize. This overrides the server's default
  // marker.size for these two panels only.
  const LIVE_AUTOSIZE_PANELS = ["plot-load-map", "plot-infid-map"];
  const _autoSizeNN = {};   // divId -> {len, nn}  (nn = median pitch, data units)

  // Median nearest-neighbor distance via a uniform spatial hash (O(n) for a
  // lattice; the array is roughly regular). Returns null for < 2 points.
  function _medianNearestNeighbor(xs, ys) {
    const n = Math.min(xs.length, ys.length);
    if (n < 2) return null;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      if (xs[i] < minX) minX = xs[i];
      if (xs[i] > maxX) maxX = xs[i];
      if (ys[i] < minY) minY = ys[i];
      if (ys[i] > maxY) maxY = ys[i];
    }
    const w = Math.max(maxX - minX, 1e-9), h = Math.max(maxY - minY, 1e-9);
    // Cell ~ expected pitch so each cell holds ~1 site; a 3x3 neighbor
    // sweep then finds the true nearest neighbor for a regular lattice.
    const cell = Math.max(Math.sqrt((w * h) / n), 1e-9);
    const buckets = new Map();
    const ci = (v, lo) => Math.floor((v - lo) / cell);
    const key = (cx, cy) => cx + "," + cy;
    for (let i = 0; i < n; i++) {
      const k = key(ci(xs[i], minX), ci(ys[i], minY));
      let b = buckets.get(k);
      if (!b) { b = []; buckets.set(k, b); }
      b.push(i);
    }
    const dists = [];
    for (let i = 0; i < n; i++) {
      const gx = ci(xs[i], minX), gy = ci(ys[i], minY);
      let best = Infinity;
      for (let ox = -1; ox <= 1; ox++) {
        for (let oy = -1; oy <= 1; oy++) {
          const b = buckets.get(key(gx + ox, gy + oy));
          if (!b) continue;
          for (let t = 0; t < b.length; t++) {
            const j = b[t];
            if (j === i) continue;
            const dx = xs[i] - xs[j], dy = ys[i] - ys[j];
            const d2 = dx * dx + dy * dy;
            if (d2 < best) best = d2;
          }
        }
      }
      if (best < Infinity) dists.push(Math.sqrt(best));
    }
    if (!dists.length) return null;
    dists.sort((a, b) => a - b);
    return dists[dists.length >> 1];
  }

  function autoFitDotSize(divId) {
    const el = $(divId);
    if (!el || !window.Plotly || !el._fullLayout || !el.data || !el.data[0]) return;
    const tr = el.data[0];
    if (!tr.x || !tr.x.length) return;
    const xa = el._fullLayout.xaxis;
    if (!xa || !xa.range || !xa._length) return;
    const span = Math.abs(xa.range[1] - xa.range[0]);
    if (!span) return;
    const pxPerData = xa._length / span;
    if (!isFinite(pxPerData)) return;
    let st = _autoSizeNN[divId];
    if (!st || st.len !== tr.x.length) {
      st = { len: tr.x.length, nn: _medianNearestNeighbor(tr.x, tr.y) };
      _autoSizeNN[divId] = st;
    }
    if (!st.nn) return;
    // 0.85 leaves a hairline gap (and absorbs the 0.5px white outline).
    let sz = st.nn * pxPerData * 0.85;
    sz = Math.max(2, Math.min(60, sz));
    const cur = (tr.marker && tr.marker.size) || 0;
    if (Math.abs(cur - sz) > 0.3) {
      try { Plotly.restyle(el, { "marker.size": sz }, [0]); } catch (e) {}
    }
  }

  // Re-fit on zoom / pan / dblclick-reset. Restyle does NOT fire
  // plotly_relayout, so there's no feedback loop. Wired once per panel.
  function wireLiveMapAutoFit(divId) {
    const el = $(divId);
    if (!el || el._autofitWired) return;
    el._autofitWired = true;
    el.on("plotly_relayout", () => autoFitDotSize(divId));
  }

  async function pollLiveFigures() {
    // Set "waiting..." placeholders FIRST so we have visible state
    // regardless of whether Plotly loaded. Old code did this AFTER the
    // window.Plotly check, leaving panels black when the CDN failed.
    const plotlyOk = !!window.Plotly && !window.__plotlyLoadFailed;
    if (!plotlyOk) {
      showPlotlyMissingBanner();
      LIVE_FIG_PANELS.forEach(([name, divId]) => {
        const el = $(divId);
        if (el && !el.querySelector(".plotly")) {
          el.innerHTML =
            '<div class="hint" style="padding:24px;text-align:center;color:#f85149;">' +
            'Plotly.js missing — see banner</div>';
        }
      });
      return;
    }
    let resp = null;
    // Autoscale switch (scan card) rides on the batch fetch: the /api/live/
    // figures endpoint reads cbar_scale and applies it to the scan figure.
    const figUrl = scanAutoscale
      ? "/api/live/figures?cbar_scale=auto"
      : "/api/live/figures";
    try { resp = await api(figUrl); }
    catch (e) {
      console.warn("figures fetch failed", e);
      LIVE_FIG_PANELS.forEach(([name, divId]) => {
        const el = $(divId);
        if (el && !el.querySelector(".plotly")) {
          el.innerHTML =
            '<div class="hint" style="padding:24px;text-align:center;color:#f85149;">' +
            'fetch failed: ' + escHtml(e.message || String(e)) + '</div>';
        }
      });
      return;
    }
    const figures = (resp && resp.figures) || {};
    LIVE_FIG_PANELS.forEach(([name, divId]) => {
      const el = $(divId);
      if (!el) return;
      const f = figures[name];
      if (!f || !Array.isArray(f.data)) {
        if (!el.querySelector(".plotly")) {
          el.innerHTML =
            '<div class="hint" style="padding:24px;text-align:center;">waiting…</div>';
        }
        return;
      }
      // f.data is an array (possibly empty for _waiting() placeholders).
      // Plotly.react with empty data + annotations-only layout renders
      // the annotation text -- so panels with no data still SHOW the
      // "Waiting for data..." label rather than going black.
      try {
        Plotly.react(el, f.data, f.layout || {}, {
          displayModeBar: false, responsive: true,
        });
        // Loading-rate + infidelity site maps: size dots to the tweezer
        // pitch so they nearly touch without overlapping, and keep them
        // correctly sized on zoom/pan (rAF lets Plotly finish layout first).
        if (name === "load" || name === "infid") {
          wireLiveMapAutoFit(divId);
          requestAnimationFrame(() => autoFitDotSize(divId));
        }
      } catch (err) {
        console.error("plot render failed for", name, err);
        el.innerHTML =
          '<div class="hint" style="padding:24px;text-align:center;color:#f85149;">' +
          'render error: ' + escHtml(err.message || String(err)) + '</div>';
      }
    });
  }

  function renderLiveArray(divId, snap, shapeKey, vloKey, vhiKey,
                           logicalsKey, gridKey, imageUrl) {
    // NOTE: the snap object includes _img_shape / _img_vlo / _img_vhi
    // (the data URI itself is excluded from /api/snapshot for size and
    // fetched separately as PNG bytes from imageUrl). The Plotly heatmap
    // is replaced with a layout.images entry pointing at imageUrl.
    // shapeKey can be passed as positional 3rd arg; rewrite for clarity.
  }
  // The 6-positional-arg signature above was getting unwieldy; rewrite:
  function renderLiveArray(divId, snap,
                           uriKey, shapeKey, vloKey, vhiKey,
                           logicalsKey, gridKey, imageUrl) {
    if (!window.Plotly) return;
    const el = $(divId);
    if (!el) return;
    const shape = snap[shapeKey];
    const grid  = snap[gridKey];
    if (!shape || shape.length !== 2) {
      el.innerHTML =
        '<div class="hint" style="padding:24px;text-align:center;">no image yet</div>';
      return;
    }
    const [H, W] = shape;
    const vlo = snap[vloKey] != null ? snap[vloKey] : 0;
    const vhi = snap[vhiKey] != null ? snap[vhiKey] : 255;
    const logicals = snap[logicalsKey] || [];
    const boxSize = snap.box_size || 11;

    // Image as a background layout image. Cache-bust on every poll.
    const layoutImages = [{
      source: imageUrl + "?t=" + Date.now(),
      xref:   "x", yref: "y",
      x: 0, y: 0, sizex: W, sizey: H,
      sizing: "stretch", layer: "below",
      opacity: 1,
    }];

    const traces = [];
    // Invisible scatter at corners for colorbar.
    traces.push({
      x: [0, W, 0, W], y: [0, 0, H, H],
      mode: "markers", type: "scatter",
      marker: { size: 0.1, opacity: 0,
                color: [vlo, vhi, vlo, vhi],
                colorscale: "Greys", reversescale: true,
                cmin: vlo, cmax: vhi, showscale: true,
                colorbar: { title: { text: "Counts" }, len: 0.9 }},
      hoverinfo: "skip", showlegend: false,
    });

    // Tweezer overlay: green box = loaded (logical=1), red = empty.
    // Fail-loud when the live state has no gridLocations / no sites
    // — same root cause as a .h5 with all-zero per-site logicals
    // (detection wasn't initialized before the scan started). Without
    // this hint the operator sees the bare camera image and may not
    // realize detection isn't wired up.
    const noGrid = !grid || !grid.length;
    const noGridWarn = noGrid && (snap.num_sites === 0 ||
                                    snap.num_sites == null);
    if (grid && grid.length) {
      const n = grid.length;
      const occ = new Array(n);
      for (let i = 0; i < n; i++) {
        occ[i] = logicals && logicals.length >= n
          ? (logicals[i] ? 1 : 0) : 0;
      }
      // WebGL box outlines via scattergl with mode='lines'. Each site
      // contributes a closed 5-corner loop separated by NaN.
      // grid[i] is [y, x] (row, col) per the existing Dash code.
      const half = boxSize / 2;
      const ox = [-half,  half,  half, -half, -half, NaN];
      const oy = [-half, -half,  half,  half, -half, NaN];
      const loadedX = [], loadedY = [];
      const emptyX  = [], emptyY  = [];
      for (let i = 0; i < n; i++) {
        const [y, x] = grid[i];
        const dst = occ[i] ? [loadedX, loadedY] : [emptyX, emptyY];
        for (let k = 0; k < 6; k++) {
          dst[0].push(isFinite(x + ox[k]) ? x + ox[k] : null);
          dst[1].push(isFinite(y + oy[k]) ? y + oy[k] : null);
        }
      }
      if (loadedX.length) {
        traces.push({
          x: loadedX, y: loadedY, mode: "lines", type: "scattergl",
          line: { color: "#00ff88", width: 1.5 },
          hoverinfo: "skip", showlegend: false,
        });
      }
      if (emptyX.length) {
        traces.push({
          x: emptyX, y: emptyY, mode: "lines", type: "scattergl",
          line: { color: "#ff4444", width: 1.5 },
          hoverinfo: "skip", showlegend: false,
        });
      }
      // Site number labels for small arrays.
      if (n <= 200) {
        traces.push({
          x: grid.map(p => p[1]),
          y: grid.map(p => p[0] - half - 3),
          mode: "text", type: "scatter",
          text: grid.map((_, i) => String(i + 1)),
          textfont: { color: "#ffdd44", size: 7 },
          hoverinfo: "skip", showlegend: false,
        });
      }
    }

    const layout = Object.assign(plotLayout({
      xaxis: { range: [0, W], showgrid: false, zeroline: false, visible: false },
      yaxis: { range: [H, 0], scaleanchor: "x", scaleratio: 1,
               showgrid: false, zeroline: false, visible: false },
      margin: { l: 0, r: 60, t: 4, b: 4 },
      annotations: noGridWarn ? [{
        xref: "paper", yref: "paper", x: 0.5, y: 0.04,
        text: "no gridLocations — detection not initialized<br>"
              + "<span style='font-size:10px'>(run init / threshold setup before scanning)</span>",
        showarrow: false, align: "center",
        font: { color: "#ffdd44", size: 13, family: "Inter, sans-serif" },
        bgcolor: "rgba(20, 18, 8, 0.85)",
        bordercolor: "#ffdd44", borderwidth: 1, borderpad: 6,
      }] : [],
    }), { images: layoutImages });
    Plotly.react(el, traces, layout, plotConfig());
  }

  // Scan curve — port of _fig_scan_curve (1D scatter+errors; 2D heatmap).
  // sc shape from yb_dash_data.pkl:
  //   1D: {mode, ndim:1, scan_x:[], y_mean:[], y_sem:[], n_reps:[]}
  //   2D: {mode, ndim:2, x_values, y_values, heatmap, n_reps, sem,
  //        x_name, y_name, current?:[{x_idx,y_idx}]}
  function renderScanCurve(snap) {
    if (!window.Plotly) return;
    const el = $("plot-scan");
    if (!el) return;
    const sc = snap.scan_curve;
    if (!sc || sc.mode === "undefined") {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no scan curve yet</div>';
      return;
    }
    if ((sc.ndim || 1) >= 2) {
      renderScanCurve2D(snap, sc, el);
      return;
    }
    // 1D
    const x = sc.scan_x || [];
    const y = sc.y_mean || [];
    const err = sc.y_sem || [];
    const nReps = sc.n_reps || [];
    if (!x.length || !nReps.some(n => n > 0)) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no scan curve yet</div>';
      return;
    }
    const idx = nReps.map((n, i) => n > 0 ? i : -1).filter(i => i >= 0);
    const scale = snap.plot_scale && snap.plot_scale !== 1 ? snap.plot_scale : 1;
    const fx = idx.map(i => x[i] * scale);
    const fy = idx.map(i => y[i]);
    const fe = idx.map(i => err[i]);
    const fn = idx.map(i => nReps[i]);
    const xLabel = snap.scan_param_path || snap.scan_name || "x";
    const yLabel = sc.mode === "survival" ? "Survival"
                 : sc.mode === "rearrangement" ? "Rearrangement"
                 : "Loading rate";
    Plotly.react(el, [{
      x: fx, y: fy,
      error_y: { type: "data", array: fe, visible: true,
                 thickness: 1.5, color: "#44aaff" },
      mode: "markers", type: "scatter",
      marker: { size: 6, color: "#44aaff" },
      hovertemplate: `${xLabel}=%{x:.4g}<br>${yLabel}=%{y:.3f}±%{error_y.array:.3f}<extra></extra>`,
    }], plotLayout({
      title: { text: scanTitle(snap), font: { size: 13 }},
      xaxis: { title: xLabel },
      yaxis: { title: yLabel, range: [-0.05, 1.05] },
    }), plotConfig());
  }
  function scanTitle(snap) {
    const name = snap.scan_name || "Scan";
    const file = snap.scan_filename || "";
    const run = file.endsWith(".h5") ? file.slice(0, -3) : file;
    return run ? `${name}  — ${run}` : name;
  }
  function renderScanCurve2D(snap, sc, el) {
    const z = sc.heatmap || [];
    const nReps = sc.n_reps || [];
    const xVals = sc.x_values || [];
    const yVals = sc.y_values || [];
    if (!z.length || !xVals.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no scan data yet</div>';
      return;
    }
    // Mask cells with no reps -> null so they render as gaps.
    const zMasked = z.map((row, j) =>
      row.map((v, i) => (nReps[j] && nReps[j][i] && nReps[j][i] > 0) ? v : null));
    Plotly.react(el, [{
      z: zMasked,
      x: xVals.map((_, i) => i), y: yVals.map((_, i) => i),
      type: "heatmap", colorscale: "Viridis",
      zmin: 0, zmax: 1,
      colorbar: { title: { text: sc.mode === "survival" ? "Survival" : "Loading" },
                  len: 0.9 },
    }], plotLayout({
      title: { text: scanTitle(snap), font: { size: 13 }},
      xaxis: { title: sc.x_name || "x",
               tickmode: "array",
               tickvals: xVals.map((_, i) => i),
               ticktext: xVals.map(v => v.toPrecision(3)) },
      yaxis: { title: sc.y_name || "y",
               tickmode: "array",
               tickvals: yVals.map((_, i) => i),
               ticktext: yVals.map(v => v.toPrecision(3)) },
    }), plotConfig());
  }

  // Atom intensities — port of _fig_intens (threshold dots + current
  // dots colored by occupancy + ±1σ bands for loaded/empty).
  function renderIntensities(snap) {
    if (!window.Plotly) return;
    const el = $("plot-intensities");
    if (!el) return;
    const t = snap.thresholds;
    if (!t || !t.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no thresholds yet</div>';
      return;
    }
    const n = t.length;
    const sites = Array.from({length: n}, (_, i) => i + 1);
    const curSize = Math.max(6, Math.min(13, 1800 / n));
    const thrSize = Math.max(4, curSize - 2);
    const ci = snap.cur_intensities || [];
    const logicals = snap.logicals || [];
    const occ = sites.map((_, i) =>
      logicals.length >= n ? (logicals[i] ? 1 : 0) : 0);

    const traces = [{
      x: sites, y: t, name: "Threshold",
      mode: "markers", type: "scattergl",
      marker: { size: thrSize, color: "#777",
                line: { width: 1, color: "#999" }},
    }];
    let ymin = Math.min(...t), ymax = Math.max(...t);
    if (ci.length) {
      traces.push({
        x: sites, y: ci, name: "Current",
        mode: "markers", type: "scattergl",
        marker: { size: curSize, color: occ,
                  colorscale: [[0, "#e44"], [1, "#0c6"]],
                  cmin: 0, cmax: 1,
                  line: { width: 1, color: "white" }},
      });
      ymin = Math.min(ymin, Math.min(...ci));
      ymax = Math.max(ymax, Math.max(...ci));
    }

    const shapes = [];
    const annotations = [];
    if (ci.length && logicals.length >= n) {
      const loadedVals = [], emptyVals = [];
      for (let i = 0; i < n; i++) {
        (logicals[i] ? loadedVals : emptyVals).push(ci[i]);
      }
      const meanStd = (arr) => {
        if (!arr.length) return null;
        const mu = arr.reduce((s, x) => s + x, 0) / arr.length;
        const sd = Math.sqrt(arr.reduce((s, x) => s + (x - mu) ** 2, 0) / arr.length);
        return { mu, sd };
      };
      const sLoaded = meanStd(loadedVals);
      const sEmpty  = meanStd(emptyVals);
      if (sLoaded) {
        shapes.push({ type: "rect", xref: "paper", x0: 0, x1: 1,
          y0: sLoaded.mu - sLoaded.sd, y1: sLoaded.mu + sLoaded.sd,
          fillcolor: "rgba(0,204,102,0.12)", line: { width: 0 }, layer: "below" });
        shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1,
          y0: sLoaded.mu, y1: sLoaded.mu,
          line: { color: "#0c6", width: 1.5, dash: "dash" }});
        annotations.push({ text: `Loaded: ${sLoaded.mu.toFixed(1)} ± ${sLoaded.sd.toFixed(1)}`,
          xref: "paper", x: 0.99, y: sLoaded.mu, showarrow: false,
          xanchor: "right", yanchor: "bottom",
          font: { color: "#0c6", size: 10 },
          bgcolor: "rgba(20,20,40,0.6)" });
        ymin = Math.min(ymin, sLoaded.mu - sLoaded.sd);
        ymax = Math.max(ymax, sLoaded.mu + sLoaded.sd);
      }
      if (sEmpty) {
        shapes.push({ type: "rect", xref: "paper", x0: 0, x1: 1,
          y0: sEmpty.mu - sEmpty.sd, y1: sEmpty.mu + sEmpty.sd,
          fillcolor: "rgba(238,68,68,0.12)", line: { width: 0 }, layer: "below" });
        shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1,
          y0: sEmpty.mu, y1: sEmpty.mu,
          line: { color: "#e44", width: 1.5, dash: "dash" }});
        annotations.push({ text: `Empty: ${sEmpty.mu.toFixed(1)} ± ${sEmpty.sd.toFixed(1)}`,
          xref: "paper", x: 0.99, y: sEmpty.mu, showarrow: false,
          xanchor: "right", yanchor: "top",
          font: { color: "#e44", size: 10 },
          bgcolor: "rgba(20,20,40,0.6)" });
        ymin = Math.min(ymin, sEmpty.mu - sEmpty.sd);
        ymax = Math.max(ymax, sEmpty.mu + sEmpty.sd);
      }
      if (sLoaded && sEmpty) {
        const delta = sLoaded.mu - sEmpty.mu;
        annotations.push({ text: `Δ = ${delta.toFixed(2)}`,
          xref: "paper", yref: "paper", x: 0.5, y: 1.0, showarrow: false,
          font: { size: 12, color: "#ffdd44", family: "monospace" },
          bgcolor: "rgba(20,20,40,0.8)" });
      }
    }
    const pad = Math.max((ymax - ymin) * 0.2, 1);
    Plotly.react(el, traces, plotLayout({
      title: { text: "Atom Intensities", font: { size: 12 }},
      xaxis: { title: "Site", dtick: Math.max(1, Math.floor(n / 20)) },
      yaxis: { title: "Intensity", range: [ymin - pad, ymax + pad] },
      shapes, annotations,
      legend: { x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0.3)" },
      showlegend: true,
    }), plotConfig());
  }

  // Loading-rate live history — port of _fig_loading_live.
  function renderLoadingHistory(snap) {
    if (!window.Plotly) return;
    const el = $("plot-loading-live");
    if (!el) return;
    const hist = snap.loading_history;
    if (!hist || !hist.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no history yet</div>';
      return;
    }
    const n = hist.length;
    const x = Array.from({length: n}, (_, i) => i + 1);
    const avg_ = hist.reduce((s, x) => s + x, 0) / n;
    const logicals = snap.logicals || [];
    const cur = logicals.length
      ? logicals.reduce((s, x) => s + (x ? 1 : 0), 0) / logicals.length
      : null;
    const annotations = [{
      text: `Avg: ${(100 * avg_).toFixed(1)}%`,
      xref: "paper", x: 0.99, y: avg_, showarrow: false,
      xanchor: "right", yanchor: "bottom",
      font: { color: "#ffdd44", size: 10 },
    }];
    if (cur != null) {
      annotations.push({
        text: `Current: ${(100 * cur).toFixed(1)}%`,
        xref: "paper", yref: "paper", x: 0.5, y: 1.0,
        showarrow: false,
        font: { size: 18, color: "#0c6", family: "monospace" },
        bgcolor: "rgba(20,20,40,0.8)",
      });
    }
    Plotly.react(el, [{
      x, y: hist, mode: "lines+markers", type: "scatter",
      line: { color: "#0c6", width: 1.5 },
      marker: { size: 4, color: "#0c6" },
    }], plotLayout({
      title: { text: "Loading Rate", font: { size: 12 }},
      xaxis: { title: "Shot # (oldest → latest)" },
      yaxis: { title: "Fraction loaded", tickformat: ".0%", autorange: true },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1,
                 y0: avg_, y1: avg_,
                 line: { color: "#ffdd44", width: 1.5, dash: "dash" }}],
      annotations,
      showlegend: false,
    }), plotConfig());
  }

  // Per-site loading rate map — port of _fig_loading.
  function renderLoadingMap(snap) {
    if (!window.Plotly) return;
    const el = $("plot-load-map");
    if (!el) return;
    const grid = snap.grid_locations;
    const rates = snap.loading_rates;
    if (!grid || !grid.length || !rates || !rates.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no data</div>';
      return;
    }
    const n = grid.length;
    const x = grid.map(p => p[1]);
    const y = grid.map(p => p[0]);
    const sz = n < 500 ? 14 : Math.max(6, Math.min(14, 800 / Math.sqrt(n)));
    const mode = n < 100 ? "markers+text" : "markers";
    const text = n < 100 ? rates.map(r => (100 * r).toFixed(0) + "%") : undefined;
    Plotly.react(el, [{
      x, y, mode, type: "scattergl",
      marker: { size: sz, color: rates,
                colorscale: "RdYlGn", cmin: 0, cmax: 1,
                colorbar: { title: { text: "Rate" }, len: 0.9 },
                line: { width: 0.5, color: "white" }},
      text, textfont: { size: 7, color: "black" },
      textposition: "middle center",
      hovertemplate: "Site %{pointNumber}: %{marker.color:.1%}<extra></extra>",
    }], plotLayout({
      title: { text: `Loading Rates (${n} sites)`, font: { size: 12 }},
      xaxis: { visible: false },
      yaxis: { autorange: "reversed", scaleanchor: "x", scaleratio: 1, visible: false },
      margin: { l: 10, r: 60, t: 30, b: 10 },
    }), plotConfig());
  }

  // Per-site infidelity map — port of _fig_infid (log colour).
  function renderInfidMap(snap) {
    if (!window.Plotly) return;
    const el = $("plot-infid-map");
    if (!el) return;
    const grid = snap.grid_locations;
    const inf  = snap.infidelities;
    if (!grid || !grid.length || !inf || !inf.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no data</div>';
      return;
    }
    const n = grid.length;
    const x = grid.map(p => p[1]);
    const y = grid.map(p => p[0]);
    const sz = n < 500 ? 14 : Math.max(6, Math.min(14, 800 / Math.sqrt(n)));
    const logInf = inf.map(v => Math.log10(Math.max(v, 1e-6)));
    const mode = n < 100 ? "markers+text" : "markers";
    const text = n < 100 ? inf.map(v => v.toExponential(0)) : undefined;
    Plotly.react(el, [{
      x, y, mode, type: "scattergl",
      marker: { size: sz, color: logInf,
                colorscale: "Magma", reversescale: true,
                cmin: -4, cmax: -0.3,
                colorbar: { title: { text: "log10" }, len: 0.9 },
                line: { width: 0.5, color: "white" }},
      text, textfont: { size: 6, color: "white" },
      textposition: "middle center",
      customdata: inf,
      hovertemplate: "Site %{pointNumber}: %{customdata:.2e}<extra></extra>",
    }], plotLayout({
      title: { text: `Infidelity (${n} sites)`, font: { size: 12 }},
      xaxis: { visible: false },
      yaxis: { autorange: "reversed", scaleanchor: "x", scaleratio: 1, visible: false },
      margin: { l: 10, r: 60, t: 30, b: 10 },
    }), plotConfig());
  }

  // Grid-shift heatmap — port of _fig_shift.
  function renderShiftHeatmap(snap) {
    if (!window.Plotly) return;
    const el = $("plot-shift");
    if (!el) return;
    const hm = snap.grid_shift_heatmap;
    if (!hm || !hm.length || !hm[0] || !hm[0].length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no shift data yet</div>';
      return;
    }
    const R = Math.floor((hm.length - 1) / 2);
    const xs = Array.from({length: hm[0].length}, (_, i) => i - R);
    const ys = Array.from({length: hm.length},    (_, i) => i - R);
    const traces = [{
      z: hm, x: xs, y: ys, type: "heatmap",
      colorscale: "Viridis", showscale: true,
      colorbar: { len: 0.9 },
    }];
    let titleText = "Grid Shift Heatmap";
    const histArr = snap.grid_shift_history || [];
    if (histArr.length) {
      const last = histArr[histArr.length - 1];
      const dy = last[0], dx = last[1];
      traces.push({
        x: [dx], y: [dy], mode: "markers", type: "scatter",
        marker: { symbol: "x", size: 14, color: "red",
                  line: { width: 2, color: "red" }},
        showlegend: false, hoverinfo: "skip",
      });
      titleText = `Grid Shift (dy=${dy}, dx=${dx})`;
    }
    Plotly.react(el, traces, plotLayout({
      title: { text: titleText, font: { size: 12 }},
      xaxis: { title: "dx" },
      yaxis: { title: "dy", autorange: "reversed" },
    }), plotConfig());
  }

  // Avg histogram — port of _fig_avghist (live bars + fit curves).
  function renderAvgHist(snap) {
    if (!window.Plotly) return;
    const el = $("plot-hist-avg");
    if (!el) return;
    const liveHist = snap.live_hist_data;
    const liveFits = snap.live_gauss_fits;
    const loadedFits = snap.loaded_gauss_fits;
    if (!liveHist || !liveHist.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:18px;text-align:center;font-size:10px;">no hist</div>';
      return;
    }
    const traces = [];
    // Aggregate bars: avg counts across sites on a common x-axis.
    const allCenters = [];
    liveHist.forEach(h => (h.bin_centers || []).forEach(c => allCenters.push(c)));
    if (allCenters.length) {
      const minC = Math.min(...allCenters), maxC = Math.max(...allCenters);
      const N = 50;
      const centers = Array.from({length: N},
        (_, i) => minC + (maxC - minC) * i / (N - 1));
      const avg_ = new Array(N).fill(0);
      liveHist.forEach(h => {
        const bc = h.bin_centers || [];
        const cnt = h.counts || [];
        if (!bc.length) return;
        for (let i = 0; i < N; i++) {
          // Linear interp; left/right = 0
          const x = centers[i];
          if (x <= bc[0] || x >= bc[bc.length - 1]) continue;
          let lo = 0;
          while (lo + 1 < bc.length && bc[lo + 1] < x) lo++;
          const t = (x - bc[lo]) / (bc[lo + 1] - bc[lo]);
          avg_[i] += cnt[lo] * (1 - t) + (cnt[lo + 1] || 0) * t;
        }
      });
      for (let i = 0; i < N; i++) avg_[i] /= liveHist.length;
      const bw = (maxC - minC) / (N - 1) * 0.85;
      traces.push({
        x: centers, y: avg_, type: "bar",
        marker: { color: "#4488cc", opacity: 0.8 },
        width: Array(N).fill(bw),
        name: `Live (${snap.n_accum_shots || 0})`,
      });
    }
    Plotly.react(el, traces, plotLayout({
      title: { text: "Avg Histogram", font: { size: 10 }},
      margin: { l: 32, r: 8, t: 22, b: 26 },
      barmode: "overlay",
      xaxis: { title: "" }, yaxis: { title: "" },
      showlegend: false,
    }), plotConfig());
  }

  // Per-site rep histograms — port of _figs_reps + _build_hist.
  function renderRepHists(snap) {
    const targets = ["plot-hist-rep0", "plot-hist-rep1",
                     "plot-hist-rep2", "plot-hist-rep3"];
    const labels = ["Best", "Worst", "Random", "Random"];
    const sites = snap.hist_rep_sites || [];
    const liveHist = snap.live_hist_data || [];
    targets.forEach((id, k) => {
      const el = $(id);
      if (!el || !window.Plotly) return;
      const idx = sites[k];
      if (idx == null || !liveHist[idx] || !liveHist[idx].counts) {
        Plotly.purge(el);
        el.innerHTML = '<div class="hint" style="padding:18px;text-align:center;font-size:10px;">no hist</div>';
        return;
      }
      const h = liveHist[idx];
      Plotly.react(el, [{
        x: h.bin_centers, y: h.counts, type: "bar",
        marker: { color: "#58a6ff" },
      }], plotLayout({
        title: { text: `${labels[k]}: Site ${idx + 1}`, font: { size: 10 }},
        margin: { l: 28, r: 6, t: 22, b: 24 },
        xaxis: { title: "" }, yaxis: { title: "" },
      }), plotConfig());
    });
  }

  function plotLayout(over) {
    return Object.assign({
      paper_bgcolor: "#1a2030",
      plot_bgcolor:  "#0b0e13",
      font: { color: "#d8dee9", size: 11,
              family: 'ui-monospace, "Segoe UI", sans-serif' },
      margin: { l: 50, r: 20, t: 30, b: 40 },
      xaxis: { gridcolor: "#2a3242", zerolinecolor: "#2a3242" },
      yaxis: { gridcolor: "#2a3242", zerolinecolor: "#2a3242" },
      showlegend: false,
      uirevision: "fixed",
    }, over || {});
  }

  // Variant where the inner plot-area background matches the card so
  // the chart visually flushes with the card (no darker inner rectangle
  // around the data). Used by the sweep vis + per-iteration chart in
  // the Analysis tab -- the per-site maps deliberately keep the darker
  // contrast since the dots read better against it.
  function plotLayoutFlush(over) {
    return plotLayout(Object.assign({
      plot_bgcolor: "#1a2030",
      xaxis: { gridcolor: "#2a3242", zerolinecolor: "#2a3242" },
      yaxis: { gridcolor: "#2a3242", zerolinecolor: "#2a3242" },
    }, over || {}));
  }
  function plotConfig() {
    return { displayModeBar: false, responsive: true };
  }

  // =====================================================================
  // HARDWARE TAB — single iframe to the SLM dashboard. No probe;
  // the iframe loads or it doesn't, and the corner reload/open-in-tab
  // buttons handle recovery if it looks empty.
  // =====================================================================

  async function pollHardware() {
    pollHwHealth();
    pollHwDevices();
    pollHwGpu();
    pollHwLocks();
    pollHwClients();
    pollHwLogs();
    pollHwRearrangeDiag();
    // Phase + camera PNGs (cache-busted via t param)
    const ts = Date.now();
    refreshImg("phase-wrap", "phase-placeholder",
               `/api/slm/phase/png?t=${ts}`, "phase-ts");
    refreshImg("cam-wrap", "cam-placeholder",
               `/api/slm/camera/png?t=${ts}`, "cam-ts");
  }

  // -- shared helper: render an object as a kv <dl> ------------------
  // ``rows`` is an Array of [labelText, value, optionalClass]. ``value``
  // may be a string, number, null/undefined (→ "—"), bool (yes/no), or
  // an object/array (JSON-stringified compactly).
  function renderKv(elId, rows) {
    const el = $(elId);
    if (!el) return;
    const html = rows.map(([k, v, cls]) => {
      let val;
      if (v == null) { val = "—"; cls = cls || ""; }
      else if (typeof v === "boolean") { val = v ? "yes" : "no"; }
      else if (typeof v === "object")  { val = JSON.stringify(v); }
      else                              { val = String(v); }
      const c = cls ? ` class="${cls}"` : "";
      const titleAttr = val.length > 40 ? ` title="${escHtml(val)}"` : "";
      return `<dt>${escHtml(k)}</dt><dd${c}${titleAttr}>${escHtml(val)}</dd>`;
    }).join("");
    el.innerHTML = html;
  }

  async function pollHwHealth() {
    try {
      const h = await api("/api/slm/health");
      renderKv("kv-server", [
        ["Status",  h.status || "up",
          (h.status || "up") === "up" ? "ok" : "warn"],
        ["Uptime",  h.uptime_s != null ? fmtNum(h.uptime_s, 1) + " s" : null],
        ["Version", h.version || null],
        ["Host",    h.host || h.hostname || null],
        ["Started", h.started_iso || (h.started_epoch ? fmtTs(h.started_epoch) : null)],
        ["Pending ops", h.pending_ops != null ? h.pending_ops : null],
      ]);
    } catch (e) {
      renderKv("kv-server", [
        ["Status", e.status === 503 ? "offline" : "error", "bad"],
        ["Error",  String(e.message || e), "bad"],
      ]);
    }
  }

  async function pollHwDevices() {
    try {
      const d = await api("/api/slm/devices");
      // /devices returns {slm: {connected, shape, type, lut_basename, ...},
      //                   camera: {connected, shape, type, ...}, warnings: [...]}.
      const slm = d.slm || {};
      const cam = d.camera || {};
      renderKv("kv-slm", [
        ["Connected", slm.connected,
          slm.connected ? "ok" : "bad"],
        ["Type",       slm.type || null],
        ["Shape",      slm.shape ? slm.shape.join(" × ") : null],
        ["Pitch (µm)", slm.pitch_um
          ? slm.pitch_um.map((v) => v.toFixed(2)).join(" × ") : null],
        ["LUT",        slm.lut_basename || null],
        ["Direct writer", slm.direct_writer != null
          ? slm.direct_writer : null],
      ]);
      renderKv("kv-camera", [
        ["Connected", cam.connected,
          cam.connected ? "ok" : "bad"],
        ["Type",      cam.type || null],
        ["Shape",     cam.shape ? cam.shape.join(" × ") : null],
        ["Pitch (µm)", cam.pitch_um
          ? cam.pitch_um.map((v) => v.toFixed(2)).join(" × ") : null],
        ["Exposure (s)", cam.exposure_s != null
          ? fmtNum(cam.exposure_s, 4) : null],
        ["ROI",       cam.roi ? cam.roi.join(", ") : null],
      ]);
      // Phase-card detail block
      if (slm.shape) setText("phase-shape", slm.shape.join(" × "));
      if (cam.exposure_s != null)
        setText("cam-exp-detail", fmtNum(cam.exposure_s, 4) + " s");
    } catch (e) {
      renderKv("kv-slm", [
        ["Connected", "offline", "bad"],
        ["Error",     String(e.message || e), "bad"],
      ]);
      renderKv("kv-camera", [
        ["Connected", "offline", "bad"],
        ["Error",     String(e.message || e), "bad"],
      ]);
    }
  }

  async function pollHwGpu() {
    try {
      const g = await api("/api/slm/gpu");
      // /gpu/info shape varies between server versions; pull the most
      // common keys defensively.
      const dev   = g.device || g.gpu || g.name || null;
      const util  = g.util != null ? g.util
                   : (g.utilization != null ? g.utilization : null);
      const memU  = g.mem_used_mb || g.memory_used_mb || null;
      const memT  = g.mem_total_mb || g.memory_total_mb || null;
      const temp  = g.temp_c || g.temperature_c || null;
      const pwr   = g.power_w || null;
      renderKv("kv-gpu", [
        ["Device",       dev],
        ["Utilization",  util != null ? util.toFixed(0) + " %" : null],
        ["Memory",       (memU != null && memT != null)
          ? `${memU.toFixed(0)} / ${memT.toFixed(0)} MB` : null],
        ["Temp",         temp != null ? temp.toFixed(0) + " °C" : null],
        ["Power",        pwr != null ? pwr.toFixed(0) + " W" : null],
        ["Clock-lock",   g.clock_locked != null ? g.clock_locked : null],
      ]);
      setText("gpu-clock-pill",
        g.clock_locked == null ? "—" :
        (g.clock_locked ? "clocks pinned" : "clocks unpinned"));
    } catch (e) {
      renderKv("kv-gpu", [
        ["Status", e.status === 503 ? "offline" : "error", "bad"],
      ]);
      setText("gpu-clock-pill", "—");
    }
  }

  async function pollHwLocks() {
    try {
      const ls = await api("/api/slm/lock/status");
      renderLocksTable(ls);
    } catch {
      $("lock-table").querySelector("tbody").innerHTML =
        '<tr><td colspan="7" class="muted">SLM offline</td></tr>';
      setText("lock-count", "0");
    }
  }
  function renderLocksTable(locks) {
    const tbody = $("lock-table").querySelector("tbody");
    // Server returns {device: lock_dict | null, ...}. Filter out nulls.
    const entries = Object.entries(locks || {})
      .filter(([k, v]) => v && typeof v === "object" && k !== "warnings");
    setText("lock-count", String(entries.length));
    if (!entries.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="muted">No locks held.</td></tr>';
      return;
    }
    tbody.innerHTML = entries.map(([dev, lk]) => `
      <tr>
        <td class="mono">${escHtml(dev)}</td>
        <td>${escHtml(lk.mode || "—")}</td>
        <td>${escHtml(lk.client_id || lk.holder || "—")}</td>
        <td>${escHtml(lk.description || "—")}</td>
        <td class="right">${typeof lk.held_for_s === "number"
          ? lk.held_for_s.toFixed(1) + " s"
          : (lk.age_s != null ? lk.age_s.toFixed(1) + " s" : "—")}</td>
        <td class="right">${lk.timeout_s != null ? lk.timeout_s + " s" : "—"}</td>
        <td></td>
      </tr>`).join("");
  }

  async function pollHwClients() {
    try {
      const cl = await api("/api/slm/clients");
      renderClientsTable(cl);
    } catch {
      $("client-table").querySelector("tbody").innerHTML =
        '<tr><td colspan="6" class="muted">SLM offline</td></tr>';
      setText("client-count", "—");
    }
  }
  function renderClientsTable(payload) {
    const tbody = $("client-table").querySelector("tbody");
    const list = (payload && payload.clients) || [];
    const totalSeen = payload && payload.total_seen;
    setText("client-count", totalSeen != null && totalSeen !== list.length
      ? `${list.length} / ${totalSeen}`
      : String(list.length));
    if (!list.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="muted">No clients seen yet.</td></tr>';
      return;
    }
    // Server returns clients sorted newest-first; render as-is but cap at 20.
    const top = list.slice(0, 20);
    tbody.innerHTML = top.map((c) => {
      const holds = c.holds && c.holds.length ? c.holds.join(", ") : "—";
      const last  = (c.last_method && c.last_path)
        ? `${c.last_method} ${c.last_path}` : "—";
      return `<tr>
        <td class="mono">${escHtml(c.client_id || "—")}</td>
        <td class="${holds !== "—" ? "ok" : ""}">${escHtml(holds)}</td>
        <td class="right">${c.requests != null ? c.requests : "—"}</td>
        <td class="truncate" title="${escHtml(last)}">${escHtml(last)}</td>
        <td>${fmtSago(c.last_seen_s_ago)}</td>
        <td>${fmtSago(c.first_seen_s_ago)}</td>
      </tr>`;
    }).join("");
    if (list.length > 20) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="6" class="muted center">+${list.length - 20} more</td>`;
      tbody.appendChild(tr);
    }
  }
  function fmtSago(s) {
    if (s == null) return "—";
    if (s < 60)    return s.toFixed(1) + " s";
    if (s < 3600)  return (s / 60).toFixed(1) + " min";
    return (s / 3600).toFixed(1) + " h";
  }

  async function pollHwLogs() {
    try {
      const lg = await api("/api/slm/logs?lines=200");
      const lines = (lg && lg.lines) || [];
      const el = $("slm-log");
      if (!el) return;
      if (lg.path) setText("log-path", lg.path);
      if (!lines.length) { el.textContent = "(no log lines)"; return; }
      el.innerHTML = lines.slice(-200).map((ln) => {
        let level = "INFO";
        const s = String(ln);
        const m = s.match(/\b(DEBUG|INFO|WARNING|ERROR)\b/);
        if (m) level = m[1];
        return `<span class="L-${level}">${escHtml(s)}</span>`;
      }).join("\n");
      el.scrollTop = el.scrollHeight;
    } catch (e) {
      $("slm-log").textContent = "(logs unavailable: " + (e.message || e) + ")";
      setText("log-path", "—");
    }
  }

  async function pollHwRearrangeDiag() {
    try {
      const rd = await api("/api/slm/rearrange/diag");
      renderRearrangeDiag(rd);
    } catch {
      $("rearrange-diag-body").innerHTML =
        '<div class="hint">SLM offline</div>';
    }
  }

  function refreshImg(wrapId, phId, url, tsId) {
    const wrap = $(wrapId);
    if (!wrap) return;
    let img = wrap.querySelector("img");
    if (!img) {
      img = document.createElement("img");
      img.style.cssText =
        "max-width:100%;max-height:100%;display:none;image-rendering:pixelated;";
      img.onerror = () => {
        img.style.display = "none";
        const ph = $(phId);
        if (ph) { ph.style.display = ""; ph.textContent = "image unavailable"; }
      };
      img.onload = () => {
        img.style.display = "";
        const ph = $(phId);
        if (ph) ph.style.display = "none";
        if (tsId) setText(tsId, fmtTs(Date.now() / 1000));
      };
      wrap.appendChild(img);
    }
    img.src = url;
  }

  function renderRearrangeDiag(rd) {
    const body = $("rearrange-diag-body");
    if (!body) return;
    const entries = (rd && rd.entries) || [];
    if (!entries.length) {
      body.innerHTML = '<div class="hint">no recent rearrange shots</div>';
      return;
    }
    const rows = entries.slice(-10).map((e) => {
      const d = e.diag || {};
      return `<tr>
        <td class="mono">${escHtml(e.ts_iso || "")}</td>
        <td class="mono">${escHtml(String(e.scan_id || ""))}</td>
        <td class="right">${e.seq_id != null ? e.seq_id : ""}</td>
        <td class="right">${d.total_ms != null ? d.total_ms.toFixed(1) : ""}</td>
        <td class="right">${d.n_loaded != null ? d.n_loaded : ""}</td>
        <td>${d.aborted ? '<span class="bad">yes</span>' : ""}</td>
      </tr>`;
    }).join("");
    body.innerHTML = `
      <table class="dense">
        <thead><tr><th>ts</th><th>scan_id</th><th>seq</th>
          <th>total_ms</th><th>n_loaded</th><th>aborted</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // =====================================================================
  // ANALYSIS TAB
  // =====================================================================
  let runsCache = [];
  let selectedScanId = null;
  let groupsCache = {};
  let activeGroupId = null;

  async function loadRunsList() {
    // Only flip status to "loading list" if we're not already mid-analysis
    // for some specific scan -- the analysis status takes precedence.
    if (!selectedScanId) {
      setAnalysisStatus("loading", "loading runs…", "warn");
    }
    try {
      const data = await api("/api/runs/list");
      runsCache = data.runs || [];
      renderRunsTable();
      populateDateFilter();
      setText("runs-count", String(runsCache.length));
      // Auto-pick a run for first paint. Only fires if no run is currently
      // selected. Priority:
      //   1. A saved MULTI-run tray -> restore every chip + re-run the
      //      group analysis (so a refresh holds the whole selection, not
      //      one of them).
      //   2. The localStorage-remembered single selection.
      //   3. The most recent complete run.
      if (!selectedScanId && traySet.size === 0) {
        const exists = (sid) => runsCache.some((r) => r.scan_id === sid);
        const savedTray = savedTrayAtLoad.filter(exists);
        if (savedTray.length > 1) {
          traySet.clear();
          savedTray.forEach((s) => traySet.add(s));
          renderTray();
          analyzeTrayGroup();
        } else {
          const remembered = (() => {
            try { return localStorage.getItem("yb_dashboard_selected_scan"); }
            catch { return null; }
          })();
          let pick = savedTray[0]
            || (remembered && exists(remembered) ? remembered : null);
          if (!pick && runsCache.length) {
            // List is newest-first.
            pick = runsCache[0].scan_id;
          }
          if (pick) {
            trayReplace(pick);
            loadAnalysis(pick);
          }
        }
      }
    } catch (e) {
      const wrap = $("runs-table");
      if (wrap) wrap.innerHTML = `<div class="run-row"><div class="run-info muted">${escHtml(e.message)}</div></div>`;
    }
  }
  function populateDateFilter() {
    const dates = Array.from(new Set(
      runsCache.map((r) => (r.scan_id || "").slice(0, 8))
    )).filter(Boolean).sort().reverse();
    const sel = $("runs-date-filter");
    const cur = sel.value;
    sel.innerHTML = '<option value="">All dates</option>' +
      dates.map((d) => `<option value="${d}">${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}</option>`).join("");
    sel.value = cur;
  }
  function renderRunsTable() {
    const wrap = $("runs-table");
    if (!wrap) return;
    const search = ($("runs-search").value || "").toLowerCase();
    const date = $("runs-date-filter").value || "";
    const filtered = runsCache.filter((r) => {
      if (date && (r.scan_id || "").slice(0, 8) !== date) return false;
      if (search) {
        const blob = ((r.scan_id || "") + " " + (r.name || "")).toLowerCase();
        if (!blob.includes(search)) return false;
      }
      return true;
    });
    if (!filtered.length) {
      wrap.innerHTML =
        '<div class="run-row"><div class="run-info muted">no runs match</div></div>';
      return;
    }
    // SLM-style row: [+] button | scan_id | name | dim-swept-info.
    // Clicking the [+] adds/toggles in tray; clicking elsewhere on the
    // row REPLACES the tray with just that run.
    wrap.innerHTML = filtered.map((r) => {
      const id = r.scan_id || "";
      const idShort = id.length === 14
        ? `${id.slice(4,6)}/${id.slice(6,8)} ${id.slice(8,10)}:${id.slice(10,12)}:${id.slice(12,14)}`
        : id;
      const inTray = traySet.has(id);
      return `
        <div class="run-row ${inTray ? "in-tray" : ""}"
             data-scan-id="${id}">
          <button class="run-add" data-tray-toggle="${id}"
                  title="${inTray ? "Remove from tray" : "Add to tray"}">${inTray ? "✓" : "+"}</button>
          <div class="run-info" title="${id}">
            ${idShort}
            <span class="run-dim"> · ${escHtml(r.name || "—")}</span>
            <span class="run-dim"> · ${escHtml(r.swept || "—")}</span>
          </div>
        </div>`;
    }).join("");
    $$(".run-row", wrap).forEach((row) => {
      row.addEventListener("click", (e) => {
        // Add-button click is its own handler.
        if (e.target.closest(".run-add")) return;
        // STOP propagation -- loadAnalysis synchronously calls
        // renderRunsTable which detaches the clicked .run-row from
        // the DOM. By the time the click bubbles to the document
        // handler, e.target is no longer inside any card, and the
        // document handler would collapse the runs card to peek
        // (then edge). Stop bubbling here so the card state stays
        // exactly where it was when the user clicked.
        e.stopPropagation();
        const sid = row.dataset.scanId;
        trayReplace(sid);
        loadAnalysis(sid);
      });
    });
    $$(".run-add", wrap).forEach((btn) => {
      btn.addEventListener("click", (e) => {
        // Same rationale as the row click above: renderRunsTable
        // rebuilds the DOM, detaches this button -- stop propagation
        // so the document/card handlers don't see the click as
        // "outside" and collapse the runs card.
        e.stopPropagation();
        const sid = btn.dataset.trayToggle;
        trayToggle(sid);
        renderRunsTable();
      });
    });
    syncTrayHighlight();
  }
  $("runs-search").addEventListener("input", renderRunsTable);
  $("runs-date-filter").addEventListener("change", renderRunsTable);
  $("runs-refresh").addEventListener("click", () => {
    loadRunsList();
    toast("Runs refreshed", "warn");
  });

  // Click-to-copy on scan_id tiles (Analysis tab + anywhere else
  // the .scan-id-copy class is dropped). Delegated so it picks up
  // tiles rendered after page load.
  document.addEventListener("click", (e) => {
    const el = e.target.closest && e.target.closest(".scan-id-copy");
    if (!el) return;
    const sid = el.dataset.scanId || el.textContent.trim();
    if (!sid) return;
    const announce = () => {
      toast(`Copied scan_id ${sid}`, "ok");
      el.classList.add("just-copied");
      setTimeout(() => el.classList.remove("just-copied"), 600);
    };
    // Modern path; falls back to a hidden textarea for browsers / iframes
    // where clipboard.writeText is unavailable.
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(sid).then(announce, () => {
        toast("Copy failed — clipboard permission denied", "warn");
      });
    } else {
      try {
        const ta = document.createElement("textarea");
        ta.value = sid;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        announce();
      } catch {
        toast("Copy not supported in this browser", "warn");
      }
    }
  });

  // Active analysis state lives module-level so the filter card and the
  // per-iteration metric toggles can refetch + re-render against the
  // currently-loaded run without re-asking the user to pick.
  let activeAnalysis = null;       // last loaded analysis result
  let activeFilters  = {};         // {axis_name: [allowed values]}
  let perIterToggles = {};         // {metric_key: bool} -- legend on/off
  let perIterDiagKey = "";         // diag column overlaid on the right axis ("" = none)
  // Rearrangement scatter axis picks (persisted), keys into the scatter
  // variable list: "loaded_frac" | "survival_frac" | "fp_frac" | "diag:<col>".
  const scatterAxes = (() => {
    try { return JSON.parse(localStorage.getItem("yb_scatter_axes")) || {}; }
    catch (e) { return {}; }
  })();
  function saveScatterAxes() {
    try { localStorage.setItem("yb_scatter_axes", JSON.stringify(scatterAxes)); }
    catch (e) {}
  }

  // Plotly.purge throws "DOM element provided is null or undefined" if
  // its arg is null, which rejects the whole async loadAnalysis promise
  // and silently freezes the "loading..." state. Guard every purge so a
  // missing div doesn't kill the render path.
  function safePurge(divId) {
    const el = $(divId);
    if (el && window.Plotly) try { Plotly.purge(el); } catch {}
  }

  // Status indicator state machine: idle / loading-list / analyzing /
  // analyzed (with N shots) / error. Painted next to the scan_id badge
  // in the Run summary card. Colors come from the .status-pill classes
  // already in the CSS (ok=green, warn=yellow, bad=red, dim=grey).
  function setAnalysisStatus(state, label, klass) {
    const pill = document.getElementById("analysis-status");
    if (!pill) return;
    const lbl = pill.querySelector(".status-label");
    pill.className = "status-pill " + (klass || "dim");
    pill.style.marginLeft = "6px";
    pill.style.fontSize = "11px";
    if (lbl) lbl.textContent = label;
  }

  async function loadAnalysis(scanId, opts) {
    opts = opts || {};
    selectedScanId = scanId;
    setText("selected-scan-id", scanId);
    setAnalysisStatus("analyzing", "analyzing…", "warn");
    renderRunsTable();
    // Mirror the loaded scan into the paste-box so the user can see /
    // copy the canonical 14-digit form. (Doesn't fire the listener
    // attached below because we set .value directly.)
    const pasteEl = document.getElementById("manual-scan-id");
    if (pasteEl) pasteEl.value = scanId || "";
    // Persist so refresh keeps you on the same scan.
    try { localStorage.setItem("yb_dashboard_selected_scan", scanId || ""); }
    catch { /* private mode */ }
    if (!opts.keepFilters) { activeFilters = {}; recomputeInfid = false; }
    const body = $("analysis-detail-body");
    body.innerHTML = '<div class="hint">loading…</div>';
    ["plot-analysis-scan", "plot-analysis-scan-lines", "plot-site-loading",
     "plot-site-survival", "plot-site-fp", "plot-site-infid",
     "plot-site-inthist", "plot-per-iter", "plot-per-iter-hist",
     "plot-svd", "plot-avg-image", "plot-rearrange-scatter"].forEach(safePurge);
    try {
      let url = `/api/runs/${scanId}/analysis`;
      const qs = [];
      if (Object.keys(activeFilters).length) {
        qs.push('filter=' + encodeURIComponent(JSON.stringify(activeFilters)));
      }
      if (recomputeInfid) qs.push('recompute_infidelity=1');
      if (opts.forceRecache) qs.push('force_recache=1');
      if (qs.length) url += '?' + qs.join('&');
      const r = await api(url);
      if (!r || typeof r !== "object") {
        throw new Error("server returned non-object response");
      }
      activeAnalysis = r;
      renderAnalysisDetail(r);
      renderAnalysisFilters(r);
      renderPerSiteMaps(r);
      renderSurvivalVsDistance(r);
      renderPerIteration(r);
      renderSeqSpecific(r);
      renderAvgImage(r);
      renderRearrangeScatter(r);
      // Status: analyzed · N shots (green). Falls back to n_params if
      // n_shots is missing (e.g. 0d aggregate scan).
      const ns = (typeof r.n_shots === "number" && r.n_shots > 0)
        ? r.n_shots
        : (typeof r.n_params === "number" ? r.n_params : 0);
      setAnalysisStatus("analyzed",
        `analyzed · ${ns.toLocaleString()} shot${ns === 1 ? "" : "s"}`, "ok");
    } catch (e) {
      console.error("analysis fetch failed", e);
      setAnalysisStatus("error", "error", "bad");
      const status = e.status != null ? ` (HTTP ${e.status})` : "";
      const detail = e.body && e.body.error ? `\n${e.body.error}` : "";
      body.innerHTML = `
        <div style="background:rgba(248,81,73,0.08);
                    border:1px solid var(--err);
                    border-radius:4px;padding:12px 16px;
                    font-family:var(--mono);font-size:12px;">
          <div class="bad" style="font-weight:600;margin-bottom:6px;">
            analysis failed${status}
          </div>
          <div style="white-space:pre-wrap;color:var(--text);">${escHtml(e.message + detail)}</div>
          <div class="hint" style="margin-top:8px;">
            Check the lab-PC <code>run_monitor</code> log for the full traceback.
          </div>
        </div>`;
    }
  }

  // Survival-vs-transit-distance panel (Phase 5.5 Track B). Hidden unless
  // run_analysis emitted a curve; shows a badge if the lab couldn't match
  // target coords to its own lattice ('lattice_mismatch').
  function renderSurvivalVsDistance(r) {
    const card = $("analysis-svd-card");
    if (!card) return;
    const root = r && r.survival_vs_distance;
    const reason = r && r.survival_vs_distance_skipped_reason;
    if (!root && !reason) { card.hidden = true; return; }
    card.hidden = false;
    // Total (top-level) vs per-step (root.per_step). Default total. Falls
    // back to total when per-step isn't available (diag had no nsteps).
    const perStepAvail = !!(root && root.per_step);
    const modeSel = $("svd-mode");
    if (modeSel) {
      const ps = modeSel.querySelector('option[value="per_step"]');
      if (ps) {
        ps.disabled = !perStepAvail;
        ps.text = perStepAvail ? "per-step" : "per-step (n/a)";
      }
      if (svdMode === "per_step" && !perStepAvail) modeSel.value = "total";
      else modeSel.value = svdMode;
    }
    const usePerStep = svdMode === "per_step" && perStepAvail;
    const svd = usePerStep ? root.per_step : root;
    const distKind = usePerStep ? "per-step distance" : "transit distance";
    const info = $("svd-info");
    const el = $("plot-svd");
    if (reason) {
      if (el) {
        safePurge("plot-svd");
        el.innerHTML =
          '<div class="hint" style="padding:24px;text-align:center;">' +
          'not computable — ' + escHtml(reason) + '</div>';
      }
      if (info) info.textContent = reason;
      return;
    }
    const centers = svd.centers || [];
    const mean = svd.survival_mean || [];
    const sem = svd.survival_sem || [];
    const counts = svd.n_pairs_per_bin || [];
    // Drop empty bins so the line connects only populated points.
    const X = [], Y = [], E = [], C = [];
    for (let i = 0; i < centers.length; i++) {
      if (mean[i] == null) continue;
      X.push(centers[i]); Y.push(mean[i]);
      E.push(sem[i] || 0); C.push(counts[i] || 0);
    }
    if (info) {
      const units = svd.distance_units || "px";
      // Pair counts live on the root (total) payload.
      info.textContent =
        `${root.n_total_pairs} pairs · ${root.n_unmatched} unmatched · ${units}`;
    }
    if (!window.Plotly || !el) return;
    if (!X.length) {
      safePurge("plot-svd");
      el.innerHTML =
        '<div class="hint" style="padding:24px;text-align:center;">no pairs</div>';
      return;
    }
    const trace = {
      x: X, y: Y, type: "scatter", mode: "lines+markers",
      marker: {size: 7, color: "#58a6ff"},
      line: {color: "#58a6ff"},
      error_y: {type: "data", array: E, visible: true, color: "#58a6ff"},
      hovertext: C.map((n, i) => `n=${n}`),
      hoverinfo: "x+y+text",
      name: "survival",
    };
    const units = svd.distance_units === "camera_pixels" ? "camera px"
      : svd.distance_units === "knm_pixels" ? "knm px" : "distance";
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: "#c9d1d9", size: 11},
      margin: {l: 48, r: 12, t: 8, b: 42},
      xaxis: {title: `${distKind} (${units})`, gridcolor: "#1a1a30"},
      yaxis: {title: "survival", range: [0, 1.02], gridcolor: "#1a1a30"},
      showlegend: false,
    };
    Plotly.react(el, [trace], layout, {displayModeBar: false, responsive: true});
  }

  function renderAnalysisDetail(r) {
    const body = $("analysis-detail-body");
    const sweep = r.sweep || {};
    const summary = r.summary || {};
    const diag = r.diag_aggregate;
    const code = r.code || {};
    const grid = r.grid || {};
    // Phase 5a: when target_aware is populated (rearrangement scans
    // either from cached SLM analysis or, eventually, lab-computed
    // from paths_per_shot), the "survival" curve is actually the TP
    // (target-aware) survival, not per-site survival. Surface that
    // distinction in labels + extra TP/FP stat tiles.
    // Headline tiles read the GLOBAL (filter-independent) block so the
    // top bar never moves when a filter is applied (operator request).
    const sg = r.summary_global || summary || {};
    const ta = r.target_aware_global || r.target_aware || null;
    const targetAware = !!(sg.survival_source
                            || (ta && ta.overall_mean != null));
    const disc = r.discrimination || null;
    const imf = r.imaging_fidelity || null;   // only set when recompute pressed
    // 1-image (loading-only) scans have no img2 → no survival. Show
    // loading as the headline and turn the discrimination map on by
    // default (the "is this data trash?" check).
    // 1-image (loading-only) scans have no img2 → survival is all-NaN.
    // (Don't key off avg_image — it's now present for every scan.)
    const oneImg = !hasFinite(sg.survival_mean);
    const sweepG = r.sweep_all || sweep || {};
    const nDims = sweepG.n_dims != null ? sweepG.n_dims
                  : (sweepG.dims || []).filter((d) => d > 1).length;
    const nPts = r.n_params_global != null ? r.n_params_global : r.n_params;
    const dateStr = scanIdToDate(r.scan_id);
    const ti = r.thresholds_info || null;
    // Surface unpack errors prominently so empty charts have a reason.
    let warnHtml = "";
    if (r.unpack_error) {
      warnHtml = `
        <div style="background:rgba(210,153,34,0.08);
                    border:1px solid var(--warn);
                    border-radius:4px;padding:10px 14px;
                    font-family:var(--mono);font-size:12px;
                    margin-bottom:10px;">
          <div class="warn" style="font-weight:600;margin-bottom:4px;">
            unpack_scan_logicals failed — charts may be empty
          </div>
          <div style="white-space:pre-wrap;color:var(--text);">${escHtml(r.unpack_error)}</div>
          ${r.data_shapes ? `
            <div class="hint" style="margin-top:6px;">
              data shapes: <code>${escHtml(JSON.stringify(r.data_shapes))}</code>
            </div>` : ""}
        </div>`;
    } else if (r.n_params === 0) {
      warnHtml = `
        <div style="background:rgba(210,153,34,0.08);
                    border:1px solid var(--warn);
                    border-radius:4px;padding:10px 14px;
                    font-family:var(--mono);font-size:12px;
                    margin-bottom:10px;">
          <div class="warn" style="font-weight:600;">
            no scan points unpacked (n_params=0) — check that logicals + Params + seq_ids are consistent in this scan
          </div>
          ${r.data_shapes ? `
            <div class="hint" style="margin-top:6px;">
              data shapes: <code>${escHtml(JSON.stringify(r.data_shapes))}</code>
            </div>` : ""}
        </div>`;
    }
    // Survival/TP headline tile — only for 2-image scans (1-image has
    // no img2). Source badge marks lab-computed vs SLM-cache.
    const survTile = oneImg ? "" : `
        <div class="stat-tile">
          <span class="stat-label">${targetAware ? "TP (target)" : "survival"}${
            targetAware && sg.survival_source
              ? ` <span class="src-badge src-${
                  sg.survival_source.startsWith("lab") ? "lab" : "slm"
                }">${sg.survival_source.startsWith("lab") ? "lab" : "SLM cache"}</span>`
              : ""
          }</span>
          <span class="stat-value" title="whole-scan (filter-independent)">${
            targetAware && ta && ta.overall_mean != null
              ? fmtPct(ta.overall_mean)
              : fmtPct(avg(sg.survival_mean))
          }</span>
        </div>`;
    // Discrimination infidelity — shown for ANY scan (data-quality signal).
    // Use the MEDIAN site (robust: the mean is dragged up by a few junk/edge
    // sites whose double-Gaussian fit is meaningless). Lower = better.
    const discVal = disc
      ? (disc.median_infidelity != null ? disc.median_infidelity : disc.mean_infidelity)
      : null;
    const discFromRun = !!(disc && disc.source === "recomputed_from_run");
    const discSrc = discFromRun ? "this run" : "scan-start";
    const calAge = ti && ti.calibration_age_human ? ti.calibration_age_human : null;
    // Only color-code (green/red) when it's the RUN's actual value. For the
    // stored scan-start calibration, stay NEUTRAL — it may be stale and must
    // not falsely read as "great" (operator request).
    const discColor = discFromRun ? infidColor(discVal) : "var(--text-dim)";
    const discTitle = discFromRun
      ? `median per-site infidelity from THIS run's data. Lower is better. mean=${fmtInfid(disc.mean_infidelity)}, max=${fmtInfid(disc.max_infidelity)}.`
      : `STORED scan-start calibration infidelity${calAge ? ` — calibrated ${calAge} before this run` : ""}. MAY BE STALE: the run's actual value can differ — click "recompute from this run". mean=${fmtInfid(disc.mean_infidelity)}.`;
    const discTile = disc ? `
        <div class="stat-tile" title="${discTitle}">
          <span class="stat-label">infidelity&darr; <span class="src-badge src-${discFromRun ? "lab" : "slm"}">${discSrc}</span></span>
          <span class="stat-value" style="color:${discColor}">${fmtInfid(discVal)}${
            discFromRun ? "" : ` <span class="muted" style="font-size:10px;">cal${calAge ? " · " + escHtml(calAge) : ""}</span>`
          }</span>
        </div>` : "";
    const params = r.run_parameters || [];
    body.innerHTML = warnHtml + `
      <div class="run-head">
        <span class="run-name" title="${escHtml(r.scan_filename || "")}">${escHtml(r.scan_name || "(unnamed scan)")}</span>
        ${dateStr ? `<span class="run-date mono">${dateStr}</span>` : ""}
      </div>
      ${r.scan_description ? `<div class="run-desc">${escHtml(r.scan_description)}</div>` : ""}
      <div class="run-head-body">
      <div class="stat-grid">
        <div class="stat-tile">
          <span class="stat-label">scan_id</span>
          <span class="stat-value mono scan-id-copy" style="font-size:14px;"
                data-scan-id="${r.scan_id}"
                title="Click to copy">${r.scan_id}</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">sweep</span>
          <span class="stat-value" title="${escHtml((sweepG.cols || []).join(', '))}">${nDims}D · ${nPts} pts</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">shots</span>
          <span class="stat-value" title="actual recorded shots${
            r.n_shots_scheduled != null ? ` · scheduled ${r.n_shots_scheduled}` : ""
          }">${r.n_shots}${
            r.n_shots_scheduled != null && r.n_shots_scheduled !== r.n_shots
              ? ` <span class="muted" style="font-size:11px;">/ ${r.n_shots_scheduled}</span>` : ""
          }</span>
        </div>
        ${survTile}
        <div class="stat-tile">
          <span class="stat-label">loading</span>
          <span class="stat-value">${fmtPct(avg(sg.loading_rate))}</span>
        </div>
        ${discTile}
        ${targetAware && ta && ta.fp_overall != null ? `
        <div class="stat-tile">
          <span class="stat-label">FP</span>
          <span class="stat-value">${fmtPct(ta.fp_overall)}</span>
        </div>` : ""}
        ${diag ? `
        <div class="stat-tile">
          <span class="stat-label">mean total_ms</span>
          <span class="stat-value">${fmtNum(diag.mean_total_ms)}</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">aborted</span>
          <span class="stat-value">${diag.aborted_count}</span>
        </div>` : ""}
      </div>
      <div class="run-meta">
        <div>swept: <span class="mono">${(sweepG.cols || []).join(", ") || "(none)"}</span></div>
        ${ti ? `<div title="${escHtml(ti.source_note || "")}">thresholds: <span class="mono">${ti.source || "?"}</span>${
          (ti.patterns && ti.patterns.length) ? ` · pattern${ti.patterns.length === 1 ? "" : "s"}: <span class="mono">${escHtml(ti.patterns.join(", "))}</span>` : ""
        } · mean ${fmtNum(ti.mean, 1)} (${ti.n} sites)${
          ti.mean_infidelity != null ? ` · mean infid <span class="muted">${fmtInfid(ti.mean_infidelity)} (cal)</span>` : ""
        }${
          ti.calibration_age_human ? ` · <span style="color:${(ti.calibration_age_s || 0) > 86400 ? "#d29922" : "var(--text-dim)"}" title="calibration source: ${escHtml(ti.calibration_source || "")}${ti.calibration_age_basis === "file_mtime" ? " (from file mtime — approximate)" : ""}">calibrated ${escHtml(ti.calibration_age_human)} before run${ti.calibration_age_basis === "file_mtime" ? "~" : ""}</span>` : ""
        }</div>` : ""}
        <div>
          code snapshot: <span class="${code.present ? "ok" : "muted"}">${code.present ? code.n_files + " files" : "none"}</span>
          · grid sidecar: <span class="${grid.present ? "ok" : "muted"}">${grid.present ? grid.n_sites + " sites" : "none"}</span>
          ${disc ? `· discrimination: <span class="src-badge src-${discFromRun ? "lab" : "slm"}">${discSrc}</span>
            <button class="ghost" id="recompute-infid"
                title="Refit per-site discrimination from THIS run's intensities (also computes imaging fidelity at the used thresholds). Non-destructive; cached after the first run; default uses the scan-start calibration.">${
              discFromRun ? "✓ using this run (click for scan-start)" : "recompute from this run"
            }</button>` : ""}
          · <button class="ghost" id="reanalyze-btn"
              title="Clear this run's cached analysis (double-Gaussian fits, focus metrics) and recompute from scratch.">↻ re-analyze</button>
        </div>
        ${imf ? `<div title="Discrimination fidelity at the ACTUALLY-USED thresholds (initThresholds), measured on THIS run's intensities — i.e. how trustworthy the bitstrings/logicals the run actually produced were. 1 − infidelity at the used cut, averaged over sites.">
          imaging fidelity (used thresholds, this run):
          <span style="color:${infidColor(1 - (imf.mean_fidelity || 0))};font-weight:600;">${fmtPct(imf.mean_fidelity)}</span>
          mean · ${fmtPct(imf.median_fidelity)} median
          <span class="muted">(${imf.n_sites} sites)</span>
        </div>` : ""}
      </div>
      </div>
      ${params.length ? `
      <details class="calib-details run-params-details">
        <summary>Details — ${params.length} parameter${params.length === 1 ? "" : "s"} set this run
          <span class="hint">(swept params highlighted)</span>
        </summary>
        <table class="run-params-table mono">
          ${params.map((p) => {
            const swept = p.group === "swept";
            return `<tr class="${swept ? "rp-swept" : ""}">
            <td class="rp-name">${escHtml(p.name)}</td>
            <td class="rp-val">${escHtml(Array.isArray(p.value)
              ? "[" + p.value.join(", ") + "]"
              : typeof p.value === "object"
                ? JSON.stringify(p.value) : String(p.value))}</td>
            <td class="rp-grp">${escHtml(swept ? "swept" : p.group)}</td></tr>`;
          }).join("")}
        </table>
      </details>` : ""}
    `;
    // Recompute-from-this-run toggle (non-destructive; re-fetches analysis
    // with recompute_infidelity so the discrimination metric reflects the
    // run's own data instead of the scan-start calibration).
    const recompBtn = document.getElementById("recompute-infid");
    if (recompBtn) recompBtn.addEventListener("click", () => {
      recomputeInfid = !recomputeInfid;
      if (selectedScanId) loadAnalysis(selectedScanId, {keepFilters: true});
    });
    // Re-analyze: invalidate the on-disk analysis cache + recompute.
    const reanBtn = document.getElementById("reanalyze-btn");
    if (reanBtn) reanBtn.addEventListener("click", () => {
      if (selectedScanId)
        loadAnalysis(selectedScanId, {keepFilters: true, forceRecache: true});
    });
    // The dedicated Survival / Loading cards were replaced by the
    // per-site maps + per-iteration chart further down the tab. Only
    // the sweep visualization needs to render here.
    plotAnalysisScanCurve(r);
  }

  // Sanitize an error array to plain finite numbers (Plotly error_y needs
  // numbers; nulls/NaN → 0 = "no bar at this point").
  function _cleanErr(arr) {
    if (!Array.isArray(arr)) return null;
    return arr.map((v) => (v == null || !isFinite(v)) ? 0 : v);
  }
  // Pick the per-param error array for the chosen metric + signal.
  //   sem_pershot (default) | std_pershot | sem_site | none
  function _errArray(summary, isSurv, mode) {
    if (mode === "none") return null;
    if (mode === "std_pershot")
      return _cleanErr(summary[isSurv ? "survival_std_pershot" : "loading_std_pershot"]);
    if (mode === "sem_site")
      return _cleanErr(summary[isSurv ? "survival_sem" : "loading_rate_sem"]);
    return _cleanErr(summary[isSurv ? "survival_sem_pershot" : "loading_sem_pershot"]);
  }
  const _errLabel = (m) => ({sem_pershot: "SEM (per-shot)",
    std_pershot: "STD (per-shot)", sem_site: "SEM (per-site)",
    none: "no errors"})[m] || m;

  function plotAnalysisScanCurve(r) {
    const el = $("plot-analysis-scan");
    const elLines = $("plot-analysis-scan-lines");
    if (!el || !window.Plotly) return;
    const sweep = r.sweep || {};
    const summary = r.summary || {};
    const dims = sweep.dims || [];
    const cols = sweep.cols || [];
    const sm = summary.survival_mean || [];
    const lr = summary.loading_rate  || [];
    // 1-image (loading-only) scans show LOADING as the main signal; all
    // others default to survival/TP (already TP-overridden in summary).
    const oneImg = !hasFinite(sm);
    const isSurv = !oneImg && sm.length > 0;
    const useY = isSurv ? sm : lr;
    const useE = _errArray(summary, isSurv, sweepPrefs.errMode);
    const targetAware = !!summary.survival_source;
    const yLabel = isSurv
        ? (targetAware ? "TP (target survival)" : "survival")
        : "loading rate";
    const baseMargin = {l: 70, r: 50, t: 14, b: 56};
    const nDimsReal = dims.filter((d) => d > 1).length;

    // Controls bar: show error dropdown for everything; 2D block only
    // for 2D sweeps.
    const ctrls = $("sweep-controls");
    if (ctrls) {
      ctrls.hidden = false;
      const only2d = ctrls.querySelector(".only-2d");
      if (only2d) only2d.style.display = (nDimsReal >= 2) ? "" : "none";
    }
    // Lines container only used for 2D "both"/"lines".
    const showLines = (nDimsReal >= 2)
      && (sweepPrefs.view === "both" || sweepPrefs.view === "lines");
    const showHeat = (nDimsReal < 2)
      || (sweepPrefs.view === "both" || sweepPrefs.view === "heatmap");
    if (elLines) elLines.hidden = !showLines;
    el.hidden = (nDimsReal >= 2 && !showHeat);

    // ---- 0D (single point) ----
    if (nDimsReal === 0) {
      const yMean = useY.length ? useY[0] : null;
      const yErr  = useE && useE.length ? useE[0] : null;
      Plotly.react(el, [{
        x: [0, r.n_shots || 1], y: [yMean, yMean], mode: "lines",
        line: {color: "#58a6ff", width: 2, dash: "dash"}, name: yLabel,
      }], plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: "shot # (0d, single point)" }, tickformat: ".0f" },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05], tickformat: ".2f" },
        annotations: [{
          text: `${yLabel} = ${fmtNum(yMean, 3)}${yErr != null ? " ± " + fmtNum(yErr, 3) : ""}`,
          xref: "paper", yref: "paper", x: 0.5, y: 0.5,
          showarrow: false, font: { size: 14, color: "#ffdd44" },
          bgcolor: "rgba(20,20,40,0.7)",
        }],
      }), plotConfig());
      setText("analysis-scan-info", `0d · ${r.n_shots || 0} shots · ${_errLabel(sweepPrefs.errMode)}`);
      return;
    }

    // ---- 1D ----
    if (nDimsReal === 1) {
      // The single real axis may not be axis 0 (e.g. a 2-axis scan
      // filtered to one value on the other axis). Find it.
      const axisIdx = Math.max(0, dims.findIndex((d) => d > 1));
      const xs = (sweep.values && sweep.values[axisIdx]) || useY.map((_, i) => i + 1);
      const xLabel = cols[axisIdx] || "scan param";
      Plotly.react(el, [{
        x: xs, y: useY,
        error_y: useE ? {type: "data", array: useE, visible: true, color: "#1f6feb"} : undefined,
        mode: "markers+lines",
        marker: {size: 8, color: "#58a6ff"}, line: {color: "#1f6feb", width: 2},
        hovertemplate: `${xLabel}=%{x:.4g}<br>${yLabel}=%{y:.3f}<extra></extra>`,
      }], plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: xLabel } },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05], tickformat: ".2f" },
      }), plotConfig());
      setText("analysis-scan-info",
        `1d · ${xLabel} · ${dims[axisIdx]} pts · ${_errLabel(sweepPrefs.errMode)}`);
      _wireSweepZoomFilter(el, r, [{prefix: "xaxis", col: axisIdx}]);
      return;
    }

    // ---- 2D (or higher; >2 reduced via the Filter card) ----
    // Identify the first two real (size>1) axes.
    const realAxes = dims.map((d, i) => [d, i]).filter(([d]) => d > 1).map(([, i]) => i);
    if (realAxes.length > 2) {
      Plotly.purge(el);
      if (elLines) { Plotly.purge(elLines); elLines.hidden = true; }
      el.hidden = false;
      el.innerHTML = `<div class="hint" style="padding:32px;text-align:center;">
        ${realAxes.length}D sweep (${realAxes.map((i) => cols[i]).join(" × ")}).<br>
        Pin extra axes to a single value in the <b>Filter</b> card to reduce to a 2-D view.</div>`;
      setText("analysis-scan-info", `${realAxes.length}d · use the Filter card to reduce`);
      return;
    }
    let ax0 = realAxes[0], ax1 = realAxes[1];
    if (sweepPrefs.axisSwap) { const t = ax0; ax0 = ax1; ax1 = t; }
    const xVals = (sweep.values || [])[ax0] || [];
    const yVals = (sweep.values || [])[ax1] || [];
    const nx = xVals.length, ny = yVals.length;
    const xLabel = cols[ax0] || "x", yLabel2 = cols[ax1] || "y";
    // useY is flattened in column-major over the original axes; map by
    // (i over ax0, j over ax1) -> original linear index.
    const dim0orig = dims[realAxes[0]];
    const swap = sweepPrefs.axisSwap;
    const valAt = (ix, jy) => {
      // ix indexes the displayed-x axis (ax0), jy the displayed-y (ax1).
      const i0 = swap ? jy : ix;   // index along original realAxes[0]
      const i1 = swap ? ix : jy;   // index along original realAxes[1]
      const lin = i1 * dim0orig + i0;   // column-major over (axis0, axis1)
      return useY[lin] ?? null;
    };

    if (showHeat) {
      const z = [];
      for (let jy = 0; jy < ny; jy++) {
        const row = [];
        for (let ix = 0; ix < nx; ix++) row.push(valAt(ix, jy));
        z.push(row);
      }
      // Square cells = categorical equal spacing; default = numeric
      // coords (cells sized by the separation between sweep values).
      const sq = sweepPrefs.square;
      const trace = {
        z, type: "heatmap", colorscale: "Viridis", zmin: 0, zmax: 1,
        colorbar: {title: {text: yLabel}, len: 0.9, tickformat: ".2f"},
        x: sq ? xVals.map((v) => Number(v).toPrecision(4)) : xVals,
        y: sq ? yVals.map((v) => Number(v).toPrecision(4)) : yVals,
      };
      Plotly.react(el, [trace], plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: xLabel }, type: sq ? "category" : undefined },
        yaxis: { title: { text: yLabel2 }, type: sq ? "category" : undefined },
      }), plotConfig());
      // Box-zoom filters in both numeric (value range) and square
      // (category-index range) modes.
      _wireSweepZoomFilter(el, r, sq
        ? [{prefix: "xaxis", col: ax0, cats: xVals},
           {prefix: "yaxis", col: ax1, cats: yVals}]
        : [{prefix: "xaxis", col: ax0}, {prefix: "yaxis", col: ax1}]);
    }

    if (showLines && elLines) {
      // One trace per displayed-y value; X = displayed-x axis.
      const traces = [];
      for (let jy = 0; jy < ny; jy++) {
        const yy = [];
        for (let ix = 0; ix < nx; ix++) yy.push(valAt(ix, jy));
        traces.push({
          x: xVals, y: yy, mode: "markers+lines",
          name: `${yLabel2}=${Number(yVals[jy]).toPrecision(4)}`,
          marker: {size: 6}, line: {width: 1.5},
          hovertemplate: `${xLabel}=%{x:.4g}<br>${yLabel}=%{y:.3f}<extra>${yLabel2}=${Number(yVals[jy]).toPrecision(4)}</extra>`,
        });
      }
      Plotly.react(elLines, traces, plotLayoutFlush({
        margin: {l: 70, r: 16, t: 14, b: 56},
        xaxis: { title: { text: xLabel } },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05], tickformat: ".2f" },
        showlegend: true, legend: {font: {size: 9}},
      }), plotConfig());
      _wireSweepZoomFilter(elLines, r, [{prefix: "xaxis", col: ax0}]);
    }
    setText("analysis-scan-info",
      `2d · ${xLabel} × ${yLabel2} · ${nx}×${ny}${sweepPrefs.square ? " · square" : ""}`);
  }

  // Zoom-to-filter: box-zoom on a sweep plot maps the visible range of each
  // mapped axis to the swept values inside it and commits them to the same
  // activeFilters the chip UI uses. Double-click (autorange) clears those
  // axes. `specs` is a list of {prefix:'xaxis'|'yaxis', col:<sweep col idx>,
  // cats?:<displayed numeric values>}. When `cats` is set the axis is
  // CATEGORICAL (square-cells heatmap) and the relayout range is in index
  // space; otherwise it's numeric data space.
  function _wireSweepZoomFilter(el, r, specs) {
    if (!el || !el.on) return;
    const sweepAll = r.sweep_all || r.sweep || {};
    try { el.removeAllListeners && el.removeAllListeners("plotly_relayout"); }
    catch { /* not a plotly gd yet */ }
    el.on("plotly_relayout", (ev) => {
      if (!ev) return;
      const updates = {};   // axisName -> array | null (clear)
      for (const spec of specs) {
        const name = (sweepAll.cols || [])[spec.col];
        if (!name) continue;
        const pfx = spec.prefix;
        if (ev[pfx + ".autorange"]) { updates[name] = null; continue; }
        // Range comes as indexed keys OR a single array (varies by
        // interaction / trace type — heatmaps often use the array form).
        let r0 = ev[pfx + ".range[0]"], r1 = ev[pfx + ".range[1]"];
        const ra = ev[pfx + ".range"];
        if ((r0 == null || r1 == null) && Array.isArray(ra) && ra.length === 2) {
          r0 = ra[0]; r1 = ra[1];
        }
        if (r0 == null || r1 == null) continue;
        const lo = Math.min(Number(r0), Number(r1));
        const hi = Math.max(Number(r0), Number(r1));
        let sel;
        if (spec.cats) {
          // Categorical axis: range is in category-index space.
          const cats = spec.cats;
          const i0 = Math.max(0, Math.ceil(lo));
          const i1 = Math.min(cats.length - 1, Math.floor(hi));
          sel = [];
          for (let i = i0; i <= i1; i++) sel.push(Number(cats[i]));
          if (sel.length && sel.length < cats.length) updates[name] = sel;
        } else {
          const allVals = (sweepAll.values || [])[spec.col] || [];
          sel = allVals.filter((v) => Number(v) >= lo && Number(v) <= hi);
          if (sel.length && sel.length < allVals.length)
            updates[name] = sel.map(Number);
        }
      }
      if (!Object.keys(updates).length) return;
      let changed = false;
      for (const [name, sel] of Object.entries(updates)) {
        if (sel == null) {
          if (activeFilters[name]) { delete activeFilters[name]; changed = true; }
        } else {
          const prev = activeFilters[name] || [];
          if (prev.length !== sel.length
              || sel.some((v, i) => Number(v) !== Number(prev[i]))) {
            activeFilters[name] = sel.map(Number);
            changed = true;
          }
        }
      }
      if (changed && selectedScanId)
        loadAnalysis(selectedScanId, {keepFilters: true});
    });
  }

  // Wire the sweep-control bar (error metric + 2D view / swap / square).
  // State lives in sweepPrefs (persisted); each change just re-renders
  // the current analysis (no refetch needed — all data is already local).
  (function wireSweepControls() {
    const elErr = $("sweep-err"), elView = $("sweep-view"),
          elSwap = $("sweep-swap"), elSq = $("sweep-square");
    const rerender = () => { if (activeAnalysis) plotAnalysisScanCurve(activeAnalysis); };
    if (elErr) {
      elErr.value = sweepPrefs.errMode;
      elErr.addEventListener("change", () => {
        sweepPrefs.errMode = elErr.value; saveSweepPrefs(); rerender();
      });
    }
    if (elView) {
      elView.value = sweepPrefs.view;
      elView.addEventListener("change", () => {
        sweepPrefs.view = elView.value; saveSweepPrefs(); rerender();
      });
    }
    if (elSwap) elSwap.addEventListener("click", () => {
      sweepPrefs.axisSwap = !sweepPrefs.axisSwap; saveSweepPrefs(); rerender();
    });
    if (elSq) {
      elSq.checked = !!sweepPrefs.square;
      elSq.addEventListener("change", () => {
        sweepPrefs.square = elSq.checked; saveSweepPrefs(); rerender();
      });
    }
  })();

  // =====================================================================
  // FILTER PANEL — one chip-row per swept-param axis. Clicking a chip
  // toggles inclusion in the allowed-values set. Each change triggers a
  // refetch of /api/runs/<id>/analysis with the filter encoded.
  // =====================================================================
  function renderAnalysisFilters(r) {
    const wrap = $("analysis-filters");
    const body = $("filter-body");
    // ALWAYS use the unfiltered sweep so chips show every possible
    // value -- the user has to know what they can still pick after
    // narrowing. (Old bug: derived chips from r.sweep, which got
    // narrowed to the filtered subset and made the others vanish.)
    const sweep = r.sweep_all || r.sweep || {};
    const cols  = sweep.cols   || [];
    const vals  = sweep.values || [];
    if (!cols.length || !vals.length) {
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;
    let html = "";
    cols.forEach((name, axisIdx) => {
      const axisVals = vals[axisIdx] || [];
      if (!axisVals.length) return;
      const allowed = activeFilters[name] || [];
      html += `<div class="filter-row" data-axis="${escHtml(name)}">
        <span class="filter-axis-label">${escHtml(name)}</span>
        <div class="filter-chip-grid" data-axis="${escHtml(name)}">`;
      axisVals.forEach((v, vi) => {
        const selected = allowed.length === 0
          ? false
          : allowed.some((a) => Math.abs(Number(a) - Number(v)) < 1e-6);
        const av = Math.abs(Number(v));
        const lbl = (av >= 1e4 || (av < 1e-2 && av > 0))
          ? Number(v).toExponential(2) : Number(v).toPrecision(4);
        html += `<button class="filter-chip ${selected ? "selected" : ""}"
                    data-axis="${escHtml(name)}"
                    data-val="${v}" data-idx="${vi}">
          ${escHtml(lbl)}
        </button>`;
      });
      html += `</div></div>`;
    });
    body.innerHTML = html;
    updateFilterStatus();
    updateFilterDetail(sweep);
    wireFilterChipInteractions(body);
  }

  // Detail line below the filter status -- shows axis-by-axis breakdown
  // of how many values are selected vs. how many exist. Visible in both
  // peek and expanded so the status line above it never moves.
  function updateFilterDetail(sweep) {
    const el = document.getElementById("filter-detail");
    if (!el) return;
    const cols = (sweep && sweep.cols) || [];
    const vals = (sweep && sweep.values) || [];
    if (!cols.length) {
      el.innerHTML = '<span class="muted">no swept axis on this run</span>';
      return;
    }
    el.innerHTML = cols.map((name, i) => {
      const total = (vals[i] || []).length;
      const sel = (activeFilters[name] || []).length || total;
      const all = sel === total || sel === 0;
      const counts = all ? `all ${total}` : `${sel} of ${total}`;
      return `<span class="axis-pill">
        <span class="axis-name">${escHtml(name)}</span>
        <span class="axis-counts">${escHtml(counts)}</span>
      </span>`;
    }).join("");
  }

  // Module-level drag state (NOT per-render closures, which leaked
  // event listeners and left "drag active" sticking between selects).
  const filterDrag = {
    active: false,
    mode:   null,   // "select" | "deselect"
    axis:   null,
    dirty:  false,
    local:  {},     // axis -> Set<number> (provisional)
  };

  function _filterChipSet(axis) {
    if (!filterDrag.local[axis]) {
      filterDrag.local[axis] = new Set(
        (activeFilters[axis] || []).map(Number));
    }
    return filterDrag.local[axis];
  }
  function _filterApplyToChip(chip) {
    if (!chip || chip.dataset.axis !== filterDrag.axis) return;
    const v = Number(chip.dataset.val);
    const set = _filterChipSet(filterDrag.axis);
    const isSel = chip.classList.contains("selected");
    if (filterDrag.mode === "select" && !isSel) {
      set.add(v); chip.classList.add("selected"); filterDrag.dirty = true;
    } else if (filterDrag.mode === "deselect" && isSel) {
      for (const a of set) {
        if (Math.abs(a - v) < 1e-6) { set.delete(a); break; }
      }
      chip.classList.remove("selected"); filterDrag.dirty = true;
    }
  }
  function _filterCommit() {
    for (const [axis, set] of Object.entries(filterDrag.local)) {
      if (set.size) activeFilters[axis] = Array.from(set);
      else delete activeFilters[axis];
    }
    filterDrag.local = {};
    if (selectedScanId) loadAnalysis(selectedScanId, {keepFilters: true});
  }

  // GLOBAL listeners -- attached ONCE at script load, NOT per-render.
  // - mousedown on a chip starts a drag (also acts as a single click;
  //   mouseup commits even if no other chip entered).
  // - mousemove only advances the drag while the LEFT button is held;
  //   `e.buttons & 1` checks the live state on every move, so if the
  //   user releases outside any chip, the next move with the button
  //   up bails immediately.
  // - mouseup ALWAYS ends the drag (or releases the just-committed
  //   single click).
  document.addEventListener("mousedown", (e) => {
    const chip = e.target.closest && e.target.closest(".filter-chip");
    if (!chip) return;
    e.preventDefault();
    filterDrag.active = true;
    filterDrag.dirty  = false;
    filterDrag.axis   = chip.dataset.axis;
    filterDrag.mode   = chip.classList.contains("selected") ? "deselect" : "select";
    _filterApplyToChip(chip);
  });
  document.addEventListener("mousemove", (e) => {
    if (!filterDrag.active) return;
    // Mouse-button NOT held? -> drag is over even though no mouseup
    // fired (can happen if release lands on a non-document target).
    if ((e.buttons & 1) === 0) {
      filterDrag.active = false;
      if (filterDrag.dirty) _filterCommit();
      else filterDrag.local = {};
      return;
    }
    const chip = e.target.closest && e.target.closest(".filter-chip");
    if (chip) _filterApplyToChip(chip);
  });
  document.addEventListener("mouseup", () => {
    if (!filterDrag.active) return;
    filterDrag.active = false;
    if (filterDrag.dirty) _filterCommit();
    else filterDrag.local = {};
  });

  // Hook function used by renderAnalysisFilters -- no-op now (all
  // listeners are global). Kept so the call site stays the same.
  function wireFilterChipInteractions() { /* no-op: global listeners */ }

  function updateFilterStatus() {
    const el = $("filter-status");
    const n = Object.keys(activeFilters).reduce(
      (sum, k) => sum + (activeFilters[k] || []).length, 0);
    if (n === 0) {
      el.textContent = "no filter";
      el.classList.remove("active");
    } else {
      const axes = Object.keys(activeFilters).length;
      el.textContent = `${n} value${n === 1 ? "" : "s"} across ${axes} axis${axes === 1 ? "" : "es"}`;
      el.classList.add("active");
    }
    // Mirror the active-filter count onto the filter card's data
    // attribute so the edge tab can highlight when filtered.
    const filtCard = document.getElementById("analysis-filters");
    if (filtCard) filtCard.dataset.floatCount = String(n);
  }

  $("filter-clear").addEventListener("click", () => {
    activeFilters = {};
    if (selectedScanId) loadAnalysis(selectedScanId, {keepFilters: true});
  });
  $("analysis-filters").querySelector(".filter-header")
    .addEventListener("click", (e) => {
      // Don't collapse when the Clear button is clicked.
      if (e.target.tagName === "BUTTON") return;
      $("analysis-filters").classList.toggle("collapsed");
    });

  // =====================================================================
  // PER-SITE MAPS — three side-by-side scattergl panels coloring each
  // tweezer site by its per-metric scalar. Dot size is operator-tunable
  // via the small slider in the Loading-map card header (single source
  // of truth for size across all 3 panels).
  // =====================================================================
  let siteDotSize = 4;     // user-controlled marker size

  // Phase 5a paths-overlay state (kept module-level so handlers can
  // re-render without a full analysis refetch).
  let pathsOverlayState = {
    payload: null,    // r.paths_overlay
    enabled: false,
    shotIdx: 0,       // index into payload.shot_indices
  };

  function _siteCard(cardId) {
    return document.querySelector(`[data-card-id="${cardId}"]`);
  }

  // Pooled camera-intensity histogram (data-quality), with the median
  // detection threshold(s) marked: scan-start always, plus this-run when
  // discrimination was recomputed (so both cuts are visible). The infidelity
  // metric itself is threshold-independent (optimal-cut); these are
  // reference lines for the empty/atom split.
  function plotIntensityHist(r) {
    const card = $("analysis-site-inthist-card");
    const el = $("plot-site-inthist");
    const ih = r && r.intensity_hist;
    if (!card || !el) return;
    if (!ih || !Array.isArray(ih.counts) || !ih.counts.length) {
      card.hidden = true;
      if (el) Plotly.purge(el);
      return;
    }
    card.hidden = false;
    if (!window.Plotly) return;
    const trace = {
      x: ih.bin_centers, y: ih.counts, type: "bar",
      marker: {color: "#58a6ff"}, hoverinfo: "x+y", name: "intensity",
    };
    const markers = ih.threshold_markers || [];
    const shapes = markers.map((m) => ({
      type: "line", x0: m.value, x1: m.value, yref: "paper", y0: 0, y1: 1,
      line: {color: m.source === "recomputed_from_run" ? "#4cc762" : "#f0883e",
             width: 1.5, dash: "dash"},
    }));
    const annos = markers.map((m, i) => ({
      x: m.value, yref: "paper", y: 1 - i * 0.07, text: m.label,
      showarrow: false, font: {size: 9,
        color: m.source === "recomputed_from_run" ? "#4cc762" : "#f0883e"},
      xanchor: "left", bgcolor: "rgba(20,20,40,0.6)",
    }));
    Plotly.react(el, [trace], plotLayoutFlush({
      margin: {l: 48, r: 14, t: 10, b: 40},
      xaxis: {title: {text: "camera intensity"}},
      yaxis: {title: {text: "count"}, type: "linear"},
      shapes, annotations: annos, bargap: 0.02, showlegend: false,
    }), plotConfig());
    const srcTxt = markers.length
      ? markers.map((m) => `${m.label.split(" ")[0]} ${fmtNum(m.value, 0)}`).join(" · ")
      : "";
    setText("site-inthist-info",
      `${(ih.n_samples || 0).toLocaleString()} samples${srcTxt ? " · thr " + srcTxt : ""}`);
  }

  function renderPerSiteMaps(r) {
    const ps = r.per_site;
    const infidCard = $("analysis-site-infid-card");
    if (!ps) {
      ["plot-site-loading", "plot-site-survival", "plot-site-fp",
       "plot-site-infid", "plot-site-inthist"].forEach((id) => {
        const el = $(id);
        if (el) Plotly.purge(el);
      });
      if (infidCard) infidCard.hidden = true;
      const ihc = $("analysis-site-inthist-card");
      if (ihc) ihc.hidden = true;
      setupPathsOverlay(null);
      return;
    }
    // 1-image (loading-only) scans have no img2 → survival / FP are empty.
    // Hide those two cards and lead with loading + discrimination.
    const oneImg = !hasFinite(ps.survival_mean);
    const survCard = _siteCard("analysis-site-survival");
    const fpCard   = _siteCard("analysis-site-fp");
    if (survCard) survCard.hidden = oneImg;
    if (fpCard)   fpCard.hidden   = oneImg;
    // Per-site discrimination-infidelity map (data-quality) — always shown.
    const haveInfid = Array.isArray(ps.infidelity) && ps.infidelity.length > 0;
    if (infidCard) infidCard.hidden = !haveInfid;
    if (haveInfid) {
      plotSiteMap("plot-site-infid", "site-infid-info",
        ps.x, ps.y, ps.infidelity, "Magma", "infidelity", {mode: "infid"});
    } else {
      const ie = $("plot-site-infid");
      if (ie) Plotly.purge(ie);
    }
    // Avg camera-intensity histogram (pooled over sites) next to it.
    plotIntensityHist(r);
    if (oneImg) {
      ["plot-site-survival", "plot-site-fp"].forEach((id) => {
        const el = $(id); if (el) Plotly.purge(el);
      });
      plotSiteMap("plot-site-loading", "site-loading-info",
        ps.x, ps.y, ps.loading_rate, "Cividis", "loading", {});
      setupPathsOverlay(null);
      return;
    }
    // Phase 5a paths-overlay: surface the picker on runs that have
    // paths data (lab-paths source). Hidden otherwise.
    setupPathsOverlay(r.paths_overlay);
    // Phase 5a: when the per_site source is lab-only filtered (legacy
    // run + filter active), surface a small hint so the operator
    // knows TP/FP markers were intentionally suppressed for this view.
    const infoSuffix = ps.note ? ` <span class="muted">(${ps.note})</span>` : "";
    // Phase 5a: when per_site has TP/FP markers (from
    // slm_analysis.json or, eventually, lab-paths-derived), pass them
    // so the survival map only colors TARGET sites and the FP map
    // only colors NON-TARGET sites. Non-applicable sites render as
    // small grey background dots so the operator can SEE the target
    // pattern without losing the array context.
    const tgtMask = ps.is_target_site || null;
    const ntgtMask = ps.is_nontarget_site || null;
    plotSiteMap("plot-site-loading", "site-loading-info",
      ps.x, ps.y, ps.loading_rate, "Cividis", "loading", {infoSuffix});
    plotSiteMap("plot-site-survival", "site-survival-info",
      ps.x, ps.y, ps.survival_mean, "Viridis",
      tgtMask ? "TP (target survival)" : "survival",
      {mask: tgtMask, maskLabel: "target site", infoSuffix,
       pathsOverlay: currentPathsOverlaySegments()});
    plotSiteMap("plot-site-fp", "site-fp-info",
      ps.x, ps.y, ps.fp_rate, "Plasma", "FP",
      {mask: ntgtMask, maskLabel: "non-target site", infoSuffix});
  }

  // Returns the trace index that holds the colored per-site markers
  // (the one whose marker.size should track the dot-size slider / zoom).
  // The other traces are: [bg dots]?, [paths polyline]?, [target endpoints]?.
  // Convention: layout always orders traces as
  //   [bg-greyed]? -> [colored data] -> [paths line]? -> [endpoint markers]?
  function _siteMapDataTraceIndex(el) {
    if (!el || !el.data) return 0;
    // The colored data trace is the first scattergl 'markers' trace
    // whose marker.colorscale is set (the others lack one).
    for (let i = 0; i < el.data.length; i++) {
      const tr = el.data[i];
      if (tr.mode === "markers" && tr.marker && tr.marker.colorscale) {
        return i;
      }
    }
    return el.data.length >= 2 ? 1 : 0;
  }

  function currentPathsOverlaySegments() {
    const s = pathsOverlayState;
    if (!s.enabled || !s.payload || !s.payload.all_segments) return null;
    const seg = s.payload.all_segments[s.shotIdx];
    if (!seg) return null;
    return {
      x: seg.x, y: seg.y,
      shot_index: seg.shot_index,
      seq_id: seg.seq_id,
      n_paths: seg.n_paths,
    };
  }

  function setupPathsOverlay(payload) {
    const ctl = $("paths-overlay-ctl");
    const toggle = $("paths-overlay-toggle");
    const sel = $("paths-overlay-shot");
    if (!ctl || !toggle || !sel) return;
    if (!payload || !payload.shot_indices || !payload.shot_indices.length) {
      ctl.hidden = true;
      pathsOverlayState = {payload: null, enabled: false, shotIdx: 0};
      return;
    }
    ctl.hidden = false;
    pathsOverlayState.payload = payload;
    // Reset shotIdx to default if out of bounds.
    if (pathsOverlayState.shotIdx >= payload.shot_indices.length) {
      pathsOverlayState.shotIdx = payload.default_idx || 0;
    }
    // Repopulate the shot picker.
    sel.innerHTML = payload.shot_indices.map((sh, i) => {
      const sid = payload.seq_ids[i];
      const seg = (payload.all_segments && payload.all_segments[i]) || {};
      const np = seg.n_paths != null ? ` · ${seg.n_paths} paths` : "";
      return `<option value="${i}">shot ${sh} (seq ${sid}${np})</option>`;
    }).join("");
    sel.value = String(pathsOverlayState.shotIdx);
    sel.disabled = !pathsOverlayState.enabled;
    toggle.checked = pathsOverlayState.enabled;
  }

  // Toggle / shot-picker handlers — re-render only the survival map.
  // Phase 5.5 prep: control sidebar toggle. The sidebar is
   // position:fixed on the right edge; the Live tab pane reserves
   // padding-right for it so chart content isn't covered. Default
   // OPEN (Phase 5.5 spec). Operator preference persists in
   // localStorage so a closed-then-refresh stays closed.
  (function wireLiveSidebarToggle() {
    const tab     = document.getElementById("tab-live");
    const sidebar = document.querySelector(".live-sidebar");
    const btn     = document.getElementById("live-sidebar-toggle");
    if (!tab || !sidebar || !btn) return;
    const KEY = "yb-dash-live-sidebar-collapsed";
    let collapsed = (() => {
      try { return localStorage.getItem(KEY) === "1"; }
      catch { return false; }
    })();
    const apply = () => {
      sidebar.classList.toggle("collapsed", collapsed);
      tab.classList.toggle("sidebar-collapsed", collapsed);
      tab.classList.toggle("sidebar-open", !collapsed);
      try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch {}
    };
    btn.addEventListener("click", () => { collapsed = !collapsed; apply(); });
    apply();
  })();

  // Live-tab image-row switches: Downsample (img1) + Middle frame (img2).
  (function wireLiveImageSwitches() {
    // Downsample toggle -> POST to the server, which writes the
    // browser->main reverse-channel control file; the main process reads
    // it when encoding the next live frame. Default ON (full-sensor
    // frames are ~12 MB of base64 and choke the browser).
    const ds = document.getElementById("downsample-live");
    if (ds) {
      ds.addEventListener("change", async () => {
        try {
          await api("/api/control/downsample", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({on: ds.checked}),
          });
        } catch (e) {
          console.warn("downsample toggle failed", e);
          toast("Downsample toggle failed: " + (e.message || e), "bad");
          ds.checked = !ds.checked;   // revert on failure
        }
      });
    }

    // Middle-frame toggle -> pure client-side show/hide (persisted).
    const mid = document.getElementById("show-mid-live");
    if (mid) {
      mid.checked = showMidFrame;
      mid.addEventListener("change", () => {
        showMidFrame = mid.checked;
        try { localStorage.setItem("yb-dash-show-mid", showMidFrame ? "1" : "0"); }
        catch {}
        applyMidVisibility();
      });
    }

    // Scan-curve colorbar autoscale toggle -> client-side (changes the query
    // param the next figures fetch sends); re-poll immediately so the scan
    // plot reflects the change without waiting for the next tick.
    const auto = document.getElementById("scan-autoscale-live");
    if (auto) {
      auto.checked = scanAutoscale;
      auto.addEventListener("change", () => {
        scanAutoscale = auto.checked;
        try { localStorage.setItem("yb-dash-scan-autoscale", scanAutoscale ? "1" : "0"); }
        catch {}
        pollLiveFigures();
      });
    }
  })();

  // Control sidebar buttons (Phase 5.5 Track A). Mirrors control_panel.py.
  (function wireControlSidebar() {
    const sidebar = document.querySelector(".live-sidebar");
    if (!sidebar) return;
    const JSON_HDR = {"Content-Type": "application/json"};
    // Single knob for every hold-to-confirm button (Restart All + the
    // MATLAB/pyctrl backend switches). Change this one value to retune the
    // hold duration for all of them at once.
    const HOLD_MS = 1000;

    async function postControl(path, body) {
      if (!controlsAllowed) {
        toast("Remote controls disabled on this interface", "bad");
        return null;
      }
      const opts = {method: "POST"};
      if (body !== undefined) {
        opts.headers = JSON_HDR;
        opts.body = JSON.stringify(body);
      }
      return api(path, opts);
    }

    // --- single-click actions ---
    const SIMPLE = {
      start: "/api/control/start",
      pause: "/api/control/pause",
      "restart-dash": "/api/control/restart_dash",
    };
    Object.entries(SIMPLE).forEach(([kind, path]) => {
      const btn = sidebar.querySelector(`.ctrl-btn[data-kind="${kind}"]`);
      if (!btn) return;
      btn.addEventListener("click", async () => {
        try { await postControl(path); toast(btn.textContent.trim() + " sent"); }
        catch (e) { toast(e.message || "control failed", "bad"); }
      });
    });

    // --- Abort: immediate single click (no hold-to-confirm — aborting a
    //     scan isn't destructive and should be instantly available). Still
    //     fetches the one-shot confirm token the server requires before
    //     POSTing /api/control/abort. ---
    const abortBtn = sidebar.querySelector('.ctrl-btn[data-kind="abort"]');
    if (abortBtn) {
      abortBtn.addEventListener("click", async () => {
        if (!controlsAllowed) {
          toast("Remote controls disabled on this interface", "bad");
          return;
        }
        try {
          const tok = await api("/api/control/confirm_token?action=abort");
          await api(`/api/control/abort?confirm=${encodeURIComponent(tok.token)}`,
                    {method: "POST"});
          toast("Abort sent");
        } catch (e) {
          toast(e.message || "abort failed", "bad");
        }
      });
    }

    // --- init folder ---
    const initBtn = sidebar.querySelector('.ctrl-btn[data-kind="init-folder"]');
    const initInput = document.getElementById("ctrl-init-path");
    if (initBtn && initInput) {
      initBtn.addEventListener("click", async () => {
        const path = initInput.value.trim();
        if (!path) { toast("Enter a folder path first", "bad"); return; }
        try {
          await postControl("/api/control/init_dir", {path});
          toast("Init folder load requested");
        } catch (e) { toast(e.message || "init load failed", "bad"); }
      });
    }

    // --- dummy-mode radios ---
    sidebar.querySelectorAll('input[name="dummy-mode-ph5"]').forEach((r) => {
      r.addEventListener("change", async () => {
        if (!r.checked) return;
        lastDummyUserChangeMs = Date.now();
        try { await postControl("/api/control/dummy_mode", {mode: r.value}); }
        catch (e) { toast(e.message || "dummy mode failed", "bad"); }
      });
    });

    // --- hold-to-confirm destructive ops (Restart All + backend switch,
    //     all HOLD_MS). Abort is handled above as an immediate single
    //     click. ---
    sidebar.querySelectorAll(".ctrl-hold").forEach((btn) => {
      const bar = btn.querySelector(".hold-bar");
      const holdMs = HOLD_MS;
      const action = btn.dataset.confirmAction;
      let raf = null, downAt = 0, firing = false;

      const reset = () => {
        if (raf) cancelAnimationFrame(raf);
        raf = null; downAt = 0;
        if (bar) {
          bar.style.transition = "transform 200ms ease-out";
          bar.style.transform = "scaleX(0)";
        }
      };
      const fire = async () => {
        if (firing) return;
        firing = true;
        if (raf) cancelAnimationFrame(raf);
        raf = null;
        if (!controlsAllowed) {
          toast("Remote controls disabled on this interface", "bad");
          firing = false; reset(); return;
        }
        // Backend switch to the already-active backend is a no-op.
        const target = btn.dataset.backendTarget;
        if (action === "set_backend" && target && target === currentBackend) {
          toast(`${target} already active`);
          firing = false; reset(); return;
        }
        try {
          const tok = await api(
            `/api/control/confirm_token?action=${action}`);
          let path, label;
          if (action === "abort") {
            path = "/api/control/abort";
            label = "Abort";
          } else if (action === "set_backend") {
            path = `/api/control/set_backend?target=${encodeURIComponent(target)}`;
            label = `Switch to ${target}`;
          } else {
            path = "/api/control/restart_all";
            label = "Restart All";
          }
          const sep = path.includes("?") ? "&" : "?";
          await api(`${path}${sep}confirm=${encodeURIComponent(tok.token)}`,
                    {method: "POST"});
          toast(label + " sent");
        } catch (e) {
          toast(e.message || "action failed", "bad");
        } finally {
          firing = false; reset();
        }
      };
      const tick = () => {
        const frac = Math.min(1, (performance.now() - downAt) / holdMs);
        if (bar) {
          bar.style.transition = "none";
          bar.style.transform = `scaleX(${frac})`;
        }
        if (frac >= 1) { fire(); return; }
        raf = requestAnimationFrame(tick);
      };
      const begin = (e) => {
        if (firing || downAt) return;
        if (e && e.preventDefault) e.preventDefault();
        downAt = performance.now();
        raf = requestAnimationFrame(tick);
      };
      const cancel = () => { if (!firing) reset(); };

      btn.addEventListener("pointerdown", begin);
      btn.addEventListener("pointerup", cancel);
      btn.addEventListener("pointerleave", cancel);
      btn.addEventListener("keydown", (e) => {
        if (e.key === " " || e.key === "Enter") begin(e);
      });
      btn.addEventListener("keyup", (e) => {
        if (e.key === " " || e.key === "Enter") cancel();
      });
    });
  })();

  // Camera controls (Phase 5.5) — mirror of the Tkinter CameraPane's
  // Connect / Disconnect / Apply buttons. ROI/exposure are POSTed to the
  // server, which spools them to the main process (it owns the ZMQ client
  // + expConfig.m persistence). Gated by the same controlsAllowed policy.
  (function wireCameraControls() {
    const sidebar = document.querySelector(".live-sidebar");
    if (!sidebar) return;

    function readCamFields() {
      const num = (id) => {
        const v = parseFloat($(id).value);
        return Number.isFinite(v) ? v : null;
      };
      return {
        roi: [num("cam-roi-x"), num("cam-roi-y"),
              num("cam-roi-w"), num("cam-roi-h")],
        exposure: num("cam-exposure"),
      };
    }

    // Mark fields touched so the status poll doesn't clobber a pending edit.
    sidebar.querySelectorAll(".cam-num").forEach((el) => {
      el.addEventListener("input", () => { camFieldsTouched = Date.now(); });
    });

    sidebar.querySelectorAll("[data-cam]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!controlsAllowed) {
          toast("Remote controls disabled on this interface", "bad");
          return;
        }
        const action = btn.dataset.cam;
        try {
          if (action === "disconnect") {
            await api("/api/control/camera/disconnect", {method: "POST"});
            toast("Camera disconnect sent");
            return;
          }
          const {roi, exposure} = readCamFields();
          if (roi.some((v) => v == null)) {
            toast("Bad ROI values", "bad"); return;
          }
          if (exposure == null || !(exposure > 0)) {
            toast("Bad exposure value", "bad"); return;
          }
          const path = action === "connect"
            ? "/api/control/camera/connect" : "/api/control/camera/apply";
          await api(path, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({roi, exposure}),
          });
          toast(action === "connect"
            ? "Camera connect sent" : "Camera settings applied");
        } catch (e) {
          toast(e.message || "camera command failed", "bad");
        }
      });
    });
  })();

  // "full ›" link (and any other data-goto-tab element): jump to a tab.
  document.querySelectorAll("[data-goto-tab]").forEach((el) => {
    el.addEventListener("click", () => setTab(el.dataset.gotoTab));
  });

  (function wirePathsOverlayHandlers() {
    const toggle = document.getElementById("paths-overlay-toggle");
    const sel    = document.getElementById("paths-overlay-shot");
    if (!toggle || !sel) return;
    const rerenderSurvival = () => {
      if (!activeAnalysis || !activeAnalysis.per_site) return;
      const ps = activeAnalysis.per_site;
      const tgtMask = ps.is_target_site || null;
      const infoSuffix = ps.note ? ` <span class="muted">(${ps.note})</span>` : "";
      plotSiteMap("plot-site-survival", "site-survival-info",
        ps.x, ps.y, ps.survival_mean, "Viridis",
        tgtMask ? "TP (target survival)" : "survival",
        {mask: tgtMask, maskLabel: "target site", infoSuffix,
         pathsOverlay: currentPathsOverlaySegments()});
    };
    toggle.addEventListener("change", () => {
      pathsOverlayState.enabled = toggle.checked;
      sel.disabled = !pathsOverlayState.enabled;
      rerenderSurvival();
    });
    sel.addEventListener("change", () => {
      pathsOverlayState.shotIdx = parseInt(sel.value, 10) || 0;
      rerenderSurvival();
    });
  })();

  function plotSiteMap(divId, infoId, x, y, values, colorscale, label, opts) {
    if (!window.Plotly) return;
    const el = $(divId);
    if (!el) return;
    if (!x || !y || !values || !x.length || values.length !== x.length) {
      Plotly.purge(el);
      el.innerHTML =
        '<div class="hint" style="padding:24px;text-align:center;">no data</div>';
      setText(infoId, "");
      return;
    }
    // When a mask is supplied, split into in-mask and out-of-mask
    // arrays. In-mask points carry the colormap (TP at target sites,
    // FP at non-target sites). Out-of-mask points render as small
    // grey background dots so the array geometry is still visible.
    const mask = opts && opts.mask;
    const maskLabel = opts && opts.maskLabel;
    const n = x.length;
    const xIn = [], yIn = [], vIn = [], iIn = [];
    const xOut = [], yOut = [], iOut = [];
    if (mask && mask.length === n) {
      for (let k = 0; k < n; k++) {
        if (mask[k]) {
          xIn.push(x[k]); yIn.push(y[k]); vIn.push(values[k]); iIn.push(k);
        } else {
          xOut.push(x[k]); yOut.push(y[k]); iOut.push(k);
        }
      }
    } else {
      for (let k = 0; k < n; k++) {
        xIn.push(x[k]); yIn.push(y[k]); vIn.push(values[k]); iIn.push(k);
      }
    }
    const infid = !!(opts && opts.mode === "infid");
    const finite = vIn.filter((v) => v != null && isFinite(v));
    const vmin = 0;
    const vmax = 1;
    const mean = finite.length ? finite.reduce((a, b) => a + b, 0) / finite.length : null;
    let infoTxt = mean != null
      ? (infid ? `mean ${fmtInfid(mean)}` : `mean ${fmtNum(mean, 3)}`)
      : "";
    if (mask && mask.length === n) {
      infoTxt += ` · ${xIn.length}/${n} ${maskLabel || "marked"}`;
    }
    const infoSuffix = opts && opts.infoSuffix;
    const el2 = $(infoId);
    if (el2) {
      el2.innerHTML = `${infoTxt}${infoSuffix || ""}`;
    } else {
      setText(infoId, infoTxt);
    }
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const baseXSpan = Math.max(1, xMax - xMin);
    el._sitemapBaseXSpan = baseXSpan;
    el._sitemapBaseSize = siteDotSize;
    const traces = [];
    // Background (out-of-mask) trace first so colored points sit on top.
    if (xOut.length) {
      traces.push({
        x: xOut, y: yOut, type: "scattergl", mode: "markers",
        marker: {
          size: Math.max(1, siteDotSize - 1),
          color: "#3a3a3a", line: {width: 0},
          opacity: 0.4,
        },
        hoverinfo: "skip",
        showlegend: false,
        name: "non-marked",
      });
    }
    if (infid) {
      // Log-scale infidelity (matches the live view _fig_infid): color by
      // log10 of the clipped infidelity, Magma_r, range [-4, -0.3];
      // raw value shown via customdata in scientific notation.
      const logv = vIn.map((v) => Math.log10(
        Math.min(1, Math.max(1e-6, (v == null || !isFinite(v)) ? 1e-6 : v))));
      traces.push({
        x: xIn, y: yIn, type: "scattergl", mode: "markers",
        customdata: vIn,
        marker: {
          size: siteDotSize, color: logv, colorscale: "Magma",
          reversescale: true, cmin: -4, cmax: -0.3, line: {width: 0},
          colorbar: {title: {text: "log10 infid"}, len: 0.9},
        },
        hovertemplate: "site %{pointNumber}: %{customdata:.2e}<extra></extra>",
        name: label,
      });
    } else {
      traces.push({
        x: xIn, y: yIn, type: "scattergl", mode: "markers",
        marker: {
          size: siteDotSize, color: vIn, colorscale,
          cmin: vmin, cmax: vmax,
          line: {width: 0},
          colorbar: {title: {text: label}, len: 0.9, tickformat: ".2f"},
        },
        hovertemplate: "site %{pointNumber}: %{marker.color:.3f}<extra></extra>",
        name: maskLabel || label,
      });
    }
    // Phase 5a: paths overlay — NaN-separated polyline drawn on top
    // of the markers. Light cyan with low opacity so the underlying
    // colormap stays legible.
    const po = opts && opts.pathsOverlay;
    if (po && Array.isArray(po.x) && Array.isArray(po.y) && po.x.length) {
      traces.push({
        x: po.x, y: po.y,
        type: "scattergl", mode: "lines",
        line: {color: "rgba(120, 220, 255, 0.55)", width: 1.2},
        hoverinfo: "skip",
        showlegend: false,
        name: `paths shot ${po.shot_index}`,
      });
      // Bonus: tiny end-cap markers at every target endpoint of the
      // segments so the user can see exactly where the atoms LANDED.
      // Pull endpoints out of the polyline (every 3rd entry pair = the
      // segment's end before the NaN separator).
      const xEnds = [], yEnds = [];
      for (let i = 1; i < po.x.length; i += 3) {
        xEnds.push(po.x[i]); yEnds.push(po.y[i]);
      }
      traces.push({
        x: xEnds, y: yEnds,
        type: "scattergl", mode: "markers",
        marker: {size: 3, color: "rgba(120, 220, 255, 0.9)",
                 symbol: "circle-open"},
        hovertemplate: "target endpoint<extra></extra>",
        showlegend: false,
        name: "target endpoints",
      });
    }
    Plotly.react(el, traces, plotLayoutFlush({
      xaxis: {visible: false},
      yaxis: {visible: false, autorange: "reversed",
              scaleanchor: "x", scaleratio: 1},
      margin: {l: 10, r: 60, t: 10, b: 10},
      showlegend: false,
    }), plotConfig());
    // (Re)wire the relayout handler -- Plotly fires this on zoom / pan
    // / dblclick reset. We scale the marker size proportional to the
    // zoom factor so dots maintain their relative size against the
    // image. Plotly.restyle is cheap; called once per zoom interaction.
    if (!el._sitemapZoomWired) {
      el._sitemapZoomWired = true;
      el.on("plotly_relayout", () => {
        if (!el._sitemapBaseXSpan) return;
        const xa = el._fullLayout && el._fullLayout.xaxis;
        if (!xa || !xa.range) return;
        const curSpan = Math.abs(xa.range[1] - xa.range[0]);
        if (!curSpan) return;
        const zoom = el._sitemapBaseXSpan / curSpan;
        const newSize = Math.max(1, el._sitemapBaseSize * zoom);
        // Resize ONLY the colored-data trace (skip background bg dots
        // at index 0 if present, and skip the paths-overlay lines /
        // endpoint markers which use their own fixed size).
        try {
          const idx = _siteMapDataTraceIndex(el);
          Plotly.restyle(el, {"marker.size": newSize}, [idx]);
        } catch {}
      });
    }
  }

  // Wire the dot-size slider: live-restyle all three site maps (no
  // refetch) so the user can fine-tune visibility for thousands of
  // sites without round-tripping the analysis call.
  const siteSizeInput = document.getElementById("site-dot-size");
  if (siteSizeInput) {
    siteSizeInput.addEventListener("input", () => {
      siteDotSize = Number(siteSizeInput.value);
      if (!window.Plotly) return;
      ["plot-site-loading", "plot-site-survival", "plot-site-fp",
       "plot-site-infid"]
        .forEach((id) => {
          const el = $(id);
          if (el && el.data) {
            // Update the base size used by the zoom-scaler too, and
            // apply at the CURRENT zoom level.
            el._sitemapBaseSize = siteDotSize;
            let scaled = siteDotSize;
            if (el._sitemapBaseXSpan && el._fullLayout
                && el._fullLayout.xaxis && el._fullLayout.xaxis.range) {
              const curSpan = Math.abs(
                el._fullLayout.xaxis.range[1] - el._fullLayout.xaxis.range[0]);
              if (curSpan) {
                scaled = Math.max(1, siteDotSize * (el._sitemapBaseXSpan / curSpan));
              }
            }
            try {
              const idx = _siteMapDataTraceIndex(el);
              Plotly.restyle(el, {"marker.size": scaled}, [idx]);
            } catch {}
          }
        });
    });
  }

  // =====================================================================
  // PER-ITERATION — single chart, X = shot index, toggleable traces.
  // Available metrics: loaded_frac, survival_frac, fp_frac, plus each
  // swept-param axis (param_values.<axis_name>). Click a chip to toggle.
  // =====================================================================
  const METRIC_COLORS = {
    loaded_frac:    "#58a6ff",
    survival_frac:  "#3fb950",
    fp_frac:        "#f85149",
  };

  // Context for the shot-image popup: the current per-iteration payload,
  // its scan_id, and frames-per-shot. Set on every per-iteration render so
  // the plotly_click handler can map a clicked point to its camera frames.
  let perIterCtx = null;

  function renderPerIteration(r) {
    const pi = r.per_iteration;
    perIterCtx = pi && pi.shot_index
      ? {pi, scanId: r.scan_id, numImages: pi.num_images || 1}
      : null;
    const toggleWrap = $("per-iter-toggles");
    if (!pi || !pi.shot_index || !pi.shot_index.length) {
      toggleWrap.innerHTML = '<span class="muted">no per-shot data</span>';
      Plotly.purge($("plot-per-iter"));
      setText("per-iter-info", "");
      const dw = $("per-iter-diag-wrap");
      if (dw) dw.style.display = "none";
      return;
    }
    // Available metrics.
    const metrics = [];
    if (pi.loaded_frac)    metrics.push({key: "loaded_frac",   label: "loaded frac",  isFrac: true});
    if (pi.survival_frac)  metrics.push({
      key: "survival_frac",
      // Phase 5a: when target-aware, the per-iteration curve is per-shot TP
      // (not per-site survival). Label changes accordingly so the legend
      // doesn't lie. `survival_source` is set by run_analysis when the
      // override applied (lab_paths or slm_server_cached).
      label: pi.survival_label || "survival",
      isFrac: true,
    });
    if (pi.fp_frac)        metrics.push({key: "fp_frac",       label: "FP rate",      isFrac: true});
    const paramVals = pi.param_values || {};
    Object.keys(paramVals).forEach((name) => {
      metrics.push({key: "param:" + name, label: name, isFrac: false,
                    paramName: name});
    });
    // Default: loaded + survival visible, others off.
    metrics.forEach((m) => {
      if (perIterToggles[m.key] === undefined) {
        perIterToggles[m.key] = (m.key === "loaded_frac" || m.key === "survival_frac");
      }
    });
    // Render chips.
    toggleWrap.innerHTML = metrics.map((m) => `
      <span class="metric-chip ${perIterToggles[m.key] ? "on" : ""}"
            data-metric="${m.key}"
            style="color: ${METRIC_COLORS[m.key] || "#ffdd44"}">
        <span class="dot"></span>${escHtml(m.label)}
      </span>`).join("");
    $$(".metric-chip", toggleWrap).forEach((chip) => {
      chip.addEventListener("click", () => {
        perIterToggles[chip.dataset.metric] = !perIterToggles[chip.dataset.metric];
        chip.classList.toggle("on", perIterToggles[chip.dataset.metric]);
        renderPerIterPlot(pi, metrics);
        renderPerIterHist(pi, metrics);
      });
    });
    // Rearrangement-only: dropdown to overlay a numeric diag column on the
    // right axis (its own units). Populated from pi.diag_series.
    const diagSel = $("per-iter-diag-select");
    const diagWrap = $("per-iter-diag-wrap");
    const diagSeries = pi.diag_series || {};
    const diagKeys = Object.keys(diagSeries);
    if (diagSel && diagWrap) {
      if (diagKeys.length) {
        diagWrap.style.display = "";
        diagSel.innerHTML = '<option value="">none</option>' +
          diagKeys.map((k) => {
            const d = diagSeries[k];
            const u = d.unit ? ` (${d.unit})` : "";
            return `<option value="${escHtml(k)}">${escHtml(d.label || k)}${u}</option>`;
          }).join("");
        if (!(perIterDiagKey in diagSeries)) perIterDiagKey = "";
        diagSel.value = perIterDiagKey;
        diagSel.onchange = () => {
          perIterDiagKey = diagSel.value;
          renderPerIterPlot(pi, metrics);
        };
      } else {
        diagWrap.style.display = "none";
        perIterDiagKey = "";
      }
    }
    setText("per-iter-info",
            `${pi.shot_index.length} shots · click a point to view its image`);
    renderPerIterPlot(pi, metrics);
    renderPerIterHist(pi, metrics);
  }

  function renderPerIterHist(pi, metrics) {
    if (!window.Plotly) return;
    const el = $("plot-per-iter-hist");
    if (!el) return;
    const traces = [];
    metrics.forEach((m) => {
      if (!perIterToggles[m.key]) return;
      // Histogram fractions only (skip swept-param-value metrics since
      // they're a different scale than 0..1).
      if (!m.isFrac) return;
      const y = pi[m.key] || [];
      const vals = y.filter(
        (v) => v != null && typeof v === "number" && isFinite(v));
      if (!vals.length) return;
      traces.push({
        x: vals,
        type: "histogram",
        // 0.5% bins from 0 to 1.
        xbins: {start: 0, end: 1.005, size: 0.005},
        marker: {color: METRIC_COLORS[m.key] || "#ffdd44", opacity: 0.6},
        name: m.label,
        autobinx: false,
      });
    });
    if (!traces.length) {
      Plotly.purge(el);
      el.innerHTML =
        '<div class="hint" style="padding:14px;text-align:center;">enable a fraction metric to histogram</div>';
      return;
    }
    Plotly.react(el, traces, plotLayoutFlush({
      margin: {l: 70, r: 90, t: 10, b: 40},
      xaxis: {title: {text: "value (0.5% bins)"}, range: [0, 1.005],
              tickformat: ".2f"},
      yaxis: {title: {text: "shots"}, rangemode: "nonnegative"},
      barmode: "overlay",
      showlegend: true,
      legend: {x: 0.99, y: 0.99, xanchor: "right",
               bgcolor: "rgba(0,0,0,0.3)"},
    }), plotConfig());
  }

  function renderPerIterPlot(pi, metrics) {
    if (!window.Plotly) return;
    const el = $("plot-per-iter");
    if (!el) return;
    const x = pi.shot_index || [];
    const traces = [];
    let needsRightAxis = false;
    metrics.forEach((m) => {
      if (!perIterToggles[m.key]) return;
      let y;
      if (m.key.startsWith("param:")) {
        y = (pi.param_values || {})[m.paramName] || [];
        needsRightAxis = true;
      } else {
        y = pi[m.key] || [];
      }
      traces.push({
        x: x, y: y, name: m.label,
        type: "scattergl", mode: "lines+markers",
        line: {width: 1.2, color: METRIC_COLORS[m.key] || "#ffdd44"},
        marker: {size: 4, color: METRIC_COLORS[m.key] || "#ffdd44"},
        yaxis: m.key.startsWith("param:") ? "y2" : "y",
        connectgaps: false,
      });
    });
    // Rearrangement diag overlay (right axis, its own units). Added on top
    // of whatever fraction/param traces are toggled on.
    let rightUnit = null;
    const diagSeries = pi.diag_series || {};
    if (perIterDiagKey && diagSeries[perIterDiagKey]) {
      const d = diagSeries[perIterDiagKey];
      needsRightAxis = true;
      rightUnit = d.unit || "";
      traces.push({
        x: x, y: d.values || [], name: d.label || perIterDiagKey,
        type: "scattergl", mode: "lines+markers",
        line: {width: 1.2, color: "#ffdd44", dash: "dot"},
        marker: {size: 4, color: "#ffdd44"},
        yaxis: "y2", connectgaps: false,
      });
    }
    if (!traces.length) {
      Plotly.purge(el);
      el.innerHTML =
        '<div class="hint" style="padding:24px;text-align:center;">enable a metric to plot</div>';
      return;
    }
    // Right Y axis label: the diag overlay's label+unit takes precedence
    // (it's the explicit pick); else the toggled-on swept-param name.
    let rightLabel = "sweep value";
    metrics.forEach((m) => {
      if (perIterToggles[m.key] && m.key.startsWith("param:")) {
        rightLabel = m.paramName || rightLabel;
      }
    });
    if (perIterDiagKey && diagSeries[perIterDiagKey]) {
      const d = diagSeries[perIterDiagKey];
      rightLabel = (d.label || perIterDiagKey) + (rightUnit ? ` (${rightUnit})` : "");
    }
    // Generous right margin so the right-Y axis label has room AND
    // wide-format frequency ticks (e.g. "1.057×10^8") aren't clipped.
    const layout = plotLayoutFlush({
      margin: {l: 70, r: 90, t: 18, b: 56},
      xaxis: {title: {text: "shot # (iteration / time order)"}},
      yaxis: {title: {text: "fraction"}, tickformat: ".2f",
              range: [0, 1.05]},
      showlegend: true,
      legend: {x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0.3)"},
    });
    if (needsRightAxis) {
      layout.yaxis2 = {
        title: {text: rightLabel, font: {size: 12, color: "#d8dee9"},
                standoff: 10},
        overlaying: "y", side: "right",
        gridcolor: "#2a3242", zerolinecolor: "#2a3242",
        // ".5g" cuts 105100000 to "1.05e8" -- ambiguous. Use ".6g" or
        // SI prefix so 105_100_000 reads as "1.051×10^8" or "105.1M".
        tickformat: "~s",   // SI prefix: 105M, 1.05G, etc.
        hoverformat: ".6g",
        ticks: "outside",
        showgrid: false,
        automargin: true,
      };
    }
    Plotly.react(el, traces, layout, plotConfig());
    // Click a point -> open the shot-image popup. Rebind each render
    // (removeAllListeners avoids stacking duplicate handlers across
    // re-renders / metric toggles). p.x is the shot_index value;
    // p.pointNumber indexes the per-iteration arrays for param lookups.
    if (el.removeAllListeners) el.removeAllListeners("plotly_click");
    if (el.on) {
      el.on("plotly_click", (data) => {
        if (!data || !data.points || !data.points.length) return;
        const p = data.points[0];
        openShotModal(p.pointNumber, p.x);
      });
    }
    el.style.cursor = "pointer";
  }

  // =====================================================================
  // REARRANGEMENT SCATTER (protocol-specific) — scatter ANY two per-shot
  // quantities (loading / TP / FP / numeric diag). Reads the SAME
  // per_iteration arrays as the per-iteration plot, so it's filter-aware
  // for free (they're already sliced by the active sweep filter). Points
  // are colored by shot # so temporal drift is visible. Rearrangement only.
  // =====================================================================
  function scatterVarList(pi) {
    const vars = [];
    if (pi.loaded_frac)   vars.push({key: "loaded_frac",   label: "loading",                     unit: "", frac: true});
    if (pi.survival_frac) vars.push({key: "survival_frac", label: pi.survival_label || "survival", unit: "", frac: true});
    if (pi.fp_frac)       vars.push({key: "fp_frac",       label: "FP rate",                     unit: "", frac: true});
    const ds = pi.diag_series || {};
    Object.keys(ds).forEach((k) => vars.push({
      key: "diag:" + k, label: ds[k].label || k, unit: ds[k].unit || "", frac: false}));
    return vars;
  }
  function scatterArr(pi, key) {
    if (key && key.indexOf("diag:") === 0) {
      const d = (pi.diag_series || {})[key.slice(5)];
      return d ? (d.values || []) : [];
    }
    return pi[key] || [];
  }
  function renderRearrangeScatter(r) {
    const card = $("analysis-rearrange-scatter-card");
    const el = $("plot-rearrange-scatter");
    const xs = $("scatter-x");
    const ys = $("scatter-y");
    if (!card || !el || !xs || !ys) return;
    const pi = r && r.per_iteration;
    const isRearrange = pi && pi.shot_index && pi.shot_index.length
      && (r.paths_n_shots_with_pairing || 0) > 0;
    if (!isRearrange) {
      card.hidden = true;
      if (window.Plotly) Plotly.purge(el);
      return;
    }
    const vars = scatterVarList(pi);
    if (vars.length < 2) { card.hidden = true; return; }
    card.hidden = false;
    const byKey = {};
    vars.forEach((v) => { byKey[v.key] = v; });
    const opts = vars.map((v) =>
      `<option value="${escHtml(v.key)}">${escHtml(v.label)}${v.unit ? ` (${v.unit})` : ""}</option>`).join("");
    let xk = scatterAxes.x, yk = scatterAxes.y;
    if (!byKey[xk]) xk = vars[0].key;
    if (!byKey[yk]) yk = (vars[1] || vars[0]).key;
    xs.innerHTML = opts; ys.innerHTML = opts;
    xs.value = xk; ys.value = yk;

    const grid = {gridcolor: "#2a3242", zerolinecolor: "#2a3242"};
    const draw = () => {
      if (!window.Plotly) return;
      const vx = byKey[xs.value] || vars[0];
      const vy = byKey[ys.value] || vars[0];
      const ax = scatterArr(pi, xs.value);
      const ay = scatterArr(pi, ys.value);
      const si = pi.shot_index || [];
      const X = [], Y = [], C = [];
      const n = Math.min(ax.length, ay.length);
      for (let i = 0; i < n; i++) {
        const a = ax[i], b = ay[i];
        if (a == null || b == null || !isFinite(a) || !isFinite(b)) continue;
        X.push(a); Y.push(b); C.push(si[i] != null ? si[i] : i);
      }
      setText("scatter-info", `${X.length} shots`);
      Plotly.react(el, [{
        x: X, y: Y, mode: "markers", type: "scattergl",
        marker: {size: 6, color: C, colorscale: "Viridis", showscale: true,
                 colorbar: {title: {text: "shot #", side: "right"},
                            thickness: 12, len: 0.9}},
        hovertemplate: `${escHtml(vx.label)}: %{x}<br>${escHtml(vy.label)}: %{y}`
                       + `<br>shot %{marker.color}<extra></extra>`,
      }], plotLayoutFlush({
        margin: {l: 78, r: 20, t: 14, b: 60},
        xaxis: Object.assign({title: {text: vx.label + (vx.unit ? ` (${vx.unit})` : "")}},
                             grid, vx.frac ? {tickformat: ".0%"} : {}),
        yaxis: Object.assign({title: {text: vy.label + (vy.unit ? ` (${vy.unit})` : "")}},
                             grid, vy.frac ? {tickformat: ".0%"} : {}),
        showlegend: false,
      }), plotConfig());
    };
    xs.onchange = () => { scatterAxes.x = xs.value; saveScatterAxes(); draw(); };
    ys.onchange = () => { scatterAxes.y = ys.value; saveScatterAxes(); draw(); };
    draw();
  }

  // ---- Shot-image popup (click a per-iteration point) ----------------
  function openShotModal(pointIdx, shotNum) {
    const ctx = perIterCtx;
    if (!ctx || !ctx.pi) return;
    const pi = ctx.pi;
    let shot = shotNum;
    if (shot == null && Array.isArray(pi.shot_index)) {
      shot = pi.shot_index[pointIdx];
    }
    if (shot == null) return;
    const numImages = ctx.numImages || 1;
    const scanId = ctx.scanId;

    setText("shot-modal-title", `Shot ${shot}`);

    // Subtitle line 1: sweep param values for this shot. Line 2: metrics.
    const pv = pi.param_values || {};
    const paramBits = Object.keys(pv)
      .map((name) => {
        const v = (pv[name] || [])[pointIdx];
        return v != null ? `${name} = ${fmtNum(v)}` : null;
      })
      .filter(Boolean);
    const metricBits = [];
    if (pi.loaded_frac && pi.loaded_frac[pointIdx] != null) {
      metricBits.push(`loaded ${fmtPct(pi.loaded_frac[pointIdx])}`);
    }
    if (pi.survival_frac && pi.survival_frac[pointIdx] != null) {
      metricBits.push(
        `${pi.survival_label || "survival"} ${fmtPct(pi.survival_frac[pointIdx])}`);
    }
    if (pi.fp_frac && pi.fp_frac[pointIdx] != null) {
      metricBits.push(`FP ${fmtPct(pi.fp_frac[pointIdx])}`);
    }
    $("shot-modal-sub").innerHTML =
      [paramBits.join("  ·  "), metricBits.join("  ·  ")]
        .filter(Boolean).map(escHtml).join("<br>");

    // Body: one camera frame per NumImages. Each frame is a Plotly figure
    // (PNG as a layout image + scaleanchor axes) so it zooms/pans exactly
    // like the live-view array images — drag a box to zoom, scroll to zoom,
    // double-click to reset.
    const frameLabel = (j) => {
      if (numImages === 1) return "image";
      if (j === 0) return "img 1 (initial)";
      if (j === numImages - 1) return "img 2 (final)";
      return `img ${j + 1}`;
    };
    const body = $("shot-modal-body");
    body.innerHTML = "";
    body.classList.toggle("multi", numImages > 1);
    for (let j = 0; j < numImages; j++) {
      const src = `/api/runs/${encodeURIComponent(scanId)}/shot_image`
        + `?shot=${shot}&frame=${j}&num_images=${numImages}`;
      const fig = document.createElement("figure");
      fig.className = "shot-frame";
      const plot = document.createElement("div");
      plot.className = "shot-frame-plot";
      const cap = document.createElement("figcaption");
      cap.textContent = frameLabel(j);
      fig.appendChild(plot);
      fig.appendChild(cap);
      body.appendChild(fig);
      renderShotFramePlot(plot, src, `shot ${shot} ${frameLabel(j)}`);
    }
    $("shot-modal").hidden = false;
  }

  // Render one shot frame into `container` as a zoomable Plotly image,
  // mirroring _fig_array's layout-image + scaleanchor approach. Falls back
  // to a plain <img> if Plotly didn't load.
  function renderShotFramePlot(container, src, altText) {
    if (!window.Plotly) {
      container.innerHTML =
        `<img src="${src}" alt="${escHtml(altText)}" class="shot-frame-img" ` +
        `onerror="this.replaceWith(document.createTextNode('frame unavailable'));">`;
      return;
    }
    // Preload to learn the pixel dims, then size the axes to the image so
    // aspect ratio is locked (scaleanchor) and zoom maps to real pixels.
    const probe = new Image();
    probe.onload = () => {
      const W = probe.naturalWidth || 1;
      const H = probe.naturalHeight || 1;
      const layout = {
        paper_bgcolor: "#0d1220", plot_bgcolor: "#000",
        margin: {l: 0, r: 0, t: 0, b: 0},
        xaxis: {visible: false, range: [0, W], showgrid: false,
                zeroline: false, constrain: "domain"},
        yaxis: {visible: false, range: [H, 0], scaleanchor: "x",
                scaleratio: 1, showgrid: false, zeroline: false,
                constrain: "domain"},
        images: [{source: src, xref: "x", yref: "y", x: 0, y: 0,
                  sizex: W, sizey: H, sizing: "stretch", layer: "below"}],
        uirevision: "shot",
      };
      // scrollZoom on top of the live-view defaults (box-zoom + dblclick
      // reset). No modebar, matching the live array panels.
      Plotly.newPlot(container, [], layout,
                     {displayModeBar: false, responsive: true,
                      scrollZoom: true});
    };
    probe.onerror = () => {
      container.innerHTML =
        '<div class="shot-frame-err">frame unavailable</div>';
    };
    probe.src = src;
  }

  function closeShotModal() {
    const m = $("shot-modal");
    if (m) m.hidden = true;
    // Purge Plotly instances + clear the body so image loads stop and
    // memory frees.
    const body = $("shot-modal-body");
    if (body) {
      if (window.Plotly) {
        body.querySelectorAll(".shot-frame-plot").forEach((el) => {
          try { Plotly.purge(el); } catch (e) {}
        });
      }
      body.innerHTML = "";
    }
  }

  (function wireShotModal() {
    const modal = $("shot-modal");
    if (!modal) return;
    modal.querySelectorAll("[data-shot-close]").forEach((el) => {
      el.addEventListener("click", closeShotModal);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !modal.hidden) closeShotModal();
    });
  })();

  // =====================================================================
  // SEQUENCE-SPECIFIC ANALYSIS (placeholder).
  // Right now just shows the scan name + a "Coming soon" message.
  // Per-Seq custom analyses will be plugged in here.
  // =====================================================================
  function renderSeqSpecific(r) {
    setText("seq-specific-name", r.scan_name || "(unknown)");
    const body = $("seq-specific-body");
    if (!body) return;

    // (The averaged-image card lives in its own section after this one —
    // see renderAvgImage / #analysis-avg-image-card.)
    let html = '';

    // Focus / discrimination metrics vs the swept param (LoadingDefocus):
    // several spot-count-robust per-site metrics so the operator can pick
    // the best SLM defocus by whichever metric they trust.
    const ss = r.seq_specific;
    const hasFocus = ss && ss.type === "focus_metrics" && ss.metrics;
    if (hasFocus) {
      html += `
        <div style="padding:8px 0;">
          <h3 style="margin:0 0 4px;font-size:13px;color:var(--text-dim);
                     text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">
            Focus / discrimination vs ${escHtml(ss.x_label || "param")}
            <span class="muted mono" style="text-transform:none;margin-left:8px;
                  letter-spacing:0;font-weight:400;">
              per-site, spot-count-robust · each curve normalised 0–1 · ★ = optimum
            </span>
          </h3>
          <div class="plot-container" id="plot-focus-metrics"
               style="height:360px;"></div>
          <div class="hint mono" id="focus-metrics-info" style="margin-top:6px;"></div>
        </div>`;
    }

    // Always include the placeholder text so future per-Seq panels
    // have a home. When avg_image renders above, this is just a tiny
    // muted footer.
    html += `
      <div style="padding:8px 0;font-family:var(--mono);font-size:11px;
                  color:var(--text-dim);">
        scan_name = ${escHtml(r.scan_name || "(none)")}
        ${hasFocus ? '' : '· (no Seq-specific panel for this scan type yet)'}
      </div>`;
    body.innerHTML = html;

    if (hasFocus) renderFocusMetrics(ss);
  }

  // Averaged camera image — shown for ANY scan with an /imgs dataset (its
  // own half-row section after Sequence-specific). Mean of the FIRST image of
  // each shot; the PNG is computed lazily + cached by /api/runs/<id>/avg_image
  // on first request. Rendered as a Plotly image so it ZOOMS/pans like the
  // other image panels (box-zoom drag; double-click resets).
  function renderAvgImage(r) {
    const card = $("analysis-avg-image-card");
    const el = $("plot-avg-image");
    if (!card || !el) return;
    const ai = r && r.avg_image;
    if (!ai || !(ai.available || ai.computable)) {
      card.hidden = true;
      if (window.Plotly) Plotly.purge(el);
      setText("avg-image-info", "");
      return;
    }
    card.hidden = false;
    const sh = ai.image_shape || [];
    const H = sh.length === 2 ? sh[0] : 0;
    const W = sh.length === 2 ? sh[1] : 0;
    const navg = ai.n_avg || ai.n_shots || 0;
    const sampNote = ai.sampled ? ` (sampled of ${ai.n_shots})` : "";
    setText("avg-image-info", ai.available
      ? `cached · mean of ${navg} first-images${sampNote} · ${H}×${W}`
      : `computes on first load (~${Math.max(1, Math.round(0.025 * navg))} s, one-time) · `
        + `mean of ${navg} first-images${sampNote} · ${H}×${W}`);
    if (!window.Plotly || !W || !H) return;
    const src = `/api/runs/${encodeURIComponent(r.scan_id || "")}`
              + `/avg_image?v=${encodeURIComponent(r.scan_id || "")}`;
    // Transparent corner trace establishes the axes (so box-zoom works);
    // the PNG is a below-layer layout image filling the pixel extent.
    Plotly.react(el, [{
      x: [0, W], y: [0, H], mode: "markers",
      marker: {opacity: 0}, hoverinfo: "skip", showlegend: false,
    }], plotLayoutFlush({
      margin: {l: 0, r: 0, t: 0, b: 0},
      xaxis: {visible: false, range: [0, W], constrain: "domain"},
      yaxis: {visible: false, range: [H, 0], scaleanchor: "x", scaleratio: 1},
      images: [{
        source: src, xref: "x", yref: "y",
        x: 0, y: 0, sizex: W, sizey: H,
        xanchor: "left", yanchor: "top", sizing: "stretch", layer: "below",
      }],
    }), plotConfig());
  }

  // Overlay each focus metric. The metrics have DIFFERENT UNITS (px,
  // counts, ratio) so they can't share a real y-axis: each is min–max
  // normalised AND oriented so "up = better" for every curve (lower-is-
  // better metrics like spot width are flipped). Real value + unit are
  // always on hover and in the info line; the y-axis label says so
  // explicitly so the normalisation is never mistaken for real units.
  // Each curve's optimum defocus is starred.
  function renderFocusMetrics(ss) {
    if (!window.Plotly) return;
    const el = $("plot-focus-metrics");
    if (!el) return;
    const x = ss.x || [];
    const COLORS = {
      spot_width: "#58a6ff", spot_peak: "#3fb950",
      spot_contrast: "#d29922", n_spots: "#8b949e",
    };
    const traces = [];
    const stars = {x: [], y: [], text: [], type: "scatter", mode: "markers",
      marker: {symbol: "star", size: 15, color: "#f0f6fc",
               line: {color: "#000", width: 1}},
      name: "optimum", hoverinfo: "text", showlegend: false};
    const infoBits = [];
    Object.entries(ss.metrics).forEach(([key, m], ci) => {
      const vals = (m.values || []).map((v) => (v == null ? NaN : v));
      const finite = vals.filter((v) => isFinite(v));
      if (!finite.length) return;
      const lo = Math.min(...finite), hi = Math.max(...finite);
      const span = hi - lo;
      const higherBetter = m.higher_better !== false;
      const unit = m.unit ? ` ${m.unit}` : "";
      // Normalise to [0,1]; flip lower-is-better so up = better everywhere.
      const norm = vals.map((v) => {
        if (!isFinite(v)) return NaN;
        const n = span > 0 ? (v - lo) / span : 0.5;
        return higherBetter ? n : 1 - n;
      });
      const color = COLORS[key] || `hsl(${(ci * 70) % 360},60%,60%)`;
      const label = `${m.label || key} (${higherBetter ? "↑" : "↓"} better, ${m.unit || "a.u."})`;
      traces.push({
        x: x, y: norm, type: "scatter", mode: "lines+markers",
        name: label, line: {color}, marker: {color, size: 6},
        text: vals.map((v) => isFinite(v) ? `${v.toPrecision(4)}${unit}` : "—"),
        hovertemplate: `${m.label || key}: %{text}<br>${escHtml(ss.x_label || "x")}=%{x}<extra></extra>`,
      });
      // Optimum = best raw value (min for lower-better, max otherwise).
      let bi = -1, bv = higherBetter ? -Infinity : Infinity;
      vals.forEach((v, i) => {
        if (!isFinite(v)) return;
        if (higherBetter ? v > bv : v < bv) { bv = v; bi = i; }
      });
      if (bi >= 0) {
        stars.x.push(x[bi]); stars.y.push(norm[bi]);
        stars.text.push(`${m.label || key} best @ ${escHtml(ss.x_label || "x")}=${x[bi]} (${bv.toPrecision(4)}${unit})`);
        infoBits.push(`${m.label || key}: best @ ${x[bi]} (${bv.toPrecision(4)}${unit})`);
      }
    });
    if (!traces.length) {
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no spots detected / no metric data</div>';
      setText("focus-metrics-info", "");
      return;
    }
    traces.push(stars);
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: "#c9d1d9", size: 11},
      margin: {l: 56, r: 12, t: 8, b: 42},
      xaxis: {title: ss.x_label || "param", gridcolor: "#1a1a30"},
      yaxis: {title: "normalised, ↑=better (units differ — hover for real values)",
              range: [-0.05, 1.08], gridcolor: "#1a1a30"},
      legend: {orientation: "h", y: 1.14, font: {size: 10}},
    };
    Plotly.react(el, traces, layout, {displayModeBar: false, responsive: true});
    setText("focus-metrics-info",
            "⚠ curves min–max normalised, different units (px / counts / ratio); "
            + "raw values on hover.  " + infoBits.join("  ·  "));
  }

  function avg(arr) {
    if (!arr || !arr.length) return null;
    const vs = arr.filter((v) => v != null);
    if (!vs.length) return null;
    return vs.reduce((a, b) => a + b, 0) / vs.length;
  }

  function plotAnalysisCurve(divId, ys, es, sweep, title) {
    if (!window.Plotly) return;
    const el = $(divId);
    if (!el || !ys || !ys.length) {
      if (el) Plotly.purge(el);
      return;
    }
    const values = sweep.values || [];
    const xs = (values.length === 1 && values[0].length === ys.length)
      ? values[0]
      : ys.map((_, i) => i + 1);
    const xlabel = (values.length === 1 && sweep.cols && sweep.cols[0])
      ? sweep.cols[0] : "scan point";
    const trace = {
      x: xs, y: ys, mode: "markers+lines",
      marker: { color: "#58a6ff", size: 8 },
      line:   { color: "#1f6feb", width: 2 },
      error_y: es ? {type: "data", array: es, color: "#1f6feb", visible: true} : undefined,
    };
    Plotly.react(el, [trace], plotLayout({
      title: { text: title, font: { size: 12, color: "#d8dee9" }},
      xaxis: { title: xlabel },
    }), plotConfig());
  }

  // ---- Selection tray (multi-run picker for Analysis tab) ----
  // The tray is the source of truth for "what runs to analyze". Each
  // chip is one scan_id. "Analyze tray" runs the group analysis API;
  // "Save" persists the current tray to the legacy run-groups store
  // so it can be reloaded later. Selecting from "Load saved group..."
  // overwrites the tray with that group's members.
  const traySet = new Set();   // scan_id strings, insertion-ordered
  const TRAY_LS_KEY = "yb_dashboard_tray";   // persists the full multi-run tray

  // Persist the whole tray (not just the single selected scan) so a page
  // reload restores every chip, not one of them. Called from renderTray —
  // the single choke point every mutation funnels through.
  function persistTray() {
    try {
      localStorage.setItem(TRAY_LS_KEY, JSON.stringify(Array.from(traySet)));
    } catch (e) { /* storage full / disabled — non-fatal */ }
  }
  function loadSavedTray() {
    try {
      const raw = localStorage.getItem(TRAY_LS_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr.map(String) : [];
    } catch (e) { return []; }
  }
  // Snapshot the persisted tray at script-init, BEFORE the DOMContentLoaded
  // renderTray() (empty set) can overwrite it with []. Used once by the
  // first loadRunsList to restore the multi-run selection.
  const savedTrayAtLoad = loadSavedTray();

  function syncTrayHighlight() {
    // Highlight rows in the SLM-style .run-picker that are in the tray.
    $$("#runs-table .run-row").forEach((row) => {
      row.classList.toggle("in-tray", traySet.has(row.dataset.scanId));
      const addBtn = row.querySelector(".run-add");
      if (addBtn) addBtn.textContent = traySet.has(row.dataset.scanId) ? "✓" : "+";
    });
  }

  function renderTray() {
    const wrap = $("tray-chips");
    if (!wrap) return;
    persistTray();
    setText("tray-count", String(traySet.size));
    // Push the count onto the runs card so the edge-tab indicator
    // reflects the current selection at a glance.
    const runsCard = document.getElementById("analysis-runs-card");
    if (runsCard) runsCard.dataset.floatCount = String(traySet.size);
    if (!traySet.size) {
      wrap.innerHTML =
        '<span class="tray-empty">none yet — click a row below</span>';
      syncTrayHighlight();
      return;
    }
    wrap.innerHTML = Array.from(traySet).map((sid) => `
      <span class="tray-chip" data-scan-id="${sid}">
        <span class="chip-label" title="${sid}">${sid}</span>
        <button class="chip-remove" title="Remove" data-remove="${sid}">×</button>
      </span>`).join("");
    $$(".chip-remove", wrap).forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        traySet.delete(btn.dataset.remove);
        renderTray();
      });
    });
    syncTrayHighlight();
  }

  function trayToggle(sid) {
    if (traySet.has(sid)) traySet.delete(sid);
    else traySet.add(sid);
    renderTray();
  }
  function trayReplace(sid) {
    traySet.clear();
    traySet.add(sid);
    renderTray();
  }

  $("tray-clear").addEventListener("click", () => {
    traySet.clear();
    renderTray();
  });

  // Multi-run tray analysis: use an ephemeral group through the existing
  // /api/runs/groups/<id>/analysis aggregator. Create -> add -> analyze ->
  // delete. Extracted so restore-on-reload can re-run it (not just the
  // Analyze button).
  async function analyzeTrayGroup() {
    if (traySet.size < 2) return;
    const name = '__tray_' + Date.now();
    const body = $("analysis-detail-body");
    body.innerHTML = '<div class="hint">running tray analysis (' +
      traySet.size + ' runs)…</div>';
    let gid = null;
    try {
      const r = await api("/api/runs/groups", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      gid = r.group_id;
      for (const sid of traySet) {
        await api(`/api/runs/groups/${gid}/add/${sid}`, {method: "POST"});
      }
      const r2 = await api(`/api/runs/groups/${gid}/analysis`);
      // Render the FULL panel suite (not just the detail header) so the
      // combined view shows per-site maps, survival-vs-distance, and the
      // pooled seq-specific focus curve -- same as a single-run load.
      activeAnalysis = r2;
      renderAnalysisDetail(r2);
      renderPerSiteMaps(r2);
      renderSurvivalVsDistance(r2);
      renderPerIteration(r2);
      renderSeqSpecific(r2);
      renderAvgImage(r2);   // hides the card for group views (no avg_image)
      renderRearrangeScatter(r2);
      setText("selected-scan-id", `tray:${traySet.size} runs`);
    } catch (e) {
      body.innerHTML = `<div class="hint bad">tray analysis failed: ${escHtml(e.message)}</div>`;
    } finally {
      if (gid) {
        try { await api(`/api/runs/groups/${gid}`, {method: "DELETE"}); } catch {}
        loadGroups();
      }
    }
  }

  $("tray-analyze").addEventListener("click", async () => {
    // Paste-box takes precedence: if a 14-digit scan_id is sitting
    // there AND doesn't match the current selection, treat Analyze as
    // "load that scan into the tray then analyze it".
    const pasted = ($("manual-scan-id").value || "").trim();
    if (/^\d{14}$/.test(pasted) && pasted !== selectedScanId) {
      trayReplace(pasted);
      loadAnalysis(pasted);
      return;
    }
    if (!traySet.size) { toast("tray is empty", "warn"); return; }
    if (traySet.size === 1) {
      loadAnalysis(Array.from(traySet)[0]);
      return;
    }
    await analyzeTrayGroup();
  });

  $("tray-save").addEventListener("click", async () => {
    if (!traySet.size) { toast("tray is empty", "warn"); return; }
    const name = ($("tray-save-name").value || "").trim();
    if (!name) { toast("group name required", "warn"); return; }
    try {
      const r = await api("/api/runs/groups", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ name }),
      });
      for (const sid of traySet) {
        await api(`/api/runs/groups/${r.group_id}/add/${sid}`, {method: "POST"});
      }
      $("tray-save-name").value = "";
      toast(`saved group "${name}" (${traySet.size} runs)`);
      await loadGroups();
    } catch (e) {
      toast("save failed: " + e.message, "bad");
    }
  });

  $("tray-load-group").addEventListener("change", (e) => {
    const gid = e.target.value;
    if (!gid) return;
    const g = groupsCache[gid];
    if (!g) return;
    traySet.clear();
    (g.members || []).forEach((m) => traySet.add(m.scan_id));
    renderTray();
    toast(`loaded "${g.name}" (${traySet.size} runs)`);
  });

  $("tray-delete-group").addEventListener("click", async () => {
    const gid = $("tray-load-group").value;
    if (!gid) { toast("pick a group to delete", "warn"); return; }
    if (!confirm("Delete this saved group?")) return;
    try {
      await api(`/api/runs/groups/${gid}`, {method: "DELETE"});
      await loadGroups();
      toast("deleted");
    } catch (e) { toast("delete failed: " + e.message, "bad"); }
  });

  // ---- Run groups (legacy persistence layer for the tray) ----
  async function loadGroups() {
    try {
      const r = await api("/api/runs/groups");
      groupsCache = r.groups || {};
      // Hidden groups starting with __tray_ are ephemeral; skip them.
      const cleaned = {};
      for (const [gid, g] of Object.entries(groupsCache)) {
        if (!(g.name || "").startsWith("__tray_")) cleaned[gid] = g;
      }
      groupsCache = cleaned;
      // Populate the "Load saved group..." dropdown.
      const sel = $("tray-load-group");
      if (sel) {
        const cur = sel.value;
        sel.innerHTML = '<option value="">Load group…</option>' +
          Object.entries(groupsCache).map(([gid, g]) =>
            `<option value="${gid}">${escHtml(g.name || gid)} (${(g.members||[]).length})</option>`
          ).join("");
        sel.value = cur;
      }
    } catch {
      // groups endpoint may not be implemented yet
    }
  }
  // (Legacy group chip + group detail UI removed; the tray section
  // above replaces both. Saved groups now feed the "Load saved
  // group..." picker in the tray.)

  // Protocol source viewer (lazy)
  $("show-protocol-src").addEventListener("click", async () => {
    if (!selectedScanId) { toast("select a run first", "warn"); return; }
    const pre = $("protocol-src-body");
    pre.style.display = "";
    pre.textContent = "fetching…";
    try {
      const r = await fetch(`/api/runs/${selectedScanId}/code`);
      const data = await r.json();
      pre.textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      pre.textContent = "fetch failed: " + e.message;
    }
  });

  // =====================================================================
  // QUEUE TAB
  // =====================================================================
  let seqCatalog = [];           // [{name, file, n_steps, n_params}, ...]
  let seqDetailCache = {};       // name -> full {name, file, steps, params, runp}
  let selectedSeqName = null;

  async function pollQueue() {
    try {
      const q = await api("/api/queue");
      renderQueueTable(q);
      renderQueueStatus(q);
    } catch (e) {
      $("queue-table").querySelector("tbody").innerHTML =
        `<tr><td colspan="6" class="muted">runner unreachable: ${escHtml(e.message)}</td></tr>`;
    }
  }

  function renderQueueStatus(q) {
    const running = q.running;
    const queued = q.queued || [];
    const history = q.history || [];
    if (running) {
      $("queue-active-name").innerHTML = `${escHtml(running.label || running.seqName || "—")} <span class="badge mono">#${running.id}</span>`;
      const dur = running.start_ts ? (Date.now() / 1000 - running.start_ts) : 0;
      $("queue-active-progress").textContent =
        `running ${dur.toFixed(0)}s · file_id=${running.file_id || "—"}`;
    } else {
      $("queue-active-name").innerHTML = '<span class="muted">(none)</span>';
      $("queue-active-progress").textContent = "—";
    }
    setText("queue-depth",      String(queued.filter(r => (r.kind||"job") === "job").length));
    setText("queue-desc-depth", String(queued.filter(r => r.kind === "descriptor").length));
    setText("queue-history-count", String(history.length));
    // Active detail panel
    if (running) {
      $("queue-active-detail-body").innerHTML = `
        <dl class="kv">
          <dt>id</dt><dd class="mono">${running.id}</dd>
          <dt>kind</dt><dd>${running.kind || "job"}</dd>
          <dt>seq</dt><dd class="mono">${escHtml(running.seqName || "—")}</dd>
          <dt>label</dt><dd>${escHtml(running.label || "—")}</dd>
          <dt>started</dt><dd>${fmtTs(running.start_ts)}</dd>
          <dt>file_id</dt><dd class="mono">${running.file_id || "—"}</dd>
        </dl>`;
    } else {
      $("queue-active-detail-body").innerHTML = '<span class="muted">(no active job)</span>';
    }
  }

  function renderQueueTable(q) {
    const queued = q.queued || [];
    const running = q.running ? [q.running] : [];
    const history = q.history || [];
    setText("queue-counts",
      `${queued.length} queued · ${running.length} running · ${history.length} history`);
    const rows = [...running, ...queued].map((r) => `
      <tr>
        <td class="mono">${r.id}</td>
        <td>${r.kind || "job"}</td>
        <td>${r.state}</td>
        <td>${escHtml(r.label || r.seqName || "")}</td>
        <td class="mono">${r.file_id || ""}</td>
        <td>
          ${r.state === "queued" ? `
            <button class="ghost" data-cancel="${r.id}" style="font-size:10px;padding:2px 8px;">cancel</button>
            <button class="ghost" data-move-up="${r.id}" style="font-size:10px;padding:2px 8px;">↑</button>
            <button class="ghost" data-move-down="${r.id}" style="font-size:10px;padding:2px 8px;">↓</button>
          ` : ""}
          ${r.descriptor ? `<button class="ghost" data-requeue="${r.id}" title="Queue a new copy with exactly the same parameters" style="font-size:10px;padding:2px 8px;">re-queue</button>` : ""}
        </td>
      </tr>
    `).join("");
    $("queue-table").querySelector("tbody").innerHTML =
      rows || '<tr><td colspan="6" class="muted">queue empty</td></tr>';
    $$("[data-cancel]", $("queue-table")).forEach((btn) => {
      btn.addEventListener("click", async () => {
        try { await api(`/api/queue/cancel/${btn.dataset.cancel}`, {method: "POST"});
              toast("Cancelled");
        } catch (e) { toast("Cancel failed", "bad"); }
      });
    });
    $$("[data-move-up],[data-move-down]", $("queue-table")).forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.dataset.moveUp || btn.dataset.moveDown;
        const dir = btn.dataset.moveUp ? "up" : "down";
        try { await api(`/api/queue/move/${id}/${dir}`, {method: "POST"}); }
        catch (e) { toast("Move failed", "bad"); }
      });
    });
    // History
    $("history-table").querySelector("tbody").innerHTML =
      history.slice(0, 30).map((r) => `
        <tr>
          <td class="mono">${r.id}</td>
          <td>${r.kind || "job"}</td>
          <td>${escHtml(r.label || r.seqName || "")}</td>
          <td class="${r.status === "ok" ? "ok" : "bad"}">${escHtml(r.status || r.state || "")}</td>
          <td class="mono">${r.file_id || ""}</td>
          <td class="mono">${r.built_job_id != null ? r.built_job_id : ""}</td>
          <td class="muted">${escHtml((r.error_message || "").slice(0, 80))}</td>
          <td>${r.descriptor ? `<button class="ghost" data-requeue="${r.id}" title="Queue a new copy with exactly the same parameters (uses today's code)" style="font-size:10px;padding:2px 8px;">re-queue</button>${r.file_id ? `<button class="ghost" data-requeue-code="${r.id}" title="Re-queue with the EXACT code that ran originally — replays this run's code snapshot (YbSeqs/YbSteps/YbScans)" style="font-size:10px;padding:2px 6px;">+code</button>` : ""}` : ""}</td>
        </tr>`).join("")
      || '<tr><td colspan="8" class="muted">empty</td></tr>';
    // Re-queue buttons live in both the queue and history tables: replay the original
    // descriptor (same params). "+code" also pins the source run's code snapshot so the
    // runner replays the EXACT experiment source that ran originally (reproducibility).
    async function doRequeue(btn, id, withCode) {
      btn.disabled = true;
      try {
        const r = await api(`/api/queue/requeue/${id}${withCode ? "?code=1" : ""}`,
                            {method: "POST"});
        toast(`Re-queued #${id} → #${r.descriptor_id}${withCode ? " (orig code)" : ""}`);
        pollQueue();
      } catch (e) {
        toast("Re-queue failed: " + (e.message || e), "bad");
        btn.disabled = false;
      }
    }
    $$("[data-requeue]").forEach((btn) => {
      btn.addEventListener("click", () => doRequeue(btn, btn.dataset.requeue, false));
    });
    $$("[data-requeue-code]").forEach((btn) => {
      btn.addEventListener("click", () => doRequeue(btn, btn.dataset.requeueCode, true));
    });
  }

  $("queue-refresh-btn").addEventListener("click", pollQueue);

  $("submit-scan-btn").addEventListener("click", async () => {
    const txt = $("submit-scan-json").value;
    if (!txt.trim()) { toast("descriptor empty", "warn"); return; }
    let payload;
    try { payload = JSON.parse(txt); }
    catch (e) {
      $("submit-scan-result").textContent = "invalid JSON: " + e.message;
      $("submit-scan-result").style.color = "var(--err)";
      return;
    }
    // Auto-fill label from the seq picker if user left it.
    if (!payload.label && $("submit-label").value.trim()) {
      payload.label = $("submit-label").value.trim();
    }
    try {
      const r = await api("/api/queue/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      $("submit-scan-result").textContent = `submitted descriptor_id=${r.descriptor_id}`;
      $("submit-scan-result").style.color = "var(--ok)";
      toast("Submitted #" + r.descriptor_id);
      pollQueue();
    } catch (e) {
      $("submit-scan-result").textContent = "submit failed: " + e.message;
      $("submit-scan-result").style.color = "var(--err)";
    }
  });
  $("format-json-btn").addEventListener("click", () => {
    const txt = $("submit-scan-json").value;
    try { $("submit-scan-json").value = JSON.stringify(JSON.parse(txt), null, 2); }
    catch (e) { toast("invalid JSON", "warn"); }
  });

  // ---- Seq catalog ----
  async function loadSeqCatalog() {
    try {
      const r = await api("/api/seqs/list");
      seqCatalog = r.seqs || [];
      renderSeqCatalog();
      populateSeqSelect();
    } catch (e) {
      $("seq-catalog-table").querySelector("tbody").innerHTML =
        `<tr><td colspan="4" class="muted">catalog unavailable: ${escHtml(e.message)}</td></tr>`;
    }
  }
  function renderSeqCatalog() {
    const filter = ($("seq-filter").value || "").toLowerCase();
    const tbody = $("seq-catalog-table").querySelector("tbody");
    const filtered = seqCatalog.filter((s) =>
      !filter || s.name.toLowerCase().includes(filter));
    if (!filtered.length) {
      tbody.innerHTML = '<tr><td colspan="4" class="muted">no match</td></tr>';
      return;
    }
    tbody.innerHTML = filtered.map((s) => `
      <tr class="selectable ${s.name === selectedSeqName ? "selected" : ""}"
          data-seq="${s.name}">
        <td class="mono">${escHtml(s.name)}</td>
        <td class="right">${s.n_steps}</td>
        <td class="right">${s.n_params}</td>
        <td><button class="ghost" data-use="${s.name}"
              style="font-size:10px;padding:2px 8px;">Use</button></td>
      </tr>
    `).join("");
    $$("tr.selectable", tbody).forEach((tr) => {
      tr.addEventListener("click", (e) => {
        if (e.target.tagName === "BUTTON") return;
        loadSeqDetail(tr.dataset.seq);
      });
    });
    $$("[data-use]", tbody).forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        useSeqAsTemplate(btn.dataset.use);
      });
    });
  }
  function populateSeqSelect() {
    const sel = $("submit-seq-select");
    const cur = sel.value;
    sel.innerHTML = '<option value="">— pick a sequence —</option>' +
      seqCatalog.map((s) => `<option value="${s.name}">${s.name}</option>`).join("");
    sel.value = cur;
  }
  async function loadSeqDetail(name) {
    selectedSeqName = name;
    renderSeqCatalog();
    const wrap = $("seq-detail");
    wrap.style.display = "";
    if (!seqDetailCache[name]) {
      try { seqDetailCache[name] = await api(`/api/seqs/${name}`); }
      catch (e) {
        $("seq-detail-name").textContent = name;
        $("seq-detail-file").textContent = "fetch failed: " + e.message;
        return;
      }
    }
    const d = seqDetailCache[name];
    $("seq-detail-name").textContent = name;
    $("seq-detail-file").textContent = d.file || "";
    $("seq-detail-params").innerHTML = (d.params || []).map((p) => `
      <tr>
        <td class="mono">${escHtml(p.path)}</td>
        <td class="mono muted">${escHtml(p.step)}</td>
        <td class="mono muted truncate" title="${escHtml(p.default)}">${escHtml(p.default)}</td>
      </tr>`).join("") ||
      '<tr><td colspan="3" class="muted">no params discovered</td></tr>';
    $("seq-detail-runp").innerHTML = (d.runp || []).map((r) => `
      <tr>
        <td class="mono">${escHtml(r.field)}</td>
        <td class="mono">${escHtml(String(r.default))}</td>
        <td class="muted">${escHtml(r.comment || "")}</td>
      </tr>`).join("");
  }
  function useSeqAsTemplate(name) {
    selectedSeqName = name;
    $("submit-seq-select").value = name;
    fillDescriptorTemplate();
  }
  async function fillDescriptorTemplate() {
    const name = $("submit-seq-select").value || selectedSeqName;
    if (!name) { toast("pick a sequence first", "warn"); return; }
    if (!seqDetailCache[name]) {
      try { seqDetailCache[name] = await api(`/api/seqs/${name}`); }
      catch (e) { toast("fetch failed", "bad"); return; }
    }
    const d = seqDetailCache[name];
    // Build a starter descriptor: include all params with `null` values so
    // the operator sees the full namespace. They can delete the ones they
    // want defaulted via Consts() and edit the rest.
    const params = {};
    (d.params || []).forEach((p) => {
      params[p.path] = null;
    });
    const desc = {
      schema_version: 1,
      seq: name,
      params,
      runp: { NumPerGroup: 4000, NumImages: 2, Scramble: true },
    };
    $("submit-scan-json").value = JSON.stringify(desc, null, 2);
    if (!$("submit-label").value) $("submit-label").value = name;
    toast(`template filled (${(d.params || []).length} params)`);
  }
  $("fill-template-btn").addEventListener("click", fillDescriptorTemplate);
  $("seq-use-btn").addEventListener("click", () =>
    selectedSeqName && useSeqAsTemplate(selectedSeqName));
  $("seq-filter").addEventListener("input", renderSeqCatalog);
  $("seq-refresh-btn").addEventListener("click", async () => {
    try {
      await api("/api/seqs/refresh", { method: "POST" });
      seqDetailCache = {};
      await loadSeqCatalog();
      toast("catalog refreshed");
    } catch (e) { toast("refresh failed", "bad"); }
  });

  // ---- Manual scan_id loader ----
  // Paste-box <-> picker row sync. Typing in the paste-box scrolls the
  // matching picker row into view + highlights it (without loading it,
  // since the user might still be typing). Pressing Enter or clicking
  // Load triggers the actual analysis. Picker row click writes into
  // the paste-box (handled inside renderRunsTable -> loadAnalysis).
  $("manual-scan-load").addEventListener("click", () => {
    const id = ($("manual-scan-id").value || "").trim();
    if (!id) { toast("scan_id required", "warn"); return; }
    trayReplace(id);
    loadAnalysis(id);
  });
  $("manual-scan-id").addEventListener("input", () => {
    const id = ($("manual-scan-id").value || "").trim();
    const rows = $$("#runs-table .run-row");
    let matchedRow = null;
    rows.forEach((row) => {
      const isMatch = id && row.dataset.scanId === id;
      row.classList.toggle("paste-highlight", isMatch);
      if (isMatch) matchedRow = row;
    });
    if (matchedRow) {
      matchedRow.scrollIntoView({block: "nearest", behavior: "smooth"});
    }
  });
  $("manual-scan-id").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("manual-scan-load").click();
    }
  });

  // =====================================================================
  // DIAG TAB
  // =====================================================================
  async function pollDiag() {
    // Just refreshes the live rearrange-diag rollup; the per-scan
    // ledger lookup is on-demand via the Load button.
    try {
      const rd = await api("/api/slm/rearrange/diag");
      renderRearrangeDiag(rd);
      setText("diag-rearrange-count",
        ((rd && rd.entries) || []).length + " entries");
    } catch (e) {
      $("rearrange-diag-body").innerHTML =
        '<div class="hint">SLM offline (' + escHtml(e.message || "") + ')</div>';
    }
  }

  $("ledger-load-btn").addEventListener("click", loadLedger);
  $("ledger-scan-id").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); loadLedger(); }
  });

  async function loadLedger() {
    const id = ($("ledger-scan-id").value || "").trim();
    if (!id) { toast("scan_id required", "warn"); return; }
    setText("ledger-status", "loading…");
    try {
      const r = await api(`/api/runs/${id}/diag`);
      const entries = (r && r.entries) || [];
      setText("ledger-rows-count", entries.length + " rows");
      // Summary
      let totalMs = [], nLoaded = [], aborted = 0;
      entries.forEach((e) => {
        const d = e.diag || {};
        if (typeof d.total_ms === "number") totalMs.push(d.total_ms);
        if (typeof d.n_loaded === "number") nLoaded.push(d.n_loaded);
        if (d.aborted) aborted += 1;
      });
      const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
      renderKv("ledger-summary", [
        ["scan_id",       id],
        ["total rows",    entries.length],
        ["mean total_ms", mean(totalMs) != null ? fmtNum(mean(totalMs)) : "—"],
        ["mean n_loaded", mean(nLoaded) != null ? fmtNum(mean(nLoaded), 1) : "—"],
        ["aborted",       aborted,
          aborted > 0 ? "warn" : "ok"],
        ["source",        r.source || "—"],
      ]);
      // Rows table
      const tbody = $("ledger-table").querySelector("tbody");
      if (!entries.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="muted">no rows</td></tr>';
      } else {
        tbody.innerHTML = entries.slice(0, 500).map((e) => {
          const d = e.diag || {};
          return `<tr>
            <td class="mono">${escHtml(e.ts_iso || "")}</td>
            <td class="right">${e.seq_id != null ? e.seq_id : ""}</td>
            <td class="right">${e.retry_count != null ? e.retry_count : ""}</td>
            <td class="right">${d.total_ms != null ? d.total_ms.toFixed(1) : ""}</td>
            <td class="right">${d.n_loaded != null ? d.n_loaded : ""}</td>
            <td>${d.aborted ? '<span class="bad">yes</span>' : ""}</td>
            <td class="mono muted">${escHtml(d.two_round_phase || "")}</td>
          </tr>`;
        }).join("");
      }
      setText("ledger-status", `loaded ${entries.length} rows`);
    } catch (e) {
      setText("ledger-status", "failed: " + (e.message || e));
      $("ledger-table").querySelector("tbody").innerHTML =
        `<tr><td colspan="7" class="muted">${escHtml(e.message || "fetch failed")}</td></tr>`;
    }
  }

  // =====================================================================
  // BOOTSTRAP
  // =====================================================================
  // ---- FLOATING_ANALYSIS_CARDS implementation ----
  // Reparent the Runs picker + Filter card into the fixed-position host.
  // Hide the host whenever the user navigates away from the Analysis tab.
  // Easily reversible: set FLOATING_ANALYSIS_CARDS=false at the top.
  // 3-state state machine for the floating analysis cards.
  // Single source of truth = each card's data-float-state attribute.
  // Transitions:
  //   edge -- mouseenter --> peek
  //   peek -- mouseleave (and not expanded) --> edge
  //   any  -- click inside --> expanded
  //   expanded -- click outside the card --> peek (then mouseleave -> edge)
  //   any  -- Escape key --> edge
  function _setFloatState(card, state) {
    if (!card) return;
    card.dataset.floatState = state;
  }

  // Selectors that count as "interactive content" inside an expanded
  // card -- a click on any of these should NOT collapse the card.
  const _FLOAT_INTERACTIVE = "input, button, select, option, " +
                             ".tray-chip, .chip-remove, .run-row, " +
                             ".run-add, .filter-chip, a";

  // Per-card grace period for peek -> edge collapse. 1 second after the
  // mouse leaves before the card tucks back into the edge. If the user
  // re-enters within that window the timer is cancelled (the card stays
  // in peek), so accidentally drifting off the card doesn't immediately
  // hide it.
  const PEEK_GRACE_MS = 350;

  function _wireFloatingCard(card) {
    _setFloatState(card, "edge");
    card.addEventListener("mouseenter", () => {
      // Cancel any pending peek->edge transition.
      if (card._peekTimer) {
        clearTimeout(card._peekTimer);
        card._peekTimer = null;
      }
      if (card.dataset.floatState === "edge") _setFloatState(card, "peek");
    });
    card.addEventListener("mouseleave", () => {
      if (card.dataset.floatState !== "peek") return;
      // Schedule the collapse with a grace window; mouseenter cancels.
      if (card._peekTimer) clearTimeout(card._peekTimer);
      card._peekTimer = setTimeout(() => {
        if (card.dataset.floatState === "peek") _setFloatState(card, "edge");
        card._peekTimer = null;
      }, PEEK_GRACE_MS);
    });
    card.addEventListener("click", (e) => {
      const t = e.target;
      // Action controls inside the header fire their own action, no
      // state change. The float-toggle button is NOT in this list --
      // it's part of the toggle area.
      const isAction = t.closest(
        "input, select, option, a, " +
        ".chip-remove, .run-row, .run-add, .filter-chip, " +
        "#tray-clear, #tray-save, #tray-delete-group, " +
        "#tray-analyze, #manual-scan-load, #runs-refresh, " +
        "#tray-load-group, #filter-clear"
      );
      if (isAction) return;
      const state = card.dataset.floatState;
      // From edge or peek: any non-action click expands.
      if (state !== "expanded") {
        _setFloatState(card, "expanded");
        return;
      }
      // EXPANDED: the entire peek-view area (everything OUTSIDE the
      // picker body / chip grid) is the collapse target. So clicks on
      // header empty space, filter-peek-area padding, the float-toggle
      // button, the card's own padding -- all collapse to peek. Only
      // clicks INSIDE the picker body / chip grid keep the card open.
      const inBody = t.closest(".runs-picker-body, .filter-body, .chn-picker-body");
      if (!inBody) _setFloatState(card, "peek");
    });
  }

  function setupFloatingAnalysisCards() {
    if (!FLOATING_ANALYSIS_CARDS) return;
    const host = document.getElementById("floating-analysis-host");
    const runs = document.getElementById("analysis-runs-card");
    const filt = document.getElementById("analysis-filters");
    if (!host || !runs || !filt) return;
    host.appendChild(runs);
    host.appendChild(filt);
    runs.classList.remove("runs-collapsed");
    filt.classList.remove("collapsed");
    // Initial counts. renderTray / updateFilterStatus update these
    // on every change; this just ensures the attribute exists before
    // the first paint so the CSS active/inactive selectors apply.
    runs.dataset.floatCount = "0";
    filt.dataset.floatCount = "0";
    _wireFloatingCard(runs);
    _wireFloatingCard(filt);

    // Click anywhere outside an EXPANDED card collapses it to peek
    // (then mouseleave from peek -> edge fires naturally). Inside-card
    // clicks are skipped via card.contains() so chip-removes, paste
    // box input, etc. still work normally without collapsing the card.
    document.addEventListener("click", (e) => {
      const cards = host.querySelectorAll(".card");
      cards.forEach((card) => {
        if (card.contains(e.target)) return;
        if (card.dataset.floatState === "expanded") {
          _setFloatState(card, "peek");
          // If the mouse isn't currently over this card, go to edge
          // after a short delay (gives any genuine hover time to set in).
          setTimeout(() => {
            if (!card.matches(":hover") &&
                card.dataset.floatState === "peek") {
              _setFloatState(card, "edge");
            }
          }, 120);
        }
      });
    });
    // Keyboard shortcuts (only when Analysis tab is active AND focus
    // isn't in a form control -- otherwise typing 'r' or 'f' in the
    // paste-box would trigger the shortcut).
    document.addEventListener("keydown", (e) => {
      if (activeTab !== "analysis") return;
      if (e.target.matches("input, textarea, select")) return;
      // Esc: collapse everything back to edge.
      if (e.key === "Escape") {
        host.querySelectorAll(".card").forEach((card) => {
          if (card.dataset.floatState !== "edge") _setFloatState(card, "edge");
        });
        return;
      }
      // r -> Runs card expanded; f -> Filter card expanded.
      if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        _setFloatState(document.getElementById("analysis-runs-card"), "expanded");
        return;
      }
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        _setFloatState(document.getElementById("analysis-filters"), "expanded");
        return;
      }
    });

    // Click-target shortcuts in the Run summary card:
    //   - clicking the FIRST stat tile (scan_id) toggles the Runs card
    //   - clicking any OTHER stat tile toggles the Filter card
    // "Toggle" = expand if not expanded, otherwise collapse to edge.
    const detailBody = document.getElementById("analysis-detail-body");
    function _toggleFloatCard(cardId) {
      const card = document.getElementById(cardId);
      if (!card) return;
      _setFloatState(card,
        card.dataset.floatState === "expanded" ? "edge" : "expanded");
    }
    if (detailBody) {
      detailBody.addEventListener("click", (e) => {
        const tile = e.target.closest(".stat-tile");
        if (!tile) return;
        const grid = tile.parentElement;
        if (!grid) return;
        const tiles = Array.from(grid.children).filter(
          (c) => c.classList && c.classList.contains("stat-tile"));
        const idx = tiles.indexOf(tile);
        e.stopPropagation();
        _toggleFloatCard(idx === 0 ? "analysis-runs-card" : "analysis-filters");
      });
      const styleTiles = () => {
        detailBody.querySelectorAll(".stat-tile").forEach((t, i, all) => {
          const isFirst = (Array.from(t.parentElement.children)
            .filter((c) => c.classList.contains("stat-tile"))[0] === t);
          t.style.cursor = "pointer";
          t.title = isFirst
            ? "Click to open the Runs picker (click again to close)"
            : "Click to open the Filter card (click again to close)";
        });
      };
      const obs = new MutationObserver(styleTiles);
      obs.observe(detailBody, {childList: true, subtree: true});
      styleTiles();
    }

    host.hidden = (activeTab !== "analysis");
  }

  // ===================== Sequence tab (flattened .seq viewer) ==========
  // Reads .seq files from a scan's sequence/ folder (auto-dump) or any
  // folder of .seq files via /api/sequence/*. On-demand; no polling.
  const seqState = { index: null, query: null, file: null };
  let seqScansCache = [];   // scans with sequence dumps (the picker list)

  function seqFmt(v) {
    if (typeof v === "number") {
      if (v !== 0 && (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3))
        return v.toExponential(3);
      return String(+v.toPrecision(6));
    }
    return String(v);
  }

  function seqEsc(s) {
    return String(s).replace(/[&<>]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  function seqQueryString(extra) {
    const p = new URLSearchParams(seqState.query || {});
    if (extra) for (const k in extra) p.set(k, extra[k]);
    return p.toString();
  }

  async function seqLoad(query) {
    const status = $("seq-source-status");
    try {
      if (status) status.textContent = "loading…";
      const idx = await api("/api/sequence/list?" +
                            new URLSearchParams(query).toString());
      seqState.index = idx;
      seqState.query = query;
      seqPopulate(idx);
      if (status) {
        const nf = (idx.files || []).length, np = (idx.points || []).length;
        status.textContent = `${np} point(s), ${nf} file(s)` +
          (idx.has_manifest ? "" : "  (no manifest)");
      }
    } catch (e) {
      seqState.index = null;
      if (status) status.textContent = "load failed: " + (e.message || e);
      toast("Sequence load failed: " + (e.message || e), "err");
    }
  }

  // ---- Scan picker (modeled on the Analysis runs picker) --------------------
  // Only the most-recent N scans (Refresh re-fetches): keeps the list light and the
  // per-scan config read (has_seq / has_snapshot) fast.
  const SEQ_SCANS_LIMIT = 30;
  async function loadSeqScans() {
    const wrap = $("seq-scan-table");
    try {
      const data = await api("/api/sequence/scans?max=" + SEQ_SCANS_LIMIT);
      seqScansCache = data.scans || [];
      seqPopulateScanDate();
      renderSeqScans();           // also updates the dumps/total count badge
    } catch (e) {
      if (wrap) wrap.innerHTML =
        `<div class="run-row"><div class="run-info muted">${seqEsc(e.message || e)}</div></div>`;
    }
  }

  function seqPopulateScanDate() {
    const sel = $("seq-scan-date");
    if (!sel) return;
    const dates = Array.from(new Set(
      seqScansCache.map((s) => (s.scan_id || "").slice(0, 8))
    )).filter(Boolean).sort().reverse();
    const cur = sel.value;
    sel.innerHTML = '<option value="">All dates</option>' +
      dates.map((d) => `<option value="${d}">${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}</option>`).join("");
    sel.value = cur;
  }

  function renderSeqScans() {
    const wrap = $("seq-scan-table");
    if (!wrap) return;
    const nSeq = seqScansCache.filter((s) => s.has_seq).length;
    setText("seq-scan-count", `${nSeq}/${seqScansCache.length}`);
    const search = ($("seq-scan-search").value || "").toLowerCase();
    const date = $("seq-scan-date").value || "";
    const filtered = seqScansCache.filter((s) => {
      if (date && (s.scan_id || "").slice(0, 8) !== date) return false;
      if (search) {
        const blob = ((s.scan_id || "") + " " + (s.name || "")).toLowerCase();
        if (!blob.includes(search)) return false;
      }
      return true;
    });
    if (!filtered.length) {
      wrap.innerHTML =
        '<div class="run-row"><div class="run-info muted">no scans match</div></div>';
      return;
    }
    const cur = (seqState.query && seqState.query.scan_id) || "";
    // Same row layout as the Analysis picker, with THREE states (§12.4):
    //   Ready          -- has a .seq dump            -> click plots it.
    //   Reconstructable -- no .seq but a code snapshot -> dark; click reconstructs
    //                      offline from the snapshot (+ captured runtime globals).
    //   Unrecoverable  -- no .seq and no snapshot     -> grayed, inert.
    wrap.innerHTML = filtered.map((s) => {
      const id = s.scan_id || "";
      const idShort = id.length === 14
        ? `${id.slice(4,6)}/${id.slice(6,8)} ${id.slice(8,10)}:${id.slice(10,12)}:${id.slice(12,14)}`
        : id;
      let state, tag, title;
      if (s.has_seq) {
        state = "ready"; tag = `${s.n_seq} seq`; title = id;
      } else if (s.has_snapshot && s.has_descriptor) {
        state = "reconstructable"; tag = "reconstruct ⟳";
        title = id + " — no .seq; click to reconstruct from the code snapshot";
      } else if (s.has_snapshot) {
        // Has a code snapshot but no descriptor -> predates self-contained
        // reconstruction; the ScanGroup/seq can't be rebuilt, so NOT a dead button.
        state = "unrecoverable"; tag = "no descriptor";
        title = id + " — has a code snapshot but predates self-contained " +
                "reconstruction (no descriptor); cannot rebuild the ScanGroup";
      } else {
        state = "unrecoverable"; tag = "no dump";
        title = id + " — no .seq and no code snapshot; cannot reconstruct";
      }
      return `
        <div class="run-row seq-row-${state} ${id === cur ? "in-tray" : ""}"
             data-scan-id="${id}" data-state="${state}" title="${seqEsc(title)}">
          <div class="run-info">
            ${idShort}
            <span class="run-dim"> · ${seqEsc(s.name || "—")}</span>
            <span class="run-dim"> · ${seqEsc(s.swept || "—")}</span>
            <span class="seq-row-tag"> · ${tag}</span>
          </div>
        </div>`;
    }).join("");
    $$(".run-row", wrap).forEach((row) => {
      if (!row.dataset.scanId) return;
      const state = row.dataset.state;
      if (state === "ready") {
        row.addEventListener("click", () => {
          const fin = $("seq-folder"); if (fin) fin.value = "";
          seqLoad({ scan_id: row.dataset.scanId }).then(renderSeqScans);
        });
      } else if (state === "reconstructable") {
        row.addEventListener("click", () => seqReconstruct(row.dataset.scanId, row));
      }
      // unrecoverable: inert (no handler)
    });
  }

  // Reconstruct a scan's missing .seq(s) offline: the dashboard POSTs to
  // /api/sequence/reconstruct, which spawns the engine-python (py3.8 + libnacs,
  // use_dummy_device) driver. It replays the run's code snapshot + captured
  // runtime globals, regenerates the .seq(s) into <scan>/sequence/, and the row
  // flips to Ready. CPU-heavy + may be deferred while a live scan runs.
  async function seqReconstruct(scanId, rowEl) {
    if (!scanId || seqState._reconstructing) return;
    seqState._reconstructing = true;
    if (rowEl) rowEl.classList.add("seq-row-working");
    toast("Reconstructing " + scanId + "… (engine subprocess, may take a while)", "");
    try {
      const r = await api("/api/sequence/reconstruct?scan_id=" +
                          encodeURIComponent(scanId), { method: "POST" });
      if (r && r.deferred) {
        toast("Reconstruct deferred: " + (r.reason || "a scan is running"), "warn");
        return;
      }
      toast("Reconstructed " +
            (r && r.n_seq != null ? r.n_seq + " sequence(s)" : "") +
            (r && r.approximate ? " (some channels approximate)" : ""), "ok");
      await loadSeqScans();                      // the row flips to Ready
      await seqLoad({ scan_id: scanId });        // load + plot the regenerated .seq
      renderSeqScans();
    } catch (e) {
      toast("Reconstruct failed: " + (e.message || e), "err");
    } finally {
      seqState._reconstructing = false;
      if (rowEl) rowEl.classList.remove("seq-row-working");
    }
  }

  function seqPopulate(idx) {
    const axEl = $("seq-scanned-axes");
    if (axEl) {
      const ax = idx.scanned_axes || [];
      axEl.textContent = ax.length
        ? "scanned: " + ax.map((a) => a.path).join(", ") : "";
    }
    const psel = $("seq-point-select");
    if (psel) {
      psel.innerHTML = "";
      (idx.points || []).forEach((pt, i) => {
        const o = document.createElement("option");
        o.value = String(i);
        const sc = pt.scanned && Object.keys(pt.scanned).length
          ? "  " + Object.entries(pt.scanned)
              .map(([k, v]) => `${k}=${seqFmt(v)}`).join(", ")
          : "";
        o.textContent = `#${pt.n != null ? pt.n : i + 1}${sc}`;
        psel.appendChild(o);
      });
      if ((idx.points || []).length) { psel.selectedIndex = 0; seqOnPoint(); }
    }
  }

  function seqCurrentPoint() {
    const idx = seqState.index;
    if (!idx) return null;
    const psel = $("seq-point-select");
    const i = psel ? parseInt(psel.value, 10) : 0;
    const pts = idx.points || [];
    return pts[isNaN(i) ? 0 : i] || pts[0] || null;
  }

  function seqOnPoint() {
    const pt = seqCurrentPoint();
    if (!pt) return;
    seqState.file = pt.file;
    const fileEntry = (seqState.index.files || []).find((f) => f.file === pt.file);
    const ssel = $("seq-seq-select");
    if (ssel) {
      ssel.innerHTML = "";
      const seqs = (fileEntry && fileEntry.sequences) || [];
      seqs.forEach((s) => {
        const o = document.createElement("option");
        o.value = String(s.seq_idx);
        o.textContent = `${s.name} (idx ${s.seq_idx}, ${s.nchns} ch)`;
        ssel.appendChild(o);
      });
      if (seqs.length) ssel.selectedIndex = 0;
    }
    seqOnSeq();
  }

  function seqOnSeq() {
    const fileEntry = (seqState.index.files || []).find((f) => f.file === seqState.file);
    const ssel = $("seq-seq-select");
    const seqIdx = ssel ? ssel.value : null;
    const seqs = (fileEntry && fileEntry.sequences) || [];
    const seq = seqs.find((s) => String(s.seq_idx) === String(seqIdx)) || seqs[0];
    const csel = $("seq-chn-select");
    const prev = csel ? new Set(Array.from(csel.selectedOptions).map((o) => o.value))
                      : new Set();
    if (csel && seq) {
      csel.innerHTML = "";
      (seq.channels || []).forEach((name) => {
        const o = document.createElement("option");
        o.value = name; o.textContent = name;
        if (prev.has(name)) o.selected = true;
        csel.appendChild(o);
      });
    }
    seqUpdateChnCount();
    seqRenderPlot();
    seqRenderParams();
  }

  function seqSelectedChannels() {
    const csel = $("seq-chn-select");
    return csel ? Array.from(csel.selectedOptions).map((o) => o.value) : [];
  }

  // Reflect the # of selected channels on the floating card (drives the
  // edge-tab count badge + the active accent palette).
  function seqUpdateChnCount() {
    const card = document.getElementById("sequence-chn-card");
    if (card) card.dataset.floatCount = String(seqSelectedChannels().length);
  }

  // Hover-driven floating card: edge (minimized) -> expanded on mouseenter ->
  // edge on mouseleave, after a 400 ms grace. Never collapses while the mouse
  // is still over it OR while a field inside has focus (so typing in the scan
  // search box holds the panel open even if the cursor drifts off).
  function _wireSeqFloatCard(card) {
    if (!card) return;
    _setFloatState(card, "edge");
    let timer = null;
    card.addEventListener("mouseenter", () => {
      if (timer) { clearTimeout(timer); timer = null; }
      _setFloatState(card, "expanded");
    });
    card.addEventListener("mouseleave", () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        if (!card.matches(":hover") && !card.contains(document.activeElement))
          _setFloatState(card, "edge");
        timer = null;
      }, 400);
    });
  }

  async function seqRenderPlot() {
    const el = $("plot-sequence");
    if (!el || !seqState.index || !seqState.file || !window.Plotly) return;
    const ssel = $("seq-seq-select");
    const q = seqQueryString({
      file: seqState.file,
      seq: ssel ? ssel.value : "",
      chns: seqSelectedChannels().join(","),
    });
    try {
      const fig = await fetch("/api/sequence/figure?" + q).then((r) => r.json());
      await Plotly.react(el, fig.data || [], fig.layout || {},
                         { responsive: true, displayModeBar: true });
      seqWirePlotClick(el);          // click point -> segment params + formula
      seqWirePlotHover(el);          // hover point -> SVG-overlay highlight of the pulse
      const _hp = _seqHoverPath(el, true);
      if (_hp) _hp.setAttribute("points", "");     // clear stale hover line after a rebuild
      if (seqState._emphParam) seqEmphasizeParamRegion(seqState._emphParam);  // survive rebuild
    } catch (e) { console.warn("seq figure", e); }
  }

  // Click a point in the plot -> highlight the params whose value appears in that
  // channel's waveform (option 3). Bound once; survives Plotly.react re-renders.
  function seqWirePlotClick(el) {
    if (el._seqClickWired) return;
    el._seqClickWired = true;
    el.on("plotly_click", (data) => {
      const pt = data && data.points && data.points[0];
      const chn = pt && pt.data && pt.data.name;
      if (!chn || chn === "Selected") return;
      // The clicked point's customdata IS its pulse id: prefer per-pulse (segment-specific)
      // provenance so we focus ONLY the params that derive THIS segment, not every param
      // that touches the channel anywhere. Falls back to whole-channel.
      seqFocusPoint(chn, pt.customdata, pt.y, pt.x);
    });
  }

  async function seqRenderParams() {
    const box = $("seq-params");
    if (!box || !seqState.index || !seqState.file) return;
    seqClearFocus();                 // selection context changes when params reload
    const ssel = $("seq-seq-select");
    const qs = seqQueryString({ file: seqState.file, seq: ssel ? ssel.value : "" });
    let r;
    try {
      r = await api("/api/sequence/params?" + qs);
    } catch (e) {
      box.innerHTML = '<span class="muted">params unavailable: ' +
                      seqEsc(e.message || e) + "</span>";
      seqState.params = null; return;
    }
    if (!r.has_params || !r.params) {
      box.innerHTML = '<span class="muted">No parameters for this scan.</span>';
      seqState.params = null; seqState.scannedPaths = new Set(); seqState.xref = null;
      return;
    }
    seqState.params = r.params;
    seqState.scannedPaths = new Set(r.scanned_paths || []);
    // Param<->channel cross-reference (build-time provenance). Best-effort.
    try { seqState.xref = await api("/api/sequence/xref?" + qs); }
    catch (e) { seqState.xref = null; }
    seqRenderParamTree();
    // No xref yet (or a pre-region one)? Kick a BACKGROUND build/upgrade (non-blocking) and
    // light up the affordance once it lands. The .seq plot is already shown.
    seqMaybeBuildXref(qs);
  }

  // Build (or upgrade) sequence/xref.json in the background for the loaded scan, then poll
  // until it lands and re-render the param tree. Non-blocking; bails if the user navigates
  // away mid-build. Only for scan_id loads (a folder load may lack the descriptor).
  //   * no xref        -> build.
  //   * pre-region xref (available but no per-pulse `pulses`) -> force-UPGRADE once/session.
  //   * region-ready xref -> nothing.
  async function seqMaybeBuildXref(qs) {
    const scanId = seqState.query && seqState.query.scan_id;
    if (!scanId || seqState._xrefBuilding === scanId) return;
    const xr = seqState.xref;
    if (seqXrefComplete(xr)) return;                  // already has per-pulse maps + formulas
    const stale = !!(xr && xr.available);             // exists but older schema -> upgrade
    if (stale) {
      seqState._xrefUpgraded = seqState._xrefUpgraded || new Set();
      if (seqState._xrefUpgraded.has(scanId)) return; // upgrade at most once per session
      seqState._xrefUpgraded.add(scanId);
    }
    seqState._xrefBuilding = scanId;
    try {
      let r;
      try {
        r = await api("/api/sequence/build_xref?scan_id=" + encodeURIComponent(scanId) +
                      (stale ? "&force=1" : ""), { method: "POST" });
      } catch (e) { return; }                         // best-effort: stay dormant on failure
      if (!r || (!r.started && !r.available)) return; // e.g. scan has no descriptor
      if (r.started) toast((stale ? "Upgrading" : "Building") + " param↔channel map…", "");
      for (let i = 0; i < 40; i++) {                  // ~60 s budget (provenance is quick)
        if ((seqState.query && seqState.query.scan_id) !== scanId) return;  // navigated away
        let x = null;
        try { x = await api("/api/sequence/xref?" + qs); } catch (e) {}
        // When upgrading, wait for the COMPLETE artifact; otherwise accept availability.
        if (x && (stale ? seqXrefComplete(x) : x.available)) {
          seqState.xref = x;
          seqRenderParamTree();                       // affordance lights up
          if (r.started) toast("param↔channel map ready", "ok");
          return;
        }
        await new Promise((res) => setTimeout(res, 1500));
      }
    } finally {
      if (seqState._xrefBuilding === scanId) seqState._xrefBuilding = null;
    }
  }

  // The current xref schema/format version the viewer expects. Older artifacts (aggregate
  // only, per-pulse without formulas, or verbose pre-cleanup formulas) carry a lower/absent
  // version and are rebuilt in place on load. Keep in lock-step with XREF_VERSION in
  // pyctrl/tools/provenance_scan.py.
  const SEQ_XREF_VERSION = 4;
  function seqXrefComplete(xr) {
    return !!(xr && xr.available && (xr.version || 0) >= SEQ_XREF_VERSION);
  }

  // "Rebuild ⟳" button: force-regenerate xref.json for the loaded scan, even if one already
  // exists (e.g. to pick up new formulas), then re-render the param tree when it lands.
  async function seqForceRebuildXref() {
    const scanId = seqState.query && seqState.query.scan_id;
    if (!scanId) { toast("Load a scan from the Scans picker first", "warn"); return; }
    if (seqState._xrefBuilding === scanId) { toast("Already rebuilding…", ""); return; }
    seqState._xrefBuilding = scanId;
    const ssel = $("seq-seq-select");
    const qs = seqQueryString({ file: seqState.file || "", seq: ssel ? ssel.value : "" });
    try {
      let r;
      try {
        r = await api("/api/sequence/build_xref?scan_id=" + encodeURIComponent(scanId) +
                      "&force=1", { method: "POST" });
      } catch (e) { toast("Rebuild failed: " + (e.message || e), "err"); return; }
      if (!r || r.ok === false) {
        toast("Rebuild: " + ((r && r.error) || "no descriptor / unavailable"), "warn");
        return;
      }
      toast("Rebuilding param↔channel map…", "");
      for (let i = 0; i < 60; i++) {                 // ~90 s (a multi-point scan is slower)
        if ((seqState.query && seqState.query.scan_id) !== scanId) return;  // navigated away
        let x = null;
        try { x = await api("/api/sequence/xref?" + qs); } catch (e) {}
        if (x && seqXrefComplete(x)) {
          seqState.xref = x;
          seqRenderParamTree();
          toast("param↔channel map rebuilt", "ok");
          return;
        }
        await new Promise((res) => setTimeout(res, 1500));
      }
      toast("Rebuild timed out", "warn");
    } finally {
      if (seqState._xrefBuilding === scanId) seqState._xrefBuilding = null;
    }
  }

  // Re-render the param tree applying the search box + the config/modified/scanned
  // show-hide toggles. Cheap (client-side) so it runs on every keystroke / toggle.
  function seqRenderParamTree() {
    const box = $("seq-params");
    if (!box || !seqState.params) return;
    const search = (($("seq-param-search") || {}).value || "").trim().toLowerCase();
    const filters = {
      config:   seqFilterOn("seq-filter-config"),
      modified: seqFilterOn("seq-filter-modified"),
      scanned:  seqFilterOn("seq-filter-scanned"),
    };
    const html = seqParamTree(seqState.params, seqState.scannedPaths || new Set(),
                              "", search, filters);
    box.innerHTML = html || '<span class="muted">no parameters match the filter.</span>';
  }
  function seqFilterOn(id) {
    const el = $(id);
    return el ? el.checked : true;
  }

  function seqIsLeaf(v) {
    return v && typeof v === "object" && !Array.isArray(v) &&
      Object.prototype.hasOwnProperty.call(v, "value") &&
      Object.prototype.hasOwnProperty.call(v, "type");
  }

  function seqLeafClass(leaf, isScanned) {
    const hasCfg = leaf.config_value !== undefined && leaf.config_value !== null;
    const modified = leaf.type === 2 || leaf.type === 3 ||
                     (hasCfg && leaf.config_value !== leaf.value);
    if (isScanned) return "seq-scanned";
    if (modified) return "seq-modified";
    if (hasCfg) return "seq-config";
    return "seq-default";
  }

  // Show a leaf iff its category toggle is on AND it matches the search. "default"
  // leaves (type 0, non-scanned) aren't governed by the three category toggles.
  function seqParamPass(cls, path, search, filters) {
    if (cls === "seq-scanned"  && !filters.scanned)  return false;
    if (cls === "seq-modified" && !filters.modified) return false;
    if (cls === "seq-config"   && !filters.config)   return false;
    if (search && !path.toLowerCase().includes(search)) return false;
    return true;
  }

  function seqParamTree(node, scanned, prefix, search, filters) {
    let html = "";
    for (const key of Object.keys(node)) {
      const v = node[key];
      const path = prefix ? prefix + "." + key : key;
      if (seqIsLeaf(v)) {
        const isScanned = scanned.has(path);
        const cls = seqLeafClass(v, isScanned);
        if (seqParamPass(cls, path, search, filters))
          html += seqParamLeaf(key, path, v, isScanned, cls);
      } else if (v && typeof v === "object" && !Array.isArray(v)) {
        const inner = seqParamTree(v, scanned, path, search, filters);
        if (inner)                       // drop branches with no surviving leaves
          html += `<details open><summary>${seqEsc(key)}</summary>` +
                  `<div class="seq-indent">${inner}</div></details>`;
      } else if (!search || path.toLowerCase().includes(search)) {
        html += `<div class="seq-leaf">${seqEsc(key)}: ${seqEsc(String(v))}</div>`;
      }
    }
    return html;
  }

  function seqParamLeaf(key, path, leaf, isScanned, cls) {
    const hasCfg = leaf.config_value !== undefined && leaf.config_value !== null;
    let rhs = seqEsc(seqFmt(leaf.value));
    if (cls === "seq-modified" && hasCfg) {
      rhs = `<span class="seq-config">${seqEsc(seqFmt(leaf.config_value))}</span> ⇒ ${rhs}`;
    }
    const badge = isScanned ? ' <span class="seq-scanned-badge">scanned</span>' : '';
    // Param<->channel matching is GATED behind real build-time provenance (xref.json,
    // produced by the engine build). When absent (available=false) leaves are NOT
    // clickable and no channel reveal is shown -- value-coincidence matching was
    // dropped as inaccurate (plan §8).
    const xrefOn = !!(seqState.xref && seqState.xref.available);
    const chns = xrefOn && seqState.xref.param_to_channels
      ? seqState.xref.param_to_channels[path] : null;
    const reveal = chns
      ? ` <span class="seq-leaf-chns">→ ${seqEsc(chns.join(", "))}</span>` : "";
    const clickable = xrefOn
      ? ` data-param-path="${seqEsc(path)}"${chns ? ' data-has-xref="1"' : ''}` : "";
    return `<div class="seq-leaf ${cls}"${clickable}>` +
           `<b>${seqEsc(key)}</b>: ${rhs}${badge}${reveal}</div>`;
  }

  // Select + PROMOTE the channels a param drives: mark them selected and move them to the
  // TOP of the channel list (front), then re-render so the regions exist before emphasis.
  async function seqOnParamChannels(path) {
    if (!seqState.xref || !seqState.xref.available) return;
    const chns = (seqState.xref.param_to_channels &&
                  seqState.xref.param_to_channels[path]) || [];
    if (!chns.length) return;
    const csel = $("seq-chn-select");
    if (!csel) return;
    const want = new Set(chns);
    const opts = Array.from(csel.options);
    const driven = opts.filter((o) => want.has(o.value));
    const rest = opts.filter((o) => !want.has(o.value));
    driven.forEach((o) => { o.selected = true; });
    driven.concat(rest).forEach((o) => csel.appendChild(o));   // reorder: driven to front
    seqUpdateChnCount();
    await seqRenderPlot();
  }

  // Param click (leaf or focus chip) -> promote its channels, emphasize its regions, and
  // fill the focus panel with the gathered params + formula.
  async function seqSelectParam(path) {
    const pbox = $("seq-params");
    if (pbox) {
      $$(".seq-leaf.seq-leaf-active", pbox).forEach((el) =>
        el.classList.remove("seq-leaf-active"));
      const row = pbox.querySelector('.seq-leaf[data-param-path="' +
                                     path.replace(/"/g, '\\"') + '"]');
      if (row) row.classList.add("seq-leaf-active");
    }
    await seqOnParamChannels(path);
    seqEmphasizeParamRegion(path);
    seqFocusParam(path);
  }

  // Click a channel chip -> ensure that channel is shown (select + render), keeping any
  // active param's region emphasis.
  async function seqShowChannel(name) {
    const csel = $("seq-chn-select");
    if (!csel) return;
    let changed = false;
    Array.from(csel.options).forEach((o) => {
      if (o.value === name && !o.selected) { o.selected = true; changed = true; }
    });
    if (changed) { seqUpdateChnCount(); await seqRenderPlot();
                   if (seqState._emphParam) seqEmphasizeParamRegion(seqState._emphParam); }
  }

  // Overlay (not marker-resize) emphasis: a thicker LINE drawn on top of the matched pulse
  // segments. Distinct reserved trace names so the param-selection and the hover overlays
  // don't clobber each other or the base channel traces.
  const SEQ_HILITE = "·param-region";

  // Build thick-line overlay trace(s) over the points whose pid is in `pidSet`, grouped by
  // y-axis (primary/secondary), with null breaks between non-contiguous runs so each pulse
  // segment is its own polyline.
  function _seqOverlayTraces(el, pidSet, name, color, width) {
    const byAxis = {};
    el.data.forEach((tr) => {
      if (!tr.customdata || tr.name === SEQ_HILITE || tr.name === "Selected") return;
      const ax = tr.yaxis || "y";
      const acc = byAxis[ax] || (byAxis[ax] = { x: [], y: [] });
      let inRun = false;
      for (let k = 0; k < tr.customdata.length; k++) {
        if (pidSet.has(Number(tr.customdata[k]))) {
          acc.x.push(tr.x[k]); acc.y.push(tr.y[k]); inRun = true;
        } else if (inRun) { acc.x.push(null); acc.y.push(null); inRun = false; }
      }
      if (inRun) { acc.x.push(null); acc.y.push(null); }
    });
    const traces = [];
    Object.keys(byAxis).forEach((ax) => {
      const a = byAxis[ax];
      if (!a.x.length) return;
      const t = { x: a.x, y: a.y, mode: "lines", name: name,
                  line: { color: color, width: width }, hoverinfo: "skip",
                  showlegend: false, cliponaxis: false };
      if (ax !== "y") t.yaxis = ax;
      traces.push(t);
    });
    return traces;
  }

  function _seqRemoveTraces(el, name) {
    if (!el || !el.data) return;
    const idx = [];
    el.data.forEach((tr, i) => { if (tr.name === name) idx.push(i); });
    if (idx.length) Plotly.deleteTraces(el, idx);
  }

  // Param-region highlight: a thick line over the channel segments the param drives, plus
  // shaded time bands for any wait/timing regions it controls (waits have no channel output).
  function seqEmphasizeParamRegion(path) {
    seqClearEmphasis();
    const el = $("plot-sequence");
    if (!el || !el.data || !window.Plotly) return;
    const xr = seqState.xref;
    if (!xr || !xr.available) return;
    const pids = new Set(((xr.param_to_pids || {})[path] || []).map(Number));
    const overlays = pids.size ? _seqOverlayTraces(el, pids, SEQ_HILITE, "#ffd166", 6) : [];
    if (overlays.length) Plotly.addTraces(el, overlays);
    const regions = (xr.time_regions || {})[path] || [];
    if (regions.length) {
      Plotly.relayout(el, { shapes: regions.map(([t0, t1]) => ({
        type: "rect", xref: "x", yref: "paper", x0: t0, x1: t1, y0: 0, y1: 1,
        fillcolor: "rgba(255,209,102,0.13)", line: { width: 0 }, layer: "below" })) });
    }
    seqState._emphParam = path;
    const bits = [];
    if (overlays.length) bits.push("waveform region");
    if (regions.length) bits.push(regions.length + " time band(s)");
    toast(bits.length ? (path + " → " + bits.join(" + ")) : ("No region for " + path),
          bits.length ? "ok" : "");
  }

  // Remove the param-selection overlay + its time bands.
  function seqClearEmphasis() {
    seqState._emphParam = null;
    const el = $("plot-sequence");
    if (!el || !el.data || !window.Plotly) return;
    _seqRemoveTraces(el, SEQ_HILITE);
    if (el.layout && el.layout.shapes && el.layout.shapes.length)
      Plotly.relayout(el, { shapes: [] });
  }

  // Hover highlight (2c): a thick line over the pulse under the cursor, drawn on a plain SVG
  // overlay ON TOP of the plot -- NOT a Plotly trace. So hovering triggers ZERO Plotly work
  // (no redraw, no autorange recompute); it just sets one <polyline points="..."> attribute.
  // Pixels are computed by linear interpolation from each axis' data range + pixel geometry
  // (correct for the linear sequence axes), so primary/secondary channels both map right.
  function seqWirePlotHover(el) {
    if (el._seqHoverWired) return;
    el._seqHoverWired = true;
    el.on("plotly_hover", (data) => {
      const pt = data && data.points && data.points[0];
      const tr = pt && pt.data;
      if (!tr || !tr.customdata || pt.customdata == null) return;
      if (tr.name === SEQ_HILITE || tr.name === "Selected") return;
      const xa = pt.xaxis, ya = pt.yaxis;          // the hovered point's own axes (handles y2)
      if (!xa || !ya || !xa.range || !ya.range) return;
      const key = String(pt.customdata) + "@" + tr.name;
      if (el._seqHoverPid === key) return;          // still on the same pulse -> nothing to do
      el._seqHoverPid = key;
      const x0 = xa.range[0], xw = xa.range[1] - xa.range[0], xoff = xa._offset, xl = xa._length;
      const y0 = ya.range[0], yw = ya.range[1] - ya.range[0], yoff = ya._offset, yl = ya._length;
      const pid = Number(pt.customdata), out = [];
      for (let k = 0; k < tr.customdata.length; k++) {   // scan ONLY the hovered channel
        if (Number(tr.customdata[k]) !== pid) continue;
        const px = xoff + (tr.x[k] - x0) / xw * xl;
        const py = yoff + (1 - (tr.y[k] - y0) / yw) * yl;
        if (isFinite(px) && isFinite(py)) out.push(px.toFixed(1) + "," + py.toFixed(1));
      }
      _seqHoverPath(el).setAttribute("points", out.join(" "));
    });
    el.on("plotly_unhover", () => { el._seqHoverPid = null; _seqClearHover(el); });
    el.on("plotly_relayout", () => { el._seqHoverPid = null; _seqClearHover(el); });
  }

  // The transient hover line is a plain SVG <polyline> overlaid on the plot div (created once,
  // pointer-events:none), so updating it never touches Plotly.
  function _seqHoverPath(el, existingOnly) {
    let ov = el.querySelector(":scope > svg.seq-hover-ov");
    if (!ov) {
      if (existingOnly) return null;
      const NS = "http://www.w3.org/2000/svg";
      ov = document.createElementNS(NS, "svg");
      ov.setAttribute("class", "seq-hover-ov");
      ov.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;" +
                         "pointer-events:none;z-index:6;overflow:hidden;";
      const pl = document.createElementNS(NS, "polyline");
      pl.setAttribute("fill", "none");
      pl.setAttribute("stroke", "#7ee787");
      pl.setAttribute("stroke-width", "5");
      pl.setAttribute("stroke-linejoin", "round");
      pl.setAttribute("stroke-linecap", "round");
      ov.appendChild(pl);
      if (getComputedStyle(el).position === "static") el.style.position = "relative";
      el.appendChild(ov);
    }
    return ov.firstChild;
  }

  function _seqClearHover(el) {
    const p = _seqHoverPath(el, true);
    if (p) p.setAttribute("points", "");
  }

  // ---- Selection focus region (top of the Parameters panel) -----------------
  // Render the gathered params + derivation formula + driven channels for the current
  // selection, with a Clear button. opts: {title, formula?, params:[], channels:[]}.
  function seqSetFocus(opts) {
    const box = $("seq-focus");
    if (!box) return;
    const chip = (txt, kind, data) =>
      '<span class="seq-chip' + (kind === "chan" ? " chan" : "") + '" ' + data + ">" +
      seqEsc(txt) + "</span>";
    const parts = [
      '<div class="seq-focus-head"><span>' + seqEsc(opts.title || "selection") +
      '</span><span class="grow"></span>' +
      '<span class="seq-focus-clear" id="seq-focus-clear">clear ✕</span></div>',
    ];
    if (opts.formula)
      parts.push('<div class="seq-focus-formula">= ' + seqEsc(opts.formula) + "</div>");
    if (opts.params && opts.params.length)
      parts.push('<div class="seq-focus-row"><span class="lbl">params</span>' +
        opts.params.map((p) => {
          const path = (typeof p === "string") ? p : p.path;
          const v = (typeof p === "object" && p.value != null) ? seqFmt(p.value) : null;
          return '<span class="seq-chip" data-focus-param="' + seqEsc(path) + '">' +
            seqEsc(path) + (v != null ? ' <span class="seq-chip-val">' + seqEsc(v) +
            "</span>" : "") + "</span>";
        }).join("") + "</div>");
    if (opts.channels && opts.channels.length)
      parts.push('<div class="seq-focus-row"><span class="lbl">channels</span>' +
        opts.channels.map((c) => chip(c, "chan", 'data-focus-chan="' + seqEsc(c) + '"'))
          .join("") + "</div>");
    box.innerHTML = parts.join("");
    box.hidden = false;
  }

  // The current value of a dotted param path, from the loaded params tree (for the focus chips).
  function seqParamValue(path) {
    let node = seqState.params;
    if (!node || !path) return undefined;
    for (const k of path.split(".")) {
      if (node && typeof node === "object" &&
          Object.prototype.hasOwnProperty.call(node, k)) node = node[k];
      else return undefined;
    }
    return (node && typeof node === "object" && "value" in node) ? node.value : undefined;
  }

  function seqClearFocus() {
    const box = $("seq-focus");
    if (box) { box.innerHTML = ""; box.hidden = true; }
    const pbox = $("seq-params");
    if (pbox) {
      $$(".seq-leaf.seq-leaf-xref-hit", pbox).forEach((el) =>
        el.classList.remove("seq-leaf-xref-hit"));
      $$(".seq-leaf.seq-leaf-active", pbox).forEach((el) =>
        el.classList.remove("seq-leaf-active"));
    }
    seqClearEmphasis();
  }

  // Click a plot point -> focus the params that derive THAT segment (per-pulse), with the
  // formula; falls back to whole-channel for idle/default (pid=-1) points.
  function seqFocusPoint(chn, pid, value, time) {
    const xr = seqState.xref;
    if (!xr || !xr.available) return;
    const box = $("seq-params");
    if (box) $$(".seq-leaf.seq-leaf-xref-hit", box).forEach((el) =>
      el.classList.remove("seq-leaf-xref-hit"));
    const pulse = (pid != null && xr.pulses) ? xr.pulses[String(pid)] : null;
    let params, formula = null, idle = false;
    if (pulse) {
      params = pulse.params || []; formula = pulse.expr || null;
    } else {
      params = (xr.channel_to_params && xr.channel_to_params[chn]) || [];
      idle = true;                                  // pid=-1 / no per-pulse data
    }
    let first = null;
    if (box) params.forEach((p) => {
      const row = box.querySelector('.seq-leaf[data-param-path="' +
                                    p.replace(/"/g, '\\"') + '"]');
      if (row) { row.classList.add("seq-leaf-xref-hit"); if (!first) first = row; }
    });
    if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
    const tStr = (time != null && isFinite(time)) ? Number(time).toFixed(3) + " ms" : "";
    const vStr = (value != null && isFinite(value)) ? seqFmt(value) : "";
    let title = chn + (tStr ? " @ " + tStr : "") + (vStr ? " = " + vStr : "");
    if (idle && !params.length) title += "  (idle / no pulse here)";
    else if (idle) title += "  (whole channel)";
    seqSetFocus({ title, formula, channels: [chn],
                  params: params.map((p) => ({ path: p, value: seqParamValue(p) })) });
  }

  // Click a param -> gather the params co-deriving its segments + the formula + driven
  // channels into the focus region. (Channel promotion + region emphasis happen in
  // seqSelectParam, which calls this.)
  function seqFocusParam(path) {
    const xr = seqState.xref;
    if (!xr || !xr.available) return;
    const pids = (xr.param_to_pids && xr.param_to_pids[path]) || [];
    const channels = new Set((xr.param_to_channels && xr.param_to_channels[path]) || []);
    const related = new Set([path]);
    const exprs = new Set();
    pids.forEach((p) => {
      const pu = xr.pulses && xr.pulses[String(p)];
      if (!pu) return;
      (pu.params || []).forEach((x) => related.add(x));
      if (pu.channel) channels.add(pu.channel);
      if (pu.expr) exprs.add(pu.expr);
    });
    const ex = Array.from(exprs);
    const formula = ex.length === 1 ? ex[0]
      : (ex.length > 1 ? ex.length + " formulas (varies by segment)" : null);
    const tr = (xr.time_regions && xr.time_regions[path]) || [];
    const pv = seqParamValue(path);
    let title = path + (pv != null ? " = " + seqFmt(pv) : "");
    if (!channels.size && tr.length) title += "  (" + tr.length + " wait region(s))";
    seqSetFocus({ title, formula, channels: Array.from(channels),
                  params: Array.from(related).map((p) => ({ path: p, value: seqParamValue(p) })) });
  }

  function loadSequence() {
    if (seqState._refreshToggle) seqState._refreshToggle();
    // Load the scan list once (it carries Analysis-style meta, so it's a touch
    // heavy); the Refresh button re-fetches on demand.
    if (!seqScansCache.length) loadSeqScans();
    if (seqState.index) seqRenderPlot();
  }

  function initSequenceTab() {
    // "Load current" -> the running / most-recent scan's sequence dump.
    // (seq-folder is now a hidden staging field for Browse…; there is no
    // typed-path box anymore.)
    const loadCurBtn = $("seq-load-btn");
    if (loadCurBtn) loadCurBtn.addEventListener("click", async () => {
      try {
        const st = await api("/api/status");
        const sid = st && (st.scan_id || st.scanId);
        if (sid) seqLoad({ scan_id: String(sid) });
        else toast("No current scan_id", "warn");
      } catch (e) { toast("status failed: " + (e.message || e), "err"); }
    });
    const rebuildBtn = $("seq-rebuild-xref-btn");
    if (rebuildBtn) rebuildBtn.addEventListener("click", seqForceRebuildXref);
    const psel = $("seq-point-select");
    if (psel) psel.addEventListener("change", seqOnPoint);
    const ssel = $("seq-seq-select");
    if (ssel) ssel.addEventListener("change", seqOnSeq);
    const csel = $("seq-chn-select");
    if (csel) csel.addEventListener("change", () => { seqUpdateChnCount(); seqRenderPlot(); });

    // Params card: search box + config/modified/scanned toggles re-render the tree.
    const pSearch = $("seq-param-search");
    if (pSearch) pSearch.addEventListener("input", seqRenderParamTree);
    ["seq-filter-config", "seq-filter-modified", "seq-filter-scanned"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("change", seqRenderParamTree);
    });
    // Click a param leaf -> promote its channels, emphasize its regions, fill the focus
    // panel. Click the same (active) leaf again to clear. Delegated (the tree's innerHTML
    // is rebuilt on every filter change).
    const pbox = $("seq-params");
    if (pbox) pbox.addEventListener("click", (e) => {
      const row = e.target.closest(".seq-leaf[data-param-path]");
      if (!row || !pbox.contains(row)) return;
      if (row.classList.contains("seq-leaf-active")) { seqClearFocus(); return; }  // toggle off
      seqSelectParam(row.dataset.paramPath);
    });
    // Focus region: Clear button + clickable param/channel chips.
    const fbox = $("seq-focus");
    if (fbox) fbox.addEventListener("click", (e) => {
      if (e.target.closest("#seq-focus-clear")) { seqClearFocus(); return; }
      const pc = e.target.closest("[data-focus-param]");
      if (pc) { seqSelectParam(pc.dataset.focusParam); return; }
      const cc = e.target.closest("[data-focus-chan]");
      if (cc) { seqShowChannel(cc.dataset.focusChan); }
    });

    // Floating pickers (Channels on the right, Scans on the left): hover-driven,
    // like the Analysis selector -- a thin edge tab that expands on hover and
    // auto-minimizes on leave (see _wireSeqFloatCard).
    _wireSeqFloatCard(document.getElementById("sequence-chn-card"));
    _wireSeqFloatCard(document.getElementById("seqscan-card"));

    // Scan picker (Analysis-style): search / date filter / refresh.
    const scanSearch = $("seq-scan-search");
    if (scanSearch) scanSearch.addEventListener("input", renderSeqScans);
    const scanDate = $("seq-scan-date");
    if (scanDate) scanDate.addEventListener("change", renderSeqScans);
    const scanRefresh = $("seq-scan-refresh");
    if (scanRefresh) scanRefresh.addEventListener("click", loadSeqScans);

    // Browse: native OS folder picker on the lab PC (server-side Tk subprocess).
    const browseBtn = $("seq-browse-btn");
    if (browseBtn) browseBtn.addEventListener("click", async () => {
      const label = browseBtn.textContent;
      browseBtn.textContent = "Picking…"; browseBtn.disabled = true;
      try {
        const r = await api("/api/sequence/pick_folder", { method: "POST" });
        if (r && r.path) {
          const fin = $("seq-folder"); if (fin) fin.value = r.path;
          seqLoad({ folder: r.path });
        }
      } catch (e) {
        toast("Folder picker failed: " + (e.message || e), "err");
      } finally {
        browseBtn.textContent = label; browseBtn.disabled = false;
      }
    });

    // Auto-dump toggle: rides the pyctrl runtime_state mmap flag via /api/sequence/dump_toggle.
    const autosave = $("seq-autosave");
    if (autosave) {
      const refresh = async () => {
        try { const r = await api("/api/sequence/dump_toggle"); autosave.checked = !!r.on; }
        catch (e) { /* backend store not present yet -> leave unchecked */ }
      };
      autosave.addEventListener("change", async () => {
        try {
          const r = await api("/api/sequence/dump_toggle?on=" + (autosave.checked ? 1 : 0),
                              { method: "POST" });
          autosave.checked = !!r.on;
          toast("Sequence auto-dump " + (r.on ? "ON" : "OFF"), r.on ? "ok" : "");
        } catch (e) {
          toast("toggle failed: " + (e.message || e), "err");
          refresh();   // revert the checkbox to the real state
        }
      });
      seqState._refreshToggle = refresh;
      refresh();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSequenceTab();
    setupFloatingAnalysisCards();
    startPolling();
    loadRunsList();
    loadGroups();
    loadSeqCatalog();
    renderTray();
    // Header click anywhere (except form controls) toggles the picker.
    const runsHdr = document.getElementById("runs-card-header");
    const runsCard = document.getElementById("analysis-runs-card");
    if (runsHdr && runsCard) {
      runsHdr.addEventListener("click", (e) => {
        const t = e.target;
        // Don't collapse when the click hits an input, button, select,
        // option, or a chip's remove-x (the inner button).
        if (t.closest("input, button, select, option, .tray-chip")) return;
        runsCard.classList.toggle("runs-collapsed");
      });
    }
    // Hardware iframe reload button
    const reload = $("hw-iframe-reload");
    if (reload) reload.addEventListener("click", () => {
      const ifr = $("hw-iframe");
      if (ifr) ifr.src = ifr.src;   // re-trigger load
    });
  });
})();
