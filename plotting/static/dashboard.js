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
  const TABS = ["live", "hardware", "analysis", "queue", "diag"];
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
      // Show/hide the middle-image card based on NumImages. The
      // compressed top row holds {img1, [mid?], img2, scan curve}
      // — col-4 each when 2 images (img1/img2/scan), or col-3 each
      // when 3 (img1/mid/img2/scan).
      const nImg = snap.num_images != null ? Number(snap.num_images) : 0;
      const hasMid = nImg >= 3;
      const cardMid    = $("card-array-mid");
      const card1      = $("card-array1");
      const card2      = $("card-array2");
      const cardScan   = $("card-scan-curve");
      const _swapCol = (el, from, to) => {
        if (!el) return;
        el.classList.remove(from);
        el.classList.add(to);
      };
      if (cardMid && card1 && card2) {
        if (hasMid) {
          cardMid.hidden = false;
          _swapCol(card1,   "col-4", "col-3");
          _swapCol(cardMid, "col-4", "col-3");
          _swapCol(card2,   "col-4", "col-3");
          _swapCol(cardScan,"col-4", "col-3");
        } else {
          cardMid.hidden = true;
          _swapCol(card1,   "col-3", "col-4");
          _swapCol(card2,   "col-3", "col-4");
          _swapCol(cardScan,"col-3", "col-4");
        }
      }
      // Site picker bounds + info block.
      const ns = Math.max(1, Number(snap.num_sites || 1));
      const picker = $("site-pick");
      if (picker) {
        picker.max = String(ns);
        if (Number(picker.value) > ns) picker.value = String(ns);
      }
      renderSiteInfo(snap, selectedSiteIdx);
    }
    // Mini queue: glanceable "what's running + what's next" tiles in
    // the Live status strip. Same /api/queue source as the Queue tab,
    // but only the bare facts.
    try {
      const q = await api("/api/queue");
      renderMiniQueue(q);
    } catch (e) { /* keep last-known state on transient error */ }
    // Plots come pre-built from the server's /api/live/figures.
    await pollLiveFigures();
    // Per-site hist refreshes too (separate endpoint with site index).
    await pollSiteHist();
  }

  function renderMiniQueue(q) {
    if (!q) return;
    const running = q.running;
    const queued  = q.queued || [];
    const aTile = $("mini-queue-active-tile");
    const nTile = $("mini-queue-next-tile");
    if (!aTile || !nTile) return;
    // Reset state classes.
    aTile.classList.remove("idle", "running", "loading");
    nTile.classList.remove("empty", "has-queue");
    if (running) {
      const label = running.label || running.seqName || `job #${running.id}`;
      let state = "running";
      if ((running.state || "").toLowerCase() === "building" ||
          (running.kind === "descriptor" && running.state !== "running")) {
        state = "loading";
      }
      aTile.classList.add(state);
      setText("mini-queue-active",
        `${state === "loading" ? "loading… " : ""}${label}`);
    } else {
      aTile.classList.add("idle");
      setText("mini-queue-active", "(idle)");
    }
    if (!queued.length) {
      nTile.classList.add("empty");
      setText("mini-queue-next", "(empty)");
    } else {
      nTile.classList.add("has-queue");
      const first = queued[0];
      const firstLabel = first.label || first.seqName || `#${first.id}`;
      const more = queued.length > 1 ? ` (+${queued.length - 1})` : "";
      setText("mini-queue-next", `${firstLabel}${more}`);
    }
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
    try { resp = await api("/api/live/figures"); }
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
      // Auto-pick a run for first paint: prefer the localStorage-remembered
      // selection (so a page refresh holds steady), fall back to the most
      // recent complete run. Only fires if no run is currently selected.
      if (!selectedScanId) {
        const remembered = (() => {
          try { return localStorage.getItem("yb_dashboard_selected_scan"); }
          catch { return null; }
        })();
        const exists = (sid) => runsCache.some((r) => r.scan_id === sid);
        let pick = remembered && exists(remembered) ? remembered : null;
        if (!pick && runsCache.length) {
          // List is newest-first.
          pick = runsCache[0].scan_id;
        }
        if (pick) {
          trayReplace(pick);
          loadAnalysis(pick);
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
  $("runs-refresh").addEventListener("click", loadRunsList);

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
    if (!opts.keepFilters) activeFilters = {};
    const body = $("analysis-detail-body");
    body.innerHTML = '<div class="hint">loading…</div>';
    ["plot-analysis-scan", "plot-site-loading", "plot-site-survival",
     "plot-site-fp", "plot-per-iter", "plot-per-iter-hist"].forEach(safePurge);
    try {
      let url = `/api/runs/${scanId}/analysis`;
      if (Object.keys(activeFilters).length) {
        url += '?filter=' + encodeURIComponent(JSON.stringify(activeFilters));
      }
      const r = await api(url);
      if (!r || typeof r !== "object") {
        throw new Error("server returned non-object response");
      }
      activeAnalysis = r;
      renderAnalysisDetail(r);
      renderAnalysisFilters(r);
      renderPerSiteMaps(r);
      renderPerIteration(r);
      renderSeqSpecific(r);
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
    const ta = r.target_aware || null;
    const targetAware = !!(summary.survival_source
                            || (ta && ta.overall_mean != null));
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
    body.innerHTML = warnHtml + `
      <div class="stat-grid">
        <div class="stat-tile">
          <span class="stat-label">scan_id</span>
          <span class="stat-value mono scan-id-copy" style="font-size:14px;"
                data-scan-id="${r.scan_id}"
                title="Click to copy">${r.scan_id}</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">params</span>
          <span class="stat-value">${r.n_params}</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">shots</span>
          <span class="stat-value">${r.n_shots}</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">${targetAware ? "TP (target)" : "survival"}${
            targetAware && summary.survival_source
              ? ` <span class="src-badge src-${
                  summary.survival_source.startsWith("lab") ? "lab" : "slm"
                }">${
                  summary.survival_source.startsWith("lab")
                    ? "lab"
                    : "SLM cache"
                }</span>`
              : ""
          }</span>
          <span class="stat-value" title="${targetAware ? "eligibility-weighted across all shots in the current filter" : "arithmetic mean over scan points"}">${
            targetAware && ta && ta.overall_mean != null
              ? fmtPct(ta.overall_mean)
              : fmtPct(avg(summary.survival_mean))
          }</span>
        </div>
        <div class="stat-tile">
          <span class="stat-label">loading</span>
          <span class="stat-value">${fmtPct(avg(summary.loading_rate))}</span>
        </div>
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
      <div style="margin-top:12px;font-size:12px;color:var(--text-dim);">
        sweep cols: <span class="mono">${(sweep.cols || []).join(", ") || "(none)"}</span>
        · code snapshot: <span class="${code.present ? "ok" : "muted"}">${code.present ? code.n_files + " files" : "none"}</span>
        · grid sidecar: <span class="${grid.present ? "ok" : "muted"}">${grid.present ? grid.n_sites + " sites" : "none"}</span>
      </div>
    `;
    // The dedicated Survival / Loading cards were replaced by the
    // per-site maps + per-iteration chart further down the tab. Only
    // the sweep visualization needs to render here.
    plotAnalysisScanCurve(r);
  }

  function plotAnalysisScanCurve(r) {
    const el = $("plot-analysis-scan");
    if (!el || !window.Plotly) return;
    const sweep = r.sweep || {};
    const summary = r.summary || {};
    const dims = sweep.dims || [];
    const cols = sweep.cols || [];
    const sm = summary.survival_mean || [];
    const lr = summary.loading_rate  || [];
    const useY = sm.length ? sm : lr;
    const useE = sm.length ? (summary.survival_sem || [])
                           : (summary.loading_rate_sem || []);
    // Phase 5a: relabel as "TP" (target-aware) when the survival
    // curve was overridden by the SLM cache / lab paths join.
    const targetAware = !!summary.survival_source;
    const yLabel = sm.length
        ? (targetAware ? "TP (target survival)" : "survival")
        : "loading rate";
    // Inset margins so plots breathe inside their card without
    // touching the borders (user asked for side padding).
    const baseMargin = {l: 70, r: 50, t: 14, b: 56};
    // 0d (no swept axis): aggregate line + annotation.
    if (dims.length === 0 || (dims.length === 1 && dims[0] === 1)) {
      const yMean = useY.length ? useY[0] : null;
      const yErr  = useE.length ? useE[0] : null;
      Plotly.react(el, [{
        x: [0, r.n_shots || 1],
        y: [yMean, yMean],
        mode: "lines",
        line: {color: "#58a6ff", width: 2, dash: "dash"},
        name: yLabel,
      }], plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: "shot # (0d, single point)" },
                 tickformat: ".0f" },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05],
                 tickformat: ".2f" },
        annotations: [{
          text: `${yLabel} = ${fmtNum(yMean, 3)}${yErr != null ? " ± " + fmtNum(yErr, 3) : ""}`,
          xref: "paper", yref: "paper", x: 0.5, y: 0.5,
          showarrow: false, font: { size: 14, color: "#ffdd44" },
          bgcolor: "rgba(20,20,40,0.7)",
        }],
      }), plotConfig());
      setText("analysis-scan-info",
        `0d · ${r.n_shots || 0} shots aggregated`);
      return;
    }
    // 1d: scatter with errors against the swept-param values.
    if (dims.length === 1) {
      const xs = (sweep.values && sweep.values[0]) || useY.map((_, i) => i + 1);
      const xLabel = cols[0] || "scan param";
      Plotly.react(el, [{
        x: xs, y: useY,
        error_y: useE.length ? {type: "data", array: useE, visible: true,
                                color: "#1f6feb"} : undefined,
        mode: "markers+lines",
        marker: {size: 8, color: "#58a6ff"},
        line:   {color: "#1f6feb", width: 2},
        hovertemplate: `${xLabel}=%{x:.4g}<br>${yLabel}=%{y:.3f}<extra></extra>`,
      }], plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: xLabel } },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05],
                 tickformat: ".2f" },
      }), plotConfig());
      setText("analysis-scan-info",
        `1d · ${xLabel} · ${dims[0]} pts`);
      return;
    }
    // 2d: heatmap on (dim0, dim1) grid.
    if (dims.length >= 2) {
      const xVals = (sweep.values || [[]])[0] || [];
      const yVals = (sweep.values || [[],[]])[1] || [];
      const n = dims[0] * dims[1];
      const z = [];
      for (let j = 0; j < dims[1]; j++) {
        const row = [];
        for (let i = 0; i < dims[0]; i++) {
          row.push(useY[j * dims[0] + i] ?? null);
        }
        z.push(row);
      }
      Plotly.react(el, [{
        z, x: xVals, y: yVals,
        type: "heatmap", colorscale: "Viridis", zmin: 0, zmax: 1,
        colorbar: {title: {text: yLabel}, len: 0.9, tickformat: ".2f"},
      }], plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: cols[0] || "x" } },
        yaxis: { title: { text: cols[1] || "y" } },
      }), plotConfig());
      setText("analysis-scan-info",
        `2d · ${cols.slice(0, 2).join(" × ")} · ${dims[0]}×${dims[1]} = ${n} pts`);
      return;
    }
  }

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

  function renderPerSiteMaps(r) {
    const ps = r.per_site;
    if (!ps) {
      ["plot-site-loading", "plot-site-survival", "plot-site-fp"].forEach((id) => {
        const el = $(id);
        if (el) Plotly.purge(el);
      });
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
    const finite = vIn.filter((v) => v != null && isFinite(v));
    const vmin = 0;
    const vmax = 1;
    const mean = finite.length ? finite.reduce((a, b) => a + b, 0) / finite.length : null;
    let infoTxt = mean != null ? `mean ${fmtNum(mean, 3)}` : "";
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
      ["plot-site-loading", "plot-site-survival", "plot-site-fp"]
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

  function renderPerIteration(r) {
    const pi = r.per_iteration;
    const toggleWrap = $("per-iter-toggles");
    if (!pi || !pi.shot_index || !pi.shot_index.length) {
      toggleWrap.innerHTML = '<span class="muted">no per-shot data</span>';
      Plotly.purge($("plot-per-iter"));
      setText("per-iter-info", "");
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
    setText("per-iter-info", `${pi.shot_index.length} shots`);
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
    if (!traces.length) {
      Plotly.purge(el);
      el.innerHTML =
        '<div class="hint" style="padding:24px;text-align:center;">enable a metric to plot</div>';
      return;
    }
    // Find the swept-param NAME (if any toggled-on metric is a param)
    // so the right Y axis can be labeled with the actual parameter.
    let rightLabel = "sweep value";
    metrics.forEach((m) => {
      if (perIterToggles[m.key] && m.key.startsWith("param:")) {
        rightLabel = m.paramName || rightLabel;
      }
    });
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
  }

  // =====================================================================
  // SEQUENCE-SPECIFIC ANALYSIS (placeholder).
  // Right now just shows the scan name + a "Coming soon" message.
  // Per-Seq custom analyses will be plugged in here.
  // =====================================================================
  function renderSeqSpecific(r) {
    setText("seq-specific-name", r.scan_name || "(unknown)");
    const body = $("seq-specific-body");
    if (!body) return;

    // Phase 5a: single-image scans (NumImages=1) get an averaged
    // camera image card. The per-site survival + paths panels are
    // empty for these scans (no second image), but the averaged
    // loading frame IS what the experimenter wants to see.
    const ai = r.avg_image;
    let html = '';
    if (ai && (ai.available || ai.computable)) {
      const sh = ai.image_shape || [];
      const dims = sh.length === 2 ? `${sh[0]}×${sh[1]}` : '';
      // Append the scan_id query so the browser refreshes when the
      // operator switches scans (and bypasses cache on the new run).
      const src = `/api/runs/${encodeURIComponent(r.scan_id || '')}`
                  + `/avg_image?v=${encodeURIComponent(r.scan_id || '')}`;
      const hint = ai.available
        ? `cached · ${ai.n_shots} shots averaged · ${dims}`
        : `will compute on first load (~${Math.round(0.03 * (ai.n_shots || 1))} s) ·`
          + ` ${ai.n_shots} shots · ${dims}`;
      html += `
        <div class="avg-image-card" style="padding:8px 0;">
          <h3 style="margin:0 0 8px;font-size:13px;color:var(--text-dim);
                     text-transform:uppercase;letter-spacing:0.5px;font-weight:600;">
            Averaged loading image
            <span class="muted mono" style="text-transform:none;margin-left:8px;
                                              letter-spacing:0;font-weight:400;">
              ${hint}
            </span>
          </h3>
          <div style="position:relative;background:#000;border-radius:4px;
                       overflow:hidden;max-width:780px;">
            <img src="${src}" alt="averaged loading image"
                 style="display:block;width:100%;height:auto;image-rendering:pixelated;"
                 onerror="this.style.display='none';this.nextElementSibling.style.display='block';">
            <div class="hint" style="display:none;padding:24px;text-align:center;">
              avg image fetch failed
            </div>
          </div>
        </div>`;
    }

    // Always include the placeholder text so future per-Seq panels
    // have a home. When avg_image renders above, this is just a tiny
    // muted footer.
    html += `
      <div style="padding:8px 0;font-family:var(--mono);font-size:11px;
                  color:var(--text-dim);">
        scan_name = ${escHtml(r.scan_name || "(none)")}
        ${ai ? '' : '· (no Seq-specific panel for this scan type yet)'}
      </div>`;
    body.innerHTML = html;
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
    // Multi-run: use an ephemeral group through the existing
    // /api/runs/groups/<id>/analysis aggregator. Create -> add -> analyze -> delete.
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
      renderAnalysisDetail(r2);
      setText("selected-scan-id", `tray:${traySet.size} runs`);
    } catch (e) {
      body.innerHTML = `<div class="hint bad">tray analysis failed: ${escHtml(e.message)}</div>`;
    } finally {
      if (gid) {
        try { await api(`/api/runs/groups/${gid}`, {method: "DELETE"}); } catch {}
        loadGroups();
      }
    }
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
        </tr>`).join("")
      || '<tr><td colspan="7" class="muted">empty</td></tr>';
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
      const inBody = t.closest(".runs-picker-body, .filter-body");
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

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
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
