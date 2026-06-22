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
    live:     500,
    // The coherent SINGLE-SHOT group (both camera frames + that frame's intensities +
    // the scan-result red current-cell outline) on its OWN loop, fetched together so
    // they always show ONE shot (pollSnapshot / group=snapshot). Gated by the ~5 MB
    // image payload, so it may skip shots -- but coherently. The fast `live` loop
    // streams everything else (maps/histograms) independently. setTimeout-after-
    // completion, so the real period is (fetch time + this gap) -- keep it small so the
    // shot view refreshes ~as fast as the images can stream.
    snapshot: 500,
    hardware: 10000,
    queue:    500,
    // Shared "Scans" picker (Analysis + Sequence tabs): cheap incremental list
    // refresh so it self-updates as scans complete. List-only (analysis re-run
    // is gated to shot growth), and the rebuild is skipped when unchanged.
    scans:    3000,
    molecube: 1500,
    // NI DAC monitor (Hardware/Molecube sub-view). Slower than molecube: each
    // poll spawns an engine-python subprocess to read the card, server-cached
    // ~5 s, so a tight loop would just re-serve the cache.
    nidaq: 6000,
    // Global control-panel refresh on NON-Live tabs (queue / runner / camera).
    // Live is covered by pollLive's 500ms; elsewhere this keeps it fresh.
    ctrlpanel: 1500,
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
  // Ultra-compact elapsed-since, no suffix: "5s" / "3m" / "2h" / "4d".
  const fmtAgoShort = (epoch) => {
    if (!epoch) return "";
    const s = (Date.now() / 1000 - epoch);
    if (s < 60)    return Math.round(s) + "s";
    if (s < 3600)  return Math.round(s / 60) + "m";
    if (s < 86400) return Math.round(s / 3600) + "h";
    return Math.round(s / 86400) + "d";
  };
  const fmtPct = (v) => (v == null || Number.isNaN(v))
    ? "—" : (100 * v).toFixed(1) + "%";
  const fmtNum = (v, n) => (v == null || Number.isNaN(v))
    ? "—" : v.toFixed(n || 2);
  // ~`sig` significant figures, trailing zeros trimmed (nice but informative).
  const fmtSig = (v, sig) => (v == null || !isFinite(v))
    ? "—" : (v === 0 ? "0" : String(Number(v.toPrecision(sig || 2))));
  // Error bar as a percentage at ~2 sig figs (e.g. 0.0123 → "1.2%").
  const errPct = (v) => (v == null || !isFinite(v)) ? null : fmtSig(100 * v, 2) + "%";
  const setText = (id, t) => { const el = $(id); if (el) el.textContent = t; };

  // ---- Discrimination-infidelity formatting (matches the LIVE view:
  // infidelity as a raw fraction in scientific notation, color-coded
  // green<1% / yellow<5% / red). See dashboard.py:_fig_infid. ----
  const fmtInfid = (v) => (v == null || Number.isNaN(v))
    ? "—" : Number(v).toExponential(1);
  // Fidelity = 1 − infidelity, as a percent to 1 decimal. Never reads 100%
  // unless infidelity is exactly 0 — anything above 99.9% is clamped to 99.9%
  // (e.g. infidelity 1.8e-2 → 98.2%, 4e-4 → 99.9%, 0 → 100.0%).
  const fmtFidelity = (v) => {
    if (v == null || Number.isNaN(v)) return "—";
    if (v <= 0) return "100.0%";
    let pct = 100 * (1 - v);
    if (pct > 99.9) pct = 99.9;
    if (pct < 0) pct = 0;
    return pct.toFixed(1) + "%";
  };
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
                       axisSwap: false, square: false, cbarAuto: false };
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
  const TABS = ["live", "analysis", "hardware", "logs"];
  // Hardware is a merged tab with four sub-views; "monitor"/"molecube"/"scope"
  // are no longer top tabs (they redirect into Hardware -- see setTab).
  const HW_SUBVIEWS = ["overview", "slm", "monitor", "molecube", "scope"];
  let activeTab = "live";
  // Analysis tab sub-view: "data" (the analysis cards) or "sequence" (the
  // flattened-sequence viewer folded in from the old Sequence tab).
  let analysisSubMode = "data";
  // Whether the loaded run has any swept axis (-> the Filter picker has chips
  // to show). renderAnalysisFilters sets it; updateAnalysisFloatingHosts uses
  // it so we don't pop an empty Filter card open just because we're in Data
  // sub-mode. (The Filter card moved into the shared LEFT column, so its
  // visibility is now decided centrally rather than by a lone wrap.hidden.)
  let _filterHasAxes = false;
  // Hardware tab sub-view: one of HW_SUBVIEWS (slm / monitor / molecube / scope).
  let hwSubview = "slm";

  function setTab(tab) {
    // The full-queue popup is a Live-tab overlay (fixed-position, outside
    // #tab-live); hide it on ANY tab switch so it doesn't float over other tabs.
    // This also covers clicking a popup run row (which calls setTab("analysis")).
    closeQueuePopup();
    // The Sequence view lives INSIDE the Analysis tab as a sub-mode; the
    // standalone Sequence tab was removed. This redirect keeps any lingering
    // setTab("sequence") / "#sequence" deep-link working.
    if (tab === "sequence") { setTab("analysis"); setAnalysisSubMode("sequence"); return; }
    // Monitor / Molecube / Scope merged into the Hardware tab as sub-views;
    // redirect their old top-tab names / "#..." deep-links into Hardware.
    if (tab === "monitor" || tab === "molecube" || tab === "scope") {
      setTab("hardware"); setHwSubview(tab); return;
    }
    if (!TABS.includes(tab)) return;
    activeTab = tab;
    // Expose the active tab on <html> so CSS can hide the page scrollbar on the
    // scrolling tabs (live / analysis / logs) -- it would otherwise sit over the
    // always-on right control panel. (Hardware is handled separately.)
    document.documentElement.dataset.tab = tab;
    TABS.forEach((t) => {
      $("tab-btn-" + t).classList.toggle("active", t === tab);
      $("tab-btn-" + t).setAttribute("aria-selected", String(t === tab));
      $("tab-" + t).hidden = (t !== tab);
    });
    // Floating overlays (Scans picker, Filter card, Channels picker) track the
    // active tab AND the Analysis sub-mode (Data vs Sequence). Centralized so
    // the sub-toggle and tab switch stay in sync.
    updateAnalysisFloatingHosts(tab);
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
    // Seed the <html> data-tab attribute for the default tab (no #hash case,
    // where setTab isn't called on load) so the scrollbar-hiding CSS applies
    // immediately.
    document.documentElement.dataset.tab = activeTab;
    TABS.forEach((t) => {
      $("tab-btn-" + t).addEventListener("click", () => setTab(t));
    });
    const hash = location.hash.replace("#", "");
    if (TABS.includes(hash)) setTab(hash);
  }

  // ---- Hardware tab sub-views (SLM / Monitor / Molecube / Scope) ----
  // One Hardware tab with a sub-tab bar. SLM/Monitor/Scope are full-bleed
  // iframes; Molecube is the native cards UI (reparented in from #tab-molecube
  // at load -- IDs preserved so all /api/molecube wiring still works).
  // Friendly labels for the Channels (molecube) cards (used as Overview tile labels).
  const MC_LABELS = { "mc-status": "Channels · Status", "mc-dds": "Channels · DDS",
                      "mc-ttl": "Channels · TTL", "mc-startup": "Channels · Startup",
                      "mc-nidaq": "NI DAC · Dev1" };

  function foldHardwareSubtabs() {
    const pane = document.getElementById("hw-pane-molecube");
    const src = document.getElementById("tab-molecube");
    if (pane && src) Array.from(src.children).forEach((c) => pane.appendChild(c));
    // ★ "add to Overview" on each molecube card. Molecube is native (not an
    // iframe), so the star toggles hwTiles directly (no postMessage). The tile
    // in the Overview is a live, read-only clone of the card (mirrorMolecubeTiles).
    if (pane) $$("section.card[data-card-id]", pane).forEach((card) => {
      const h2 = card.querySelector("h2");
      const id = card.dataset.cardId;
      if (!h2 || !id || h2.querySelector(".hw-mc-star")) return;
      const star = document.createElement("button");
      star.className = "hw-mc-star"; star.dataset.mcCard = id; star.textContent = "☆";
      star.title = "Star — add to the Hardware Overview";
      star.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleHwStar("molecube", id, MC_LABELS[id] || id);
      });
      h2.insertBefore(star, h2.firstChild);
    });
    refreshMcStars();
  }

  // Reflect which molecube cards are starred (★) vs not (☆).
  function refreshMcStars() {
    $$(".hw-mc-star").forEach((s) => {
      const on = hwTiles.some((t) => t.source === "molecube" && t.tab === s.dataset.mcCard);
      s.textContent = on ? "★" : "☆";
      s.classList.toggle("on", on);
    });
  }

  // A molecube Overview tile shows a LIVE clone of the real card (molecube is
  // native + single-instance, so we can't iframe it -- we mirror its DOM instead,
  // ID-stripped to avoid duplicate ids). It is INTERACTIVE: the mirror is wired
  // to the same DDS/TTL handlers (mcControlClick/Keydown), which are container-
  // relative so they work despite the stripped ids. Refreshed each molecube poll
  // + on (re)render -- but a mirror holding the focused field is left alone so we
  // don't wipe a value mid-edit (same idea as the real card's focus guard).
  function mirrorMolecubeTiles() {
    const grid = $("hw-overview-grid");
    if (!grid) return;
    $$(".hw-mc-mirror", grid).forEach((m) => {
      const card = document.querySelector(
        '#hw-pane-molecube section.card[data-card-id="' + m.dataset.mcCard + '"]');
      if (!card) { m.textContent = "(molecube card unavailable)"; return; }
      if (!m._mcWired) {   // wire once per mirror element (survives child rebuilds)
        m._mcWired = true;
        m.addEventListener("click", mcControlClick);
        m.addEventListener("keydown", mcControlKeydown);
        m.addEventListener("mousedown", mcControlMousedown);
        m.addEventListener("focusout", mcNameFocusOut);
      }
      if (m.contains(document.activeElement)) return;   // editing here -> don't clobber
      const clone = card.cloneNode(true);
      clone.querySelectorAll("[id]").forEach((e) => e.removeAttribute("id"));
      clone.querySelectorAll(".hw-mc-star").forEach((e) => e.remove());
      m.replaceChildren(clone);
    });
  }

  function setHwSubview(view) {
    if (!HW_SUBVIEWS.includes(view)) return;
    hwSubview = view;
    try { localStorage.setItem("ybHwSubview", view); } catch (e) {}
    HW_SUBVIEWS.forEach((v) => {
      const pane = document.getElementById("hw-pane-" + v);
      if (pane) pane.hidden = (v !== view);
    });
    const bar = document.querySelector(".hw-subtabs");
    if (bar) bar.querySelectorAll("button").forEach((b) => {
      const on = b.dataset.hwview === view;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", String(on));
    });
    // Molecube readback: poll immediately when it (or an Overview molecube tile)
    // becomes visible (the recurring poll is gated in startPolling).
    if (view === "molecube" && activeTab === "hardware") {
      refreshMcStars();
      try { pollMolecube().then(mirrorMolecubeTiles); } catch (e) {}
      // NI DAC monitor lives in this sub-view too (separate data path -- not molecube).
      try { pollNidaq().then(mirrorMolecubeTiles); } catch (e) {}
    }
    if (view === "overview") {
      renderHwOverview();   // also calls mirrorMolecubeTiles
      if (hwTiles.some((t) => t.source === "molecube") && activeTab === "hardware") {
        try { pollMolecube().then(mirrorMolecubeTiles); } catch (e) {}
      }
    }
  }

  function initHwSubtabs() {
    foldHardwareSubtabs();
    const bar = document.querySelector(".hw-subtabs");
    if (bar) bar.addEventListener("click", (e) => {
      // The inline ↗ opens that view's source in a new tab (no sub-view switch).
      const opener = e.target.closest(".hw-subtab-open");
      if (opener) {
        e.stopPropagation();
        const url = opener.closest("button").dataset.openUrl;
        if (url) window.open(url, "_blank", "noopener");
        return;
      }
      const btn = e.target.closest("button[data-hwview]");
      if (btn) setHwSubview(btn.dataset.hwview);
    });
    let saved = "overview";
    try { saved = localStorage.getItem("ybHwSubview") || "overview"; } catch (e) {}
    setHwSubview(HW_SUBVIEWS.includes(saved) ? saved : "overview");
    initHwOverview();   // load starred tiles + wire the cross-origin star protocol
  }

  // ---- Hardware Overview: starred tiles aggregated from the sub-views ----
  // Sources come from the #yb-hw-sources JSON (single source of truth). Each
  // iframe source emits ★ toggles over postMessage; molecube is native. The
  // tile list (order = layout) is persisted server-side via /api/hw/overview.
  const HW_SOURCES = (() => {
    try { return JSON.parse(document.getElementById("yb-hw-sources").textContent); }
    catch (e) { return {}; }
  })();
  // iframe element id <-> source id (for postMessage sync back to the embeds).
  const HW_SOURCE_IFRAME = { slm: "hw-iframe", monitor: "mon-iframe", scope: "scope-iframe" };
  let hwTiles = [];   // [{source, tab, label}] -- server-persisted; order == layout

  async function initHwOverview() {
    window.addEventListener("message", onHwTileMessage);
    try {
      const r = await api("/api/hw/overview");
      hwTiles = Array.isArray(r.tiles) ? r.tiles : [];
    } catch (e) { hwTiles = []; }
    renderHwOverview();
    broadcastHwStars();   // tell each embed which of its tabs are starred
    refreshMcStars();     // reflect starred state on the native molecube cards
  }

  async function saveHwTiles() {
    try {
      await api("/api/hw/overview", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tiles: hwTiles }),
      });
    } catch (e) { /* best-effort; next load re-syncs */ }
  }

  function hwTileIndex(source, tab) {
    return hwTiles.findIndex((t) => t.source === source && t.tab === tab);
  }
  function toggleHwStar(source, tab, label) {
    const i = hwTileIndex(source, tab);
    if (i >= 0) hwTiles.splice(i, 1);
    else hwTiles.push({ source, tab, label: label || tab });
    saveHwTiles();
    renderHwOverview();
    broadcastHwStars(source);
    refreshMcStars();   // keep the native molecube card ★ in sync
  }

  // Reply to a source embed (or all) with the list of ITS starred tabs, so the
  // embed renders ★/☆ correctly. Origin-agnostic post (the embeds filter on yb).
  function broadcastHwStars(only) {
    Object.keys(HW_SOURCE_IFRAME).forEach((source) => {
      if (only && source !== only) return;
      const ifr = $(HW_SOURCE_IFRAME[source]);
      if (!ifr || !ifr.contentWindow) return;
      const stars = hwTiles.filter((t) => t.source === source).map((t) => t.tab);
      try { ifr.contentWindow.postMessage({ yb: "hwtile", action: "sync", stars }, "*"); }
      catch (e) {}
    });
  }

  function _hwOriginTrusted(origin, source) {
    const base = (HW_SOURCES[source] || {}).base || "";
    return !origin || !base || base.indexOf(origin) === 0 || origin === location.origin;
  }
  function onHwTileMessage(e) {
    const d = e.data || {};
    if (!d || d.yb !== "hwtile") return;
    if (d.source && !_hwOriginTrusted(e.origin, d.source)) return;   // ignore foreign posts
    if (d.action === "ready") { broadcastHwStars(d.source); return; }
    if (d.action === "toggle" && d.source && d.tab) {
      toggleHwStar(d.source, String(d.tab), d.label);
    }
  }

  // CHROME-LESS, freely-placed, resizable tiles on an absolute canvas: just the
  // embedded view (no title bar / border). A hover top strip (the ⠿ grip + a
  // label hint) drags the tile to ANY x/y; a hover bottom-right corner resizes.
  // Both position (x,y) AND size (w,h) persist per tile in hwTiles. Drag a tile
  // onto the 🗑 (shown the moment you grab any tile) to remove it.
  function renderHwOverview() {
    const grid = $("hw-overview-grid");
    if (!grid) return;
    if (!hwTiles.length) {
      grid.innerHTML =
        '<div class="hw-ov-empty">No starred views yet.<br>Open a hardware sub-tab ' +
        '(SLM / Scope / Molecube) and click the ★ on a view to pin it here.<br>' +
        'Click a tile&rsquo;s top strip to open its full view; hold &amp; drag it to ' +
        'move (or onto the 🗑 to remove); drag the bottom-right corner to resize. ' +
        'Position + size are saved.</div>';
      grid.style.minHeight = "";
      return;
    }
    // Auto-place any tile with no saved position (newly starred), packing left
    // to right / top to bottom; explicit drags override this afterward.
    const gw = grid.clientWidth || 1200;
    let ax = 12, ay = 0, rh = 0, placed = false;   // ay=0: no top margin on the overview canvas
    hwTiles.forEach((t) => {
      if (!t.w) t.w = 440;
      if (!t.h) t.h = 300;
      if (t.x == null || t.y == null) {
        if (ax + t.w > gw - 12 && ax > 12) { ax = 12; ay += rh + 12; rh = 0; }
        t.x = ax; t.y = ay; ax += t.w + 12; rh = Math.max(rh, t.h); placed = true;
      }
    });
    grid.innerHTML = hwTiles.map((t) => {
      const src = HW_SOURCES[t.source] || {};
      const inner = src.native
        ? '<div class="hw-mc-mirror" data-mc-card="' + escHtml(t.tab) + '"></div>'
        : '<iframe class="hw-tile-frame" src="' +
            escHtml((src.base || "") + "?embed=1&tab=" + encodeURIComponent(t.tab)) +
            '" referrerpolicy="no-referrer" allow="fullscreen"></iframe>';
      const label = escHtml((src.label || t.source) + " · " + (t.label || t.tab));
      return '<div class="hw-tile" data-src="' + escHtml(t.source) + '" data-tab="' +
        escHtml(t.tab) + '" style="left:' + t.x + "px;top:" + t.y + "px;width:" +
        t.w + "px;height:" + t.h + 'px">' +
        '<div class="hw-tile-grip" title="Click to open this view · hold &amp; drag to move (or onto the trash to remove)">' +
          '<span class="hw-grip-dots">⠏</span>' +
          '<span class="hw-grip-label">' + label + "</span></div>" +
        inner +
        '<div class="hw-tile-resize" title="Drag to resize"></div></div>';
    }).join("");
    _hwUpdateCanvasHeight();
    wireHwTiles();
    mirrorMolecubeTiles();   // fill any molecube tiles with the current card content
    if (placed) saveHwTiles();   // persist the initial auto-placement
  }

  // Grow the canvas so tiles placed low are reachable by scrolling.
  function _hwUpdateCanvasHeight() {
    const grid = $("hw-overview-grid");
    if (!grid) return;
    let maxB = 0;
    hwTiles.forEach((t) => { if (t.y != null && t.h) maxB = Math.max(maxB, t.y + t.h); });
    // No bottom pad: canvas = exactly the lowest tile's bottom. With the grid's
    // CSS min-height:100%, a set of tiles that fits the pane needs no scroll at all.
    grid.style.minHeight = maxB ? maxB + "px" : "";
  }

  function _hwTileRec(tile) {
    return hwTiles.find((t) => t.source === tile.dataset.src && t.tab === tile.dataset.tab);
  }

  function wireHwTiles() {
    const grid = $("hw-overview-grid");
    if (!grid) return;
    $$(".hw-tile-grip", grid).forEach((g) => g.addEventListener("pointerdown", startHwDrag));
    $$(".hw-tile-resize", grid).forEach((h) => h.addEventListener("pointerdown", startHwResize));
  }

  // The grip is dual-purpose:
  //   * a quick CLICK opens that source's full Hardware sub-view (SLM / Monitor /
  //     Scope / Molecube) -- like clicking the sub-tab itself;
  //   * a press-and-HOLD (or a drag past a few px) enters rearrange mode and
  //     free-moves the tile.
  // Drag mode arms on EITHER a short hold timer OR movement past a threshold, so
  // a fast drag still feels immediate while a plain tap navigates. The grip is a
  // real (non-iframe) overlay, so we get the pointerdown; while dragging,
  // .hw-tile-frame is pointer-events:none (CSS via body.hw-dragging) so the move
  // glides over iframes. The 🗑 shows the moment drag mode arms; dropping onto it
  // removes the tile, else the new x/y persists.
  const HW_DRAG_HOLD_MS = 200;   // press-and-hold this long -> rearrange mode
  const HW_DRAG_MOVE_PX = 6;     // ...or move this far first
  function startHwDrag(e) {
    e.preventDefault(); e.stopPropagation();
    const tile = e.target.closest(".hw-tile");
    if (!tile) return;
    const x0 = e.clientX, y0 = e.clientY, l0 = tile.offsetLeft, t0 = tile.offsetTop;
    let dragging = false, holdTimer = 0;
    const arm = () => {
      if (dragging) return;
      dragging = true;
      if (holdTimer) { clearTimeout(holdTimer); holdTimer = 0; }
      document.body.classList.add("hw-dragging");   // reveal the trash
      tile.classList.add("hw-tile-dragging");
      tile.style.zIndex = "10";
    };
    holdTimer = setTimeout(arm, HW_DRAG_HOLD_MS);   // held in place -> rearrange
    const move = (ev) => {
      if (!dragging) {
        if (Math.abs(ev.clientX - x0) < HW_DRAG_MOVE_PX &&
            Math.abs(ev.clientY - y0) < HW_DRAG_MOVE_PX) return;
        arm();   // moved far enough -> rearrange
      }
      tile.style.left = Math.max(0, l0 + (ev.clientX - x0)) + "px";
      tile.style.top = Math.max(0, t0 + (ev.clientY - y0)) + "px";
    };
    const up = (ev) => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      if (holdTimer) { clearTimeout(holdTimer); holdTimer = 0; }
      if (!dragging) {
        // A plain click (no hold, no drag) -> open that source's sub-view.
        setHwSubview(tile.dataset.src);
        return;
      }
      // Hit-test the trash BEFORE hiding it: the 🗑 is display:none unless
      // body.hw-dragging is set, and a display:none element reports an all-zero
      // rect -- so the test must happen while the class is still on the body.
      const trash = $("hw-trash");
      let onTrash = false;
      if (trash) {
        const r = trash.getBoundingClientRect();
        onTrash = ev.clientX >= r.left && ev.clientX <= r.right &&
                  ev.clientY >= r.top && ev.clientY <= r.bottom;
      }
      document.body.classList.remove("hw-dragging");
      tile.classList.remove("hw-tile-dragging");
      tile.style.zIndex = "";
      if (onTrash) {
        toggleHwStar(tile.dataset.src, tile.dataset.tab);   // dropped on 🗑 -> remove
        return;
      }
      const t = _hwTileRec(tile);
      if (t) { t.x = tile.offsetLeft; t.y = tile.offsetTop; }
      _hwUpdateCanvasHeight();
      saveHwTiles();
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
  }

  // Custom resize (a corner handle z-above the iframe; native CSS resize can't be
  // grabbed over an iframe). pointer-events:none on the iframe during the drag so
  // it doesn't swallow the move. Size persisted on release.
  function startHwResize(e) {
    e.preventDefault(); e.stopPropagation();
    const tile = e.target.closest(".hw-tile");
    if (!tile) return;
    const x0 = e.clientX, y0 = e.clientY, w0 = tile.offsetWidth, h0 = tile.offsetHeight;
    document.body.classList.add("hw-resizing");
    const move = (ev) => {
      tile.style.width = Math.max(220, w0 + (ev.clientX - x0)) + "px";
      tile.style.height = Math.max(140, h0 + (ev.clientY - y0)) + "px";
    };
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      document.body.classList.remove("hw-resizing");
      const t = _hwTileRec(tile);
      if (t) { t.w = tile.offsetWidth; t.h = tile.offsetHeight; }
      _hwUpdateCanvasHeight();
      saveHwTiles();
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
  }

  // ---- Analysis Data/Sequence sub-view ----
  // The Sequence view is folded into the Analysis tab as a sub-mode. The run
  // summary card is shared; the Data and Sequence panes below it swap. The
  // Scans picker (left dock) is shared; the Filter card (right dock) is Data-
  // only; the Channels picker (right dock) is Sequence-only.

  // One-time: move the flattened-sequence viewer cards out of #tab-sequence
  // into the Analysis "Sequence" pane, so there is a single sequence view.
  // IDs are preserved (getElementById is location-independent), so all the
  // sequence wiring keeps working. #tab-sequence is left empty; its top-tab
  // button redirects into Analysis+Sequence (see setTab).
  function foldSequenceIntoAnalysis() {
    const pane = document.getElementById("analysis-sequence-pane");
    const seqTab = document.getElementById("tab-sequence");
    if (!pane || !seqTab) return;
    Array.from(seqTab.children).forEach((c) => pane.appendChild(c));
  }

  // Show/hide the three floating overlays for the current tab + sub-mode.
  function updateAnalysisFloatingHosts(tab) {
    const inAnalysis = (tab === "analysis");
    // All three pickers now live in the LEFT host (#floating-seqscan-host),
    // stacked. Show the host on the Analysis tab; toggle the per-mode CARDS
    // (Filter = Data sub-mode, Channels = Sequence sub-mode) individually.
    // Runs is always shown in Analysis.
    const scanHost = document.getElementById("floating-seqscan-host");
    const filtCard = document.getElementById("analysis-filters");
    const chanCard = document.getElementById("sequence-chn-card");
    if (scanHost) scanHost.hidden = !inAnalysis;
    // Filter shows only when in Analysis+Data AND the run actually has a swept
    // axis (otherwise there are no chips -- don't pop an empty card).
    if (filtCard) filtCard.hidden =
      !(inAnalysis && analysisSubMode === "data" && _filterHasAxes);
    if (chanCard) chanCard.hidden = !(inAnalysis && analysisSubMode === "sequence");
    // The old RIGHT-docked hosts are now empty (Filter + Channels moved left).
    const rFilt = document.getElementById("floating-analysis-host");
    const rChan = document.getElementById("floating-sequence-host");
    if (rFilt) rFilt.hidden = true;
    if (rChan) rChan.hidden = true;
  }

  function setAnalysisSubMode(mode) {
    if (mode !== "data" && mode !== "sequence") return;
    analysisSubMode = mode;
    try { localStorage.setItem("ybAnalysisSubMode", mode); } catch (e) {}
    const dataPane = document.getElementById("analysis-data-pane");
    const seqPane  = document.getElementById("analysis-sequence-pane");
    if (dataPane) dataPane.hidden = (mode !== "data");
    if (seqPane)  seqPane.hidden  = (mode !== "sequence");
    const tg = document.getElementById("analysis-mode-toggle");
    if (tg) tg.querySelectorAll("button").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", String(on));
    });
    updateAnalysisFloatingHosts(activeTab);
    // Lazily load the view that just became visible.
    if (mode === "sequence") { try { loadSequence(); } catch (e) {} }
    else { try { ensureAnalysisShown(); } catch (e) {} }
    // Re-fit Plotly plots now that their container is visible.
    setTimeout(() => {
      if (!window.Plotly) return;
      const pane = mode === "sequence" ? seqPane : dataPane;
      if (pane) $$(".plot-container", pane).forEach((el) => {
        try { Plotly.Plots.resize(el); } catch (e) {}
      });
    }, 60);
  }

  function initAnalysisModeToggle() {
    const tg = document.getElementById("analysis-mode-toggle");
    if (tg) tg.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => setAnalysisSubMode(b.dataset.mode));
    });
    // Restore the last-used sub-view across reloads (default Data).
    let saved = "data";
    try { saved = localStorage.getItem("ybAnalysisSubMode") || "data"; } catch (e) {}
    setAnalysisSubMode(saved === "sequence" ? "sequence" : "data");
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
    if (tab === "live")     { pollSnapshot(); return pollLive(); }   // coherent shot on switch-in; stream now
    // Hardware tab: the SLM/Monitor/Scope sub-views are self-driving iframes;
    // only the native Molecube sub-view needs a lab-side poll.
    if (tab === "hardware") { if (hwSubview === "molecube") return pollMolecube(); return; }
    // (The Queue tab was removed; the full-queue popup is driven by pollLive.)
    if (tab === "logs")     return pollLogs();
    if (tab === "sequence") return loadSequence();
    // Analysis: refresh the (cheap, incremental) runs list and sync the view to
    // the current selection. We do NOT unconditionally re-analyze here -- that's
    // the item-2 fix: ensureAnalysisShown() is a no-op when the selection is
    // already on screen, and loadRunsList re-analyzes the selected run only when
    // its shot count grew (a live scan). So a reload/poll has no delay.
    if (tab === "analysis") {
      loadRunsList();
      ensureAnalysisShown();
      return;
    }
  }

  // Hook the iframe reload button now that the DOM is ready (wired
  // at bootstrap, not here).

  function startPolling() {
    // gate() (optional) overrides the default "this tab is active" check -- used
    // by the Molecube sub-view, which is active only on Hardware + its sub-tab.
    function loop(tab, fn, interval, gate) {
      const tick = async () => {
        if (autoRefresh && (gate ? gate() : activeTab === tab)) {
          try { await fn(); } catch (e) { console.warn(tab, e); }
        }
        timers[tab] = setTimeout(tick, interval);
      };
      tick();
    }
    loop("live",     pollLive,     POLL.live);
    // The coherent single-shot group (frames + intensities + scan red-outline) on its
    // own loop, gated to the Live tab. Fetched together so they always show ONE shot;
    // the fast `live` loop above streams the aggregate maps/histograms independently.
    loop("snapshot", pollSnapshot,  POLL.snapshot,   () => activeTab === "live");
    // Hardware tab is a self-contained iframe to the SLM dashboard --
    // it owns its own polling. We DON'T poll /api/slm/* here, or we
    // get null-querySelector crashes against UI elements that only
    // existed in the old manual-port version of this tab.
    // (The Queue tab was removed; the full-queue popup refreshes via pollLive.)
    // Analysis + Sequence: keep the shared "Scans" picker self-updating as new
    // scans complete. loadRunsList is the cheap incremental cached path; it's
    // list-only (the analysis re-run is gated to actual shot growth), so this
    // doesn't churn the analysis. Skips the DOM rebuild when nothing changed.
    loop("analysis", loadRunsList, POLL.scans);
    loop("sequence", loadRunsList, POLL.scans);
    // Logs tab: re-list + refresh the open file every 5 s so an open log
    // "follows" a running server (tail mode auto-scrolls to the bottom).
    loop("logs",     pollLogs,     5000);
    // Molecube is a Hardware sub-view now: poll only when it's the active one.
    // Molecube polls when its sub-view is open OR the Overview has molecube
    // tile(s) to keep live; mirror the cards into those tiles after each poll.
    loop("molecube",
         async () => { await pollMolecube(); mirrorMolecubeTiles(); },
         POLL.molecube,
         () => activeTab === "hardware" &&
               (hwSubview === "molecube" ||
                (hwSubview === "overview" && hwTiles.some((t) => t.source === "molecube"))));
    // NI DAC monitor: same gate as molecube (it shares the sub-view) but its own
    // slower cadence. Not via molecube -- a separate /api/nidaq/monitor poll.
    loop("nidaq",
         async () => { await pollNidaq(); mirrorMolecubeTiles(); },
         POLL.nidaq,
         () => activeTab === "hardware" &&
               (hwSubview === "molecube" ||
                (hwSubview === "overview" && hwTiles.some((t) => t.source === "molecube"))));
    // The "Yb Control" sidebar is shown on EVERY tab now. pollLive refreshes it
    // on the Live tab; this loop keeps its queue / runner-state / camera live on
    // the OTHER tabs. Gated to non-Live so the two never double-poll.
    loop("ctrlpanel", pollControlPanel, POLL.ctrlpanel, () => activeTab !== "live");
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

  // Current run's scan_id (from /api/snapshot), so the scan_id / scan_name
  // status chips can open it in Analysis -- the same row-click behaviour the
  // Queue tab has. Set each live poll in pollLive.
  let _liveScanId = null;
  ["tile-scan-id", "tile-scan-name"].forEach((id) => {
    const tile = document.getElementById(id);
    if (!tile) return;
    tile.addEventListener("click", () => {
      if (!_liveScanId) return;
      selectPrimary(_liveScanId);   // sets primary + loads analysis (background)
      setTab("analysis");           // then reveal it
    });
  });

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
    ["avghist",   "plot-hist-avg"],
    ["rep0",      "plot-hist-rep0"],
    ["rep1",      "plot-hist-rep1"],
    ["rep2",      "plot-hist-rep2"],
    // rep3 (a 2nd "random") was dropped: the right-half histogram grid is now
    // best / worst / random + the selected site (#plot-hist-site).
  ];

  // The camera/tweezer-array panels bake the frame as a Plotly layout image
  // (a data: URI). If the browser can't decode that image, Plotly leaves the
  // broken-image "frowny" glyph on a white box -- which breaks the dashboard's
  // dark format. These panels get validated before rendering (see
  // renderArrayPanel) and degrade to a clean "no data" panel instead.
  const ARRAY_PANELS = new Set(["array", "array_mid", "array2"]);
  // Per-panel image-validation cache: divId -> {sig, ok}. Lets a poll skip
  // re-probing an unchanged frame (same baked data URI string).
  const _arrayImgState = {};

  // Two live fetch groups (see POLL.snapshot / POLL.live). SNAPSHOT = the coherent
  // single-shot view: both camera frames + THAT frame's per-site intensities + the
  // scan-result panel (red current-cell outline). Fetched + rendered together
  // (pollSnapshot / group=snapshot) so they always show ONE shot. STREAM = everything
  // else (survival/loading maps, histograms) -- aggregate data that just accumulates,
  // streamed fast + independently (pollLive / group=stream). Derived from
  // LIVE_FIG_PANELS so the split stays in sync if a panel is added/removed above.
  const SNAPSHOT_NAMES = new Set(["array", "array_mid", "array2", "intens", "scan"]);
  const SNAPSHOT_FIG_PANELS = LIVE_FIG_PANELS.filter(([name]) => SNAPSHOT_NAMES.has(name));
  const STREAM_FIG_PANELS   = LIVE_FIG_PANELS.filter(([name]) => !SNAPSHOT_NAMES.has(name));

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
    // Queue first: the status strip's "shot # / total" + "shot length" tiles
    // read the running entry's scheduled total + start_ts. renderSidebarQueue
    // also sets queueActiveRunning, which pollLiveDiag needs further down.
    let q = null;
    try {
      q = await api("/api/queue");
      renderSidebarQueue(q);
      if (queuePopupOpen) renderQueuePopup(q);   // keep the full-queue popup live
    } catch (e) { /* keep last-known state on transient error */ }
    const running = q && q.running;

    // Status card uses /api/snapshot (small fields).
    let snap = null;
    try { snap = await api("/api/snapshot"); } catch (e) {
      console.warn("snapshot failed", e);
    }
    if (snap) {
      setText("kv-scan-id",    snap.scan_id != null ? String(snap.scan_id) : "—");
      setText("kv-scan-name",  snap.scan_name || snap.scan_filename || "—");
      // Remember the current run's scan_id so the scan_id / scan_name chips can
      // open it in Analysis (like a queue row). The snapshot is the reliable
      // source -- the queue's running entry often has no scan_id yet.
      _liveScanId = snap.scan_id != null ? String(snap.scan_id) : null;
      ["tile-scan-id", "tile-scan-name"].forEach((id) => {
        const t = $(id);
        if (t) t.classList.toggle("is-linked", !!_liveScanId);
      });
      // shot # / total = did / supposed-to-do: numerator = the per-run shot
      // stamp (actual shots so far); denominator = the scan's PLANNED total
      // (nseqs x StackNum, StackNum honoring an explicit rep -- see
      // scan_summary.build_descriptor_summary). total_per_group is the correct
      // plan (num_per_group is often a run-until-stopped sentinel). null total
      // (run-forever, no finite plan) -> show just the actual count.
      const cur = snap.shots_this_run ?? snap.n_accum_shots ?? null;
      const total = (running && running.summary && running.summary.total_per_group)
        ? running.summary.total_per_group : null;
      setText("kv-shot",
        cur == null ? "—" : (total ? `${cur} / ${total}` : String(cur)));
      // avg shot length (s) = elapsed since the run started / shots so far.
      // Replaces the old num_sites tile (deprecated: a scan can span multiple
      // loading patterns across images, so one site count is meaningless).
      let shotLen = null;
      if (running && running.start_ts && cur) {
        const elapsed = (running.finish_ts || Date.now() / 1000) - running.start_ts;
        if (elapsed > 0) shotLen = elapsed / cur;
      }
      setText("kv-shot-length", shotLen == null ? "—"
        : (shotLen >= 60 ? fmtDur(shotLen)
           : shotLen.toFixed(shotLen < 10 ? 2 : 1) + "s"));
      const lr = avg(snap.loading_rates);
      setText("kv-loading-rate", lr != null ? fmtPct(lr) : "—");
      // Overall survival (per-shot TP) averaged over all recent shots this run,
      // replacing the old "progress" tile (which just echoed the shot #). Blank
      // for 1-image scans (no survival; loading has its own tile). NOTE:
      // survival_history is an OBJECT {target_aware, values:[...]}, not a bare
      // array -- the per-shot fractions live under .values.
      const survSeries = snap.survival_history;
      const survVals = (survSeries && Array.isArray(survSeries.values)
        ? survSeries.values : []).filter((v) => Number.isFinite(v));
      const survMean = survVals.length
        ? survVals.reduce((a, b) => a + b, 0) / survVals.length : null;
      setText("kv-survival", survMean != null ? fmtPct(survMean) : "—");
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
      renderSaveHealthChip(snap.save_health);
      renderSeqGapChip(snap.seq_reconciliation);
      // img2-panel detector badge: when img2 is detected by the spot-shape
      // model (distinct-pattern runs) show the model + mean % certainty;
      // otherwise leave it blank (img2 used the intensity threshold).
      const info2 = $("array2-info");
      if (info2) {
        const src2 = snap.logicals2_source;
        if (src2) {
          const c = snap.logicals2_certainty_mean;
          const label = String(src2).replace("gmm_shape_model_", "GMM ");
          info2.textContent = "det: " + label
            + (c != null && isFinite(c) ? " · ⌀" + fmtPct(c) + " cert" : "");
          info2.title = "img2 detected by the spot-shape GMM model (not an "
            + "intensity threshold). ⌀ = mean per-site P(loaded); the full "
            + "per-site certainties are stored as certainties_img2 in the .h5.";
        } else {
          info2.textContent = "";
          info2.title = "";
        }
      }
    }
    // (Queue already fetched + rendered at the top of pollLive so the status
    // strip could read the running entry; queueActiveRunning is set there.)
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
    // Affine-alignment card (small JSON; re-renders only when it changes).
    await pollAffine();
    // Detection-thresholds & calibration card (per-pattern audit log + health).
    await pollThresholds();
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

    // Abort-button indicator: a next scan is waiting → arrow (abort advances
    // to it); nothing queued → red ✗ (abort fully stops the experiment).
    const abInd = $("abort-ind");
    if (abInd) {
      const n = queued.length;
      if (n > 0) {
        abInd.textContent = "↪";
        abInd.className = "abort-ind ind-next";
        abInd.title = `${n} scan${n > 1 ? "s" : ""} queued — abort advances to the next`;
      } else {
        abInd.textContent = "✗";
        abInd.className = "abort-ind ind-stop";
        abInd.title = "Nothing queued — abort stops the experiment";
      }
    }

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
        // Click the active line to open the running scan in Analysis (it carries
        // a scan_id once it starts producing data). Mirrors the Queue-tab rows.
        if (running.scan_id) {
          actEl.dataset.scanId = running.scan_id;
          actEl.classList.add("q-linked");
          actEl.title = "Open this run in Analysis";
        } else {
          delete actEl.dataset.scanId;
          actEl.classList.remove("q-linked");
          actEl.removeAttribute("title");
        }
      } else {
        actEl.dataset.state = "idle";
        actText.textContent = "(idle)";
        queueActiveRunning = false;
        delete actEl.dataset.scanId;
        actEl.classList.remove("q-linked");
        actEl.removeAttribute("title");
      }
    }

    // Re-queue button(s) for a row, mirroring the Queue tab: "↻" replays the
    // entry's stored descriptor (same params); "+code" (history rows with a
    // code snapshot) also pins the original run's captured experiment code.
    // Only entries that carry a descriptor can be re-queued.
    const rqBtns = (e, withCode) => {
      if (!e || !e.descriptor) return "";
      let h = `<button class="q-rq" data-sb-requeue="${e.id}"` +
              ` title="Re-queue a copy with the same parameters">↻</button>`;
      if (withCode && e.file_id) {
        h += `<button class="q-rq" data-sb-requeue-code="${e.id}"` +
             ` title="Re-queue with the EXACT original code (replays this run's code snapshot)">+code</button>`;
      }
      return h;
    };

    // --- Preview list: next-up queued + recent history ---
    const listEl = $("ctrl-queue-list");
    if (!listEl) return;
    const rows = [];
    // A linked row (carries a scan_id) opens that run in Analysis on click —
    // same affordance as the Queue tab. Stamped via data-scan-id + q-linked.
    const linkAttrs = (e) =>
      e && e.scan_id
        ? ` data-scan-id="${e.scan_id}" class="q-linked` : ' class="';
    queued.slice(0, SIDEBAR_QUEUE_MAX).forEach((e) => {
      const label = e.label || e.seqName || `#${e.id}`;
      rows.push(
        `<li${linkAttrs(e)} q-queued"><span class="q-mark">·</span>` +
        `<span class="q-name">${escHtml(label)}</span>` +
        `<span class="q-status">#${e.id}</span>` +
        rqBtns(e, false) + `</li>`);
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
        // The +/× mark already conveys ok-vs-error, so the status slot shows
        // how long ago the run STARTED (start_ts) in compact form ("5m"/"2h"/
        // "3d"); the full status text rides along as the hover title.
        const ago = fmtAgoShort(e.start_ts);
        const statusTxt = ok ? "ok" : (e.status || e.state || "err");
        rows.push(
          `<li${linkAttrs(e)} q-history ${ok ? "q-ok" : "q-err"}">` +
          `<span class="q-mark">${mark}</span>` +
          `<span class="q-name">${escHtml(label)}</span>` +
          `<span class="q-status" title="${escHtml(statusTxt)}">` +
          `${escHtml(ago)}</span>` +
          rqBtns(e, true) + `</li>`);
      });
    }
    listEl.innerHTML = rows.length
      ? rows.join("")
      : `<li class="ctrl-queue-empty">queue empty</li>`;
  }

  // Re-queue an entry straight from the Yb Control sidebar queue preview
  // (mirrors the Queue tab's re-queue / "+code"). Replays the stored
  // descriptor; withCode also pins the source run's captured code snapshot.
  async function sidebarRequeue(btn, id, withCode) {
    if (!id) return;
    if (!controlsAllowed) {
      toast("Remote controls disabled on this interface", "bad");
      return;
    }
    btn.disabled = true;
    try {
      const r = await api(`/api/queue/requeue/${id}${withCode ? "?code=1" : ""}`,
                          {method: "POST"});
      toast(`Re-queued #${id} → #${r.descriptor_id}${withCode ? " (orig code)" : ""}`);
      // Refresh the sidebar preview now; refresh the full-queue popup too if open.
      try { renderSidebarQueue(await api("/api/queue")); } catch (e) { /* next poll */ }
      try { if (queuePopupOpen) refreshQueuePopup(); } catch (e) { /* next pollLive */ }
    } catch (e) {
      toast("Re-queue failed: " + (e.message || e), "bad");
      btn.disabled = false;
    }
  }

  // Open a sidebar queue/history row's run in Analysis — mirrors the Queue
  // tab's row-click. The element carries data-scan-id (server-stamped from
  // file_id); buttons inside it act on their own and are skipped by the caller.
  function sidebarOpenAnalysis(sid) {
    if (!sid) return;
    selectPrimary(sid);   // sets primary + loads analysis (background)
    setTab("analysis");   // then reveal it
  }

  // Delegated click handler for the sidebar queue preview. Only the per-row
  // re-queue / +code buttons act here (distinct sb-requeue attrs avoid clashing
  // with the Queue tab's document-wide [data-requeue] wiring). Clicking anywhere
  // ELSE in the queue section opens the full-queue popup -- wired in
  // wireQueuePopup() on #ctrl-queue-section, which this handler defers to by
  // returning without stopping propagation.
  (function wireSidebarQueue() {
    const listEl = document.getElementById("ctrl-queue-list");
    if (listEl) {
      listEl.addEventListener("click", (e) => {
        const rc = e.target.closest("[data-sb-requeue-code]");
        if (rc) { sidebarRequeue(rc, rc.dataset.sbRequeueCode, true); return; }
        const rq = e.target.closest("[data-sb-requeue]");
        if (rq) { sidebarRequeue(rq, rq.dataset.sbRequeue, false); return; }
        // Anything else falls through to the #ctrl-queue-section handler,
        // which opens the popup.
      });
    }
  })();

  function renderSiteInfo(snap, idx1) {
    const info = $("site-info");
    if (!info || !snap) return;
    const i = idx1 - 1;
    const t = (snap.thresholds || [])[i];
    const inf = (snap.infidelities || [])[i];
    const rate = (snap.loading_rates || [])[i];
    const lines = [
      `site ${idx1}`,
      `thr ${t != null ? fmtNum(t) : "—"}`,
      `load ${rate != null ? fmtPct(rate) : "—"}`,
      `infid ${inf != null ? inf.toExponential(2) : "—"}`,
    ];
    // Single compact line (the readout now lives inline in the per-site
    // histogram card's selection cell); full detail shows on hover.
    const txt = lines.join("  ·  ");
    info.textContent = txt;
    info.title = txt;
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
    renderShotHealthChip(st.state, st.shot_health);

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

  // Lightweight refresh of the GLOBAL control-panel sidebar (queue preview,
  // runner state, backend, dummy mode, camera) for tabs OTHER than Live. The
  // Live poll (pollLive) already does all of this plus the heavy live figures;
  // this is the subset the sidebar needs so it stays current on Analysis /
  // Hardware / Logs without dragging in the live-image pipeline.
  async function pollControlPanel() {
    if (activeTab === "live") return;   // pollLive owns the Live tab
    try {
      const q = await api("/api/queue");
      renderSidebarQueue(q);
      if (queuePopupOpen) renderQueuePopup(q);
    } catch (e) { /* keep last-known on transient error */ }
    try {
      const snap = await api("/api/snapshot");
      if (snap) renderControlSidebar(snap);
    } catch (e) { /* keep last-known */ }
    await pollControlStatus();
    await pollCameraStatus();
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

  // Per-shot health chip (status strip). Three lit states + a neutral one:
  //   green  "ok"   -> no shot errors recorded this scan
  //   yellow "warn" -> errors happened earlier but we're NOT currently failing
  //                    (recovered, or the count stopped climbing, or idle now)
  //   red    "fail" -> ACTIVELY failing: state is Running AND the per-scan error
  //                    count is still climbing
  //   grey   "none" -> backend reports no health info (e.g. MATLAB backend)
  // "Currently failing" keys off a CLIMBING count, not the server's
  // seconds_since_last: /api/control/status serves the LAST-PUBLISHED snapshot
  // from a temp file, so during a backend restart (or any publish stall) it
  // re-serves a frozen snapshot whose seconds_since_last never grows and would
  // read "recent" forever. A frozen total simply never increases -> never red.
  const SHOT_FAIL_WINDOW_S = 20;
  let _shotFail = { scanId: null, total: 0, lastClimbMs: 0 };
  // Last shot_health payload, so the click handler (open-logs) can jump to the
  // most recent error message when the chip is in a failed/failing state.
  let _lastShotHealth = null;
  function renderShotHealthChip(state, sh) {
    const tile = $("shot-health-tile");
    if (!tile) return;
    _lastShotHealth = (sh && typeof sh === "object") ? sh : null;
    if (!sh || typeof sh !== "object") {
      tile.dataset.state = "none";
      setText("shot-health-readout", "—");
      return;
    }
    const total = sh.total || 0;
    const scanId = sh.scan_id != null ? String(sh.scan_id) : null;
    if (scanId !== _shotFail.scanId) {
      // New scan (or first sample): start delta tracking fresh -- don't
      // retro-flag a count we've only just begun observing.
      _shotFail = { scanId: scanId, total: total, lastClimbMs: 0 };
    } else if (total > _shotFail.total) {
      _shotFail.total = total;
      _shotFail.lastClimbMs = Date.now();
    }
    const running = statusToState(state) === "running";
    const climbing = _shotFail.lastClimbMs > 0 &&
      (Date.now() - _shotFail.lastClimbMs) < SHOT_FAIL_WINDOW_S * 1000;
    const since = sh.seconds_since_last, sinceOk = sh.seconds_since_ok;
    const recovered = sinceOk != null && since != null && sinceOk < since;
    const activelyFailing = running && total > 0 && climbing && !recovered;

    let stateCls, text;
    if (activelyFailing)  { stateCls = "fail"; text = `failing · ${total}`; }
    else if (total > 0)   { stateCls = "warn"; text = `${total} failed`; }
    else                  { stateCls = "ok";   text = "ok"; }
    tile.dataset.state = stateCls;
    setText("shot-health-readout", text);
    tile.title = (total > 0 && sh.last_message)
      ? `Per-shot health — last error: ${sh.last_message}`
      : "Per-shot health: green = no failures · yellow = failed earlier, not now · red = currently failing";
  }

  // HDF5 save-health chip (status strip). Monitor-side, from /api/snapshot's
  // save_health (DataManager._save_health). The save runs in a daemon thread;
  // if append_block ultimately fails (e.g. a OneDrive lock that outlasts the
  // retries) a block of shots is lost from disk. States:
  //   green  "ok"        -> every block saved
  //   yellow "recovered" -> a block was lost earlier but saves are flowing now
  //   red    "fail"      -> a save just failed (shots being lost to disk)
  //   grey   "none"      -> no DataManager / no save info yet
  function renderSaveHealthChip(sh) {
    const tile = $("save-health-tile");
    if (!tile) return;
    if (!sh || typeof sh !== "object") {
      tile.dataset.state = "none";
      setText("save-health-readout", "—");
      tile.title = "HDF5 save health: green = all saved · yellow = recovered after an earlier loss · red = saves failing (shots lost to disk)";
      return;
    }
    const lost = sh.lost_seqs || 0;
    const state = sh.state || "ok";
    let stateCls, text;
    if (state === "fail")          { stateCls = "fail"; text = lost ? `lost ${lost}` : "failing"; }
    else if (lost > 0)             { stateCls = "warn"; text = `lost ${lost}`; }   // recovered
    else                           { stateCls = "ok";   text = "ok"; }
    tile.dataset.state = stateCls;
    setText("save-health-readout", text);
    tile.title = sh.reason
      ? `HDF5 save — ${sh.reason}`
      : "HDF5 save health: green = all saved · yellow = recovered after an earlier loss · red = saves failing (shots lost to disk)";
  }

  // ZMQ delivery gap chip: detects seq_ids that were never received because
  // a grab_imgs() poll timeout raced a large buffer drain from the server.
  //   green  "ok"       -> no missing shots this scan
  //   yellow "lost N"   -> N shots inferred missing from seq_id gap(s)
  //   grey   "none"     -> no DataManager yet
  function renderSeqGapChip(sr) {
    const tile = $("seq-gap-tile");
    if (!tile) return;
    if (!sr || typeof sr !== "object") {
      tile.dataset.state = "none";
      setText("seq-gap-readout", "—");
      tile.title = "ZMQ delivery: green = no missing shots -- yellow = shots lost to timeout this scan";
      return;
    }
    const n = sr.gap_count || 0;
    if (n > 0) {
      tile.dataset.state = "warn";
      setText("seq-gap-readout", "lost " + n);
      const ids = (sr.gap_ids || []).slice(0, 10).join(", ");
      tile.title = "ZMQ delivery gap: " + n + " shot(s) not delivered this scan (get_imgs timeout when large batch accumulated). First missing: " + ids;
    } else {
      tile.dataset.state = "ok";
      setText("seq-gap-readout", "ok");
      tile.title = "ZMQ delivery: no missing shots detected this scan";
    }
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
    // group=stream: the aggregate maps + histograms (survival/loading/infidelity + the
    // hist panels). These just accumulate, so they stream fast + independently here.
    // The coherent single-shot group (frames + intensities + scan red-outline) is
    // fetched separately by pollSnapshot so those always show ONE shot.
    try { resp = await api("/api/live/figures?group=stream"); }
    catch (e) {
      console.warn("stream figures fetch failed", e);
      STREAM_FIG_PANELS.forEach(([name, divId]) => {
        const el = $(divId);
        if (el && !el.querySelector(".plotly")) {
          el.innerHTML =
            '<div class="hint" style="padding:24px;text-align:center;color:#f85149;">' +
            'fetch failed: ' + escHtml(e.message || String(e)) + '</div>';
        }
      });
      return;
    }
    renderFigurePanels((resp && resp.figures) || {}, STREAM_FIG_PANELS);
  }

  // The coherent SINGLE-SHOT group: both camera frames + THAT frame's per-site
  // intensities + the scan-result panel (red current-cell outline). Fetched TOGETHER
  // (group=snapshot -> one server-side _read_data snapshot) so they always show ONE
  // shot -- never img2 ahead of img1, nor the red outline ahead of the images. On its
  // own loop (POLL.snapshot), gated by the ~5 MB image payload, so it may skip shots
  // but stays coherent. Autoscale (scan colorbar) rides this fetch -- scan lives here.
  async function pollSnapshot() {
    if (!window.Plotly || window.__plotlyLoadFailed) return;   // the stream loop owns the missing-Plotly banner
    let resp = null;
    const url = scanAutoscale
      ? "/api/live/figures?group=snapshot&cbar_scale=auto"
      : "/api/live/figures?group=snapshot";
    try { resp = await api(url); }
    catch (e) {
      console.warn("snapshot figures fetch failed", e);
      SNAPSHOT_FIG_PANELS.forEach(([name, divId]) => {
        const el = $(divId);
        if (el && !el.querySelector(".plotly")) {
          el.innerHTML =
            '<div class="hint" style="padding:24px;text-align:center;color:#f85149;">' +
            'fetch failed: ' + escHtml(e.message || String(e)) + '</div>';
        }
      });
      return;
    }
    renderFigurePanels((resp && resp.figures) || {}, SNAPSHOT_FIG_PANELS);
  }

  // Render a batch of {name: figure} into the given [name, divId] panels. Shared by
  // the light (per-shot curves) and heavy (camera images) live loops above.
  function renderFigurePanels(figures, panels) {
    panels.forEach(([name, divId]) => {
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
      // Camera/array panels: validate the baked frame image loads before
      // showing it, so a broken/undecodable data URI becomes a clean dark
      // "no data" panel instead of the browser's broken-image frowny glyph.
      if (ARRAY_PANELS.has(name)) {
        renderArrayPanel(el, divId, f);
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

  // Render a camera/array panel, guarding against a baked layout image that
  // the browser can't load. The frame is baked into f.layout.images[0].source
  // as a data: URI; if it's missing/empty/undecodable the browser would show
  // the broken-image "frowny" glyph, so we substitute a clean "no data" panel.
  function renderArrayPanel(el, divId, f) {
    const layout = f.layout || {};
    const cfg = { displayModeBar: false, responsive: true };
    const imgs = layout.images;
    const hasImgEntry = Array.isArray(imgs) && imgs.length > 0;
    // No image baked -> the server already returned a clean placeholder
    // (annotation-only "Waiting for data..." figure); render it as-is.
    if (!hasImgEntry) {
      try { Plotly.react(el, f.data, layout, cfg); } catch (e) {}
      return;
    }
    const src = (imgs[0] && imgs[0].source) || null;
    if (!src) {
      // An image entry with no usable source would render as a broken glyph.
      renderNoData(el, layout.title);
      return;
    }
    // Cheap signature so we don't O(n)-compare a multi-MB data URI each poll.
    const sig = src.length + "|" + src.slice(0, 24) + src.slice(-24);
    const st = _arrayImgState[divId];
    if (st && st.sig === sig) {
      if (st.ok) { try { Plotly.react(el, f.data, layout, cfg); } catch (e) {} }
      else       { renderNoData(el, layout.title); }
      return;
    }
    // New frame: confirm the data URI actually decodes before we show it.
    const probe = new Image();
    probe.onload = () => {
      _arrayImgState[divId] = { sig, ok: true };
      try { Plotly.react(el, f.data, layout, cfg); } catch (e) {}
    };
    probe.onerror = () => {
      _arrayImgState[divId] = { sig, ok: false };
      renderNoData(el, layout.title);
    };
    probe.src = src;
  }

  // A dark "no data" placeholder matching the server's _waiting() panel
  // (PANEL bg + muted annotation), used when an array frame can't be shown.
  function renderNoData(el, title) {
    try {
      Plotly.react(el, [], {
        paper_bgcolor: "#0d1220", plot_bgcolor: "#0d1220",
        font: { color: "#e0e0e0", size: 10 },
        margin: { l: 40, r: 15, t: 35, b: 30 },
        title: title || "",
        xaxis: { visible: false }, yaxis: { visible: false },
        annotations: [{
          text: "no data", x: 0.5, y: 0.5, xref: "paper", yref: "paper",
          showarrow: false, font: { size: 14, color: "#666" },
        }],
      }, { displayModeBar: false, responsive: true });
    } catch (e) {
      el.innerHTML =
        '<div class="hint" style="padding:24px;text-align:center;">no data</div>';
    }
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
                // Plain Magma (not reversed): bright/colorful = HIGH infidelity
                // = bad; dark = low infidelity = good.
                colorscale: "Magma", reversescale: false,
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

  // =====================================================================
  // AFFINE ALIGNMENT (replaced Grid-shift). Visualizes the global
  // SLM(knm)→camera(px) affine: how well aligned RIGHT NOW (current-fit RMS
  // + coverage + freshness badges), how the affine has drifted over its
  // update history (trend + translation-drift scatter), and a version table
  // with rollback. Data: GET /api/affine/history → {current, history}.
  // =====================================================================
  // Trend dropdown: ONLY the quantities that actually move under our normal
  // (drift/translation) updates — translation + the full-refit RMS. Scale /
  // rotation / det are fixed by the geometry, so they don't belong in a
  // time-trend (they're still shown in the badges + the history table).
  const AFFINE_METRICS = [
    {key: "rms_px", label: "alignment RMS", unit: "px"},
    {key: "tx",     label: "translation x", unit: "px"},
    {key: "ty",     label: "translation y", unit: "px"},
  ];
  // Guardrail thresholds (mirror affine_transform.py): rms ceiling 2.0 px,
  // coverage floor 0.85. We add a tighter "good" band for the badge colors.
  let affineData = null;        // {current, timeline:[...]}
  let affineRenderedTs = null;  // dedupe re-renders (current.updated_iso)

  async function pollAffine() {
    let resp = null;
    try { resp = await api("/api/affine/history"); }
    catch (e) { return; }
    const current = resp && resp.current;
    const history = (resp && resp.history) || [];
    // Timeline oldest→newest = history (older) then current (latest).
    const raw = current ? history.concat([current]) : history.slice();
    const timeline = raw.map((e) => {
      const A = e.A;
      const ty = (A && A[0] && A[0][2] != null) ? A[0][2] : null;  // A row0 → Y
      const tx = (A && A[1] && A[1][2] != null) ? A[1][2] : null;  // A row1 → X
      return Object.assign({}, e, {tx: tx, ty: ty});
    });
    affineData = {current: current, timeline: timeline};
    const ts = current ? (current.updated_iso || "") : "";
    if (ts === affineRenderedTs) { return; }   // unchanged → skip re-render
    affineRenderedTs = ts;
    renderAffine();
  }

  function _affineBadge(label, valHtml, cls, title) {
    return `<span class="affine-badge ${cls}" ${title ? `title="${escHtml(title)}"` : ""}>`
         + `<span class="ab-label">${escHtml(label)}</span>`
         + `<span class="ab-val">${valHtml}</span></span>`;
  }

  function renderAffine() {
    const qWrap = $("affine-quality");
    if (!qWrap) return;
    const cur = affineData && affineData.current;
    if (!cur) {
      qWrap.innerHTML = '<span class="muted">no affine calibrated yet</span>';
      setText("affine-info", "");
      ["plot-affine-trend", "plot-affine-drift"].forEach((id) => {
        const el = $(id); if (el && window.Plotly) Plotly.purge(el);
      });
      const t = $("affine-table"); if (t) t.innerHTML = "";
      return;
    }
    // ---- current-alignment quality badges ----
    // rms_px / coverage / n_pairs are only recorded on FULL refits; live drift
    // (translation) updates leave them null. Fall back to the most recent entry
    // that has them so "how aligned right now" still shows the last fit.
    const fmt = (v, d) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
    const tl0 = affineData.timeline || [];
    const lastFit = tl0.slice().reverse().find(
      (e) => e.rms_px != null && isFinite(e.rms_px));
    const rms = cur.rms_px != null ? cur.rms_px : (lastFit ? lastFit.rms_px : null);
    const cov = cur.coverage != null ? cur.coverage : (lastFit ? lastFit.coverage : null);
    const npairs = cur.n_pairs != null ? cur.n_pairs : (lastFit ? lastFit.n_pairs : null);
    // rms/cov are carried forward from the last FULL fit through drift updates;
    // tag them "(last fit)" when they didn't come from the current entry's scan.
    const carried = (cur.fit_scan_id && cur.fit_scan_id !== cur.last_scan_id)
                    || (cur.rms_px == null && rms != null);
    const fitSfx = carried ? " (last fit)" : "";
    const rmsCls = rms == null ? "ab-neutral"
      : rms <= 0.5 ? "ab-good" : rms <= 2.0 ? "ab-warn" : "ab-bad";
    const covCls = cov == null ? "ab-neutral"
      : cov >= 0.85 ? "ab-good" : cov >= 0.5 ? "ab-warn" : "ab-bad";
    const A = cur.A || null;
    const tx = (A && A[1]) ? A[1][2] : null;   // A row1 → X translation
    const ty = (A && A[0]) ? A[0][2] : null;   // A row0 → Y translation
    const badges = [
      _affineBadge("alignment RMS" + fitSfx, rms == null ? "—" : `${fmt(rms, 3)} px`, rmsCls,
                   "Affine fit residual (recorded on full refits). ≤0.5 good · ≤2 ok · >2 poor. "
                   + "Live drift updates don't measure it — this shows the last full fit."),
      _affineBadge("coverage" + fitSfx, cov == null ? "—" : fmtPct(cov), covCls,
                   "Fraction of pattern sites matched in the last full fit (≥85% good)."),
      _affineBadge("pairs", npairs == null ? "—" : String(npairs), "ab-neutral"),
      _affineBadge("offset x,y", (tx == null || ty == null) ? "—" : `${fmt(tx, 1)}, ${fmt(ty, 1)}`,
                   "ab-neutral", "Affine translation in camera px — live drift tracks this."),
      _affineBadge("scale", `${fmt(cur.scale_x, 3)} × ${fmt(cur.scale_y, 3)}`, "ab-neutral",
                   "px per knm pixel (x × y)."),
      _affineBadge("rotation", `${fmt(cur.rotation_deg, 3)}°`, "ab-neutral"),
      _affineBadge("det", fmt(cur.det, 2), "ab-neutral"),
    ];
    qWrap.innerHTML = badges.join("");
    // Freshness: how recent the current affine is + which scan set it.
    const age = cur.updated_iso ? isoAgeHuman(cur.updated_iso) : null;
    const sid = cur.last_scan_id ? ` · from scan ${escHtml(String(cur.last_scan_id))}` : "";
    const stale = cur.rolled_back ? " · rolled back" : "";
    setText("affine-info",
            (age ? `updated ${age} ago` : "") + sid + stale
            + ` · ${(affineData.timeline || []).length} versions`);
    // ---- ensure the trend-metric dropdown is populated ----
    const sel = $("affine-metric");
    if (sel && !sel.options.length) {
      sel.innerHTML = AFFINE_METRICS.map((m) =>
        `<option value="${m.key}">${escHtml(m.label)}${m.unit ? ` (${m.unit})` : ""}</option>`).join("");
      // Default to the first metric that actually has values — drift updates
      // record translation/scale but not rms/coverage, so rms_px is often all
      // null and would otherwise show an empty chart.
      const withData = AFFINE_METRICS.find(
        (m) => tl0.some((e) => e[m.key] != null && isFinite(e[m.key])));
      sel.value = (withData && withData.key) || "rms_px";
      sel.onchange = drawAffineTrend;
    }
    drawAffineTrend();
    drawAffineDrift();
    renderAffineTable();
  }

  // Approx "2.3 h" / "4 days" from an ISO timestamp to now.
  function isoAgeHuman(iso) {
    const t = Date.parse(iso);
    if (!isFinite(t)) return null;
    let s = Math.max(0, (Date.now() - t) / 1000);
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${Math.round(s / 60)} min`;
    if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
    return `${(s / 86400).toFixed(1)} days`;
  }

  function drawAffineTrend() {
    if (!window.Plotly) return;
    const el = $("plot-affine-trend");
    if (!el || !affineData) return;
    const tl = affineData.timeline || [];
    const sel = $("affine-metric");
    const key = (sel && sel.value) || "rms_px";
    const meta = AFFINE_METRICS.find((m) => m.key === key) || AFFINE_METRICS[0];
    const x = tl.map((e, i) => e.updated_iso || String(i));
    const y = tl.map((e) => {
      const v = e[key];
      return (v == null || !isFinite(v)) ? null : Number(v);
    });
    if (!y.some((v) => v != null)) {
      Plotly.purge(el);
      const msg = tl.length
        ? `no ${escHtml(meta.label)} values (drift updates don't record it — try another metric)`
        : "no affine history yet";
      el.innerHTML = `<div class="hint" style="padding:24px;text-align:center;">${msg}</div>`;
      return;
    }
    const traces = [{
      x: x, y: y, mode: "lines+markers", type: "scatter",
      line: {width: 1.6, color: "#58a6ff"}, marker: {size: 6, color: "#58a6ff"},
      connectgaps: false, name: meta.label,
    }];
    // rms guardrail line for context.
    const layout = plotLayoutFlush({
      margin: {l: 64, r: 16, t: 26, b: 60},
      title: {text: `${meta.label} over affine updates`, font: {size: 12}},
      xaxis: {title: {text: "affine version (time)"}, type: "category",
              gridcolor: "#2a3242"},
      yaxis: {title: {text: meta.unit || meta.label}, gridcolor: "#2a3242"},
      showlegend: false,
    });
    if (key === "rms_px") {
      layout.shapes = [{type: "line", xref: "paper", x0: 0, x1: 1,
                        yref: "y", y0: 2.0, y1: 2.0,
                        line: {color: "#f85149", width: 1, dash: "dash"}}];
    }
    Plotly.react(el, traces, layout, plotConfig());
  }

  function drawAffineDrift() {
    if (!window.Plotly) return;
    const el = $("plot-affine-drift");
    if (!el || !affineData) return;
    const tl = (affineData.timeline || []).filter(
      (e) => e.tx != null && e.ty != null && isFinite(e.tx) && isFinite(e.ty));
    if (!tl.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no translation history</div>';
      return;
    }
    const X = tl.map((e) => e.tx), Y = tl.map((e) => e.ty);
    const C = tl.map((e, i) => i);
    Plotly.react(el, [{
      x: X, y: Y, mode: "lines+markers", type: "scatter",
      line: {width: 1, color: "#2a3242"},
      marker: {size: 9, color: C, colorscale: "Viridis", showscale: true,
               colorbar: {title: {text: "version", side: "right"},
                          thickness: 10, len: 0.9}},
      text: tl.map((e) => e.last_scan_id ? `scan ${e.last_scan_id}` : ""),
      hovertemplate: "x=%{x:.1f} px<br>y=%{y:.1f} px<br>%{text}<extra></extra>",
    }], plotLayoutFlush({
      margin: {l: 64, r: 16, t: 26, b: 50},
      title: {text: "translation drift (camera px)", font: {size: 12}},
      xaxis: {title: {text: "X offset (px)"}, gridcolor: "#2a3242"},
      yaxis: {title: {text: "Y offset (px)"}, gridcolor: "#2a3242",
              scaleanchor: "x", scaleratio: 1},
      showlegend: false,
    }), plotConfig());
  }

  function renderAffineTable() {
    const wrap = $("affine-table");
    if (!wrap || !affineData) return;
    const tl = (affineData.timeline || []).slice().reverse();   // newest first
    if (!tl.length) { wrap.innerHTML = ""; return; }
    const fmt = (v, d) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
    const rows = tl.map((e, i) => {
      const isCur = (i === 0);
      const when = e.updated_iso ? escHtml(e.updated_iso.replace("T", " ").slice(0, 19)) : "—";
      return `<tr class="${isCur ? "af-cur" : ""}">
        <td>${isCur ? "● current" : "history"}</td>
        <td class="mono">${when}</td>
        <td class="mono">${e.last_scan_id ? escHtml(String(e.last_scan_id)) : "—"}</td>
        <td class="mono">${fmt(e.rms_px, 3)}</td>
        <td class="mono">${e.coverage == null ? "—" : fmtPct(e.coverage)}</td>
        <td class="mono">${e.n_pairs == null ? "—" : e.n_pairs}</td>
        <td class="mono">${fmt(e.scale_x, 3)}×${fmt(e.scale_y, 3)}</td>
        <td class="mono">${fmt(e.rotation_deg, 3)}°</td>
      </tr>`;
    }).join("");
    wrap.innerHTML = `<table class="af-table"><thead><tr>
      <th></th><th>updated</th><th>scan</th><th>RMS px</th><th>cov</th>
      <th>pairs</th><th>scale x×y</th><th>rot</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  // ===== Detection thresholds & calibration (per-pattern history + health) =====
  const THRESHOLD_METRICS = [
    {key: "mean_thr", label: "mean threshold", unit: "ADU"},
    {key: "std_thr", label: "spread (std)", unit: "ADU"},
    {key: "mean_infidelity", label: "mean infidelity", unit: ""},
    {key: "n_updated", label: "sites updated", unit: ""},
  ];
  const THRESHOLD_SRC_COLOR = {cheap: "#58a6ff", fit: "#3fb950", fit_rejected: "#f85149"};
  let thresholdData = null;
  let thresholdPlotsKey = null;
  // Which pattern's threshold history the card shows. null => follow the live
  // img1 pattern (the default). Set by the pattern dropdown. Deliberately a
  // plain JS var (not persisted), so it RESETS to the img1 pattern on a page
  // refresh but is KEPT across live polls (new data never clobbers the choice).
  let selectedThresholdPattern = null;

  async function pollThresholds() {
    let resp = null;
    const q = selectedThresholdPattern
      ? ("?pattern=" + encodeURIComponent(selectedThresholdPattern)) : "";
    try { resp = await api("/api/thresholds/history" + q); }
    catch (e) { return; }
    thresholdData = resp || null;
    renderThresholds();   // badges + banner refresh every poll (live countdown)
  }

  // Build/refresh the pattern dropdown: img1 first, then img2 (if a distinct
  // loading pattern), then every other loaded pattern. Defaults the selection
  // to the img1 pattern. Rebuilds <option>s only when the set changes so an
  // open dropdown isn't clobbered each poll.
  function syncThresholdPatternDropdown() {
    const sel = $("threshold-pattern");
    if (!sel || !thresholdData) return;
    const img1 = thresholdData.active_pattern || null;
    const img2 = thresholdData.active_pattern_img2 || null;   // distinct img2 only
    const all = Array.isArray(thresholdData.all_patterns) ? thresholdData.all_patterns : [];
    if (selectedThresholdPattern == null) {
      selectedThresholdPattern = img1 || thresholdData.pattern || null;
    }
    const seen = new Set();
    const opts = [];
    const add = (name, role) => {
      if (!name || seen.has(name)) return;
      seen.add(name); opts.push({name: name, role: role});
    };
    add(img1, "img1");
    add(img2, "img2");
    all.forEach((n) => add(n, "other"));
    add(selectedThresholdPattern, "other");   // ensure the selection is listable
    const key = opts.map((o) => o.role + ":" + o.name).join("|");
    if (sel.dataset.optKey !== key) {
      sel.dataset.optKey = key;
      sel.innerHTML = opts.map((o) => {
        const tag = o.role === "img1" ? " (img1)" : o.role === "img2" ? " (img2)" : "";
        return `<option value="${escHtml(o.name)}" class="popt-${o.role}">`
             + `${escHtml(o.name)}${tag}</option>`;
      }).join("");
      sel.onchange = () => {
        selectedThresholdPattern = sel.value || null;
        colorThresholdPatternSelect();
        pollThresholds();   // re-fetch the selected pattern's history at once
      };
    }
    if (selectedThresholdPattern && sel.value !== selectedThresholdPattern) {
      sel.value = selectedThresholdPattern;
    }
    colorThresholdPatternSelect();
  }

  // Tint the select green/blue/gray by whether the chosen pattern is the live
  // img1 / img2 frame of the running scan, or not used in this scan.
  function colorThresholdPatternSelect() {
    const sel = $("threshold-pattern");
    if (!sel || !thresholdData) return;
    const v = sel.value;
    const img1 = thresholdData.active_pattern || null;
    const img2 = thresholdData.active_pattern_img2 || null;
    sel.classList.remove("is-img1", "is-img2", "is-other");
    sel.classList.add(v && v === img1 ? "is-img1"
                      : v && v === img2 ? "is-img2" : "is-other");
  }

  function renderThresholds() {
    const qWrap = $("threshold-quality");
    if (!qWrap || !thresholdData) return;
    const active = thresholdData.active || {};
    const health = thresholdData.health || {};
    const hist = thresholdData.history || [];
    const fmt = (v, d) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);

    // active loading pattern + defocus chip. `pat` is the pattern whose history
    // is shown (the live one when a scan is running, else the most-recent one);
    // `livePat` is the currently-running pattern (null when idle).
    const livePat = active.pattern || thresholdData.active_pattern;
    const pat = thresholdData.pattern || livePat;
    const defoc = active.defocus;
    const idleSfx = (pat && !livePat) ? " · idle (last run)" : "";
    setText("threshold-active",
      pat ? (`${pat}` + (defoc == null ? "" : ` · defocus ${fmt(defoc, 0)}`) + idleSfx)
          : "no loading pattern");

    // loud warning banner (A3/A5)
    const banner = $("threshold-banner");
    if (banner) {
      const st = health.state || active.state;
      if (st === "degraded") {
        banner.hidden = false;
        banner.className = "threshold-banner tb-bad";
        banner.innerHTML = `⚠ thresholds DEGRADED — ${escHtml(health.reason || "")}`
          + ` · holding &amp; re-anchoring from live data (not using day folder)`;
      } else if (st === "unknown_pattern" || st === "unknown") {
        banner.hidden = false;
        banner.className = "threshold-banner tb-warn";
        banner.innerHTML = `⚠ ${escHtml(health.reason || "no loading pattern declared")}`
          + ` — using ${escHtml(active.source || "day-folder")} thresholds`;
      } else {
        banner.hidden = true; banner.innerHTML = "";
      }
    }

    // health badges
    const spread = active.spread != null ? active.spread : health.spread;
    const spreadCls = spread == null ? "ab-neutral"
      : spread <= 0.6 ? "ab-good" : spread <= 1.0 ? "ab-warn" : "ab-bad";
    const mi = active.mean_infidelity != null ? active.mean_infidelity : health.mean_infidelity;
    const miCls = mi == null ? "ab-neutral"
      : mi <= 0.05 ? "ab-good" : mi <= 0.15 ? "ab-warn" : "ab-bad";
    const lastFit = hist.slice().reverse().find((e) => e.source === "fit");
    const fitAge = (lastFit && lastFit.ts) ? isoAgeHuman(lastFit.ts) : null;
    const nextIn = active.next_fit_in;
    const rejects = hist.filter((e) => e.source === "fit_rejected");
    const lastReject = rejects.length ? rejects[rejects.length - 1] : null;
    const stateCls = (health.state === "ok") ? "ab-good"
      : (health.state === "degraded") ? "ab-bad"
      : (health.state === "unknown_pattern" || health.state === "unknown") ? "ab-warn"
      : "ab-neutral";
    const badges = [
      _affineBadge("calibration", escHtml(health.state || "—"), stateCls,
        "ok = a clean full fit anchors detection; degraded = stored thresholds rejected/holding."),
      _affineBadge("spread (std)", spread == null ? "—" : `${fmt(spread, 2)} ADU`, spreadCls,
        "Per-site threshold spread. Tight (≤0.6) good; a wide spread is the corruption symptom."),
      _affineBadge("mean infidelity", mi == null ? "—" : fmt(mi, 4), miCls,
        "Mean per-site discrimination infidelity at the last accepted full fit."),
      _affineBadge("last full fit", fitAge ? `${fitAge} ago` : "—", "ab-neutral",
        lastFit ? `accepted fit at seq ${lastFit.seq_no} (scan ${lastFit.scan_id})` : "no accepted fit yet"),
      _affineBadge("next fit in", nextIn == null ? "—" : `${nextIn} shots`, "ab-neutral",
        "Shots until the next full Gaussian refit (counts across runs of this pattern)."),
      _affineBadge("rejections", String(rejects.length), rejects.length ? "ab-bad" : "ab-good",
        lastReject ? `last: ${escHtml(lastReject.reason || "")}` : "no rejected fits"),
    ];
    qWrap.innerHTML = badges.join("");

    const age = active.updated_iso ? isoAgeHuman(active.updated_iso) : null;
    setText("threshold-info",
      (pat ? `pattern ${pat}` : "no pattern")
      + (active.source ? ` · ${escHtml(active.source)}` : "")
      + (age ? ` · health ${age} ago` : "")
      + ` · ${hist.length} updates`);

    const sel = $("threshold-metric");
    if (sel && !sel.options.length) {
      sel.innerHTML = THRESHOLD_METRICS.map((m) =>
        `<option value="${m.key}">${escHtml(m.label)}${m.unit ? ` (${m.unit})` : ""}</option>`).join("");
      sel.value = "mean_thr";
      sel.onchange = drawThresholdTrend;
    }

    // Pattern selector (img1 / img2 / other loaded patterns); drives which
    // pattern's history this card fetches + plots.
    syncThresholdPatternDropdown();

    // Redraw the heavy plots/table only when the history actually changed.
    const key = (hist.length ? (hist[hist.length - 1].ts || "") : "") + "|" + hist.length;
    if (key !== thresholdPlotsKey) {
      thresholdPlotsKey = key;
      drawThresholdTrend();
      drawThresholdEvents();
      renderThresholdTable();
    }
  }

  function drawThresholdTrend() {
    if (!window.Plotly) return;
    const el = $("plot-threshold-trend");
    if (!el || !thresholdData) return;
    const hist = thresholdData.history || [];
    const sel = $("threshold-metric");
    const key = (sel && sel.value) || "mean_thr";
    const meta = THRESHOLD_METRICS.find((m) => m.key === key) || THRESHOLD_METRICS[0];
    if (!hist.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no threshold history for this pattern yet</div>';
      return;
    }
    const x = hist.map((e, i) => e.ts || String(i));
    const y = hist.map((e) => {
      const v = e[key];
      return (v == null || !isFinite(v)) ? null : Number(v);
    });
    const colors = hist.map((e) => THRESHOLD_SRC_COLOR[e.source] || "#8b949e");
    const traces = [{
      // SVG scatter (NOT scattergl): the page already has many WebGL plots
      // (per-site maps, histograms, sequence channels); adding more exhausts the
      // browser's WebGL-context limit -> the "frowny face" placeholder. This
      // history is tiny (<=400 pts) so SVG is plenty fast.
      x: x, y: y, mode: "lines+markers", type: "scatter",
      line: {width: 1.4, color: "#30363d"}, marker: {size: 6, color: colors},
      connectgaps: false, name: meta.label,
      text: hist.map((e) => e.source + (e.reason ? ` — ${e.reason}` : "")),
      hovertemplate: "%{y}<br>%{text}<extra></extra>",
    }];
    Plotly.react(el, traces, plotLayoutFlush({
      margin: {l: 64, r: 16, t: 26, b: 60},
      title: {text: `${meta.label} over threshold updates`, font: {size: 12}},
      xaxis: {title: {text: "update (time)"}, type: "category", gridcolor: "#2a3242"},
      yaxis: {title: {text: meta.unit || meta.label}, gridcolor: "#2a3242"},
      showlegend: false,
    }), plotConfig());
  }

  function drawThresholdEvents() {
    if (!window.Plotly) return;
    const el = $("plot-threshold-events");
    if (!el || !thresholdData) return;
    const hist = thresholdData.history || [];
    if (!hist.length) {
      Plotly.purge(el);
      el.innerHTML = '<div class="hint" style="padding:24px;text-align:center;">no update events</div>';
      return;
    }
    const SRC = ["cheap", "fit", "fit_rejected"];
    const yrow = {cheap: 0, fit: 1, fit_rejected: 2};
    const traces = SRC.map((s) => {
      const pts = hist.filter((e) => e.source === s);
      return {
        x: pts.map((e) => e.ts), y: pts.map(() => yrow[s]),
        mode: "markers", type: "scatter", name: s,   // SVG, not WebGL (see trend)
        marker: {size: 9, color: THRESHOLD_SRC_COLOR[s],
                 symbol: s === "fit_rejected" ? "x" : "circle"},
        text: pts.map((e) => e.reason ? `${s} — ${e.reason}` : s),
        hovertemplate: "%{x}<br>%{text}<extra></extra>",
      };
    });
    Plotly.react(el, traces, plotLayoutFlush({
      margin: {l: 90, r: 16, t: 26, b: 50},
      title: {text: "update events (cheap · fit · rejected)", font: {size: 12}},
      xaxis: {title: {text: "time"}, type: "category", gridcolor: "#2a3242"},
      yaxis: {tickvals: [0, 1, 2], ticktext: ["cheap", "fit", "rejected"],
              gridcolor: "#2a3242", range: [-0.5, 2.5]},
      showlegend: false,
    }), plotConfig());
  }

  function renderThresholdTable() {
    const wrap = $("threshold-table");
    if (!wrap || !thresholdData) return;
    const hist = (thresholdData.history || []).slice().reverse();   // newest first
    if (!hist.length) { wrap.innerHTML = ""; return; }
    const fmt = (v, d) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
    const rows = hist.slice(0, 60).map((e, i) => {
      const isCur = (i === 0);
      const when = e.ts ? escHtml(e.ts.replace("T", " ").slice(0, 19)) : "—";
      const rejCls = e.source === "fit_rejected" ? "tb-row-rej" : "";
      return `<tr class="${isCur ? "af-cur" : ""} ${rejCls}">
        <td>${escHtml(e.source || "—")}</td>
        <td class="mono">${when}</td>
        <td class="mono">${e.scan_id ? escHtml(String(e.scan_id)) : "—"}</td>
        <td class="mono">${e.seq_no == null ? "—" : e.seq_no}</td>
        <td class="mono">${fmt(e.mean_thr, 2)}</td>
        <td class="mono">${fmt(e.std_thr, 2)}</td>
        <td class="mono">${e.mean_infidelity == null ? "—" : fmt(e.mean_infidelity, 4)}</td>
        <td class="mono">${e.n_updated == null ? "—" : e.n_updated}</td>
        <td>${e.reason ? escHtml(e.reason) : ""}</td>
      </tr>`;
    }).join("");
    wrap.innerHTML = `<table class="af-table"><thead><tr>
      <th>source</th><th>time</th><th>scan</th><th>seq</th><th>mean</th>
      <th>std</th><th>infid</th><th>sites</th><th>note</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
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

  // Per-site rep histograms — port of _figs_reps + _build_hist. (rep3, a 2nd
  // "random", was dropped; the 4th grid cell is now the selected-site hist.)
  function renderRepHists(snap) {
    const targets = ["plot-hist-rep0", "plot-hist-rep1", "plot-hist-rep2"];
    const labels = ["Best", "Worst", "Random"];
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

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  // escHtml + quote-escaping, so a value is safe inside a single- OR double-quoted
  // HTML attribute (channel names can contain ' or ").
  function escAttr(s) {
    return escHtml(s).replace(/'/g, "&#39;").replace(/"/g, "&quot;");
  }

  // =====================================================================
  // ANALYSIS TAB
  // =====================================================================
  let runsCache = [];
  let selectedScanId = null;
  let groupsCache = {};
  let activeGroupId = null;
  // What the Analysis tab is currently RENDERING: a scan_id, or "grp:<ids>" for
  // a group view. Lets ensureAnalysisShown() skip a redundant re-render when the
  // selection already matches what's on screen (cross-tab sync without churn).
  let _analysisShownFor = null;
  // Signature of the last-rendered picker list; lets a poll skip the rebuild
  // when nothing changed (avoids disrupting hover/scroll every few seconds).
  let _runsSig = null;
  // Every available data day (YYYYMMDD, newest-first), from /api/runs/dates.
  // Populates the date dropdown so ANY historical day is reachable -- the
  // polled list itself stays a bounded recent window (the picker only loads
  // the whole multi-year archive one chosen day at a time). See loadRunsList.
  let allDates = [];
  // scan_id -> recorded-shot count at the last analysis. The reload/poll gate
  // (item 2): a run is only re-analyzed when its shot count GROWS (a live
  // scan). A completed/unchanged run is never re-fetched on a poll or reload,
  // so reloads are instant (the backend payload cache makes the one first-paint
  // fetch ~ms too). Cleared per-scan when (re)analyzed.
  let lastAnalyzedShots = {};

  // Recorded-shot count for a run as the runs list currently knows it (the
  // SAME quantity analyze_scan reports as n_shots, so the live-growth gate
  // compares apples to apples). None when unknown.
  function _rowShots(scanId) {
    const row = runsCache.find((r) => r.scan_id === scanId);
    return (row && row.n_actual_shots != null) ? row.n_actual_shots : null;
  }

  // opts.force -> Full rescan (clears the backend enrichment cache, re-enriches
  // every scan). Normal calls use the cheap incremental cached path.
  async function loadRunsList(opts) {
    opts = opts || {};
    // Only flip status to "loading list" if we're not already mid-analysis
    // for some specific scan -- the analysis status takes precedence.
    if (!selectedScanId) {
      setAnalysisStatus("loading", "loading runs…", "warn");
    }
    try {
      // The unified picker uses /api/sequence/scans -- the SUPERSET: every run
      // (Analysis fields) PLUS per-run seq availability (has_seq/n_seq/
      // snapshot/descriptor) for the Sequence-tab badge + click. Shares the
      // backend enrichment cache, so it's cheap; ?force=1 = Full rescan.
      // A selected date scopes the fetch to that ONE day server-side (bounded),
      // so picking an old day never walks/enriches the whole archive; empty =
      // the default recent window. This is what extends past the 500-row cap.
      const qs = [];
      const day = (($("runs-date-filter") || {}).value) || "";
      if (day) qs.push("date=" + encodeURIComponent(day));
      if (opts.force) qs.push("force=1");
      const data = await api("/api/sequence/scans" + (qs.length ? "?" + qs.join("&") : ""));
      runsCache = data.scans || data.runs || [];
      // Skip the (DOM-rebuilding) re-render when nothing the picker shows has
      // changed -- so the 3 s auto-poll doesn't disrupt a hover/scroll. The
      // signature folds in the fields a row renders (id, shot count, seq state).
      const sig = runsCache.map((r) =>
        `${r.scan_id}:${r.n_actual_shots == null ? "" : r.n_actual_shots}:` +
        `${r.has_seq ? 1 : 0}:${r.n_seq || 0}`).join("|");
      if (sig !== _runsSig || opts.force) {
        _runsSig = sig;
        renderRunsTable();
        populateDateFilter();
        setText("runs-count", String(runsCache.length));
      } else {
        // unchanged: keep highlights fresh (selection may have changed), no rebuild
        syncTrayHighlight();
      }
      // Auto-pick a run for FIRST paint only (nothing selected yet). Priority:
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
      } else if (selectedScanId && traySet.size <= 1) {
        // Live-growth gate (item 2): re-analyze the SELECTED run only when its
        // recorded shot count has grown since we last analyzed it (i.e. it's a
        // running scan still accumulating). Completed/unchanged runs are never
        // re-fetched here, so polls + reloads don't churn the analysis.
        const cur  = _rowShots(selectedScanId);
        const prev = lastAnalyzedShots[selectedScanId];
        if (cur != null && prev != null && cur > prev) {
          // Silent: refresh in place (no blank/purge/"analyzing…" flash).
          loadAnalysis(selectedScanId, {keepFilters: true, silent: true});
        }
      }
    } catch (e) {
      const wrap = $("runs-table");
      if (wrap) wrap.innerHTML = `<div class="run-row"><div class="run-info muted">${escHtml(e.message)}</div></div>`;
    }
  }
  function populateDateFilter() {
    const sel = $("runs-date-filter");
    if (!sel) return;
    // Full archive date list (every day, from /api/runs/dates) UNIONed with the
    // dates present in the currently-loaded rows -- so a brand-new day shows up
    // immediately even before the next dates refresh. Selecting a day re-fetches
    // just that day server-side (see loadRunsList), so reaching any historical
    // run no longer depends on it being inside the recent 500-row window.
    const fromRows = runsCache.map((r) => (r.scan_id || "").slice(0, 8));
    const dates = Array.from(new Set([...allDates, ...fromRows]))
      .filter((d) => d && d.length === 8).sort().reverse();
    const cur = sel.value;
    const opt = (d) =>
      `<option value="${d}">${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}</option>`;
    sel.innerHTML = '<option value="">All dates (recent)</option>' +
      dates.map(opt).join("");
    // Preserve the current selection even if it isn't in the (possibly stale) list.
    if (cur && !dates.includes(cur)) sel.insertAdjacentHTML("beforeend", opt(cur));
    sel.value = cur;
  }
  // Fetch the full list of archived days (cheap: a dir listing, no enrichment)
  // so the date dropdown can reach the whole multi-year archive, not just the
  // days inside the recent window.
  async function fetchRunDates() {
    try {
      const d = await api("/api/runs/dates");
      allDates = (d && d.dates) || [];
      populateDateFilter();
    } catch (e) { /* keep the dates the loaded rows already provide */ }
  }
  // ---- Unified selection model (primary + group) -----------------------
  // The "tray" (traySet, insertion-ordered) IS the selected GROUP; the FIRST
  // member is the PRIMARY. The primary drives the Sequence tab + is rendered
  // slightly darker; the whole group drives the Analysis tab (group analysis)
  // and every member is highlighted (picker + Queue). One picker, shared across
  // Analysis + Sequence.
  function trayPrimary() { return traySet.size ? Array.from(traySet)[0] : null; }

  function persistSelected(sid) {
    try { localStorage.setItem("yb_dashboard_selected_scan", sid || ""); }
    catch { /* private mode */ }
    const pasteEl = document.getElementById("manual-scan-id");
    if (pasteEl && sid) pasteEl.value = sid;
  }

  // Seq-availability badge + 3-state (Ready / Reconstructable / Unrecoverable),
  // mirrored from the old Sequence picker so a row in the unified picker still
  // tells you whether a Sequence-tab click will plot or reconstruct.
  function _seqState(r) {
    if (r.has_seq) return {state: "ready", tag: `${r.n_seq || ""} seq`.trim()};
    if (r.has_snapshot && r.has_descriptor)
      return {state: "reconstructable", tag: "reconstruct ⟳"};
    if (r.has_snapshot) return {state: "unrecoverable", tag: "no descriptor"};
    return {state: "unrecoverable", tag: "no dump"};
  }

  // Load the primary's SEQUENCE (Sequence tab): plot the .seq dump if present,
  // else reconstruct from the code snapshot, else explain why it can't.
  function seqLoadForScan(sid) {
    if (!sid) return;
    const r = runsCache.find((x) => x.scan_id === sid);
    if (!r) { toast("scan not in list — Refresh", "warn"); return; }
    const st = _seqState(r).state;
    if (st === "ready") {
      const fin = $("seq-folder"); if (fin) fin.value = "";
      seqLoad({ scan_id: sid }).then(renderRunsTable);
    } else if (st === "reconstructable") {
      seqReconstruct(sid, null);
    } else {
      toast("no .seq dump and no reconstructable snapshot for this scan", "warn");
    }
  }

  // Make the analysis view match the current selection (single -> loadAnalysis,
  // group -> group analysis). No-op when it's already showing the right thing.
  function _desiredAnalysisKey() {
    if (traySet.size > 1) return "grp:" + Array.from(traySet).join(",");
    return trayPrimary();
  }
  function ensureAnalysisShown(opts) {
    const want = _desiredAnalysisKey();
    if (!want) return;
    if (_analysisShownFor === want) return;
    if (traySet.size > 1) analyzeTrayGroup();
    else loadAnalysis(trayPrimary(), opts && opts.analysisOpts);
  }
  // Make the Sequence tab show the primary's sequence (if not already).
  function ensureSeqShown() {
    const p = trayPrimary();
    if (!p) return;
    if (seqState.query && seqState.query.scan_id === p) return;
    seqLoadForScan(p);
  }
  // Debounced group re-analysis so a burst of "+" clicks coalesces.
  let _grpAnalyzeTimer = null;
  function scheduleGroupAnalyze() {
    if (_grpAnalyzeTimer) clearTimeout(_grpAnalyzeTimer);
    _grpAnalyzeTimer = setTimeout(() => {
      _grpAnalyzeTimer = null;
      if (activeTab === "analysis") ensureAnalysisShown();
    }, 350);
  }

  // Select `sid` as the SOLE primary (clears the group) and drive the active
  // tab. Used by a plain row-click (analysis OR sequence) + the paste box.
  function selectPrimary(sid) {
    if (!sid) return;
    trayReplace(sid);                 // group = {sid}; renderTray -> highlights
    selectedScanId = sid;
    setText("selected-scan-id", sid);
    persistSelected(sid);
    syncSelectionEverywhere();
    // Drive the sequence viewer when it's the visible view (legacy Sequence
    // tab OR the new Analysis "Sequence" sub-mode); otherwise the data view.
    const seqVisible = activeTab === "sequence" ||
      (activeTab === "analysis" && analysisSubMode === "sequence");
    if (seqVisible) seqLoadForScan(sid);
    else ensureAnalysisShown();
  }

  // Highlight the selected group + primary across the picker AND the queue.
  function syncSelectionEverywhere() {
    syncTrayHighlight();
    syncQueueSelectionHighlight();
  }

  function renderRunsTable() {
    const wrap = $("runs-table");
    if (!wrap) return;
    const search = ($("runs-search").value || "").toLowerCase();
    const date = $("runs-date-filter").value || "";
    const filtered = runsCache.filter((r) => {
      if (date && (r.scan_id || "").slice(0, 8) !== date) return false;
      if (search) {
        const blob = ((r.scan_id || "") + " " + (r.name || "") + " " +
                      (r.description || "")).toLowerCase();
        if (!blob.includes(search)) return false;
      }
      return true;
    });
    if (!filtered.length) {
      wrap.innerHTML =
        '<div class="run-row"><div class="run-info muted">no runs match</div></div>';
      return;
    }
    const primary = trayPrimary();
    const curSeq = (seqState && seqState.query && seqState.query.scan_id) || "";
    // SLM-style row: [+] button | when | name | swept | seq-availability badge.
    // Click the [+] to toggle a run into the GROUP; click elsewhere to make it
    // the sole PRIMARY. Behaviour is tab-aware (Analysis vs Sequence).
    wrap.innerHTML = filtered.map((r) => {
      const id = r.scan_id || "";
      const idShort = id.length === 14
        ? `${id.slice(4,6)}/${id.slice(6,8)} ${id.slice(8,10)}:${id.slice(10,12)}:${id.slice(12,14)}`
        : id;
      const inTray = traySet.has(id);
      const isPrimary = id === primary;
      const seq = _seqState(r);
      const cls = ["run-row", `seq-row-${seq.state}`];
      if (inTray) cls.push("in-tray");
      if (isPrimary) cls.push("is-primary");
      if (id === curSeq) cls.push("seq-current");
      return `
        <div class="${cls.join(" ")}" data-scan-id="${id}" data-seq-state="${seq.state}">
          <button class="run-add" data-tray-toggle="${id}"
                  title="${inTray ? "Remove from group" : "Add to group"}">${inTray ? "✓" : "+"}</button>
          <div class="run-info" title="${id}">
            ${idShort}
            <span class="run-dim"> · ${escHtml(r.name || "—")}</span>
            <span class="run-dim"> · ${escHtml(r.swept || "—")}</span>
            <span class="seq-row-tag"> · ${escHtml(seq.tag)}</span>
          </div>
        </div>`;
    }).join("");
    $$(".run-row", wrap).forEach((row) => {
      if (!row.dataset.scanId) return;
      // Full-info hover tooltip (reused from the old Sequence picker).
      const sInfo = runsCache.find((x) => (x.scan_id || "") === row.dataset.scanId);
      if (sInfo) {
        row.addEventListener("mouseenter", (e) => seqShowScanTip(e, sInfo));
        row.addEventListener("mousemove", seqMoveScanTip);
        row.addEventListener("mouseleave", seqHideScanTip);
      }
      row.addEventListener("click", (e) => {
        if (e.target.closest(".run-add")) return;   // its own handler
        // Stop the click bubbling to the float-card document handler (which
        // would collapse the card). The row stays put -- selectPrimary updates
        // highlights via class-toggle, it no longer rebuilds the table.
        e.stopPropagation();
        seqHideScanTip();
        selectPrimary(row.dataset.scanId);
      });
    });
    $$(".run-add", wrap).forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const sid = btn.dataset.trayToggle;
        trayToggle(sid);            // add/remove from the GROUP
        syncSelectionEverywhere();  // class-toggle, no full rebuild
        // Keep the picker chips/labels in sync (+ / ✓) without detaching rows.
        const b2 = wrap.querySelector(`.run-add[data-tray-toggle="${sid}"]`);
        if (b2) b2.textContent = traySet.has(sid) ? "✓" : "+";
        if (activeTab === "analysis") scheduleGroupAnalyze();
      });
    });
    syncTrayHighlight();
  }
  $("runs-search").addEventListener("input", renderRunsTable);
  // A date change re-FETCHES server-side (that one day, or the recent window for
  // "All dates"), not just a client-side filter of the loaded rows -- that's how
  // an older day is brought into the picker at all.
  $("runs-date-filter").addEventListener("change", () => loadRunsList());
  $("runs-refresh").addEventListener("click", () => {
    // Full rescan: clear the backend enrichment cache and re-enrich every scan
    // (picks up edits / new days the incremental path wouldn't re-stat).
    loadRunsList({force: true});
    fetchRunDates();   // a new day may have appeared since the last dates fetch
    toast("Rescanning all runs…", "warn");
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
    // Silent = a background live-growth refresh (the selected run is still
    // accumulating shots). Keep the current view ON SCREEN: don't blank the
    // body, purge the plots, or flash "analyzing…". The post-fetch render*()
    // calls all use Plotly.react (in-place diff), so the new data swaps in
    // atomically once it arrives -- a live scan updates smoothly instead of
    // disappearing every poll.
    const silent = !!opts.silent;
    selectedScanId = scanId;
    setText("selected-scan-id", scanId);
    if (!silent) setAnalysisStatus("analyzing", "analyzing…", "warn");
    // Update row highlights WITHOUT rebuilding the table. The old
    // renderRunsTable() here detached the just-clicked .run-row mid-click
    // (the source of the "glitchy load" + the stopPropagation workaround);
    // a class-toggle leaves the DOM intact.
    syncTrayHighlight();
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
    if (!silent) {
      body.innerHTML = '<div class="hint">loading…</div>';
      ["plot-analysis-scan", "plot-analysis-scan-lines", "plot-site-loading",
       "plot-site-survival", "plot-site-fp", "plot-site-infid",
       "plot-site-inthist", "plot-per-iter", "plot-per-iter-hist",
       "plot-svd", "plot-avg-image", "plot-rearrange-scatter"].forEach(safePurge);
    }
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
      _analysisShownFor = scanId;   // cross-tab sync: this single run is on screen
      // Remember the shot count we just analyzed so the live-growth gate only
      // re-analyzes this run when it actually grows (item 2).
      lastAnalyzedShots[scanId] = (typeof r.n_shots === "number") ? r.n_shots : null;
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
      // On a silent live-refresh, a transient fetch error must NOT blow away
      // the on-screen view -- keep the last good render and just flag the pill;
      // the next poll retries.
      if (silent) { setAnalysisStatus("analyzed", "live · update failed", "warn"); return; }
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
    // Imaging fidelity at the thresholds used THROUGHOUT the run. Default view
    // = from the logged infidelities; recompute = refit from this run's data.
    const imf = r.imaging_fidelity || null;
    // Loss #2: max rearrangement survival cap from source-site detection
    // confidence (recompute-loaded / cached).
    const cap = r.rearrange_survival_cap || null;
    // Filtered TP survival: target survival with outlier bad-fidelity target
    // spots excluded (no recompute needed — uses available per-site infids).
    const taf = (r.target_aware_filtered
                 && r.target_aware_filtered.overall_mean != null)
                ? r.target_aware_filtered : null;
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
      ? `median per-site discrimination fidelity (1 − infidelity) from THIS run's data. Higher is better; capped at 99.9% unless infidelity is exactly 0. infidelity mean=${fmtInfid(disc.mean_infidelity)}, max=${fmtInfid(disc.max_infidelity)}.`
      : `STORED scan-start calibration fidelity (1 − infidelity)${calAge ? ` — calibrated ${calAge} before this run` : ""}. MAY BE STALE: the run's actual value can differ — click "recompute from this run". infidelity mean=${fmtInfid(disc.mean_infidelity)}.`;
    const discTile = disc ? `
        <div class="stat-tile" title="${discTitle}">
          <span class="stat-label">fidelity&uarr; <span class="src-badge src-${discFromRun ? "lab" : "slm"}">${discSrc}</span></span>
          <span class="stat-value" style="color:${discColor}">${fmtFidelity(discVal)}${
            discFromRun ? "" : ` <span class="muted" style="font-size:10px;">cal${calAge ? " · " + escHtml(calAge) : ""}</span>`
          }</span>
        </div>` : "";
    // False-positive headline (↓ better). Target-aware (excludes target
    // sites) when available, else all-empty. Shown for any 2-image scan.
    const fpSrc = sg.fp_source || ((ta && ta.fp_overall != null) ? "rearrange" : "all_empty");
    const fpVal = (ta && ta.fp_overall != null) ? ta.fp_overall
                  : (sg.fp_overall != null ? sg.fp_overall : avg(sg.fp_mean));
    const fpTile = (!oneImg && fpVal != null && isFinite(fpVal)) ? `
        <div class="stat-tile" title="${
          fpSrc === "rearrange"
            ? "False positives at empty NON-target sites (target-aware) — whole-scan"
            : "False positives: atoms at sites empty in img1 (all-empty) — whole-scan"
        }">
          <span class="stat-label">FP&darr;${
            fpSrc === "rearrange" ? ' <span class="src-badge src-lab">target</span>' : ""
          }</span>
          <span class="stat-value" style="color:${infidColor(fpVal)}">${fmtPct(fpVal)}</span>
        </div>` : "";
    const params = r.run_parameters || [];
    body.innerHTML = warnHtml + `
      <div class="run-head">
        <span class="run-name" title="${escHtml(r.scan_filename || "")}">${escHtml(r.scan_name || "(unnamed scan)")}</span>
        ${dateStr ? `<span class="run-date mono">${dateStr}</span>` : ""}
      </div>
      ${r.scan_description ? `<details class="run-desc-details"><summary>description</summary><div class="run-desc">${escHtml(r.scan_description)}</div></details>` : ""}
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
          <span class="stat-value" title="actual recorded / planned ('supposed to do') shots${
            r.n_shots_scheduled != null ? ` · planned ${r.n_shots_scheduled}` : ""
          }">${r.n_shots}${
            r.n_shots_scheduled != null
              ? ` <span class="muted" style="font-size:11px;">/ ${r.n_shots_scheduled}</span>` : ""
          }</span>
        </div>
        ${survTile}
        <div class="stat-tile">
          <span class="stat-label">loading</span>
          <span class="stat-value">${fmtPct(avg(sg.loading_rate))}</span>
        </div>
        ${discTile}
        ${fpTile}
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
                title="Refit per-site discrimination from THIS run's intensities — also re-derives the imaging fidelity at the throughout-run thresholds and the rearrangement max-survival cap. Non-destructive; cached after the first run; default discrimination uses the scan-start calibration.">${
              discFromRun ? "✓ using this run (click for scan-start)" : "recompute from this run"
            }</button>` : ""}
          · <button class="ghost" id="reanalyze-btn"
              title="Clear this run's cached analysis (double-Gaussian fits, focus metrics) and recompute from scratch.">↻ re-analyze</button>
        </div>
        ${imf ? `<div title="${
          imf.source === "logged_throughout_run"
            ? `Discrimination fidelity at the thresholds the run ACTUALLY used, averaged over every shot using the per-site infidelities logged THROUGHOUT this run (scan-start seed + each live refit). How trustworthy the bitstrings/logicals the run produced were. ${imf.n_in_run_updates || 0} in-run threshold update${(imf.n_in_run_updates === 1) ? "" : "s"}. Press “recompute from this run” to re-derive it from this run's own intensities.`
            : "Discrimination fidelity at the thresholds used THROUGHOUT the run (per-shot threshold timeline), measured on THIS run's refit intensities — i.e. how trustworthy the bitstrings/logicals the run actually produced were. 1 − infidelity at the used cut, averaged over sites."
        }">
          imaging fidelity (used thresholds${
            imf.source === "logged_throughout_run" ? ", throughout run · logged" : ", throughout run · this run"
          }):
          <span style="color:${infidColor(1 - (imf.mean_fidelity || 0))};font-weight:600;">${fmtPct(imf.mean_fidelity)}</span>
          mean · ${fmtPct(imf.median_fidelity)} median
          <span class="muted">(${imf.n_sites} sites${
            imf.source === "logged_throughout_run" && imf.n_in_run_updates != null
              ? `, ${imf.n_in_run_updates} update${imf.n_in_run_updates === 1 ? "" : "s"}` : ""
          })</span>
        </div>` : ""}
        ${cap ? `<div title="Maximum rearrangement survival this run COULD reach given source-site detection confidence (loss #2). Each path starts at a site detected as loaded in img1; if that detection was a false positive the path can't deliver an atom and the target stays empty. Per source we take the posterior P(atom | its img1 intensity) from that site's double-Gaussian fit; cap = mean of P over all ${cap.n_paths} paths, ± sqrt(Σ P(1−P))/N. Detection-confidence only — excludes physical loss of a correctly-detected source atom.">
          max survival cap (source loading):
          <span style="color:${infidColor(1 - (cap.cap_mean || 0))};font-weight:600;">${fmtPct(cap.cap_mean)}</span>
          &plusmn; ${fmtPct(cap.cap_sem)}
          <span class="muted">(~${fmtNum(cap.expected_nulled, 1)} of ${cap.n_paths} paths likely had no atom${
            cap.n_no_fit ? `; ${cap.n_no_fit} no-fit→1.0` : ""
          })</span>
        </div>` : ""}
        ${taf ? `<div title="Target-aware (TP) survival recomputed with outlier bad-fidelity target spots EXCLUDED — so transport survival isn't dragged down by a few mis-detected targets. A target is dropped when its detection infidelity is both a robust outlier among the targets and above an absolute floor (cut here: infid > ${fmtInfid(taf.infidelity_threshold)}). Per-site infidelities from the ${taf.infid_source === "throughout_run" ? "throughout-run logged values" : "scan-start calibration"} — no recompute needed.">
          filtered TP survival (good targets):
          <span style="color:${infidColor(1 - (taf.overall_mean || 0))};font-weight:600;">${fmtPct(taf.overall_mean)}</span>${
            taf.overall_sem != null ? ` &plusmn; ${fmtPct(taf.overall_sem)}` : ""
          }
          <span class="muted">(${taf.n_excluded} of ${taf.n_target_sites} target spot${taf.n_target_sites === 1 ? "" : "s"} excluded${
            taf.n_excluded && taf.excluded_max_infid != null ? `, worst infid ${fmtInfid(taf.excluded_max_infid)}` : ""
          })</span>
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
    // FP overlay (false-positive rate vs param) — discoverable on the sweep
    // curve like on the per-shot timeseries. Faint red, same 0–1 axis; only
    // for 2-image scans that carry an fp_mean curve.
    const fpm = summary.fp_mean || [];
    const fpSem = summary.fp_sem || [];
    const fpHasData = isSurv && fpm.some((v) => v != null && isFinite(v));
    const fpName = (summary.fp_source === "rearrange") ? "FP (target)" : "FP";
    const _num = (v) => (v == null || !isFinite(v)) ? null : v;
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
      const fpVal0 = fpHasData ? _num(fpm[0]) : null;
      const fpErr0 = fpHasData ? _num(fpSem[0]) : null;
      const hasFp0 = fpVal0 != null;
      const traces0d = [{
        x: [0, r.n_shots || 1], y: [yMean, yMean], mode: "lines",
        line: {color: "#58a6ff", width: 2, dash: "dash"}, name: yLabel,
      }];
      if (hasFp0) {
        traces0d.push({
          x: [0, r.n_shots || 1], y: [fpVal0, fpVal0], mode: "lines",
          line: {color: "#f85149", width: 1.5, dash: "dot"}, name: fpName, opacity: 0.7,
        });
      }
      const annText = `${yLabel} = ${fmtPct(yMean)}`
        + (errPct(yErr) ? " ± " + errPct(yErr) : "")
        + (hasFp0 ? `   ·   ${fpName} = ${fmtPct(fpVal0)}`
                    + (errPct(fpErr0) ? " ± " + errPct(fpErr0) : "") : "");
      Plotly.react(el, traces0d, plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: "shot # (0d, single point)" }, tickformat: ".0f" },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05], tickformat: ".2f" },
        showlegend: hasFp0,
        legend: {x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0.3)", font: {size: 10}},
        annotations: [{
          text: annText,
          xref: "paper", yref: "paper", x: 0.5, y: 0.5,
          showarrow: false, font: { size: 14, color: "#ffdd44" },
          bgcolor: "rgba(20,20,40,0.7)",
        }],
      }), plotConfig());
      setText("analysis-scan-info", `0d · ${r.n_shots || 0} shots · ${_errLabel(sweepPrefs.errMode)}`
        + (hasFp0 ? " · +FP" : ""));
      return;
    }

    // ---- 1D ----
    if (nDimsReal === 1) {
      // The single real axis may not be axis 0 (e.g. a 2-axis scan
      // filtered to one value on the other axis). Find it.
      const axisIdx = Math.max(0, dims.findIndex((d) => d > 1));
      const xs = (sweep.values && sweep.values[axisIdx]) || useY.map((_, i) => i + 1);
      const xLabel = cols[axisIdx] || "scan param";
      // Combined per-point hover: TP (or loading) ± err AND FP ± err, errors
      // at ~2 sig figs. Same string on both traces so either is hoverable.
      const hovText = useY.map((_, i) => {
        const tpE = errPct(useE && useE[i]);
        let s = `${xLabel} = ${fmtNum(xs[i], 4)}<br>${yLabel} = ${fmtPct(useY[i])}`
                + (tpE ? ` ± ${tpE}` : "");
        if (fpHasData) {
          const fE = errPct(fpSem[i]);
          s += `<br>${fpName} = ${fmtPct(fpm[i])}` + (fE ? ` ± ${fE}` : "");
        }
        return s;
      });
      const traces1d = [{
        x: xs, y: useY, text: hovText,
        error_y: useE ? {type: "data", array: useE, visible: true, color: "#1f6feb"} : undefined,
        mode: "markers+lines", name: yLabel,
        marker: {size: 8, color: "#58a6ff"}, line: {color: "#1f6feb", width: 2},
        hovertemplate: "%{text}<extra></extra>",
      }];
      if (fpHasData) {
        traces1d.push({
          x: xs, y: fpm, text: hovText, mode: "markers+lines", name: fpName, opacity: 0.7,
          error_y: fpSem.length ? {type: "data", array: fpSem, visible: true, color: "#f85149"} : undefined,
          marker: {size: 5, color: "#f85149"},
          line: {color: "#f85149", width: 1, dash: "dot"},
          hovertemplate: "%{text}<extra></extra>",
        });
      }
      Plotly.react(el, traces1d, plotLayoutFlush({
        margin: baseMargin,
        xaxis: { title: { text: xLabel } },
        yaxis: { title: { text: yLabel }, range: [-0.05, 1.05], tickformat: ".2f" },
        showlegend: fpHasData,
        legend: {x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0.3)", font: {size: 10}},
      }), plotConfig());
      setText("analysis-scan-info",
        `1d · ${xLabel} · ${dims[axisIdx]} pts · ${_errLabel(sweepPrefs.errMode)}`
        + (fpHasData ? " · FP overlay" : ""));
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

    const linAt = (ix, jy) => {
      const i0 = swap ? jy : ix, i1 = swap ? ix : jy;
      return i1 * dim0orig + i0;
    };
    if (showHeat) {
      const z = [];
      const cd = [];   // per-cell hover: TP ± err (+ FP ± err)
      for (let jy = 0; jy < ny; jy++) {
        const row = [], cdrow = [];
        for (let ix = 0; ix < nx; ix++) {
          row.push(valAt(ix, jy));
          const lin = linAt(ix, jy);
          const tpE = errPct(useE && useE[lin]);
          let s = `${yLabel} = ${fmtPct(useY[lin])}` + (tpE ? ` ± ${tpE}` : "");
          if (fpHasData) {
            const fE = errPct(fpSem[lin]);
            s += `<br>${fpName} = ${fmtPct(fpm[lin])}` + (fE ? ` ± ${fE}` : "");
          }
          cdrow.push(s);
        }
        z.push(row); cd.push(cdrow);
      }
      // Square cells = categorical equal spacing; default = numeric
      // coords (cells sized by the separation between sweep values).
      const sq = sweepPrefs.square;
      // Colorbar range: OFF (default) pins 0–1; ON lets Plotly autoscale to the
      // data range (mirrors the live Scan-curve "Autoscale" toggle).
      const cbarPinned = !sweepPrefs.cbarAuto;
      const trace = {
        z, customdata: cd, type: "heatmap", colorscale: "Viridis",
        zauto: !cbarPinned,
        zmin: cbarPinned ? 0 : undefined, zmax: cbarPinned ? 1 : undefined,
        colorbar: {title: {text: yLabel}, len: 0.9, tickformat: ".2f"},
        x: sq ? xVals.map((v) => Number(v).toPrecision(4)) : xVals,
        y: sq ? yVals.map((v) => Number(v).toPrecision(4)) : yVals,
        hovertemplate: `${xLabel}=%{x}<br>${yLabel2}=%{y}<br>%{customdata}<extra></extra>`,
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
      `2d · ${xLabel} × ${yLabel2} · ${nx}×${ny}${sweepPrefs.square ? " · square" : ""}`
      + (sweepPrefs.cbarAuto ? " · autoscale" : ""));
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
          elSwap = $("sweep-swap"), elSq = $("sweep-square"),
          elAuto = $("sweep-autoscale");
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
    if (elAuto) {
      elAuto.checked = !!sweepPrefs.cbarAuto;
      elAuto.addEventListener("change", () => {
        sweepPrefs.cbarAuto = elAuto.checked; saveSweepPrefs(); rerender();
      });
    }
  })();

  // =====================================================================
  // FILTER PANEL — one chip-row per swept-param axis. Clicking a chip
  // toggles inclusion in the allowed-values set. Each change triggers a
  // refetch of /api/runs/<id>/analysis with the filter encoded.
  // =====================================================================
  function renderAnalysisFilters(r) {
    const body = $("filter-body");
    // ALWAYS use the unfiltered sweep so chips show every possible
    // value -- the user has to know what they can still pick after
    // narrowing. (Old bug: derived chips from r.sweep, which got
    // narrowed to the filtered subset and made the others vanish.)
    const sweep = r.sweep_all || r.sweep || {};
    const cols  = sweep.cols   || [];
    const vals  = sweep.values || [];
    if (!cols.length || !vals.length) {
      // No swept axis -> no chips. Record it + let the central visibility logic
      // hide the (now LEFT-docked) Filter card.
      _filterHasAxes = false;
      updateAnalysisFloatingHosts(activeTab);
      return;
    }
    _filterHasAxes = true;
    updateAnalysisFloatingHosts(activeTab);
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
      // Infidelity may live on a different grid than the survival/FP maps
      // (cross-grid rearrangement: infidelity is on the init detection grid).
      plotSiteMap("plot-site-infid", "site-infid-info",
        ps.infid_x || ps.x, ps.infid_y || ps.y, ps.infidelity,
        "Magma", "infidelity", {mode: "infid"});
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
        ps.loading_x || ps.x, ps.loading_y || ps.y,
        ps.loading_init || ps.loading_rate, "Cividis", "loading", {});
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
    // Loading may live on a different grid than survival/FP (cross-grid run:
    // loading on the init grid, survival/FP on the target grid).
    plotSiteMap("plot-site-loading", "site-loading-info",
      ps.loading_x || ps.x, ps.loading_y || ps.y,
      ps.loading_init || ps.loading_rate, "Cividis", "loading", {infoSuffix});
    plotSiteMap("plot-site-survival", "site-survival-info",
      ps.x, ps.y, ps.survival_mean, "Viridis",
      tgtMask ? "TP (target survival)" : "survival",
      {mask: tgtMask, maskLabel: "target site", infoSuffix,
       pathsOverlay: currentPathsOverlaySegments()});
    // No non-target sites at all (e.g. the target IS the whole final array)
    // => false positives are undefined everywhere. Hide the FP card rather
    // than show an all-grey map.
    const noNonTarget = Array.isArray(ntgtMask) && ntgtMask.length > 0
      && !ntgtMask.some(Boolean);
    if (noNonTarget) {
      if (fpCard) fpCard.hidden = true;
      const fe = $("plot-site-fp");
      if (fe) Plotly.purge(fe);
    } else {
      plotSiteMap("plot-site-fp", "site-fp-info",
        ps.x, ps.y, ps.fp_rate, "Plasma", "FP",
        {mask: ntgtMask, maskLabel: "non-target site", infoSuffix});
    }
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

  // Globalize the "Yb Control" sidebar. It's position:fixed on the right edge
  // and shown on EVERY tab: we hoist it out of #tab-live to <body> so the old
  // .tab-pane[hidden] rule can't hide it. It is ALWAYS shown (not hideable) and
  // reads as a fixed continuation of the header (square corners, header-matched
  // colour -- see dashboard.css). Reserve-space ("push content aside") is
  // applied at the BODY level (body.ctrl-open) so it works for the grid tabs
  // AND the full-bleed Hardware iframes alike. Its TOP is pinned to the BOTTOM
  // of the (sticky, full-width) header -- measured from the live DOM so it's
  // exact regardless of the header's rendered height (CSS --header-h is only a
  // first-paint fallback).
  (function globalizeControlPanel() {
    const sidebar = document.querySelector(".live-sidebar");
    if (!sidebar) return;
    // Hoist to <body> once so it lives outside every tab pane. Idempotent.
    if (sidebar.parentElement !== document.body) document.body.appendChild(sidebar);
    // Always reserve its width; never collapse.
    document.body.classList.add("ctrl-open");
    document.body.classList.remove("ctrl-collapsed");
    // Pin the panel's top to the header's bottom edge so nothing is covered.
    const positionControlPanel = () => {
      const header = document.querySelector("header");
      if (!header) return;
      const h = Math.round(header.getBoundingClientRect().height);
      if (!h) return;
      sidebar.style.top = h + "px";
      sidebar.style.height = "calc(100vh - " + h + "px)";
    };
    positionControlPanel();
    // Re-measure once fonts/layout settle and whenever the header may reflow.
    window.addEventListener("load", positionControlPanel);
    window.addEventListener("resize", positionControlPanel);
    setTimeout(positionControlPanel, 300);
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
        pollSnapshot();   // the scan figure (cbar_scale) lives in the snapshot group now
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
      // Start / Pause are glyph-only icon buttons, so prefer the aria-label
      // for the toast (otherwise it would read e.g. "▶ sent").
      const lbl = btn.getAttribute("aria-label") || btn.textContent.trim();
      btn.addEventListener("click", async () => {
        try { await postControl(path); toast(lbl + " sent"); }
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
          } else if (action === "shutdown") {
            path = "/api/control/shutdown";
            label = "Shutdown";
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
      // log10 of the clipped infidelity, plain Magma, range [-4, -0.3].
      // Plain (un-reversed) Magma => bright/colorful = HIGH infidelity = bad,
      // dark = low infidelity = good. Raw value shown via customdata.
      const logv = vIn.map((v) => Math.log10(
        Math.min(1, Math.max(1e-6, (v == null || !isFinite(v)) ? 1e-6 : v))));
      traces.push({
        x: xIn, y: yIn, type: "scattergl", mode: "markers",
        // customdata[k] = [canonical bitstring site index, raw value]. Use the
        // CANONICAL index (iIn) for the hover, not %{pointNumber} — the latter
        // is the position within the plotted subset, which differs between maps
        // when target/non-target masks hide sites.
        customdata: vIn.map((v, j) => [iIn[j], v]),
        marker: {
          size: siteDotSize, color: logv, colorscale: "Magma",
          reversescale: false, cmin: -4, cmax: -0.3, line: {width: 0},
          colorbar: {title: {text: "log10 infid"}, len: 0.9},
        },
        hovertemplate: "site %{customdata[0]}: %{customdata[1]:.2e}<extra></extra>",
        name: label,
      });
    } else {
      traces.push({
        x: xIn, y: yIn, type: "scattergl", mode: "markers",
        customdata: iIn,   // canonical bitstring site index (shared everywhere)
        marker: {
          size: siteDotSize, color: vIn, colorscale,
          cmin: vmin, cmax: vmax,
          line: {width: 0},
          colorbar: {title: {text: label}, len: 0.9, tickformat: ".2f"},
        },
        hovertemplate: "site %{customdata}: %{marker.color:.3f}<extra></extra>",
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
    // Highlight rows in the .run-picker: whole GROUP = .in-tray, the PRIMARY
    // (first-selected) also gets .is-primary (slightly darker). Class-toggle
    // only -- never rebuilds the table, so a click never detaches its row.
    const primary = trayPrimary();
    $$("#runs-table .run-row").forEach((row) => {
      const sid = row.dataset.scanId;
      row.classList.toggle("in-tray", traySet.has(sid));
      row.classList.toggle("is-primary", sid === primary);
      const addBtn = row.querySelector(".run-add");
      if (addBtn) addBtn.textContent = traySet.has(sid) ? "✓" : "+";
    });
  }

  // Highlight Queue/History rows whose scan_id is in the selected group (same
  // colour scheme; primary darker). Queue rows carry data-scan-id (server
  // stamps it from file_id). Cheap class-toggle, called on every selection
  // change + after each queue render.
  function syncQueueSelectionHighlight() {
    const primary = trayPrimary();
    $$('#queue-table tr[data-scan-id], #history-table tr[data-scan-id], ' +
       '#pq-queue-table tr[data-scan-id], #pq-history-table tr[data-scan-id]')
      .forEach((row) => {
        const sid = row.dataset.scanId;
        row.classList.toggle("in-tray", traySet.has(sid));
        row.classList.toggle("is-primary", sid === primary);
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
      syncQueueSelectionHighlight();
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
    syncQueueSelectionHighlight();
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
      _analysisShownFor = "grp:" + Array.from(traySet).join(",");
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
    // Paste-box takes precedence: a 14-digit scan_id there (not the current
    // selection) becomes the sole PRIMARY and loads -- list-independent, so it
    // works for ANY scan ever recorded, not just the most-recent in the picker.
    const pasted = ($("manual-scan-id").value || "").trim();
    if (/^\d{14}$/.test(pasted) && pasted !== selectedScanId) {
      trayReplace(pasted);
      selectedScanId = pasted;
      persistSelected(pasted);
      syncSelectionEverywhere();
      loadAnalysis(pasted);
      return;
    }
    if (!traySet.size) { toast("tray is empty", "warn"); return; }
    if (traySet.size === 1) {
      loadAnalysis(trayPrimary());
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
        `<tr><td colspan="10" class="muted">runner unreachable: ${escHtml(e.message)}</td></tr>`;
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

  // Shared formatting for the queue/history tables. "#seq" is the actual
  // sequence count after StackNum stacking (summary.total_per_group — the same
  // number the Tk monitor's "Reps" column shows); "t/seq" = run duration / that
  // count, shown only once a job has finished (a running job's elapsed/total
  // under-counts because not all sequences have run yet).
  const fmtDur = (secs) => {
    if (secs == null || !isFinite(secs) || secs < 0) return "";
    secs = Math.floor(secs);
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
  };
  const entryNseq = (r) => {
    const s = r.summary || {};
    return s.total_per_group || s.num_per_group || "";
  };
  // "#seq" cell = actual sequences run / submitted total. Actual = r.seq_num
  // from the ExptServer (live delta while running, frozen for history; 0 for a
  // not-yet-started queued job); submitted = total_per_group.
  const entrySeqCount = (r) => {
    const sub = entryNseq(r);
    let act = r.seq_num;
    if (act == null && r.state === "queued") act = 0;
    if (!sub && act == null) return "";
    return `${act != null ? act : "?"} / ${sub || "?"}`;
  };
  const entryDurSecs = (r) => {
    if (!r.start_ts) return null;
    const end = r.finish_ts || (Date.now() / 1000);
    return Math.max(0, end - r.start_ts);
  };
  const entryDur = (r) => {
    const d = entryDurSecs(r);
    return d == null ? "" : fmtDur(d);
  };
  const entryPerSeq = (r) => {
    if (!r.start_ts || !r.finish_ts) return "";   // only meaningful once finished
    // Divide by sequences ACTUALLY run (seq_num), NOT the planned total — for
    // run-until-stopped scans the planned total is a huge sentinel and would
    // make t/seq meaninglessly small.
    const n = r.seq_num;
    if (!n || n <= 0) return "";
    const t = (r.finish_ts - r.start_ts) / n;
    if (!isFinite(t) || t < 0) return "";
    return t >= 60 ? fmtDur(t) : t.toFixed(t < 10 ? 2 : 1) + "s";
  };
  // Full date+time (started/finished can span days, so include the date).
  const fmtTsFull = (epoch) =>
    epoch ? new Date(epoch * 1000).toLocaleString() : "—";
  // Hover tooltip for a queue/history row's status cell — mirrors the Tk
  // monitor's status-column tooltip (Added / Started / Duration), plus the
  // explicit Finished time.
  const entryTimesTitle = (r) => {
    const lines = [];
    if (r.enqueued_ts) lines.push(`Added:    ${fmtTsFull(r.enqueued_ts)}`);
    if (r.start_ts)    lines.push(`Started:  ${fmtTsFull(r.start_ts)}`);
    if (r.finish_ts)   lines.push(`Finished: ${fmtTsFull(r.finish_ts)}`);
    else if (r.start_ts) lines.push(`Finished: (in progress)`);
    const d = entryDur(r);
    if (d) lines.push(`Duration: ${d}`);
    return lines.join("\n");
  };

  function renderQueueTable(q) {
    const queued = q.queued || [];
    const running = q.running ? [q.running] : [];
    const history = q.history || [];
    setText("queue-counts",
      `${queued.length} queued · ${running.length} running · ${history.length} history`);
    const rows = [...running, ...queued].map((r) => `
      <tr ${r.scan_id ? `data-scan-id="${r.scan_id}" class="q-linked"` : ""}>
        <td class="mono">${r.id}</td>
        <td>${r.kind || "job"}</td>
        <td class="mono">${fmtTs(r.enqueued_ts)}</td>
        <td class="mono">${entryDur(r)}</td>
        <td class="mono">${entrySeqCount(r)}</td>
        <td class="mono">${entryPerSeq(r)}</td>
        <td title="${escHtml(entryTimesTitle(r))}">${r.state}</td>
        <td>${escHtml(r.label || r.seqName || "")}</td>
        <td class="mono">${r.file_id || ""}</td>
        <td>
          ${r.state === "queued" ? `
            <button class="ghost" data-cancel="${r.id}" style="font-size:10px;padding:2px 8px;">cancel</button>
            <button class="ghost" data-move-up="${r.id}" style="font-size:10px;padding:2px 8px;">↑</button>
            <button class="ghost" data-move-down="${r.id}" style="font-size:10px;padding:2px 8px;">↓</button>
          ` : `
            <button class="ghost q-abort" data-abort-running="${r.id}"
              title="Abort the running scan — stops after the current shot (advances to the next queued scan, if any)"
              style="font-size:10px;padding:2px 8px;color:#ff8585;border-color:rgba(248,81,73,0.4);">abort</button>
          `}
          ${r.descriptor ? `<button class="ghost" data-requeue="${r.id}" title="Queue a new copy with exactly the same parameters" style="font-size:10px;padding:2px 8px;">re-queue</button>` : ""}
        </td>
      </tr>
    `).join("");
    $("queue-table").querySelector("tbody").innerHTML =
      rows || '<tr><td colspan="10" class="muted">queue empty</td></tr>';
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
    // Abort the RUNNING scan (destructive → confirm-token then POST, same as
    // the Live-tab control sidebar). Stops after the current shot.
    $$("[data-abort-running]", $("queue-table")).forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm("Abort the running scan? It stops after the current shot"
                     + " (then advances to the next queued scan, if any).")) return;
        btn.disabled = true;
        try {
          const tok = await api("/api/control/confirm_token?action=abort");
          await api(`/api/control/abort?confirm=${encodeURIComponent(tok.token)}`,
                    {method: "POST"});
          toast("Abort sent");
          pollQueue();
        } catch (e) {
          toast(e.message || "abort failed", "bad");
          btn.disabled = false;
        }
      });
    });
    // History
    $("history-table").querySelector("tbody").innerHTML =
      history.slice(0, 30).map((r) => `
        <tr ${r.scan_id ? `data-scan-id="${r.scan_id}" class="q-linked"` : ""}>
          <td class="mono">${r.id}</td>
          <td>${r.kind || "job"}</td>
          <td>${escHtml(r.label || r.seqName || "")}</td>
          <td class="mono">${fmtTs(r.enqueued_ts)}</td>
          <td class="mono">${entryDur(r)}</td>
          <td class="mono">${entrySeqCount(r)}</td>
          <td class="mono">${entryPerSeq(r)}</td>
          <td class="${r.status === "ok" ? "ok" : "bad"}" title="${escHtml(entryTimesTitle(r))}">${escHtml(r.status || r.state || "")}</td>
          <td class="mono">${r.file_id || ""}</td>
          <td class="muted">${escHtml((r.error_message || "").slice(0, 80))}</td>
          <td>${r.descriptor ? `<button class="ghost" data-requeue="${r.id}" title="Queue a new copy with exactly the same parameters (uses today's code)" style="font-size:10px;padding:2px 8px;">re-queue</button>${r.file_id ? `<button class="ghost" data-requeue-code="${r.id}" title="Re-queue with the EXACT code that ran originally — replays this run's code snapshot (YbSeqs/YbSteps/YbScans)" style="font-size:10px;padding:2px 6px;">+code</button>` : ""}` : ""}</td>
        </tr>`).join("")
      || '<tr><td colspan="11" class="muted">empty</td></tr>';
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
    // Click a linked queue/history row (anywhere but its action buttons) to
    // open that run in Analysis. The row carries data-scan-id (server-stamped
    // from file_id).
    $$('tr.q-linked', $("queue-table")).concat($$('tr.q-linked', $("history-table")))
      .forEach((row) => {
        row.style.cursor = "pointer";
        row.addEventListener("click", (e) => {
          if (e.target.closest("button")) return;   // let action buttons act
          const sid = row.dataset.scanId;
          if (!sid) return;
          selectPrimary(sid);   // sets primary + loads analysis (background)
          setTab("analysis");   // then reveal it
        });
      });
    // Reflect the current selection on the freshly-rendered rows.
    syncQueueSelectionHighlight();
  }

  // ---- Full-queue popup (opened from the Live sidebar's "full ›") ----
  // A side panel docked left of the control sidebar showing the SAME detail as
  // the Queue tab (active job, queued+running, history; re-queue / +code /
  // cancel / move / abort; click a row to open it in Analysis). Self-contained
  // pq-* ids + delegated wiring so it never collides with the Queue tab's
  // document-wide [data-requeue]/[data-cancel] handlers. Reuses the queue-table
  // formatting helpers above. Kept live by pollLive while open. (The standalone
  // Queue tab is intentionally kept for now.)
  let queuePopupOpen = false;

  function renderQueuePopup(q) {
    if (!q) return;
    const queued  = q.queued || [];
    const running = q.running ? [q.running] : [];
    const history = q.history || [];
    setText("pq-counts",
      `${queued.length} queued · ${running.length} running · ${history.length} history`);
    const r0 = q.running;
    const nameEl = $("pq-active-name");
    const detEl  = $("pq-active-detail-body");
    if (nameEl && detEl) {
      if (r0) {
        nameEl.innerHTML =
          `${escHtml(r0.label || r0.seqName || "—")} <span class="badge mono">#${r0.id}</span>`;
        detEl.innerHTML = `
          <dl class="kv">
            <dt>id</dt><dd class="mono">${r0.id}</dd>
            <dt>kind</dt><dd>${r0.kind || "job"}</dd>
            <dt>seq</dt><dd class="mono">${escHtml(r0.seqName || "—")}</dd>
            <dt>label</dt><dd>${escHtml(r0.label || "—")}</dd>
            <dt>started</dt><dd>${fmtTs(r0.start_ts)}</dd>
            <dt>file_id</dt><dd class="mono">${r0.file_id || "—"}</dd>
          </dl>`;
      } else {
        nameEl.innerHTML = '<span class="muted">(none)</span>';
        detEl.innerHTML  = '<span class="muted">(no active job)</span>';
      }
    }
    const lbl = (r) => escHtml(r.label || r.seqName || "");
    const qrows = [...running, ...queued].map((r) => `
      <tr ${r.scan_id ? `data-scan-id="${r.scan_id}" class="q-linked"` : ""}>
        <td class="mono">${r.id}</td>
        <td>${r.kind || "job"}</td>
        <td class="mono">${fmtTs(r.enqueued_ts)}</td>
        <td class="mono">${entryDur(r)}</td>
        <td class="mono">${entrySeqCount(r)}</td>
        <td class="mono">${entryPerSeq(r)}</td>
        <td title="${escHtml(entryTimesTitle(r))}">${r.state}</td>
        <td class="qp-trunc" title="${lbl(r)}">${lbl(r)}</td>
        <td class="mono qp-trunc" title="${escHtml(String(r.file_id || ""))}">${r.file_id || ""}</td>
        <td class="qp-actions">
          ${r.state === "queued" ? `
            <button class="ghost" data-pq-cancel="${r.id}" style="font-size:10px;padding:2px 8px;">cancel</button>
            <button class="ghost" data-pq-up="${r.id}" style="font-size:10px;padding:2px 8px;">↑</button>
            <button class="ghost" data-pq-down="${r.id}" style="font-size:10px;padding:2px 8px;">↓</button>
          ` : `
            <button class="ghost q-abort" data-pq-abort="${r.id}"
              title="Abort the running scan — stops after the current shot (advances to the next queued scan, if any)"
              style="font-size:10px;padding:2px 8px;color:#ff8585;border-color:rgba(248,81,73,0.4);">abort</button>
          `}
          ${r.descriptor ? `<button class="ghost" data-pq-requeue="${r.id}" title="Queue a new copy with exactly the same parameters" style="font-size:10px;padding:2px 8px;">re-queue</button>` : ""}
        </td>
      </tr>`).join("");
    const qbody = $("pq-queue-table") && $("pq-queue-table").querySelector("tbody");
    if (qbody) qbody.innerHTML = qrows || '<tr><td colspan="10" class="muted">queue empty</td></tr>';

    const hrows = history.slice(0, 30).map((r) => `
      <tr ${r.scan_id ? `data-scan-id="${r.scan_id}" class="q-linked"` : ""}>
        <td class="mono">${r.id}</td>
        <td>${r.kind || "job"}</td>
        <td class="qp-trunc" title="${lbl(r)}">${lbl(r)}</td>
        <td class="mono">${fmtTs(r.enqueued_ts)}</td>
        <td class="mono">${entryDur(r)}</td>
        <td class="mono">${entrySeqCount(r)}</td>
        <td class="mono">${entryPerSeq(r)}</td>
        <td class="${r.status === "ok" ? "ok" : "bad"}" title="${escHtml(entryTimesTitle(r))}">${escHtml(r.status || r.state || "")}</td>
        <td class="mono qp-trunc" title="${escHtml(String(r.file_id || ""))}">${r.file_id || ""}</td>
        <td class="muted">${escHtml(r.error_message || "")}</td>
        <td class="qp-actions">${r.descriptor ? `<button class="ghost" data-pq-requeue="${r.id}" title="Queue a new copy with exactly the same parameters (uses today's code)" style="font-size:10px;padding:2px 8px;">re-queue</button>${r.file_id ? `<button class="ghost" data-pq-requeue-code="${r.id}" title="Re-queue with the EXACT code that ran originally — replays this run's code snapshot (YbSeqs/YbSteps/YbScans)" style="font-size:10px;padding:2px 6px;">+code</button>` : ""}` : ""}</td>
      </tr>`).join("");
    const hbody = $("pq-history-table") && $("pq-history-table").querySelector("tbody");
    if (hbody) hbody.innerHTML = hrows || '<tr><td colspan="11" class="muted">empty</td></tr>';

    // Every data column is truncated by CSS; expose each cell's full value on
    // hover (e.g. the complete run error). Skip the actions column.
    $$("#pq-queue-table td:not(.qp-actions), #pq-history-table td:not(.qp-actions)")
      .forEach((td) => { const txt = td.textContent.trim(); if (txt) td.title = txt; });

    syncQueueSelectionHighlight();
  }

  async function pqRequeue(btn, id, withCode) {
    if (!id) return;
    btn.disabled = true;
    try {
      const r = await api(`/api/queue/requeue/${id}${withCode ? "?code=1" : ""}`, {method: "POST"});
      toast(`Re-queued #${id} → #${r.descriptor_id}${withCode ? " (orig code)" : ""}`);
      refreshQueuePopup();
    } catch (e) { toast("Re-queue failed: " + (e.message || e), "bad"); btn.disabled = false; }
  }

  async function refreshQueuePopup() {
    try { renderQueuePopup(await api("/api/queue")); } catch (e) { /* keep last-known */ }
  }
  function openQueuePopup() {
    queuePopupOpen = true;
    const p = $("queue-popup");
    if (p) p.hidden = false;
    positionQueuePopup();                          // place beside the queue section now...
    refreshQueuePopup().then(positionQueuePopup);  // ...and again once content height is known
  }
  function closeQueuePopup() {
    queuePopupOpen = false;
    const p = $("queue-popup");
    if (p) p.hidden = true;
  }
  function toggleQueuePopup() { queuePopupOpen ? closeQueuePopup() : openQueuePopup(); }

  // Place the popup beside the control-panel queue section, vertically centered
  // on it and clamped to the viewport, with a caret pointing at it -- so it
  // reads as a popout FROM the queue area, not a detached panel.
  function positionQueuePopup() {
    const pop = $("queue-popup"), sec = $("ctrl-queue-section");
    if (!pop || !sec || pop.hidden) return;
    const r = sec.getBoundingClientRect();
    const center = r.top + r.height / 2;
    const h = pop.offsetHeight || 360;
    const margin = 12;
    let top = Math.max(margin, Math.min(center - h / 2, window.innerHeight - h - margin));
    pop.style.top = top + "px";
    // Caret tracks the queue section's center, relative to the popup's top edge.
    pop.style.setProperty("--qp-caret-top",
      Math.max(16, Math.min(center - top, h - 16)) + "px");
  }

  function wireQueuePopup() {
    // Clicking ANYWHERE in the control-panel queue section drives the popup;
    // the per-row re-queue / +code buttons are skipped (wireSidebarQueue acts
    // on those). Covers the "full ›" link too (it's inside the section).
    //   - click a run (mini-view row / active line): 1st click selects it (so
    //     it highlights in the popout) and opens the popout; clicking the SAME
    //     run again opens it in Analysis.
    //   - click anywhere else: toggle the popout open/closed.
    const section = $("ctrl-queue-section");
    if (section) section.addEventListener("click", (e) => {
      if (e.target.closest("[data-sb-requeue], [data-sb-requeue-code]")) return;
      const runEl = e.target.closest("[data-scan-id]");
      const sid = runEl && runEl.dataset.scanId;
      if (sid) {
        if (sid === selectedScanId) { setTab("analysis"); return; }  // 2nd click -> analyze
        selectPrimary(sid);    // 1st click -> select + highlight (incl. in the popout)
        openQueuePopup();
        return;
      }
      toggleQueuePopup();
    });
    const closeBtn = $("queue-popup-close");
    if (closeBtn) closeBtn.addEventListener("click", closeQueuePopup);
    const refreshBtn = $("pq-refresh");
    if (refreshBtn) refreshBtn.addEventListener("click", refreshQueuePopup);

    const pop = $("queue-popup");
    if (pop) pop.addEventListener("click", async (e) => {
      const t = e.target;
      const cancel = t.closest("[data-pq-cancel]");
      if (cancel) {
        try { await api(`/api/queue/cancel/${cancel.dataset.pqCancel}`, {method: "POST"}); toast("Cancelled"); refreshQueuePopup(); }
        catch (err) { toast("Cancel failed", "bad"); }
        return;
      }
      const up = t.closest("[data-pq-up]"), down = t.closest("[data-pq-down]");
      if (up || down) {
        const mv = up || down;
        const id = mv.dataset.pqUp || mv.dataset.pqDown;
        try { await api(`/api/queue/move/${id}/${up ? "up" : "down"}`, {method: "POST"}); refreshQueuePopup(); }
        catch (err) { toast("Move failed", "bad"); }
        return;
      }
      const abort = t.closest("[data-pq-abort]");
      if (abort) {
        if (!confirm("Abort the running scan? It stops after the current shot"
                     + " (then advances to the next queued scan, if any).")) return;
        abort.disabled = true;
        try {
          const tok = await api("/api/control/confirm_token?action=abort");
          await api(`/api/control/abort?confirm=${encodeURIComponent(tok.token)}`, {method: "POST"});
          toast("Abort sent"); refreshQueuePopup();
        } catch (err) { toast(err.message || "abort failed", "bad"); abort.disabled = false; }
        return;
      }
      const rqc = t.closest("[data-pq-requeue-code]");
      if (rqc) { pqRequeue(rqc, rqc.dataset.pqRequeueCode, true); return; }
      const rq = t.closest("[data-pq-requeue]");
      if (rq) { pqRequeue(rq, rq.dataset.pqRequeue, false); return; }
      if (t.closest("button")) return;
      const row = t.closest("tr[data-scan-id]");
      // Open the run in Analysis; setTab() also closes this popup.
      if (row && row.dataset.scanId) { selectPrimary(row.dataset.scanId); setTab("analysis"); }
    });

    // Esc closes; a mousedown outside the popup AND the queue section closes
    // (it overlays the live view, which stays interactive behind it). The
    // queue section is excluded so its own toggle handles open/close there.
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && queuePopupOpen) closeQueuePopup();
    });
    // CAPTURE phase (3rd arg true): Plotly plots stopPropagation on mousedown
    // for their drag/zoom handlers, so a bubble-phase listener never fires when
    // you click on a plot -- which is most of the screen, making "click out to
    // close" feel broken. Capture runs document-first, before any descendant
    // can swallow the event, so an outside click always closes the popup. We
    // don't preventDefault, so the plot still gets its click.
    document.addEventListener("mousedown", (e) => {
      if (!queuePopupOpen) return;
      if (e.target.closest("#queue-popup") || e.target.closest("#ctrl-queue-section")) return;
      closeQueuePopup();
    }, true);
    window.addEventListener("resize", () => { if (queuePopupOpen) positionQueuePopup(); });
  }

  // ---- Manual scan_id loader ----
  // Paste-box <-> picker row sync. Typing in the paste-box scrolls the
  // matching picker row into view + highlights it (without loading it,
  // since the user might still be typing). Pressing Enter or clicking
  // Load triggers the actual analysis. Picker row click writes into
  // the paste-box (handled inside renderRunsTable -> loadAnalysis).
  $("manual-scan-load").addEventListener("click", () => {
    const id = ($("manual-scan-id").value || "").trim();
    if (!id) { toast("scan_id required", "warn"); return; }
    // List-independent: loads by id even if the scan isn't a visible row.
    trayReplace(id);
    selectedScanId = id;
    persistSelected(id);
    syncSelectionEverywhere();
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
  // LOGS TAB  — browse + view the server/monitor/backend log files on disk
  // =====================================================================
  let logsSelected = null;     // {category, name} of the open file, or null
  let logsListCache = null;    // last /api/logs/list payload (for re-highlight)
  let logsHighlight = null;    // substring to highlight + jump to in the open
                               // file (set when opened from the shot-health chip
                               // on an error); cleared on a manual file pick.

  function fmtBytes(n) {
    if (n == null) return "—";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / 1024 / 1024).toFixed(1) + " MB";
  }

  async function pollLogs() {
    // Refresh the file list; if a file is open, refresh its contents too so
    // an open log "follows" a running server (tail mode auto-scrolls).
    try {
      logsListCache = await api("/api/logs/list");
      renderLogsList();
    } catch (e) {
      const el = $("logs-list");
      if (el) el.innerHTML = '<div class="hint">log list unavailable (' +
        escHtml(e.message || "") + ')</div>';
    }
    if (logsSelected) { try { await loadLogFile({keepScroll: true}); } catch (e) {} }
  }

  function renderLogsList() {
    const el = $("logs-list");
    if (!el) return;
    const groups = (logsListCache && logsListCache.groups) || [];
    if (!groups.length) { el.innerHTML = '<div class="hint">no log files found</div>'; return; }
    let html = "";
    for (const g of groups) {
      html += `<div class="logs-group-label">${escHtml(g.label)} ` +
        `<span class="muted">(${(g.files || []).length})</span></div>`;
      if (!g.files || !g.files.length) {
        html += '<div class="hint" style="padding:2px 10px;">— none —</div>';
        continue;
      }
      for (const f of g.files) {
        const on = (logsSelected && logsSelected.category === g.category &&
                    logsSelected.name === f.name) ? " active" : "";
        html += `<div class="logs-file${on}" data-cat="${escHtml(g.category)}" ` +
          `data-name="${escHtml(f.name)}" title="${escHtml(f.name)}">` +
          `<span class="logs-file-name mono">${escHtml(f.name)}</span>` +
          `<span class="logs-file-meta muted">${fmtBytes(f.size)} · ` +
          `${escHtml(f.mtime_iso || "")}</span></div>`;
      }
    }
    el.innerHTML = html;
    el.querySelectorAll(".logs-file").forEach((row) => {
      row.addEventListener("click", () => {
        logsSelected = {category: row.dataset.cat, name: row.dataset.name};
        logsHighlight = null;   // manual pick clears any error jump-to
        renderLogsList();
        loadLogFile({keepScroll: false});
      });
    });
  }

  async function loadLogFile(opts) {
    opts = opts || {};
    const view = $("logs-view");
    if (!view || !logsSelected) return;
    const tail = $("logs-tail-toggle") ? $("logs-tail-toggle").checked : true;
    const q = new URLSearchParams({
      category: logsSelected.category, name: logsSelected.name,
      tail: tail ? "1" : "0"});
    const wasBottom = (view.scrollTop + view.clientHeight >= view.scrollHeight - 40);
    try {
      const r = await api("/api/logs/file?" + q.toString());
      const meta = r.truncated
        ? `tail ${fmtBytes(r.returned_bytes)} of ${fmtBytes(r.total_bytes)}`
        : fmtBytes(r.total_bytes);
      setText("logs-view-title", logsSelected.name + "  (" + meta + ")");
      const raw = "/api/logs/file?" + q.toString() + "&raw=1";
      const dl = $("logs-download"); if (dl) dl.href = raw;
      const text = r.text || "(empty)";
      // When opened from the shot-health chip on an error, highlight the last
      // occurrence of the error message and scroll to it (once). Otherwise show
      // plain text and tail to "now". lastIndexOf falls back gracefully if the
      // message isn't in the loaded tail.
      const hl = logsHighlight && logsHighlight.trim();
      const idx = hl ? text.lastIndexOf(hl) : -1;
      if (idx >= 0) {
        view.innerHTML = escHtml(text.slice(0, idx))
          + '<mark id="log-error-hit" class="log-hit">'
          + escHtml(text.slice(idx, idx + hl.length)) + '</mark>'
          + escHtml(text.slice(idx + hl.length));
        if (!opts.keepScroll) {
          const hit = document.getElementById("log-error-hit");
          if (hit) hit.scrollIntoView({block: "center"});
        }
      } else {
        view.textContent = text;
        if (!opts.keepScroll || wasBottom) view.scrollTop = view.scrollHeight;
      }
    } catch (e) {
      setText("logs-view-title", logsSelected.name);
      view.textContent = "(failed to load: " + (e.message || e) + ")";
    }
  }

  if ($("logs-tail-toggle")) {
    $("logs-tail-toggle").addEventListener("change", () => {
      if (logsSelected) loadLogFile({keepScroll: false});
    });
  }
  if ($("logs-refresh-btn")) {
    $("logs-refresh-btn").addEventListener("click", () => pollLogs());
  }

  // ============================ MOLECUBE TAB ============================
  // FPGA1 DDS/TTL/clock control via /api/molecube/* (master-gated). When the
  // gate is closed every call returns 403 and we render a clear "disabled"
  // banner -- so this tab is harmless until the gate is opened. These helpers
  // only ever talk to OUR server's proxy routes, never the daemon directly.
  let _mcLastStateId = null;
  let _mcLastNameId = null;   // names bump name_id (NOT state_id) -> rebuild on either

  function _mcFmt(v, d) {
    return (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toFixed(d);
  }

  function mcSetPill(state, text) {
    const p = $("mc-conn-pill");
    if (!p) return;
    p.textContent = text;
    p.className = "mc-pill mc-pill-" + state;   // ok | warn | err | off
  }

  async function mcPost(path, body) {
    try {
      const r = await api(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body || {}),
      });
      if (r && r.ok === false) toast("molecube: " + (r.error || "failed"), "err");
      _mcLastStateId = null; _mcLastNameId = null;   // force a re-render on the next poll
      await pollMolecube();
      return r;
    } catch (e) {
      if (e.status === 403) toast("Molecube writes are disabled (read-only mode).", "warn");
      else toast("molecube: " + (e.message || e), "err");
      return null;
    }
  }

  function renderMolecubeGated(body) {
    mcSetPill("off", "disabled");
    const b = $("mc-gate-banner");
    if (b) {
      b.hidden = false;
      b.innerHTML =
        "<b>Molecube control is disabled.</b> Every <code>/api/molecube/*</code> endpoint " +
        "is closed by a master safety gate (HTTP&nbsp;403). Reads and writes are fully " +
        "implemented and tested, but no user can reach the live FPGA daemon yet.<br>" +
        "<span class='hint'>To enable later: start <code>run_monitor</code> with " +
        "<code>YB_MOLECUBE_ENABLED=1</code>, or flip <code>_molecube_gate_open()</code> " +
        "in <code>dashboard.py</code>.</span>";
    }
    setText("mc-status-line", (body && body.error) || "gate closed");
    const dds = $("mc-dds-body"); if (dds) dds.innerHTML = "<div class='hint'>—</div>";
    const ttl = $("mc-ttl-grid"); if (ttl) ttl.innerHTML = "<div class='hint'>—</div>";
    setText("mc-dds-count", ""); setText("mc-ttl-count", "");
    _mcLastStateId = null; _mcLastNameId = null;
  }

  async function pollMolecube() {
    let snap;
    try {
      snap = await api("/api/molecube/snapshot");
    } catch (e) {
      if (e.status === 403) { renderMolecubeGated(e.body); return; }
      mcSetPill("err", "error");
      setText("mc-status-line", "snapshot failed: " + (e.message || e));
      return;
    }
    const b = $("mc-gate-banner"); if (b) b.hidden = true;
    renderMolecube(snap);
  }

  function renderMolecube(snap) {
    if (!snap.connected) {
      mcSetPill("err", "unreachable");
      setText("mc-status-line", "daemon unreachable at " + (snap.url || "?") + " — " +
        ((snap.errors && snap.errors.state_id) || ""));
      return;
    }
    mcSetPill("ok", "connected");
    // Read-only banner + dimming when the write gate is closed (the default).
    const readOnly = snap.writes_enabled === false;
    const pane = $("tab-molecube");
    if (pane) pane.classList.toggle("mc-readonly", readOnly);
    const bn = $("mc-gate-banner");
    if (bn) {
      if (readOnly) {
        bn.hidden = false;
        bn.className = "mc-banner mc-banner-info";
        bn.innerHTML =
          "<b>Read-only mode.</b> Live FPGA values are shown and auto-refresh, but all " +
          "controls (Set / override / reset / TTL toggles / clock) are disabled. " +
          "<span class='hint'>Enable control later with <code>YB_MOLECUBE_WRITES=1</code>, " +
          "or <code>_molecube_writes_open()</code> in <code>dashboard.py</code>.</span>";
      } else {
        bn.hidden = true; bn.className = "mc-banner";
      }
    }
    const errs = (snap.errors && Object.keys(snap.errors).length)
      ? "  ⚠ " + Object.keys(snap.errors).join(",") : "";
    setText("mc-status-line",
      "server " + (snap.server_id != null ? snap.server_id : "?") +
      " · state " + (snap.state_id != null ? snap.state_id : "?") +
      " · clock " + (snap.clock != null ? snap.clock : "?") +
      " · max_ttl " + (snap.max_ttl != null ? snap.max_ttl : "?") + errs);

    setText("mc-clock-cur", snap.clock != null ? ("current " + snap.clock) : "");
    const ci = $("mc-clock-input");
    if (ci && document.activeElement !== ci && snap.clock != null && ci.value === "")
      ci.value = snap.clock;

    // Rebuild the DDS/TTL panels only when device state OR names changed (names
    // bump name_id, not state_id) AND no DDS value / channel-name field is being
    // edited (so we never stomp a value or a name mid-type).
    const ae = document.activeElement;
    const editing = ae && ae.classList &&
      (ae.classList.contains("mc-in") || ae.classList.contains("mc-name-in"));
    if (!editing &&
        (snap.state_id !== _mcLastStateId || snap.name_id !== _mcLastNameId)) {
      _mcLastStateId = snap.state_id;
      _mcLastNameId = snap.name_id;
      renderMcDds(snap.dds || []);
      if (snap.ttl_reads_enabled === false) renderMcTtlDisabled();
      else renderMcTtl(snap.ttl || []);
      api("/api/molecube/startup").then((r) => {
        if (r && r.startup != null) setText("mc-startup-view", r.startup || "(empty)");
      }).catch(() => {});
    }
  }

  function _mcCell(chn, type, val, dec, ovr, ovrVal) {
    const shown = _mcFmt(val, dec);
    const ovrTag = ovr
      ? "<span class='mc-ovr-tag' title='override = " + _mcFmt(ovrVal, dec) + "'>▲</span>"
      : "";
    return "<td class='mc-cell" + (ovr ? " mc-ovr" : "") + "'><div class='mc-cell-wrap'>" +
      "<input class='mc-in' type='number' step='any' value='" +
        (shown === "—" ? "" : shown) + "' data-mc-chn='" + chn +
        "' data-mc-type='" + type + "'>" +
      "<button class='mc-mini mc-set' data-mc-chn='" + chn + "' data-mc-type='" + type +
        "'>Set</button>" +
      "<button class='mc-mini mc-ovr-btn" + (ovr ? " on" : "") + "' data-mc-chn='" + chn +
        "' data-mc-type='" + type + "' title='Toggle override at the field value'>ovr</button>" +
      ovrTag + "</div></td>";
  }

  // A read-only channel-name <td>, truncated to 20 chars with an ellipsis; the full
  // name stays in the cell's title tooltip. Used by the NI DAC table, whose names are
  // expConfig aliases (config-derived, NOT the molecube daemon's editable names).
  function mcNameCell(name) {
    const s = String(name == null ? "" : name);
    const safe = escAttr(s);
    if (s.length > 20)
      return "<td class='mc-name' title='" + safe + "'>" + escHtml(s.slice(0, 20)) + "&hellip;</td>";
    return "<td class='mc-name'>" + escHtml(s) + "</td>";
  }

  // A click-to-edit channel-name widget shared by the DDS table + TTL chips. It shows
  // the name as plain TEXT (exactly as before); only when you click the text does an
  // inline editor (a box + a ✓ button) appear in its place. Confirm with Enter or ✓,
  // cancel with Escape or by clicking away. The write goes to the daemon's name store
  // via /api/molecube/<kind>/name -- clicking the text never toggles a TTL output
  // (mcControlClick returns early for any click inside a name field).
  function mcNameField(chn, kind, name) {
    const full = String(name == null ? "" : name);
    const safe = escAttr(full);
    const empty = full === "";
    return "<span class='mc-name-field' data-mc-name-chn='" + chn +
        "' data-mc-name-kind='" + kind + "'>" +
      "<span class='mc-name-text" + (empty ? " mc-name-empty" : "") + "' title='" +
        (empty ? "Click to name this channel" : safe) + "'>" +
        (empty ? "name" : escHtml(full)) + "</span>" +
      "<span class='mc-name-editbox'>" +
        "<input class='mc-name-in' type='text' maxlength='63' value='" + safe +
          "' data-mc-name-orig='" + safe + "'>" +
        "<button class='mc-mini mc-name-save' title='Save name (Enter)'>&#10003;</button>" +
      "</span></span>";
  }
  function mcDdsNameCell(chn, name) {
    return "<td class='mc-name mc-name-cell'>" + mcNameField(chn, 'dds', name) + "</td>";
  }

  function renderMcDds(rows) {
    setText("mc-dds-count", rows.length + " active");
    const host = $("mc-dds-body");
    if (!host) return;
    if (!rows.length) { host.innerHTML = "<div class='hint'>no active DDS channels</div>"; return; }
    let h = "<table class='mc-dds-table mono'><thead><tr><th>Ch</th><th>Name</th>" +
      "<th>Freq (MHz)</th><th>Amp (0–1)</th><th>Phase (°)</th><th></th></tr></thead><tbody>";
    for (const r of rows) {
      h += "<tr><td class='mc-ch'>" + r.chn + "</td>" + mcDdsNameCell(r.chn, r.name);
      h += _mcCell(r.chn, "freq",  r.freq_hz != null ? (r.freq_hz / 1e6) : null, 6,
                   r.ovr_freq != null, r.ovr_freq != null ? (r.ovr_freq / 1e6) : null);
      h += _mcCell(r.chn, "amp",   r.amp,       4, r.ovr_amp   != null, r.ovr_amp);
      h += _mcCell(r.chn, "phase", r.phase_deg, 2, r.ovr_phase != null, r.ovr_phase);
      h += "<td><button class='ghost mc-mini' data-mc-reset='" + r.chn +
           "' title='Reset/reinitialize this DDS channel'>Reset</button></td></tr>";
    }
    host.innerHTML = h + "</tbody></table>";
  }

  function renderMcTtlDisabled() {
    setText("mc-ttl-count", "disabled");
    const host = $("mc-ttl-grid");
    if (host) host.innerHTML =
      "<div class='hint'>TTL readback is disabled for now — it would issue zero-mask " +
      "<code>set_ttl</code>/<code>override_ttl</code> frames. Enable with " +
      "<code>YB_MOLECUBE_TTL_READS=1</code>.</div>";
  }

  function renderMcTtl(rows) {
    setText("mc-ttl-count", rows.length + " channels");
    const host = $("mc-ttl-grid");
    if (!host) return;
    if (!rows.length) { host.innerHTML = "<div class='hint'>no TTL channels</div>"; return; }
    let h = "";
    for (const r of rows) {
      let cls = "mc-ttl-chip" + (r.value ? " on" : "");
      if (r.ovr_lo) cls += " ovr-lo";
      if (r.ovr_hi) cls += " ovr-hi";
      const title = "ch " + r.chn + (r.name ? (" · " + r.name) : "") +
        (r.ovr_lo ? " · forced LOW" : r.ovr_hi ? " · forced HIGH" : "");
      // The name shows as text; clicking it opens the inline editor (mcNameField).
      // Clicks inside the name field are swallowed in mcControlClick, so naming a
      // channel never toggles its output.
      h += "<div class='" + cls + "' data-mc-ttl='" + r.chn + "' title='" + escAttr(title) + "'>" +
        "<span class='mc-ttl-n'>" + r.chn + "</span>" +
        mcNameField(r.chn, 'ttl', r.name) + "</div>";
    }
    host.innerHTML = h;
  }

  // Molecube DDS/TTL control logic, factored out of the wiring so BOTH the real
  // cards AND the read-only-clone Overview tiles can drive it. CONTAINER-RELATIVE:
  // it resolves inputs within e.currentTarget (the element the listener is bound
  // to), so it works on a mirror clone too -- the clone has its `id`s stripped
  // (why the old by-id wiring never reached it) but KEEPS the data-* attributes
  // + classes this logic actually keys off. /api/molecube/* is master-gated.
  const _mcValOf = (type, raw) => (type === "freq" ? raw * 1e6 : raw);   // MHz->Hz for freq

  // ----- click-to-edit channel names (DDS table + TTL chips) -----
  // Open by clicking the name text; confirm with Enter or the ✓ button; cancel with
  // Escape or by clicking/tabbing away. Opening the editor never toggles a TTL output
  // (mcControlClick returns early for any click inside a name field).
  function mcOpenNameEditor(field) {
    if (!field || field.classList.contains("editing")) return;
    field.classList.add("editing");
    const chip = field.closest(".mc-ttl-chip");
    if (chip) chip.classList.add("mc-editing");        // widen the chip to fit the box
    const inp = field.querySelector(".mc-name-in");
    if (inp) { inp.value = inp.dataset.mcNameOrig || ""; inp.focus(); inp.select(); }
  }
  function mcCloseNameEditor(field) {                  // cancel: revert text + close
    if (!field) return;
    field.classList.remove("editing");
    const chip = field.closest(".mc-ttl-chip");
    if (chip) chip.classList.remove("mc-editing");
    const inp = field.querySelector(".mc-name-in");
    if (inp) inp.value = inp.dataset.mcNameOrig || "";
  }
  function mcCommitName(field) {                       // confirm: write only if changed
    if (!field) return;
    const inp = field.querySelector(".mc-name-in");
    field.classList.remove("editing");
    const chip = field.closest(".mc-ttl-chip");
    if (chip) chip.classList.remove("mc-editing");
    if (!inp || inp.value === (inp.dataset.mcNameOrig || "")) return;   // no change -> no write
    inp.dataset.mcNameOrig = inp.value;
    mcPost("/api/molecube/" + field.dataset.mcNameKind + "/name",
           {chn: +field.dataset.mcNameChn, name: inp.value});           // triggers a re-render
  }

  function mcControlClick(e) {
    const container = e.currentTarget;
    // Channel-name editing always wins over the chip toggle / DDS controls, so a click
    // on a name (its text, its box, or its ✓) NEVER flips a TTL output.
    if (e.target.closest(".mc-name-save")) {
      mcCommitName(e.target.closest(".mc-name-field"));
      return;
    }
    const nameText = e.target.closest(".mc-name-text");
    if (nameText) { mcOpenNameEditor(nameText.closest(".mc-name-field")); return; }
    if (e.target.closest(".mc-name-field")) return;   // click inside the open editor box
    const chip = e.target.closest("[data-mc-ttl]");
    if (chip) {
      const chn = +chip.dataset.mcTtl;
      if (e.shiftKey) {     // cycle override: normal -> low -> high -> normal
        const mode = chip.classList.contains("ovr-lo") ? "high"
                   : chip.classList.contains("ovr-hi") ? "normal" : "low";
        mcPost("/api/molecube/ttl/override", {chn, mode});
      } else {
        mcPost("/api/molecube/ttl/set", {chn, on: !chip.classList.contains("on")});
      }
      return;
    }
    const set = e.target.closest(".mc-set");
    const ovr = e.target.closest(".mc-ovr-btn");
    const rst = e.target.closest("[data-mc-reset]");
    if (set) {
      const chn = +set.dataset.mcChn, type = set.dataset.mcType;
      const inp = container.querySelector(
        ".mc-in[data-mc-chn='" + chn + "'][data-mc-type='" + type + "']");
      if (!inp || inp.value === "") return;
      mcPost("/api/molecube/dds/set", {chn, type, value: _mcValOf(type, parseFloat(inp.value))});
    } else if (ovr) {
      const chn = +ovr.dataset.mcChn, type = ovr.dataset.mcType;
      if (ovr.classList.contains("on")) {
        mcPost("/api/molecube/dds/override", {chn, type, value: null});   // clear
      } else {
        const inp = container.querySelector(
          ".mc-in[data-mc-chn='" + chn + "'][data-mc-type='" + type + "']");
        if (!inp || inp.value === "") { toast("enter a value first", "warn"); return; }
        mcPost("/api/molecube/dds/override",
               {chn, type, value: _mcValOf(type, parseFloat(inp.value))});
      }
    } else if (rst) {
      if (confirm("Reset/reinitialize DDS channel " + rst.dataset.mcReset + "?"))
        mcPost("/api/molecube/dds/reset", {chn: +rst.dataset.mcReset});
    }
  }
  function mcControlKeydown(e) {
    const nameInp = e.target.closest(".mc-name-in");
    if (nameInp) {
      const field = nameInp.closest(".mc-name-field");
      if (e.key === "Enter") { e.preventDefault(); mcCommitName(field); }
      else if (e.key === "Escape") { e.preventDefault(); mcCloseNameEditor(field); nameInp.blur(); }
      return;
    }
    if (e.key !== "Enter") return;
    const inp = e.target.closest(".mc-in");
    if (!inp || inp.value === "") return;
    const chn = +inp.dataset.mcChn, type = inp.dataset.mcType;
    mcPost("/api/molecube/dds/set", {chn, type, value: _mcValOf(type, parseFloat(inp.value))});
  }
  // Keep the field focused when the ✓ is pressed (so the focusout-cancel never beats
  // the ✓ click-commit), and cancel an editor that loses focus to anything else.
  function mcControlMousedown(e) {
    if (e.target.closest(".mc-name-save")) e.preventDefault();
  }
  function mcNameFocusOut(e) {
    const inp = e.target.closest(".mc-name-in");
    if (!inp) return;
    const field = inp.closest(".mc-name-field");
    // If focus is moving to this field's own ✓ button, let its click commit first
    // (belt-and-suspenders with the mousedown preventDefault above).
    if (e.relatedTarget && field && field.contains(e.relatedTarget)) return;
    if (field && field.classList.contains("editing")) mcCloseNameEditor(field);
  }

  // Wire the REAL DDS/TTL cards + static buttons (the Overview mirror tiles are
  // wired to the same handlers in mirrorMolecubeTiles).
  (function wireMolecube() {
    const ddsBody = $("mc-dds-body");
    if (ddsBody) {
      ddsBody.addEventListener("click", mcControlClick);
      ddsBody.addEventListener("keydown", mcControlKeydown);
      ddsBody.addEventListener("mousedown", mcControlMousedown);
      ddsBody.addEventListener("focusout", mcNameFocusOut);
    }
    const ttlGrid = $("mc-ttl-grid");
    if (ttlGrid) {
      ttlGrid.addEventListener("click", mcControlClick);
      ttlGrid.addEventListener("keydown", mcControlKeydown);
      ttlGrid.addEventListener("mousedown", mcControlMousedown);
      ttlGrid.addEventListener("focusout", mcNameFocusOut);
    }
    if ($("mc-refresh-btn"))
      $("mc-refresh-btn").addEventListener("click", () => { _mcLastStateId = null; pollMolecube(); });
    if ($("mc-clock-set"))
      $("mc-clock-set").addEventListener("click", () => {
        const v = $("mc-clock-input").value;
        if (v !== "") mcPost("/api/molecube/clock/set", {clock: +v});
      });
  })();

  // ===================== NI DAC CHANNEL MONITOR (not via molecube) =====================
  // Read-only readback of the NI PCIe-6738 analog-out voltages via
  // /api/nidaq/monitor (the card's internal AO monitor, read in the engine venv).
  // Lives in the Molecube sub-view but is a wholly separate data path.
  function niSetPill(state, text) {
    const p = $("ni-mon-pill");
    if (!p) return;
    p.textContent = text;
    p.className = "mc-pill mc-pill-" + state;   // ok | warn | err | off
  }

  async function pollNidaq() {
    let res;
    try {
      res = await api("/api/nidaq/monitor");
    } catch (e) {
      if (e.status === 403) { renderNidaq(e.body || {ok: false, disabled: true}); return; }
      niSetPill("err", "error");
      setText("ni-mon-status", "monitor failed: " + (e.message || e));
      return;
    }
    renderNidaq(res);
  }

  function renderNidaq(res) {
    const host = $("ni-mon-body");
    if (res && res.disabled) {
      niSetPill("off", "disabled");
      setText("ni-mon-status", res.error || "NI DAC monitor disabled (YB_NIDAQ_MONITOR=0).");
      if (host) host.innerHTML = "<div class='hint'>—</div>";
      return;
    }
    if (!res || res.ok === false) {
      niSetPill("err", "error");
      setText("ni-mon-status", (res && (res.error || res.stderr)) || "read failed");
      return;
    }
    const chans = res.channels || [];
    if (res.paused) {
      niSetPill("warn", "paused");
      setText("ni-mon-status",
        (res.reason || "scan running -- monitor paused") +
        (res.age_s != null ? (" · last read " + res.age_s.toFixed(0) + "s ago") : "") +
        (chans.length ? "" : " · no reading yet"));
    } else {
      const nerr = chans.filter((c) => c.error).length;
      niSetPill(nerr ? "warn" : "ok", nerr ? (nerr + " unread") : "connected");
      setText("ni-mon-status",
        (res.device || "Dev1") + " · " + chans.length + " channels" +
        (res.cached ? (" · cached " + (res.age_s || 0).toFixed(0) + "s") : " · live") +
        (nerr ? ("  ⚠ " + nerr + " unreadable") : ""));
    }
    if (!host) return;
    // Don't rebuild while a Set field is being edited (would stomp typing).
    const ae = document.activeElement;
    if (ae && ae.classList && ae.classList.contains("ni-in")) return;
    if (!chans.length) { host.innerHTML = "<div class='hint'>no NI channels</div>"; return; }
    let h = "<table class='mc-dds-table mono'><thead><tr><th>Ch</th><th>Name</th>" +
      "<th>Monitored&nbsp;(V)</th><th>Default&nbsp;(V)</th><th>&Delta;&nbsp;(V)</th>" +
      "<th>Set&nbsp;(V)</th></tr></thead><tbody>";
    for (const c of chans) {
      const v = c.voltage, d = c.default;
      const hasV = (v !== null && v !== undefined && !isNaN(v));
      const hasD = (d !== null && d !== undefined && !isNaN(d));
      const delta = (hasV && hasD) ? (v - d) : null;
      const big = (delta !== null && Math.abs(delta) > 0.05);   // visibly off its default
      const vcell = c.error
        ? "<span class='ni-err' title='" + String(c.error).replace(/'/g, "") + "'>err</span>"
        : (hasV ? v.toFixed(4) : "—");
      const prefill = hasV ? v.toFixed(4) : (hasD ? d.toFixed(4) : "");
      h += "<tr><td class='mc-ch'>" + c.chn + "</td>" +
           mcNameCell(c.alias) +
           "<td class='" + (c.error ? "ni-cell-err" : "") + "'>" + vcell + "</td>" +
           "<td class='ni-default'>" + (hasD ? d.toFixed(4) : "—") + "</td>" +
           "<td class='ni-delta" + (big ? " ni-delta-big" : "") + "'>" +
             (delta !== null ? (delta >= 0 ? "+" : "") + delta.toFixed(4) : "—") + "</td>" +
           "<td class='ni-set-cell'><input class='ni-in' type='number' step='any' value='" +
             prefill + "' data-ni-chan=\"" + c.alias + "\">" +
           "<button class='mc-mini ni-set' data-ni-chan=\"" + c.alias + "\">Set</button></td>" +
           "</tr>";
    }
    host.innerHTML = h + "</tbody></table>";
  }

  // POST a single NI channel voltage (one-off DC set; server gates + defers while running).
  async function niPost(channel, voltage) {
    try {
      const r = await api("/api/nidaq/set", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({channel: channel, voltage: voltage}),
      });
      if (r && r.ok) {
        toast("NI " + (r.channel || channel) + " = " + voltage + " V" +
              (r.readback != null ? (" (rb " + Number(r.readback).toFixed(3) + ")") : ""), "ok");
      } else {
        toast("NI set: " + ((r && r.error) || "failed"), "err");
      }
      setTimeout(pollNidaq, 300);     // cache was invalidated server-side -> refresh readback
      return r;
    } catch (e) {
      if (e.status === 403) toast("NI writes disabled (YB_NIDAQ_WRITES=0).", "warn");
      else if (e.status === 503) toast("NI write deferred — a scan is running.", "warn");
      else toast("NI set: " + (e.message || e), "err");
    }
  }

  (function wireNidaqWrite() {
    const niBody = $("ni-mon-body");
    if (!niBody) return;
    niBody.addEventListener("click", (e) => {
      const btn = e.target.closest(".ni-set");
      if (!btn) return;
      const inp = niBody.querySelector(
        '.ni-in[data-ni-chan="' + btn.dataset.niChan + '"]');
      if (!inp || inp.value === "") return;
      niPost(btn.dataset.niChan, parseFloat(inp.value));
    });
    niBody.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      const inp = e.target.closest(".ni-in");
      if (!inp || inp.value === "") return;
      niPost(inp.dataset.niChan, parseFloat(inp.value));
    });
  })();

  if ($("ni-mon-refresh"))
    $("ni-mon-refresh").addEventListener("click", () => { pollNidaq(); });

  // Open the Logs tab from the shot-health chip. "ok" -> the newest backend log
  // tailed to now ("the log at the current time"); "failing"/"failed" -> the
  // same log with the last error message highlighted + scrolled into view. The
  // pyctrl backend is where shot errors land; fall back to monitor/matlab.
  async function openLogsForShotHealth() {
    setTab("logs");
    if (!logsListCache) {
      try { logsListCache = await api("/api/logs/list"); } catch (e) { /* below */ }
    }
    const groups = (logsListCache && logsListCache.groups) || [];
    const newestIn = (cat) => {
      const g = groups.find((x) => x.category === cat);
      return (g && g.files && g.files.length)
        ? { category: cat, name: g.files[0].name } : null;   // list is newest-first
    };
    const anyFile = () => {
      const g = groups.find((x) => x.files && x.files.length);
      return g ? { category: g.category, name: g.files[0].name } : null;
    };
    const sel = newestIn("pyctrl") || newestIn("monitor")
                || newestIn("matlab") || anyFile();
    if (!sel) { toast("no log files found", "warn"); return; }
    logsSelected = sel;
    // Highlight + jump only when the chip is in an error state and we know the
    // message; otherwise just tail to "now".
    const sh = _lastShotHealth;
    const failing = sh && (sh.total || 0) > 0;
    logsHighlight = (failing && sh.last_message) ? String(sh.last_message) : null;
    renderLogsList();
    await loadLogFile({ keepScroll: false });
  }
  if ($("shot-health-tile")) {
    $("shot-health-tile").addEventListener("click", openLogsForShotHealth);
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
    // The unified "Scans" picker (the old Analysis Runs card) docks LEFT in the
    // shared #floating-seqscan-host and is shown on BOTH Analysis + Sequence.
    // Filter (Data mode) + Channels (Sequence mode) now dock LEFT too, stacked
    // under it -- the old right-docked #floating-analysis-host is left empty.
    const host     = document.getElementById("floating-analysis-host");
    const scanHost = document.getElementById("floating-seqscan-host");
    const runs = document.getElementById("analysis-runs-card");
    const filt = document.getElementById("analysis-filters");
    const chn  = document.getElementById("sequence-chn-card");
    if (!host || !scanHost || !runs || !filt) return;
    // All three analysis/sequence pickers dock LEFT now, stacked in ONE flex
    // column (Runs on top; Filter [Data mode] / Channels [Sequence mode] right
    // underneath it). This frees the entire RIGHT edge for the global Yb
    // Control panel. Reparenting into a single column (rather than separate
    // right-docked hosts) means an EXPANDED upper card pushes the lower card
    // DOWN, so the lower card's edge tab is never buried under it.
    scanHost.appendChild(runs);          // top
    scanHost.appendChild(filt);          // under Runs (Data sub-mode)
    if (chn) scanHost.appendChild(chn);  // under Runs (Sequence sub-mode)
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
      const cards = [...host.querySelectorAll(".card"),
                     ...scanHost.querySelectorAll(".card")];
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
      // Esc: collapse everything back to edge (both the right Filter host and
      // the left shared Scans host).
      if (e.key === "Escape") {
        [...host.querySelectorAll(".card"),
         ...scanHost.querySelectorAll(".card")].forEach((card) => {
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

    // The right-docked host is now empty (Filter moved into the left column).
    host.hidden = true;
    updateAnalysisFloatingHosts(activeTab);
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

  // ---- Full-info hover tooltip for the scan picker (req 10) -----------------
  // A floating tip appended to <body> (so it escapes the narrow pane) showing the
  // complete scan metadata, not just the truncated row.
  function _seqScanTip() {
    let t = document.getElementById("seq-scan-tip");
    if (!t) {
      t = document.createElement("div");
      t.id = "seq-scan-tip";
      t.className = "seq-scan-tip";
      t.hidden = true;
      document.body.appendChild(t);
    }
    return t;
  }
  function seqScanTipHtml(s) {
    const id = s.scan_id || "";
    const when = (id.length === 14)
      ? `${id.slice(0,4)}-${id.slice(4,6)}-${id.slice(6,8)} ` +
        `${id.slice(8,10)}:${id.slice(10,12)}:${id.slice(12,14)}`
      : id;
    const rows = [];
    const add = (k, v) => { if (v != null && v !== "") rows.push(
      '<div class="seq-tip-row"><span class="seq-tip-k">' + seqEsc(k) +
      '</span><span class="seq-tip-v">' + seqEsc(String(v)) + "</span></div>"); };
    add("scan_id", id);
    add("name", s.name || "—");
    if (s.description) add("description", s.description);
    add("swept", s.swept || "—");
    add("points", s.n_params);
    add("reps/pt", s.n_shots);
    add("total shots", s.n_total_shots);
    if (s.has_seq) add(".seq dumps", s.n_seq);
    const state = s.has_seq ? "ready ✓"
      : (s.has_snapshot && s.has_descriptor) ? "reconstructable ⟳" : "unrecoverable";
    add("state", state);
    const flags = [];
    if (s.has_diag) flags.push("diag");
    if (s.has_code) flags.push("code");
    if (s.has_grid) flags.push("grid");
    if (s.has_snapshot) flags.push("snapshot");
    if (s.has_descriptor) flags.push("descriptor");
    if (flags.length) add("artifacts", flags.join(", "));
    return '<div class="seq-tip-title">' + seqEsc(when) + "</div>" + rows.join("");
  }
  function seqShowScanTip(ev, s) {
    const t = _seqScanTip();
    t.innerHTML = seqScanTipHtml(s);
    t.hidden = false;
    seqMoveScanTip(ev);
  }
  function seqMoveScanTip(ev) {
    const t = document.getElementById("seq-scan-tip");
    if (!t || t.hidden) return;
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;   // flip left if off-screen
    if (x < 8) x = 8;
    if (y + h > window.innerHeight - 8) y = window.innerHeight - h - 8;
    if (y < 8) y = 8;
    t.style.left = x + "px";
    t.style.top = y + "px";
  }
  function seqHideScanTip() {
    const t = document.getElementById("seq-scan-tip");
    if (t) t.hidden = true;
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
             data-scan-id="${id}" data-state="${state}">
          <div class="run-info">
            ${idShort}
            <span class="run-dim"> · ${seqEsc(s.name || "—")}</span>
            <span class="run-dim"> · ${seqEsc(s.swept || "—")}</span>
            <span class="seq-row-tag"> · ${tag}</span>
          </div>
        </div>`;
    }).join("");
    seqHideScanTip();   // clear any tooltip pinned to a now-replaced row
    $$(".run-row", wrap).forEach((row) => {
      if (!row.dataset.scanId) return;
      // Full-info hover tooltip (req 10): the row is too narrow for everything,
      // so a floating tip (escapes the pane) shows the complete scan metadata.
      const sInfo = seqScansCache.find((x) => (x.scan_id || "") === row.dataset.scanId);
      if (sInfo) {
        row.addEventListener("mouseenter", (e) => seqShowScanTip(e, sInfo));
        row.addEventListener("mousemove", seqMoveScanTip);
        row.addEventListener("mouseleave", seqHideScanTip);
      }
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
      // Force a rescan so the new .seq dump is detected (has_seq is computed
      // fresh server-side), then plot the regenerated sequence.
      await loadRunsList({ force: true });       // the row flips to Ready
      await seqLoad({ scan_id: scanId });        // load + plot the regenerated .seq
      renderRunsTable();
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
    // Custom channel picker model: chnOrder = display order (mutated by param
    // promotion), chnSel = the selected set. Preserve selection across a sequence
    // switch by intersecting with the new channel list.
    const names = (seq && seq.channels) || [];
    const prev = seqState.chnSel || new Set();
    seqState.chnOrder = names.slice();
    seqState.chnSel = new Set(names.filter((n) => prev.has(n)));
    seqRenderChnList();
    seqUpdateChnCount();
    seqRenderPlot();
    seqRenderParams();
  }

  function seqSelectedChannels() {
    const order = seqState.chnOrder || [];
    const sel = seqState.chnSel || new Set();
    return order.filter((n) => sel.has(n));   // selected, in display order
  }

  // Render the custom channel picker (replaces the native <select multiple>):
  // click an UNSELECTED row -> ADD it (others untouched); click a SELECTED row ->
  // REMOVE it (a ✕ cue slides in on the left on hover). Honors the filter box.
  function seqRenderChnList() {
    const box = $("seq-chn-list");
    if (!box) return;
    const order = seqState.chnOrder || [];
    const sel = seqState.chnSel || new Set();
    const search = (($("seq-chn-search") || {}).value || "").trim().toLowerCase();
    const rows = order
      .filter((n) => !search || n.toLowerCase().includes(search))
      .map((n) => {
        const on = sel.has(n);
        return '<div class="chn-row' + (on ? " selected" : "") + '" data-chn="' +
          seqEsc(n) + '" title="' + seqEsc(n) + '">' +
          '<span class="chn-row-x">✕</span>' +
          '<span class="chn-row-name">' + seqEsc(n) + "</span></div>";
      });
    box.innerHTML = rows.length ? rows.join("")
      : '<div class="chn-list-empty">' +
        (order.length ? "no channels match" : "no channels") + "</div>";
  }

  // Toggle one channel in/out of the selection (custom-list click handler).
  async function seqToggleChannel(name) {
    const sel = seqState.chnSel || (seqState.chnSel = new Set());
    if (sel.has(name)) sel.delete(name); else sel.add(name);
    seqRenderChnList();
    seqUpdateChnCount();
    await seqRenderPlot();
    seqReapplyEmph();
  }

  // SHIFT-click: ADD every channel between the anchor and the clicked row (over the
  // currently-displayed order, honoring the filter). Never removes -- pure additive range.
  async function seqSelectChannelRange(aName, bName) {
    const order = seqState.chnOrder || [];
    const search = (($("seq-chn-search") || {}).value || "").trim().toLowerCase();
    const disp = order.filter((n) => !search || n.toLowerCase().includes(search));
    const ia = disp.indexOf(aName), ib = disp.indexOf(bName);
    if (ia < 0 || ib < 0) { seqToggleChannel(bName); return; }   // anchor not visible -> plain
    const lo = Math.min(ia, ib), hi = Math.max(ia, ib);
    const sel = seqState.chnSel || (seqState.chnSel = new Set());
    for (let k = lo; k <= hi; k++) sel.add(disp[k]);
    seqRenderChnList();
    seqUpdateChnCount();
    await seqRenderPlot();
    seqReapplyEmph();
  }

  // Reflect the # of selected channels on the floating card (drives the
  // edge-tab count badge + the active accent palette) and the in-header badge.
  function seqUpdateChnCount() {
    const n = seqSelectedChannels().length;
    const card = document.getElementById("sequence-chn-card");
    if (card) card.dataset.floatCount = String(n);
    const badge = $("seq-chn-count-badge");
    if (badge) { badge.textContent = n ? String(n) : ""; badge.hidden = !n; }
  }

  // Hover-driven floating card. Smoothed to feel like the Analysis selector (req 5):
  // a short hover-INTENT delay before expanding kills the "brush-past" flicker that
  // made it feel bouncy, and a longer collapse grace keeps it open if the cursor
  // briefly drifts off. Never collapses while the mouse is over it OR a field inside
  // has focus (so typing in the search box holds it open). `onReveal` (optional) fires
  // when the card starts to open -- used to auto-refresh the scans list (req 8).
  function _wireSeqFloatCard(card, onReveal) {
    if (!card) return;
    _setFloatState(card, "edge");
    let openT = null, closeT = null;
    const clearTimers = () => {
      if (openT) { clearTimeout(openT); openT = null; }
      if (closeT) { clearTimeout(closeT); closeT = null; }
    };
    card.addEventListener("mouseenter", () => {
      clearTimers();
      if (card.dataset.floatState === "expanded") return;
      if (onReveal) { try { onReveal(); } catch (e) {} }
      openT = setTimeout(() => {
        if (card.matches(":hover")) _setFloatState(card, "expanded");
        openT = null;
      }, 110);   // hover-intent: ignore a quick brush across the edge tab
    });
    card.addEventListener("mouseleave", () => {
      clearTimers();
      closeT = setTimeout(() => {
        if (!card.matches(":hover") && !card.contains(document.activeElement))
          _setFloatState(card, "edge");
        closeT = null;
      }, 450);
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
      seqWireLegendToggle(el);       // hover-reveal / click-pin the channel legend (req 3)
      seqWireClearOthers(el);        // "only highlighted" -> drop the other channels (req 4)
      seqWireStepRuler(el);          // step/phase boundary ruler + toggle
      const _hp = _seqHoverPath(el, true);
      if (_hp) _hp.setAttribute("points", "");     // clear stale hover line after a rebuild
      if (seqState._legendPinned)                  // re-apply pinned legend across react
        try { Plotly.relayout(el, { showlegend: true }); } catch (e) {}
      seqReapplyEmph();                            // survive rebuild (req 6)
      seqRenderStepRuler(el);                      // draw the phase ruler (xref.steps)
    } catch (e) { console.warn("seq figure", e); }
  }

  // Legend reveal/pin control (req 3). The legend is collapsed by default
  // (figure.py: showlegend=false). A small tab at the plot's top-left reveals it
  // while hovered, and a click PINS it open. Created once; survives Plotly.react
  // (it's a plain DOM child of the plot div, like the hover overlay).
  function seqWireLegendToggle(el) {
    if (el._seqLegendWired) return;
    el._seqLegendWired = true;
    if (getComputedStyle(el).position === "static") el.style.position = "relative";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seq-legend-toggle";
    btn.title = "Show the channel legend (click to pin it open)";
    btn.innerHTML = '<span class="seq-legend-toggle-ic">☰</span> Legend';
    el.appendChild(btn);
    const show = (on) => { try { Plotly.relayout(el, { showlegend: on }); } catch (e) {} };
    let hideT = null;
    const cancelHide = () => { if (hideT) { clearTimeout(hideT); hideT = null; } };
    btn.addEventListener("mouseenter", () => { cancelHide(); if (!seqState._legendPinned) show(true); });
    // Disappear quickly once the cursor leaves the BUTTON (short grace), not only on
    // leaving the whole plot. Click to pin if you want it to stay open.
    btn.addEventListener("mouseleave", () => {
      cancelHide();
      if (!seqState._legendPinned)
        hideT = setTimeout(() => { if (!seqState._legendPinned) show(false); }, 220);
    });
    el.addEventListener("mouseleave", () => { cancelHide(); if (!seqState._legendPinned) show(false); });
    btn.addEventListener("click", (e) => {
      cancelHide();
      e.stopPropagation();
      seqState._legendPinned = !seqState._legendPinned;
      btn.classList.toggle("pinned", seqState._legendPinned);
      show(seqState._legendPinned);
    });
    if (seqState._legendPinned) btn.classList.add("pinned");
  }

  // The channels involved in the CURRENT highlight (active param's driven channels,
  // or the clicked pulse's channel). Used by the "only highlighted" button (req 4).
  function seqHighlightedChannels() {
    const e = seqState._emph, xr = seqState.xref, out = new Set();
    if (!e || !xr) return [];
    if (e.kind === "param") {
      ((xr.param_to_channels && xr.param_to_channels[e.path]) || []).forEach((c) => out.add(c));
      ((xr.param_to_pids && xr.param_to_pids[e.path]) || []).forEach((pid) => {
        const pu = xr.pulses && xr.pulses[String(pid)];
        if (pu && pu.channel) out.add(pu.channel);
      });
    } else if (e.kind === "pulse" && xr.pulses) {
      const pu = xr.pulses[String(e.pid)];
      if (pu && pu.channel) out.add(pu.channel);
    }
    return Array.from(out);
  }

  // Drop every plotted channel EXCEPT the currently-highlighted one(s) -- i.e. isolate
  // the selection on the plot (req 4). With nothing highlighted there's nothing to keep,
  // so it clears all channels.
  async function seqClearOtherChannels() {
    const keep = new Set(seqHighlightedChannels());
    const order = seqState.chnOrder || [];
    seqState.chnSel = new Set(order.filter((n) => keep.has(n)));
    seqRenderChnList();
    seqUpdateChnCount();
    await seqRenderPlot();
    seqReapplyEmph();
    toast(keep.size ? ("Isolated " + keep.size + " channel(s)") : "Cleared all channels",
          keep.size ? "ok" : "");
  }

  // "Only highlighted" button overlaid top-LEFT of the plot (legend control is top-right).
  function seqWireClearOthers(el) {
    if (el._seqClearOthersWired) return;
    el._seqClearOthersWired = true;
    if (getComputedStyle(el).position === "static") el.style.position = "relative";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seq-clear-others";
    btn.title = "Remove every channel except the highlighted one(s)";
    btn.innerHTML = '<span class="seq-clear-others-ic">⊙</span> Only highlighted';
    btn.addEventListener("click", (e) => { e.stopPropagation(); seqClearOtherChannels(); });
    el.appendChild(btn);
  }

  // ---- Step / phase ruler (xref.steps) --------------------------------------
  // Vertical boundary lines at each top-level step's start/end + a name label at the top
  // (the experiment phases: InitStep/BlueMOTStep/GreenMOTStep/...). Drawn on a plain SVG
  // overlay (like the hover line) -- NOT Plotly shapes -- so it never clobbers the param/
  // pulse emphasis (which uses layout.shapes) and costs no Plotly redraw. Recomputed on
  // zoom/pan/resize from the axis pixel geometry.
  function _seqStepOv(el, existingOnly) {
    let ov = el.querySelector(":scope > svg.seq-step-ov");
    if (!ov) {
      if (existingOnly) return null;
      const NS = "http://www.w3.org/2000/svg";
      ov = document.createElementNS(NS, "svg");
      ov.setAttribute("class", "seq-step-ov");
      ov.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;" +
                         "pointer-events:none;z-index:5;overflow:hidden;";
      if (getComputedStyle(el).position === "static") el.style.position = "relative";
      el.appendChild(ov);
      // Delegated label interactions. The <svg> is pointer-events:none; only the
      // .seq-step-tab groups opt back in -- hover shades the step span, click zooms
      // the time axis to that step (+ buffer). Attached ONCE (the svg is cached).
      ov.addEventListener("mouseover", (e) => {
        const tab = e.target.closest(".seq-step-tab");
        if (tab) seqShowStepBand(el, +tab.dataset.t0, +tab.dataset.t1);
      });
      ov.addEventListener("mouseout", (e) => {
        const tab = e.target.closest(".seq-step-tab");
        if (tab && !tab.contains(e.relatedTarget)) seqHideStepBand(el);
      });
      ov.addEventListener("click", (e) => {
        const tab = e.target.closest(".seq-step-tab");
        if (!tab) return;
        e.stopPropagation();
        seqZoomToStep(el, +tab.dataset.t0, +tab.dataset.t1);
      });
    }
    return ov;
  }

  // seq_idx of the basic sequence currently shown in the plot (the dropdown's value), or
  // null if there's no selector. Used to filter the per-bseq step ruler / wait bands so a
  // multi-basic-sequence scan (e.g. SLM rearrangement) shows only the displayed bseq's steps.
  function seqCurrentSeqIdx() {
    const ssel = $("seq-seq-select");
    const v = ssel ? ssel.value : null;
    return (v === null || v === undefined || v === "") ? null : v;
  }

  // Keep only the entries belonging to the displayed basic sequence. Backward-compatible:
  // entries with no seq_idx (pre-v9 artifacts, or a single-bseq build) are always kept.
  function seqForCurrentBseq(items, idxOf) {
    const cur = seqCurrentSeqIdx();
    if (cur == null) return items;
    return items.filter((it) => { const s = idxOf(it); return s == null || String(s) === String(cur); });
  }

  function seqRenderStepRuler(el) {
    if (!el) return;
    const on = seqState._stepsOn !== false;        // default ON
    const ov = _seqStepOv(el, !on);
    if (ov) ov.innerHTML = "";
    if (!on || !ov) return;
    // Only the displayed basic sequence's steps (each bseq's time frame restarts at 0, so
    // an unfiltered list would overlay every bseq's phases on whichever one is shown).
    const steps = seqForCurrentBseq((seqState.xref && seqState.xref.steps) || [],
                                    (s) => s.seq_idx);
    const fl = el._fullLayout;
    if (!steps.length || !fl || !fl.xaxis || !fl.yaxis) return;
    const xa = fl.xaxis, ya = fl.yaxis;
    if (!xa.range || xa._length == null || ya._length == null) return;
    const r0 = xa.range[0], r1 = xa.range[1], xw = r1 - r0;
    if (!isFinite(xw) || xw === 0) return;
    const xoff = xa._offset, xl = xa._length, ytop = ya._offset, yh = ya._length;
    const NS = "http://www.w3.org/2000/svg";
    const X = (t) => xoff + (t - r0) / xw * xl;
    // Boundary lines (unique start/end times within the visible range).
    const bounds = new Set();
    steps.forEach((s) => { bounds.add(s.t0); bounds.add(s.t1); });
    bounds.forEach((t) => {
      const px = X(t);
      if (px < xoff - 0.5 || px > xoff + xl + 0.5) return;
      const ln = document.createElementNS(NS, "line");
      ln.setAttribute("x1", px.toFixed(1)); ln.setAttribute("x2", px.toFixed(1));
      ln.setAttribute("y1", ytop.toFixed(1)); ln.setAttribute("y2", (ytop + yh).toFixed(1));
      ln.setAttribute("class", "seq-step-line");
      ov.appendChild(ln);
    });
    // Name labels at the top, staggered over 2 rows. Each is an interactive "tab"
    // (hover -> shade the step; click -> zoom to it); handlers live in _seqStepOv.
    steps.forEach((s, i) => {
      const a = Math.max(s.t0, r0), b = Math.min(s.t1, r1);
      if (b < r0 || a > r1) return;                // off-screen
      const cx = (s.t1 > s.t0) ? X((a + b) / 2) : X(s.t0);
      const ty = ytop + 11 + (i % 2) * 13;
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "seq-step-tab");
      g.dataset.t0 = String(s.t0);
      g.dataset.t1 = String(s.t1);
      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("x", cx.toFixed(1));
      txt.setAttribute("y", ty.toFixed(1));
      txt.setAttribute("text-anchor", "middle");
      txt.setAttribute("class", "seq-step-label");
      txt.textContent = s.label || "step";
      g.appendChild(txt);
      ov.appendChild(g);
      try {                                        // hit-area + hover bg sized to the text
        const bb = txt.getBBox();
        const rc = document.createElementNS(NS, "rect");
        rc.setAttribute("class", "seq-step-tab-bg");
        rc.setAttribute("x", (bb.x - 3).toFixed(1));
        rc.setAttribute("y", (bb.y - 1).toFixed(1));
        rc.setAttribute("width", (bb.width + 6).toFixed(1));
        rc.setAttribute("height", (bb.height + 2).toFixed(1));
        rc.setAttribute("rx", "3");
        g.insertBefore(rc, txt);
      } catch (e) {}
    });
  }

  // Click a step label -> zoom the time axis to that step's span + a side buffer.
  function seqZoomToStep(el, t0, t1) {
    if (!el || !window.Plotly) return;
    const span = (t1 - t0) || 0;
    const pad = Math.max(span * 0.12, 3);          // a bit of buffer on each side
    let lo = t0 - pad, hi = t1 + pad;
    if (hi <= lo) hi = lo + 1;
    try { Plotly.relayout(el, { "xaxis.range": [lo, hi] }); } catch (e) {}
  }

  // Hover a step label -> shade that step's time span (band behind the lines/labels).
  function seqShowStepBand(el, t0, t1) {
    const ov = _seqStepOv(el, true);
    const fl = el && el._fullLayout;
    if (!ov || !fl || !fl.xaxis || !fl.yaxis || !fl.xaxis.range) return;
    const xa = fl.xaxis, ya = fl.yaxis, r0 = xa.range[0], xw = xa.range[1] - r0;
    if (!isFinite(xw) || xw === 0) return;
    const x0 = xa._offset + (t0 - r0) / xw * xa._length;
    const x1 = xa._offset + (t1 - r0) / xw * xa._length;
    let band = ov.querySelector(".seq-step-hover-band");
    if (!band) {
      band = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      band.setAttribute("class", "seq-step-hover-band");
      ov.insertBefore(band, ov.firstChild);        // behind the lines + labels
    }
    band.setAttribute("x", Math.min(x0, x1).toFixed(1));
    band.setAttribute("y", ya._offset.toFixed(1));
    band.setAttribute("width", Math.max(Math.abs(x1 - x0), 2).toFixed(1));
    band.setAttribute("height", ya._length.toFixed(1));
    band.style.display = "";
  }

  function seqHideStepBand(el) {
    const ov = _seqStepOv(el, true);
    const band = ov && ov.querySelector(".seq-step-hover-band");
    if (band) band.style.display = "none";
  }

  // Toggle button + recompute-on-relayout wiring (created once per plot div).
  function seqWireStepRuler(el) {
    if (el._seqStepWired) return;
    el._seqStepWired = true;
    if (seqState._stepsOn === undefined) seqState._stepsOn = true;   // default ON
    if (getComputedStyle(el).position === "static") el.style.position = "relative";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seq-step-toggle";
    btn.title = "Show step/phase boundaries on the time axis";
    btn.innerHTML = '<span class="seq-step-toggle-ic">⊓</span> Steps';
    btn.classList.toggle("on", seqState._stepsOn);
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      seqState._stepsOn = !seqState._stepsOn;
      btn.classList.toggle("on", seqState._stepsOn);
      seqRenderStepRuler(el);
    });
    el.appendChild(btn);
    el.on("plotly_relayout", () => seqRenderStepRuler(el));   // zoom / pan / autosize
    window.addEventListener("resize", () => seqRenderStepRuler(el));
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
    // Invalidate any xref poll loop from a PRIOR load (a different seq of the same scan, or
    // a re-render): each loop carries the token snapshotted HERE and bails the moment the
    // live token moves past it, so an old loop can't re-set the "building"/"awaiting" banner
    // after a newer load cleared it. Snapshot now (before the awaits) so concurrent
    // seqRenderParams calls each tag their own loop, not whatever the token grew to later.
    const pollGen = (seqState._xrefPollGen = (seqState._xrefPollGen || 0) + 1);
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
    seqRenderStepRuler($("plot-sequence"));        // phase ruler depends on xref.steps
    // The SERVER auto-builds/upgrades the xref on view. Three states drive the notice +
    // how we wait for it: awaiting the run's globals (can't build steps/regions yet),
    // rebuilding (globals present, build in flight), or nothing pending.
    // A producer crash (e.g. a config error) takes precedence -- surface it instead of
    // spinning on "building", which is how the silent-failure mode used to look.
    const pendingG = (seqState.xref && seqState.xref.pending_globals) || 0;
    if (r.xref_build_error) { seqSetXrefNotice("error", r.xref_build_error); }
    // This scan is the one currently RUNNING: globals.json flushes incrementally, so the
    // viewed point resolves the moment its seqid runs -- none of the per-point poll loops
    // below would start, leaving the tab frozen (stale point count, no progress banner). Drive
    // a single LIVE loop instead: it refreshes the point dropdown + ruler/bands as new points
    // run and shows the scan-wide "N points awaiting globals" banner, ticking down to scan end.
    else if (r.running) { seqLiveBanner(r); seqLiveLoop(qs, pollGen); }
    else if (r.xref_awaiting_globals) { seqSetXrefNotice("awaiting", pendingG); seqAwaitGlobals(qs, pollGen); }
    else if (r.xref_building) { seqSetXrefNotice("building"); seqPollXrefUpdate(qs, pollGen); }
    // Partial map: the ruler is drawn but N bands still wait on the run's globals (e.g. the
    // EOM ramp). Show how many, and poll so they fill in once globals land + the xref rebuilds.
    else if (pendingG > 0) { seqSetXrefNotice("pending", pendingG); seqAwaitGlobals(qs, pollGen); }
    else { seqSetXrefNotice(null); seqMaybeBuildXref(qs); }
  }

  // Pick the live-run banner from a params/xref response (key names differ slightly between
  // the two routes -- normalize both). Scan-wide pending takes precedence; before any point
  // has run (no xref yet) fall back to the generic "awaiting"; then "building"; else clear.
  function seqLiveBanner(r) {
    const pp = (r.pending_points | 0), tot = (r.total_points | 0);
    const awaiting = r.awaiting_globals || r.xref_awaiting_globals;
    const building = r.building || r.xref_building;
    if (pp > 0) seqSetXrefNotice("scan-pending", { pending: pp, total: tot });
    else if (awaiting) seqSetXrefNotice("awaiting", 0);
    else if (building) seqSetXrefNotice("building");
    else seqSetXrefNotice(null);
  }

  // Append point-dropdown options for points dumped SINCE the last populate, preserving the
  // user's current selection (we only add new <option>s; we never reset selectedIndex or
  // re-render the viewed point). Keeps seqState.index fresh so a later manual point/seq pick
  // sees the new files. Used by the live loop so a running scan's point count tracks the run.
  function seqAppendNewPoints(idx) {
    const psel = $("seq-point-select");
    if (!psel || !idx) return;
    const pts = idx.points || [];
    for (let i = psel.options.length; i < pts.length; i++) {
      const pt = pts[i];
      const o = document.createElement("option");
      o.value = String(i);
      const sc = pt.scanned && Object.keys(pt.scanned).length
        ? "  " + Object.entries(pt.scanned).map(([k, v]) => `${k}=${seqFmt(v)}`).join(", ")
        : "";
      o.textContent = `#${pt.n != null ? pt.n : i + 1}${sc}`;
      psel.appendChild(o);
    }
    seqState.index = idx;            // keep files/points fresh for seqOnPoint / seqOnSeq
    const status = $("seq-source-status");
    if (status) {
      const nf = (idx.files || []).length, np = (idx.points || []).length;
      status.textContent = `${np} point(s), ${nf} file(s)` +
        (idx.has_manifest ? "" : "  (no manifest)");
    }
  }

  // Live refresh while THIS scan is running (server says r.running). globals.json flushes per
  // seqid, so we keep the Sequence tab current: every 5 s re-fetch the xref (re-render the
  // ruler/bands as the viewed point resolves + update the scan-wide pending banner), and when
  // new .seq files have appeared (n_seq_files grew past the dropdown) re-fetch the heavier
  // /api/sequence/list ONCE to append the new point options. Bails on navigation/supersede;
  // when the run ends it does a final refresh, lets the last rebuild zero the banner, and stops.
  async function seqLiveLoop(qs, gen) {
    const scanId = seqState.query && seqState.query.scan_id;
    const qsList = new URLSearchParams(seqState.query || {}).toString();
    let endedPolls = 0;
    for (let i = 0; i < 1440; i++) {                // ~2 h cap at 5 s; bails when the run ends
      await new Promise((res) => setTimeout(res, 5000));
      if (seqPollSuperseded(scanId, gen)) return;   // newer load / navigated away
      let x = null;
      try { x = await api("/api/sequence/xref?" + qs); } catch (e) { continue; }
      if (!x) continue;
      if (x.build_error) { seqSetXrefNotice("error", x.build_error); return; }   // crashed
      // New points dumped since we last populated -> append their dropdown options (one heavier
      // list fetch only when the cheap n_seq_files count says there's actually something new).
      const haveOpts = ($("seq-point-select") || { options: [] }).options.length;
      if ((x.n_seq_files | 0) > haveOpts) {
        try { seqAppendNewPoints(await api("/api/sequence/list?" + qsList)); }
        catch (e) { /* keep going; retry next poll */ }
      }
      // Re-render the viewed point's ruler/bands if its xref grew (it resolves once its seqid
      // ran) or pending dropped.
      if (x.available && (seqXrefParamCount(x) > seqXrefParamCount(seqState.xref) ||
                          seqXrefSize(x) > seqXrefSize(seqState.xref))) {
        seqApplyXref(x);
      }
      seqLiveBanner(x);
      if (!x.running) {
        // Run ended: keep polling briefly so the finalize() globals.json + final rebuild can
        // place the last points and zero the banner; then a final list refresh and stop.
        if ((x.pending_points | 0) === 0 || ++endedPolls > 12) {
          try { seqAppendNewPoints(await api("/api/sequence/list?" + qsList)); } catch (e) {}
          return;
        }
      } else { endedPolls = 0; }
    }
  }

  // Banner above the plot: shown while the step ruler / wait bands can't finish building.
  //   "awaiting" -> the run hasn't captured its globals yet (global-dependent step/timing
  //                 offsets can't resolve); the channel waveforms still plot.
  //   "building" -> globals present, the map is rebuilding.
  //   null       -> hide.
  function seqSetXrefNotice(kind, detail) {
    const el = $("seq-xref-notice");
    if (!el) return;
    if (!kind) { el.hidden = true; el.textContent = ""; el.className = "seq-notice"; return; }
    if (kind === "awaiting") {
      el.className = "seq-notice seq-notice-wait";
      const n = detail | 0;                          // pending_globals (0 = unknown/none)
      const what = n > 0
        ? (n + ' step/band' + (n === 1 ? '' : 's') + ' can’t be placed')
        : 'the step ruler &amp; timing bands can’t be placed';
      el.innerHTML = '<span class="seq-notice-ic">⏳</span> Waiting for the run’s globals ' +
        '(captured at the <b>end of the run</b>) — ' + what + ' until then. Channel waveforms ' +
        'and the param↔channel/pulse map are already available.';
    } else if (kind === "error") {
      // The background provenance build failed (e.g. a config error). Show the producer's
      // message so it's diagnosable instead of looking like "the map just doesn't work".
      el.className = "seq-notice seq-notice-error";
      el.innerHTML = '<span class="seq-notice-ic">⚠</span> Sequence map build failed: ' +
        '<code>' + seqEsc(detail || "unknown error") + '</code> — fix the cause, then ' +
        'use <b>Rebuild ⟳</b> to retry.';
    } else if (kind === "pending") {
      // The ruler is drawn but a few global-dependent bands (e.g. the EOM ramp) can't be
      // placed until the run's globals land. Say how many; they fill in automatically then.
      const n = detail | 0;
      el.className = "seq-notice seq-notice-wait";
      el.innerHTML = '<span class="seq-notice-ic">⏳</span> ' + n + ' timing band' +
        (n === 1 ? '' : 's') + ' still waiting on the run’s globals (captured at the ' +
        '<b>end of the run</b>) — they’ll fill in once it finishes. Steps, channels and the ' +
        'param↔channel map are already shown.';
    } else if (kind === "scan-pending") {
      // SCAN-WIDE (live run): globals.json flushes incrementally, so the point you're VIEWING
      // is already placed -- but other scan points haven't run yet, so their rulers/bands
      // aren't placed. Show how many remain; the count ticks down as the scan progresses.
      const d = detail || {};
      const pp = d.pending | 0, tot = d.total | 0;
      el.className = "seq-notice seq-notice-wait";
      el.innerHTML = '<span class="seq-notice-ic">⏳</span> ' + pp +
        (tot ? ' of ' + tot : '') + ' scan point' + (pp === 1 ? '' : 's') +
        ' still awaiting their globals (captured as each point first runs) — their step ' +
        'ruler &amp; timing bands fill in as the scan progresses. The point you’re viewing ' +
        'is already fully placed.';
    } else {
      el.className = "seq-notice seq-notice-build";
      el.innerHTML = '<span class="seq-notice-ic">⟳</span> Building the step / timing map…';
    }
    el.hidden = false;
  }

  // Does this xref already carry the step ruler / wait bands the banner waits for? The
  // "building" notice is ONLY about that timing map -- if it's present there's nothing to
  // wait for, regardless of what the (possibly stale) server `building` flag says.
  const seqXrefHasTimingMap = (xr) =>
    (((xr && xr.steps) || []).length > 0) ||
    (Object.keys((xr && xr.time_regions) || {}).length > 0);

  // Shared xref-shape accessors used by every poll loop below.
  const seqXrefSize = (xr) => (((xr && xr.steps) || []).length +
                               Object.keys((xr && xr.time_regions) || {}).length);
  const seqXrefParamCount = (xr) => Object.keys((xr && xr.param_to_channels) || {}).length;
  const seqXrefPending = (xr) => (xr && xr.pending_globals) || 0;

  // Adopt a freshly-fetched xref and re-render the param tree + step ruler -- the common tail
  // of every poll loop when the artifact has grown.
  function seqApplyXref(x) {
    seqState.xref = x;
    seqRenderParamTree();
    seqRenderStepRuler($("plot-sequence"));
  }

  // A poll loop must stop when a newer seqRenderParams superseded it (the token moved) or the
  // user navigated to a different scan. Snapshot scanId/gen at loop entry; check each tick.
  function seqPollSuperseded(scanId, gen) {
    return seqState._xrefPollGen !== gen ||
           ((seqState.query && seqState.query.scan_id) !== scanId);
  }

  // The server kicked off a background xref (re)build for the loaded scan -- poll until the
  // artifact GROWS (steps / time_regions appear, or the version bumps), then re-render.
  // Handles the case the version check alone misses: a current-version xref that was built
  // before the run's globals.json was finalized (empty steps/regions). Bails on navigation.
  async function seqPollXrefUpdate(qs, gen) {
    const scanId = seqState.query && seqState.query.scan_id;
    // The engine-free build can finish in the gap between the params response (which set
    // `xref_building`) and this call -- in which case seqState.xref was already loaded
    // complete and the ruler already rendered. Nothing to poll for: clear and bail, else
    // `size(x) > start` never trips (start == final size) and the banner sticks ~40 s.
    if (seqXrefHasTimingMap(seqState.xref)) { seqSetXrefNotice(null); return; }
    const start = seqXrefSize(seqState.xref);
    const startV = (seqState.xref && seqState.xref.version) || 0;
    for (let i = 0; i < 25; i++) {                  // ~40 s (engine-free build is quick)
      await new Promise((res) => setTimeout(res, 1600));
      if (seqPollSuperseded(scanId, gen)) return;   // newer load / navigated away
      let x = null;
      try { x = await api("/api/sequence/xref?" + qs); } catch (e) { continue; }
      if (x && x.build_error) { seqSetXrefNotice("error", x.build_error); return; }  // crashed
      if (x && (seqXrefSize(x) > start || (x.version || 0) > startV)) {
        seqApplyXref(x);
        seqSetXrefNotice(null);                     // built -> clear the banner
        return;
      }
    }
    seqSetXrefNotice(null);                          // timed out -> drop the banner
  }

  // Waiting on the run's globals (captured at run END): poll the xref (which re-triggers the
  // build once globals land) until the timing map is FULLY placed -- i.e. present AND nothing
  // still pending (pending_globals==0). Handles both the empty-map "awaiting" state and the
  // partial-map "N bands pending" state (a global-dependent band like the EOM ramp resolves
  // only once globals arrive). Renders the param map / steps as they grow; bails on navigation.
  async function seqAwaitGlobals(qs, gen) {
    const scanId = seqState.query && seqState.query.scan_id;
    // Fully resolved already (map present, nothing pending) -> nothing to wait for.
    if (seqXrefHasTimingMap(seqState.xref) && seqXrefPending(seqState.xref) === 0) {
      seqSetXrefNotice(null); return;
    }
    for (let i = 0; i < 150; i++) {                 // ~12 min cap (5 s poll)
      await new Promise((res) => setTimeout(res, 5000));
      if (seqPollSuperseded(scanId, gen)) return;   // newer load / navigated away
      let x = null;
      try { x = await api("/api/sequence/xref?" + qs); } catch (e) { continue; }
      if (!x) continue;
      if (x.build_error) { seqSetXrefNotice("error", x.build_error); return; }   // crashed
      // Re-render when the artifact GREW (param map / steps / bands) or pending dropped --
      // the param map renders the moment it exists (it needs no globals); steps + bands fill
      // in once globals land and the build re-resolves them.
      if (x.available && (seqXrefParamCount(x) > seqXrefParamCount(seqState.xref) ||
                          seqXrefSize(x) > seqXrefSize(seqState.xref) ||
                          seqXrefPending(x) < seqXrefPending(seqState.xref))) {
        seqApplyXref(x);
      }
      const hasMap = seqXrefHasTimingMap(x);
      if (hasMap && seqXrefPending(x) === 0) { seqSetXrefNotice(null); return; }  // fully placed
      if (!hasMap && x.awaiting_globals) seqSetXrefNotice("awaiting", seqXrefPending(x));
      else if (seqXrefPending(x) > 0) seqSetXrefNotice("pending", seqXrefPending(x));
      else if (x.building) seqSetXrefNotice("building");
      else { seqSetXrefNotice(null); return; }             // nothing pending
    }
    seqSetXrefNotice(null);
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
          seqRenderStepRuler($("plot-sequence"));     // phase ruler (xref.steps)
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
  // pyctrl/tools/provenance_scan.py (7 = + per-pulse source backtraces, read by the
  // /api/sequence/backtrace route's xref fallback; 8 = + pending_globals count;
  // 9 = + per-basic-sequence tagging of steps/time_regions so a multi-bseq scan shows only
  // the displayed basic sequence's phase ruler / wait bands).
  const SEQ_XREF_VERSION = 9;
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
          seqRenderStepRuler($("plot-sequence"));     // phase ruler (xref.steps)
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
    const xr = seqState.xref;
    if (!xr || !xr.available) return;
    // Channels a param drives: param_to_channels is the primary map, but it can
    // miss some -- so also pull each driven pulse's channel (fixes "channel
    // sometimes not showing" on param click, req 1).
    const want = new Set((xr.param_to_channels && xr.param_to_channels[path]) || []);
    ((xr.param_to_pids && xr.param_to_pids[path]) || []).forEach((pid) => {
      const pu = xr.pulses && xr.pulses[String(pid)];
      if (pu && pu.channel) want.add(pu.channel);
    });
    if (!want.size) return;
    const order = seqState.chnOrder || [];
    const present = order.filter((n) => want.has(n));    // only channels in THIS seq
    if (!present.length) return;
    const sel = seqState.chnSel || (seqState.chnSel = new Set());
    present.forEach((n) => sel.add(n));
    seqState.chnOrder = present.concat(order.filter((n) => !want.has(n)));  // driven → front
    seqRenderChnList();
    seqUpdateChnCount();
    await seqRenderPlot();
  }

  // Param click (leaf or focus chip) -> promote its channels, emphasize its regions, and
  // fill the focus panel with the gathered params + formula.
  async function seqSelectParam(path) {
    const pbox = $("seq-params");
    if (pbox) {
      $$(".seq-leaf.seq-leaf-active, .seq-leaf.seq-leaf-xref-hit", pbox).forEach((el) =>
        el.classList.remove("seq-leaf-active", "seq-leaf-xref-hit"));
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
    const sel = seqState.chnSel || (seqState.chnSel = new Set());
    if (sel.has(name)) return;
    sel.add(name);
    seqRenderChnList();
    seqUpdateChnCount();
    await seqRenderPlot();
    seqReapplyEmph();
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
      // Scattergl to match the base channel traces (figure.py): keeping every trace
      // on the one WebGL layer means trace order alone sets z-order -- this overlay is
      // appended last (addTraces) so it draws ON TOP of the channels, with no ambiguous
      // SVG-vs-GL layer stacking. (cliponaxis is a no-op under GL -- GL always clips.)
      const t = { x: a.x, y: a.y, type: "scattergl", mode: "lines", name: name,
                  line: { color: color, width: width }, hoverinfo: "skip",
                  showlegend: false };
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

  // Background time-band(s) for a set of pids: one shaded rect per pulse, spanning
  // that pulse's [min,max] time. Selecting a param/pulse shades the period its
  // pulses occupy (req 1), the same way wait regions are shaded.
  function _seqPulseBands(el, pidSet) {
    const ext = {};   // pid -> [min, max]
    el.data.forEach((tr) => {
      if (!tr.customdata || tr.name === SEQ_HILITE || tr.name === "Selected") return;
      for (let k = 0; k < tr.customdata.length; k++) {
        const p = Number(tr.customdata[k]);
        if (!pidSet.has(p)) continue;
        const x = tr.x[k];
        if (x == null || !isFinite(x)) continue;
        const e = ext[p] || (ext[p] = [x, x]);
        if (x < e[0]) e[0] = x;
        if (x > e[1]) e[1] = x;
      }
    });
    return Object.keys(ext).map((p) => ext[p]).filter((r) => r[1] > r[0]);
  }

  // Unified plot-emphasis driver -- param AND pulse clicks both route through here, so
  // the plot highlight ALWAYS matches the current selection (req 6). Draws a thick line
  // over the matched pulse segments + a shaded background band over each pulse's time
  // period + any explicit wait/timing bands. `key` records WHAT is emphasized so it can
  // be re-applied after a plot rebuild (channel toggle / Plotly.react).
  function seqEmphasize(pidSet, waitRegions, key) {
    seqClearEmphasis();
    const el = $("plot-sequence");
    if (!el || !el.data || !window.Plotly) return null;
    const overlays = (pidSet && pidSet.size)
      ? _seqOverlayTraces(el, pidSet, SEQ_HILITE, "#ffd166", 6) : [];
    if (overlays.length) Plotly.addTraces(el, overlays);
    const bands = [];
    if (pidSet && pidSet.size) _seqPulseBands(el, pidSet).forEach((r) => bands.push(r));
    (waitRegions || []).forEach((r) => { if (r && r.length === 2) bands.push(r); });
    if (bands.length) {
      Plotly.relayout(el, { shapes: bands.map((r) => ({
        type: "rect", xref: "x", yref: "paper", x0: r[0], x1: r[1], y0: 0, y1: 1,
        fillcolor: "rgba(255,209,102,0.13)", line: { width: 0 }, layer: "below" })) });
    }
    seqState._emph = key || null;
    return { overlays: overlays.length, bands: bands.length };
  }

  // Param-region highlight: thick line over the param's pulses + a shaded band over
  // their time period + any wait/timing regions it controls (waits have no channel output).
  function seqEmphasizeParamRegion(path) {
    const el = $("plot-sequence");
    const xr = seqState.xref;
    if (!el || !el.data || !window.Plotly || !xr || !xr.available) return;
    const pids = new Set(((xr.param_to_pids || {})[path] || []).map(Number));
    // Wait bands for THIS basic sequence only (v9 bands carry a 3rd seq_idx element);
    // strip it back to [t0, t1] for seqEmphasize, which expects 2-element regions.
    const regions = seqForCurrentBseq((xr.time_regions || {})[path] || [],
                                      (r) => (r.length > 2 ? r[2] : null))
      .map((r) => [r[0], r[1]]);
    const r = seqEmphasize(pids, regions, { kind: "param", path }) || {};
    const bits = [];
    if (r.overlays) bits.push("waveform region");
    if (regions.length) bits.push(regions.length + " time band(s)");
    toast(bits.length ? (path + " → " + bits.join(" + ")) : ("No region for " + path),
          bits.length ? "ok" : "");
  }

  // Re-apply the current emphasis after a plot rebuild (channel toggle / react).
  function seqReapplyEmph() {
    const e = seqState._emph;
    if (!e) return;
    if (e.kind === "param") seqEmphasizeParamRegion(e.path);
    else if (e.kind === "pulse" && e.pid != null)
      seqEmphasize(new Set([Number(e.pid)]), [], e);
  }

  // Remove the selection overlay + its shaded bands.
  function seqClearEmphasis() {
    seqState._emph = null;
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
      // Cyan: high contrast against the dark bg AND the many greenish channel
      // colours, and distinct from the yellow click-selection highlight (req 2a).
      pl.setAttribute("stroke", "#36e0ff");
      pl.setAttribute("stroke-width", "5");
      pl.setAttribute("stroke-linejoin", "round");
      pl.setAttribute("stroke-linecap", "round");
      pl.setAttribute("opacity", "0.95");
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
          // A hover-revealed ✕ deselects the param (req 7).
          return '<span class="seq-chip seq-chip-param" data-focus-param="' + seqEsc(path) +
            '" title="click to deselect">' + seqEsc(path) +
            (v != null ? ' <span class="seq-chip-val">' + seqEsc(v) + "</span>" : "") +
            '<span class="seq-chip-x" data-focus-param-remove="' + seqEsc(path) + '">✕</span>' +
            "</span>";
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
    seqState._focusPid = null;
    const pbox = $("seq-params");
    if (pbox) {
      $$(".seq-leaf.seq-leaf-xref-hit", pbox).forEach((el) =>
        el.classList.remove("seq-leaf-xref-hit"));
      $$(".seq-leaf.seq-leaf-active", pbox).forEach((el) =>
        el.classList.remove("seq-leaf-active"));
    }
    seqClearEmphasis();
  }

  // pulse_id sentinel for a channel's default / non-pulse point (== reader's
  // PULSE_ID_DEFAULT, 2**32-1): such points have no source pulse, so no backtrace.
  const SEQ_PID_DEFAULT = 4294967295;

  // Backtrace panel (click a point -> where in the source that pulse was built).
  // innermost frame first = the user's add/add_step call (e.g. PushoutSurvivalSeq.py:40).
  function seqBacktraceHtml(frames) {
    if (!frames || !frames.length)
      return '<div class="seq-focus-row seq-bt-row"><span class="lbl">source</span>' +
             '<span class="seq-bt-empty">no source line recorded for this point</span></div>';
    return '<div class="seq-focus-row seq-bt-row"><span class="lbl">source</span>' +
      '<div class="seq-bt-frames">' +
      frames.map((f) => '<div class="seq-bt-frame" title="' + seqEsc(f.file) + '">' +
        '<span class="seq-bt-loc">' + seqEsc(f.file.split(/[\\/]/).pop()) + ':' +
        seqEsc(String(f.line)) + '</span> <span class="seq-bt-fn">' + seqEsc(f.name) +
        '</span></div>').join("") +
      '</div></div>';
  }

  // Fetch the clicked pulse's source backtrace and inject it into the focus box. No-op when
  // the .seq carries no backtrace block (e.g. pyctrl until B3 lands) -> zero clutter there.
  async function seqShowBacktrace(pid) {
    if (pid == null) return;
    const ssel = $("seq-seq-select");
    const qs = seqQueryString({ file: seqState.file, seq: ssel ? ssel.value : "",
                                pulse_id: String(pid) });
    let r;
    try { r = await api("/api/sequence/backtrace?" + qs); } catch (e) { return; }
    if (!r || !r.has_bt) return;                            // seq has no backtraces at all
    if (String(seqState._focusPid) !== String(pid)) return;  // selection moved on meanwhile
    const box = $("seq-focus");
    if (box && !box.hidden) box.insertAdjacentHTML("beforeend", seqBacktraceHtml(r.frames));
  }

  // Click a plot point -> focus the params that derive THAT segment (per-pulse), with the
  // formula; falls back to whole-channel for idle/default (pid=-1) points.
  function seqFocusPoint(chn, pid, value, time) {
    const xr = seqState.xref;
    if (!xr || !xr.available) return;
    const box = $("seq-params");
    if (box) $$(".seq-leaf.seq-leaf-xref-hit, .seq-leaf.seq-leaf-active", box).forEach((el) =>
      el.classList.remove("seq-leaf-xref-hit", "seq-leaf-active"));
    const pulse = (pid != null && xr.pulses) ? xr.pulses[String(pid)] : null;
    // A REAL pulse (not the channel's default/idle value, pid sentinel 0xFFFFFFFF) gets a
    // source backtrace even when it carries no params (a constant TTL/voltage set) -- the
    // producer captured all of them. Track the focused pid so the async backtrace fetch only
    // injects into the matching selection (a fast second click supersedes it).
    const realPid = (pid != null && Number(pid) !== SEQ_PID_DEFAULT) ? Number(pid) : null;
    seqState._focusPid = realPid;
    let params, formula = null, idle = false;
    if (pulse) {
      params = pulse.params || []; formula = pulse.expr || null;
    } else {
      params = (xr.channel_to_params && xr.channel_to_params[chn]) || [];
      idle = true;                                  // pid=-1 / no per-pulse data
    }
    // Sync the PLOT highlight to the clicked pulse (req 6): so clicking a pulse
    // always re-highlights THAT pulse (thick line + shaded time band), replacing
    // any stale param/channel emphasis. Idle points clear the emphasis.
    if (pulse) seqEmphasize(new Set([Number(pid)]), [], { kind: "pulse", pid: Number(pid) });
    else seqClearEmphasis();
    if (box) params.forEach((p) => {
      const row = box.querySelector('.seq-leaf[data-param-path="' +
                                    p.replace(/"/g, '\\"') + '"]');
      if (row) row.classList.add("seq-leaf-xref-hit");
    });
    const tStr = (time != null && isFinite(time)) ? Number(time).toFixed(3) + " ms" : "";
    const vStr = (value != null && isFinite(value)) ? seqFmt(value) : "";
    let title = chn + (tStr ? " @ " + tStr : "") + (vStr ? " = " + vStr : "");
    // "idle" = the clicked point's pid has NO per-segment provenance (it's the
    // channel's initial/idle value, or a collapsed pulse not in the map). We then
    // fall back to EVERY param that touches this channel anywhere -- spell that out
    // so the long list isn't mistaken for params that derive THIS point.
    if (idle && !params.length) title += "  ·  idle / no pulse at this point";
    else if (idle) title += "  ·  no per-segment data here — showing all " +
                            params.length + " params that touch this channel";
    seqSetFocus({ title, formula, channels: [chn],
                  params: params.map((p) => ({ path: p, value: seqParamValue(p) })) });
    // Show where this pulse was built (source file:line), appended async once it returns --
    // for ANY real pulse, including param-less ones (the focus box is shown either way).
    if (realPid != null) seqShowBacktrace(pid);
    // Frame the view so BOTH the plot and the focus/highlight region are visible
    // un-clipped (req 4) -- replaces the old jump deep into the param tree.
    seqScrollFraming();
  }

  // Scroll so the plot sits just under the top bar, bringing the focus/highlight
  // region (directly below it) into view without clipping the plot (req 4).
  function seqScrollFraming() {
    const plot = document.getElementById("plot-sequence");
    const focus = document.getElementById("seq-focus");
    if (!plot || !focus || focus.hidden) return;
    const desiredTop = 70;        // clears the sticky tab bar
    const delta = plot.getBoundingClientRect().top - desiredTop;
    if (Math.abs(delta) > 4) window.scrollBy({ top: delta, behavior: "smooth" });
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
    const tr = seqForCurrentBseq((xr.time_regions && xr.time_regions[path]) || [],
                                 (r) => (r.length > 2 ? r[2] : null));
    const pv = seqParamValue(path);
    let title = path + (pv != null ? " = " + seqFmt(pv) : "");
    if (!channels.size && tr.length) title += "  (" + tr.length + " wait region(s))";
    seqSetFocus({ title, formula, channels: Array.from(channels),
                  params: Array.from(related).map((p) => ({ path: p, value: seqParamValue(p) })) });
  }

  function loadSequence() {
    if (seqState._refreshToggle) seqState._refreshToggle();
    // The Sequence tab now uses the SAME unified Scans picker as Analysis
    // (left-docked, shared selection). Populate it if empty, else just refresh
    // the badges / seq-current highlight, then show the PRIMARY's sequence.
    if (!runsCache.length) { loadRunsList().then(ensureSeqShown); }
    else { renderRunsTable(); ensureSeqShown(); }
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
    // Custom channel picker: click a row to add/remove; SHIFT-click to add the whole
    // range from the last click to here; Clear all; filter box.
    const clist = $("seq-chn-list");
    if (clist) clist.addEventListener("click", (e) => {
      const row = e.target.closest(".chn-row[data-chn]");
      if (!row || !clist.contains(row)) return;
      const name = row.dataset.chn;
      if (e.shiftKey && seqState._chnAnchor) {
        seqSelectChannelRange(seqState._chnAnchor, name);   // add the range (anchor stays)
      } else {
        seqToggleChannel(name);
        seqState._chnAnchor = name;                         // new anchor for future shift-clicks
      }
    });
    const cclear = $("seq-chn-clear");
    if (cclear) cclear.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!seqState.chnSel || !seqState.chnSel.size) return;
      seqState.chnSel = new Set();
      seqRenderChnList(); seqUpdateChnCount(); seqRenderPlot();
    });
    const csearch = $("seq-chn-search");
    if (csearch) csearch.addEventListener("input", seqRenderChnList);

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
      // Click a param chip (or its hover-✕) -> DESELECT / clear the selection (req 7).
      if (e.target.closest("[data-focus-param-remove], [data-focus-param]")) {
        seqClearFocus(); return;
      }
      const cc = e.target.closest("[data-focus-chan]");
      if (cc) { seqShowChannel(cc.dataset.focusChan); }
    });

    // Right-docked Channels picker (hover-driven edge tab). The Scans picker is
    // now the SHARED unified picker (#analysis-runs-card) docked LEFT and wired
    // by setupFloatingAnalysisCards() -- no separate Sequence scan card anymore.
    _wireSeqFloatCard(document.getElementById("sequence-chn-card"));

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

  // =====================================================================
  // CARD EXPAND — blow a Live-tab card up into a centered overlay (rest of
  // the screen dimmed); click the backdrop, the ⤢ button again, or press Esc
  // to contract. In the live-flat layout the card <h2> headers are collapsed
  // to 0px (titles are painted inside each plot), so a clickable header is
  // impossible — instead every eligible card gets a corner ⤢ button. The
  // header-click is kept as a bonus for any card that DOES show a header.
  // =====================================================================
  function initCardExpand() {
    const tab = $("tab-live");
    if (!tab) return;
    let backdrop = document.getElementById("card-expand-backdrop");
    if (!backdrop) {
      backdrop = document.createElement("div");
      backdrop.id = "card-expand-backdrop";
      document.body.appendChild(backdrop);
    }
    let expanded = null;
    const resizeSoon = (card) => {
      if (!window.Plotly || !card) return;
      [80, 280].forEach((ms) => setTimeout(() => {
        card.querySelectorAll(".plot-container").forEach((el) => {
          try { if (el.querySelector(".plotly")) Plotly.Plots.resize(el); }
          catch (e) {}
        });
      }, ms));
    };
    const contract = () => {
      if (!expanded) return;
      const c = expanded;
      expanded = null;
      c.classList.remove("card-expanded");
      backdrop.classList.remove("show");
      resizeSoon(c);
    };
    const expand = (card) => {
      if (expanded === card) { contract(); return; }
      if (expanded) expanded.classList.remove("card-expanded");
      expanded = card;
      card.classList.add("card-expanded");
      backdrop.classList.add("show");
      resizeSoon(card);
    };
    // Inject a corner ⤢ button into each eligible card.
    tab.querySelectorAll(".card:not(.live-sidebar):not(.status-strip)")
      .forEach((card) => {
        if (card.querySelector(":scope > .card-expand-btn")) return;
        if (getComputedStyle(card).position === "static") {
          card.style.position = "relative";
        }
        const btn = document.createElement("button");
        btn.className = "card-expand-btn";
        btn.type = "button";
        btn.title = "Expand / collapse this panel";
        btn.setAttribute("aria-label", "expand panel");
        btn.textContent = "⤢";
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          expand(card);
        });
        card.appendChild(btn);
      });
    // Bonus: clicking a (visible) card header also toggles expand.
    tab.addEventListener("click", (e) => {
      const h2 = e.target.closest("h2");
      if (!h2) return;
      const card = h2.closest(".card");
      if (!card || card.classList.contains("live-sidebar")
          || card.classList.contains("status-strip")) return;
      if (e.target.closest("input, button, select, label, a, option")) return;
      e.preventDefault();
      expand(card);
    });
    backdrop.addEventListener("click", contract);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") contract();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    foldSequenceIntoAnalysis();   // single sequence view, inside Analysis
    initAnalysisModeToggle();     // wire Data/Sequence toggle + default to Data
    initHwSubtabs();              // merged Hardware tab sub-views (fold + restore)
    wireQueuePopup();             // full-queue popup (Live sidebar "full ›")
    initTabs();
    initSequenceTab();
    setupFloatingAnalysisCards();
    initCardExpand();
    // Affine card: rollback button (mint nothing — direct POST).
    const afRoll = document.getElementById("affine-rollback");
    if (afRoll) afRoll.addEventListener("click", async () => {
      if (!confirm("Roll the global affine back to the previous version?")) return;
      try {
        await api("/api/affine/rollback", {method: "POST"});
        affineRenderedTs = null;   // force re-render
        await pollAffine();
        toast("Affine rolled back");
      } catch (e) { toast("Rollback failed: " + (e.message || e)); }
    });
    startPolling();
    loadRunsList();
    fetchRunDates();   // populate the date dropdown with the whole archive
    loadGroups();
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
  });
})();
