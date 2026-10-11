// Inside the Network: the pieces every chapter shares.
// Number formatting, descriptions on hover, the Words · Math · Code switch, and the shortcut to the playground.
(function () {
  "use strict";
  const SUBS = "₀₁₂₃₄₅₆₇₈₉";
  const ITN = {};
  ITN.sb = (n) => String(n).replace(/\d/g, (d) => SUBS[d]);
  ITN.fmt = function (v, d = 2) {
    if (!Number.isFinite(v)) return "–";
    const s = Math.abs(v).toFixed(d);
    return (v < 0 && Number(s) !== 0 ? "−" : "") + s;
  };
  ITN.par = (v, d = 2) => { const s = ITN.fmt(v, d); return s[0] === "−" ? "(" + s + ")" : s; };
  // a number as it would be typed in code: plain hyphen, never −0
  ITN.code = (v, d = 3) => { const x = Number(v.toFixed(d)); return (Object.is(x, -0) ? 0 : x).toFixed(d); };
  ITN.esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  ITN.signed = (v, d = 3) => { const s = ITN.fmt(v, d); return s[0] === "−" ? s : "+" + s; };
  // σ for plain numbers (the autograd's sigmoid() takes a Value); this form never overflows
  ITN.sig = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  // a name as it can be written in code: w₂₁ → w21, ŷ → yhat
  ITN.cn = (name) => name.replace(/[₀-₉]/g, (d) => String(SUBS.indexOf(d))).replace("ŷ", "yhat");
  // one scale of confidence for every sentence about a guess ŷ of a dot labeled y
  ITN.verdict = function (yh, y) {
    const margin = Math.abs(yh - 0.5);
    if (margin < 0.005) return "undecided";
    const right = (yh > 0.5 ? 1 : 0) === y;
    if (margin < 0.15) return right ? "justRight" : "justWrong";
    if (margin < 0.35) return right ? "rightish" : "wrong";
    return right ? "right" : "wrongConf";
  };
  ITN.reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // the text of a script element: its own text if it has any, otherwise the file it loads.
  // Browsers refuse to fetch files from a page opened straight from disk; say so instead of waiting forever.
  ITN.SOURCE_MISSING = "// The code could not be loaded here. Browsers block reading files from a page opened\n// straight from your disk; serve the folder instead (see the README), or read assets/autograd.js.";
  ITN.source = function (id) {
    const el = document.getElementById(id);
    if (!el) return Promise.resolve(ITN.SOURCE_MISSING);
    if (!el.src) return Promise.resolve(el.textContent.trim());
    return fetch(el.src).then((r) => (r.ok ? r.text() : "")).then((t) => t.trim() || ITN.SOURCE_MISSING).catch(() => ITN.SOURCE_MISSING);
  };

  // ---------- descriptions on hover, keyboard focus, or after a tap ----------
  // Hover is tracked by pointer position, not enter/leave events: pages redraw under a still mouse,
  // and a control is identified by what it is (tip key, selection, id), not by its DOM node.
  const TIPS = {};
  const tt = document.createElement("div");
  tt.className = "tt"; tt.id = "tt"; tt.setAttribute("role", "tooltip"); tt.setAttribute("aria-hidden", "true");
  let tipEl = null, tipPoint = null, tipTimer = 0, tipHide = 0, tipMode = "", mouse = null, hoverKey = "";
  const tipHtml = (el) => { const f = TIPS[el.dataset.tip]; return f ? f(el) : ""; };
  function placeTip() {
    const w = tt.offsetWidth, h = tt.offsetHeight;
    let x, y;
    if (tipPoint) { x = tipPoint[0] - w / 2; y = tipPoint[1] - h - 16; if (y < 8) y = tipPoint[1] + 20; }
    else { const r = tipEl.getBoundingClientRect(); x = r.left + r.width / 2 - w / 2; y = r.top - h - 8; if (y < 8) y = r.bottom + 8; }
    tt.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x)) + "px";
    tt.style.top = Math.max(8, y) + "px";
  }
  function showTip(el, point, mode) {
    clearTimeout(tipHide);
    if (tipEl && tipEl !== el) tipEl.removeAttribute("aria-describedby");
    tipEl = el; tipMode = mode; tipPoint = point && el.hasAttribute("data-follow") ? point : null;
    const html = tipHtml(el);
    if (!html) return;
    tt.innerHTML = html; tt.classList.add("on"); tt.setAttribute("aria-hidden", "false");
    el.setAttribute("aria-describedby", "tt");
    placeTip();
  }
  function hideTip() {
    clearTimeout(tipHide);
    if (tipEl) tipEl.removeAttribute("aria-describedby");
    tipEl = null; tipPoint = null; tipMode = ""; tt.classList.remove("on"); tt.setAttribute("aria-hidden", "true");
  }
  const keyOf = (el) => (el ? `${el.dataset.tip}|${el.dataset.sel || ""}|${el.id}` : "");
  function tipTargetAt(pt) { const hit = pt && document.elementFromPoint(pt[0], pt[1]); return hit && hit.closest ? hit.closest("[data-tip]") : null; }
  function hoverCheck() {
    if (!mouse) return;
    const el = tipTargetAt(mouse), key = keyOf(el);
    if (key !== hoverKey) {
      hoverKey = key; clearTimeout(tipTimer);
      if (tipMode === "hover") hideTip();
      if (el) tipTimer = setTimeout(() => {
        const now = tipTargetAt(mouse);
        if (now && keyOf(now) === hoverKey && tipMode !== "focus") showTip(now, mouse, "hover");
      }, el.hasAttribute("data-fast") ? 120 : 450);
    } else if (el && tipMode === "hover" && tipEl !== el) tipEl = el; // the same control, rebuilt under the mouse
  }
  // pages call this whenever they redraw, so an open tip shows live values
  function refreshTip() {
    hoverCheck();
    if (tipEl && tipEl.isConnected && tt.classList.contains("on")) { tt.innerHTML = tipHtml(tipEl); placeTip(); }
  }
  ITN.tips = { add: (obj) => Object.assign(TIPS, obj), refresh: refreshTip };
  document.addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse") return;
    mouse = [e.clientX, e.clientY];
    hoverCheck();
    if (tipEl && tipPoint && tipMode === "hover") { tipPoint = mouse; placeTip(); }
  });
  document.documentElement.addEventListener("pointerleave", (e) => {
    if (e.pointerType !== "mouse") return;
    mouse = null; hoverKey = ""; clearTimeout(tipTimer);
    if (tipMode === "hover") hideTip();
  });
  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) return;
    if (e.pointerType === "touch" || (!e.pointerType && matchMedia("(hover: none)").matches)) {
      requestAnimationFrame(() => { showTip(el, null, "touch"); tipHide = setTimeout(hideTip, 2600); });
    } else if (el === tipEl) requestAnimationFrame(refreshTip);
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el && e.target.matches(":focus-visible")) { clearTimeout(tipTimer); showTip(el, null, "focus"); }
  });
  document.addEventListener("focusout", () => { if (tipMode === "focus") hideTip(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { hideTip(); hoverKey = ""; } });
  window.addEventListener("scroll", () => {
    if (tipMode === "touch" || tipPoint) hideTip();
    hoverCheck();
    if (tipEl && tipEl.isConnected) placeTip();
  }, { passive: true });

  // ---------- Words · Math · Code: one choice for the whole series ----------
  const VIEW = { mode: "words", lang: "js" };
  try {
    const saved = JSON.parse(localStorage.getItem("itn-lens") || "null");
    if (saved && ["words", "math", "code", "all"].includes(saved.mode)) { VIEW.mode = saved.mode; VIEW.lang = saved.lang === "py" ? "py" : "js"; }
  } catch (e) { /* storage unavailable: keep the default */ }
  const viewListeners = [];
  // r: { words: html, math: [html lines] or mathHtml: html, js: code text, py: code text }.
  // Any field may be a function, built only when that view is showing.
  function reps(r0) {
    const want = (m) => VIEW.mode === "all" || VIEW.mode === m;
    const val = (x) => (typeof x === "function" ? x() : x);
    const r = {
      words: want("words") ? val(r0.words) : "", math: want("math") ? val(r0.math) : null, mathHtml: want("math") ? val(r0.mathHtml) : "",
      js: want("code") && VIEW.lang === "js" ? val(r0.js) : "", py: want("code") && VIEW.lang === "py" ? val(r0.py) : "",
    };
    let h = "";
    if (want("words") && r.words) h += `<div class="rep words"><div class="rep-lab">Words</div><p>${r.words}</p></div>`;
    if (want("math") && (r.math || r.mathHtml)) h += `<div class="rep math"><div class="rep-lab">Math</div>${r.mathHtml || `<div class="mb">${r.math.map((l) => `<span>${l}</span>`).join("")}</div>`}</div>`;
    if (want("code") && (r.js || r.py)) h += `<div class="rep code"><div class="rep-lab">Code · ${VIEW.lang === "py" ? "Python" : "JavaScript"}</div><pre>${ITN.esc(VIEW.lang === "py" ? r.py : r.js)}</pre></div>`;
    return h;
  }
  const CTL = (label) => `<span class="lab">${label}</span>
    <div class="seg" role="group" aria-label="${label}">
      <button type="button" data-lens="words" data-tip="lens-words">Words</button><button type="button" data-lens="math" data-tip="lens-math">Math</button><button type="button" data-lens="code" data-tip="lens-code">Code</button><button type="button" data-lens="all" data-tip="lens-all">All three</button>
    </div>
    <div class="seg lang" role="group" aria-label="Code language">
      <button type="button" data-lang="js" data-tip="lang-js">JavaScript</button><button type="button" data-lang="py" data-tip="lang-py">Python</button>
    </div>`;
  function syncView() {
    document.querySelectorAll("[data-lens-ctl]").forEach((c) => {
      if (!c.querySelector("[data-lens]")) c.innerHTML = CTL(c.dataset.label || "Explain with");
      c.querySelectorAll("[data-lens]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lens === VIEW.mode)));
      c.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === VIEW.lang)));
      c.querySelector(".lang").hidden = !(VIEW.mode === "code" || VIEW.mode === "all");
    });
    document.querySelectorAll("[data-reps]").forEach((el) => el.classList.toggle("show-all", VIEW.mode === "all"));
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-lens], [data-lang]");
    if (!b) return;
    if (b.dataset.lens) VIEW.mode = b.dataset.lens; else VIEW.lang = b.dataset.lang;
    try { localStorage.setItem("itn-lens", JSON.stringify(VIEW)); } catch (err) { /* storage unavailable */ }
    syncView();
    viewListeners.forEach((fn) => fn());
  });
  ITN.view = {
    get mode() { return VIEW.mode; },
    get lang() { return VIEW.lang; },
    reps,
    sync: syncView,
    onChange: (fn) => viewListeners.push(fn),
  };
  ITN.tips.add({
    "lens-words": () => "<b>Words</b>: explain each step in plain sentences.",
    "lens-math": () => "<b>Math</b>: the same steps as formulas, with the real numbers filled in.",
    "lens-code": () => "<b>Code</b>: the same steps as the code that does them.",
    "lens-all": () => "<b>All three</b>: words, math and code stacked, so you can see how they match line by line.",
    "lang-js": () => "<b>JavaScript</b>: the language this site runs in.",
    "lang-py": () => "<b>Python</b>: the same program in the language most machine-learning work uses.",
  });

  // ---------- the floating shortcut, shown while reading the introduction ----------
  ITN.pill = function (pillId = "jump-pill", targetId = "playground") {
    const pill = document.getElementById(pillId), target = document.getElementById(targetId);
    if (!pill || !target) return;
    let queued = false;
    window.addEventListener("scroll", () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; pill.hidden = !(window.scrollY > 500 && target.getBoundingClientRect().top > window.innerHeight * 0.5); });
    }, { passive: true });
  };

  const start = () => { document.body.appendChild(tt); syncView(); };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
  window.ITN = ITN;
})();
