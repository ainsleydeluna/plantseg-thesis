/* ============================================================
   From Zero to Dry-Run — script.js
   Vanilla JS only. No dependencies, no network.
   Features: dark mode, scrollspy, progress checkboxes,
   glossary tooltips, quiz reveal, copy buttons, mobile menu,
   back-to-top, lightweight comment highlighting.
   ============================================================ */
(function () {
  "use strict";

  var LS_THEME = "e1guide.theme";
  var LS_PROGRESS = "e1guide.progress";

  /* -------- helpers -------- */
  function $(sel, ctx) { return (ctx || document).querySelector(sel); }
  function $all(sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); }
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /* ============================================================
     1. THEME (light/dark) with persistence
     ============================================================ */
  var html = document.documentElement;
  var themeBtn = $("#themeToggle");
  function applyTheme(t) {
    html.setAttribute("data-theme", t);
    if (themeBtn) themeBtn.textContent = t === "dark" ? "☀️" : "🌙";
  }
  try {
    var savedTheme = localStorage.getItem(LS_THEME);
    if (savedTheme) applyTheme(savedTheme);
    else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) applyTheme("dark");
  } catch (e) {}
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem(LS_THEME, next); } catch (e) {}
    });
  }

  /* ============================================================
     2. PROGRESS checkboxes (persist all [data-check])
     ============================================================ */
  var store = {};
  try { store = JSON.parse(localStorage.getItem(LS_PROGRESS) || "{}") || {}; } catch (e) { store = {}; }
  function saveStore() { try { localStorage.setItem(LS_PROGRESS, JSON.stringify(store)); } catch (e) {} }

  var navChecks = $all(".sidebar [data-check]");
  function updateProgress() {
    var total = navChecks.length || 1;
    var done = navChecks.filter(function (c) { return c.checked; }).length;
    var pct = Math.round((done / total) * 100);
    var txt = $("#progressText");
    if (txt) txt.textContent = pct + "%";
  }
  $all("[data-check]").forEach(function (cb) {
    var id = cb.getAttribute("data-check");
    if (store[id]) cb.checked = true;
    cb.addEventListener("change", function () {
      store[id] = cb.checked;
      if (!cb.checked) delete store[id];
      saveStore();
      updateProgress();
    });
  });
  updateProgress();

  var resetBtn = $("#resetProgress");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      store = {}; saveStore();
      $all("[data-check]").forEach(function (cb) { cb.checked = false; });
      updateProgress();
    });
  }

  /* ============================================================
     3. QUIZ reveal
     ============================================================ */
  $all(".quiz .reveal").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var ans = btn.parentElement.querySelector(".answer");
      if (!ans) return;
      var showing = ans.classList.toggle("show");
      btn.textContent = showing ? "Hide answer" : "Show answer";
    });
  });

  /* ============================================================
     4. COPY buttons
     ============================================================ */
  $all(".cmd .copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var code = btn.parentElement.querySelector("code");
      if (!code) return;
      var text = code.textContent;
      var done = function () {
        var old = btn.textContent;
        btn.textContent = "Copied ✓"; btn.classList.add("copied");
        setTimeout(function () { btn.textContent = old; btn.classList.remove("copied"); }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
      } else { fallbackCopy(text, done); }
    });
  });
  function fallbackCopy(text, cb) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta); if (cb) cb();
  }

  /* ============================================================
     5. LIGHTWEIGHT code highlighting (comments only, safe)
     ============================================================ */
  $all("pre.code code").forEach(function (code) {
    var lines = code.textContent.split("\n");
    code.innerHTML = lines.map(function (line) {
      var i = line.indexOf("#");
      if (i === -1) return escapeHtml(line);
      return escapeHtml(line.slice(0, i)) + '<span class="cmt">' + escapeHtml(line.slice(i)) + "</span>";
    }).join("\n");
  });

  /* ============================================================
     6. GLOSSARY tooltips (shared element; hover / focus / tap)
     ============================================================ */
  var tip = $("#tooltip");
  var activeTerm = null;
  function showTip(term) {
    if (!tip) return;
    var def = term.getAttribute("data-def");
    if (!def) return;
    tip.textContent = def;
    tip.hidden = false;
    var r = term.getBoundingClientRect();
    var tr = tip.getBoundingClientRect();
    var left = window.scrollX + r.left;
    var top = window.scrollY + r.top - tr.height - 9;
    // keep inside viewport horizontally
    var maxLeft = window.scrollX + document.documentElement.clientWidth - tr.width - 10;
    if (left > maxLeft) left = maxLeft;
    if (left < window.scrollX + 8) left = window.scrollX + 8;
    // flip below if not enough room above
    if (r.top - tr.height - 9 < 0) top = window.scrollY + r.bottom + 9;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
    activeTerm = term;
  }
  function hideTip() { if (tip) { tip.hidden = true; } activeTerm = null; }

  $all(".term").forEach(function (term) {
    term.setAttribute("tabindex", "0");
    term.addEventListener("mouseenter", function () { showTip(term); });
    term.addEventListener("mouseleave", hideTip);
    term.addEventListener("focus", function () { showTip(term); });
    term.addEventListener("blur", hideTip);
    term.addEventListener("click", function (e) {
      e.stopPropagation();
      if (activeTerm === term) hideTip(); else showTip(term);
    });
  });
  document.addEventListener("click", function () { if (activeTerm) hideTip(); });
  window.addEventListener("scroll", function () { if (activeTerm) hideTip(); }, { passive: true });

  /* ============================================================
     7. SCROLLSPY — active nav link + timeline stage
     ============================================================ */
  var sections = $all("main section[id]");
  var navLinks = {};
  $all(".sidebar a[data-sec]").forEach(function (a) { navLinks[a.getAttribute("data-sec")] = a; });

  // section id -> workflow stage (1..7) for the timeline (matches the 20-section rebuild).
  // Stages: 1 Locked decisions · 2 Understand the code · 3 Smoke tests ·
  //         4 Pre-flight · 5 Dry-run · 6 Real run · 7 Later stages.
  var STAGE = {
    s1: 2, s2: 1, s3: 2,                                       // orientation (s2 holds the invariants)
    s4: 2, s5: 2, s6: 2, s7: 2, s8: 2, s9: 2, s10: 2, s11: 2,  // the code path
    s12: 3, s13: 4,                                            // smoke tests · pre-flight/launch
    s14: 2, s15: 2, s16: 2, s17: 7, s18: 2, s19: 2, s20: 4     // reference/safety (s17 future, s20 readiness)
  };
  var tlSteps = {};
  $all(".tl-step").forEach(function (s) { tlSteps[s.getAttribute("data-stage")] = s; });

  var currentId = null;
  function setActive(id) {
    if (id === currentId || !id) return;
    currentId = id;
    Object.keys(navLinks).forEach(function (k) { navLinks[k].classList.remove("active"); });
    if (navLinks[id]) navLinks[id].classList.add("active");
    var stage = STAGE[id];
    Object.keys(tlSteps).forEach(function (k) { tlSteps[k].classList.remove("tl-active"); });
    if (stage && tlSteps[stage]) tlSteps[stage].classList.add("tl-active");
  }

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () {
      var threshold = 150;
      var chosen = sections.length ? sections[0].id : null;
      for (var i = 0; i < sections.length; i++) {
        if (sections[i].getBoundingClientRect().top <= threshold) chosen = sections[i].id;
        else break;
      }
      setActive(chosen);
      var toTop = $("#toTop");
      if (toTop) toTop.hidden = window.scrollY < 500;
      ticking = false;
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", function () { if (activeTerm) hideTip(); });
  onScroll();

  /* ============================================================
     8. MOBILE menu + back-to-top
     ============================================================ */
  var sidebar = $("#sidebar");
  var menuBtn = $("#menuToggle");
  if (menuBtn && sidebar) {
    menuBtn.addEventListener("click", function () { sidebar.classList.toggle("open"); });
    $all(".sidebar a").forEach(function (a) {
      a.addEventListener("click", function () {
        if (window.matchMedia("(max-width: 860px)").matches) sidebar.classList.remove("open");
      });
    });
  }
  var toTop = $("#toTop");
  if (toTop) {
    toTop.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: "smooth" }); });
  }
})();
