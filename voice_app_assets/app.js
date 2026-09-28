// voice_app / app.js -- V2 room voice panel (vk-1826), restyled to Aurora's
// accepted mockup room-voice-v1 (A0): http://100.110.14.59/styleguide/mockups
// /room-voice-v1.html -- the "the v2v frame (inside the iframe: its own
// ground)" section is this panel; class names (.fr, .fr-top, .spk, .mute,
// .disc, .exs, .ex) mirror that markup so the two stay in lockstep.
//
// Runs inside a sandboxed (opaque-origin) iframe loaded through the
// briefing-table engine's panel proxy. No cookies, no storage, no
// allow-same-origin. Every dynamic value is rendered with textContent (or a
// fixed, non-data-derived innerHTML for the mute icon), never innerHTML of
// server/engine data. The only communication with the host is postMessage;
// the closed vocabulary this frame ever sends is:
//   {panel:"status", speaking}
//   {panel:"height", px}
//   {panel:"voice", action:"mute"|"unmute"|"disconnect"}
//   {panel:"line", lineId}
(function () {
  "use strict";

  // Fixture harness (Playwright): every message this frame posts is also
  // recorded here, in order, regardless of panel/host wiring.
  window.__posted = window.__posted || [];

  var frameEl = document.getElementById("frame");
  var devEl = document.getElementById("dev");
  var exchangesEl = document.getElementById("exchanges");
  var speakingRow = document.getElementById("speakingRow");
  var muteBtn = document.getElementById("muteBtn");
  var disconnectBtn = document.getElementById("disconnectBtn");

  var lastHeight = -1;
  // null = no poll-driven post has happened yet. The very first (hardcoded)
  // render below always posts speaking:false unconditionally -- see render()
  // -- everything after that (the seed render and every subsequent poll)
  // posts only when the value actually changed.
  var lastPostedSpeaking = null;

  var MIC_ON = '<rect x="9" y="3" width="6" height="11" rx="3"></rect>' +
    '<path d="M5 11a7 7 0 0 0 14 0M12 18v3"></path>';
  var MIC_OFF = '<path d="M9 9v2a3 3 0 0 0 5.1 2.1M15 10V6a3 3 0 0 0-5.7-1.3">' +
    '</path><path d="M5 11a7 7 0 0 0 11.7 5.2M19 11a7 7 0 0 1-.4 2.3M12 18v3M3 3l18 18"></path>';

  function post(msg) {
    window.__posted.push(msg);
    try {
      parent.postMessage(msg, "*");
    } catch (e) {
      // opaque-origin frame with no parent reachable (e.g. opened top-level
      // directly for the curl/manual-open case) -- nothing to do.
    }
  }

  function fmtWho(who) {
    if (who === "user") {
      return "You";
    }
    if (who === "voice") {
      return "Voice";
    }
    return String(who || "").replace(/[-_]/g, " ");
  }

  // "just now" / "N min" / "N h" -- since is epoch seconds.
  function fmtSince(since) {
    var secs = Math.max(0, (Date.now() / 1000) - since);
    if (secs < 60) {
      return "just now";
    }
    var mins = Math.round(secs / 60);
    if (mins < 60) {
      return mins + " min";
    }
    return Math.round(mins / 60) + " h";
  }

  // Connection state (device, since) -- blueprint sec9.5 frame content.
  // Empty when not connected: the host draws the unreachable/disconnected
  // states, never this frame.
  function renderDev(device, since, connected) {
    devEl.textContent = "";
    if (!connected || !device) {
      return;
    }
    var b = document.createElement("b");
    b.textContent = device;
    devEl.appendChild(b);
    if (since != null) {
      devEl.appendChild(document.createTextNode(" · since " + fmtSince(since)));
    }
  }

  function makePick(lineId, el) {
    return function () {
      var prev = exchangesEl.querySelectorAll(".ex.picked");
      for (var i = 0; i < prev.length; i++) {
        prev[i].classList.remove("picked");
      }
      el.classList.add("picked");
      post({ panel: "line", lineId: lineId });
    };
  }

  function renderExchanges(list, waiting) {
    exchangesEl.textContent = "";
    var items = (list || []).slice(-3);
    if (items.length === 0) {
      var empty = document.createElement("li");
      empty.className = "empty-row";
      empty.textContent = "No exchanges yet.";
      exchangesEl.appendChild(empty);
      return;
    }

    // The ONE waiting marker: the most recent "out" row, only while
    // state.waiting is true. Never any other row.
    var lastOutIdx = -1;
    items.forEach(function (ex, i) {
      if (ex.kind === "out") {
        lastOutIdx = i;
      }
    });

    items.forEach(function (ex, i) {
      var li = document.createElement("li");
      li.className = "ex";

      var lead = document.createElement("span");
      lead.className = "v";
      lead.textContent = (ex.kind === "out" && ex.vid) ? "#" + ex.vid : fmtWho(ex.who);
      li.appendChild(lead);

      var textSpan = document.createElement("span");
      textSpan.className = ex.kind === "out" ? "a" : "q";
      textSpan.textContent = ex.text || "";
      if (ex.text) {
        textSpan.title = ex.text;
      }
      if (ex.kind === "out" && i === lastOutIdx && waiting) {
        textSpan.appendChild(document.createTextNode(" · "));
        var wt = document.createElement("span");
        wt.className = "wt";
        wt.textContent = "waiting";
        textSpan.appendChild(wt);
      }
      li.appendChild(textSpan);

      // The fallback-lead note: plain dim text under an "in" exchange,
      // never amber or coral.
      if (ex.kind === "in" && ex.resolved === "fallback-lead") {
        var note = document.createElement("div");
        note.className = "nt";
        note.textContent = "heard, no seat matched";
        li.appendChild(note);
      }

      if (ex.lineId) {
        li.tabIndex = 0;
        li.setAttribute("role", "button");
        var pick = makePick(ex.lineId, li);
        li.addEventListener("click", pick);
        li.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            pick();
          }
        });
      }

      exchangesEl.appendChild(li);
    });
  }

  // The frame draws no status word for connected/muted/waiting (the host's
  // header owns that) -- speaking is the one fact the frame adds, and it
  // toggles a class on the frame shell so .spk follows the mockup's own
  // ".fr.speaking .spk" rule.
  function renderSpeaking(speaking) {
    frameEl.classList.toggle("speaking", !!speaking);
  }

  // The frame's Mute toggle shows the state the engine/state row last sent.
  // It never guesses ahead of that -- no optimistic toggle on click.
  function renderMute(muted, connected) {
    muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
    muteBtn.setAttribute("aria-label", muted ? "Unmute" : "Mute");
    muteBtn.title = muted ? "Unmute" : "Mute";
    muteBtn.innerHTML = '<svg viewBox="0 0 24 24">' + (muted ? MIC_OFF : MIC_ON) + '</svg>';
    muteBtn.disabled = !connected;
    disconnectBtn.disabled = !connected;
  }

  function postHeight() {
    var px = Math.ceil(document.documentElement.scrollHeight);
    if (px !== lastHeight) {
      lastHeight = px;
      post({ panel: "height", px: px });
    }
  }

  // forcePost: true for the very first (hardcoded, pre-data) render only --
  // it always posts speaking:false, matching the prior behaviour. Every
  // other call (the seed render and every poll) posts speaking only when the
  // value changed since the last post.
  function render(data, forcePost) {
    var speaking = !!data.speaking;
    var connected = data.connected !== false;
    renderSpeaking(speaking);
    renderDev(data.device, data.since, connected);
    renderMute(!!data.muted, connected);
    renderExchanges(data.exchanges, !!data.waiting);
    postHeight();
    if (forcePost || speaking !== lastPostedSpeaking) {
      lastPostedSpeaking = speaking;
      post({ panel: "status", speaking: speaking });
    }
  }

  muteBtn.addEventListener("click", function () {
    if (muteBtn.disabled) {
      return;
    }
    var muted = muteBtn.getAttribute("aria-pressed") === "true";
    post({ panel: "voice", action: muted ? "unmute" : "mute" });
  });

  disconnectBtn.addEventListener("click", function () {
    if (disconnectBtn.disabled) {
      return;
    }
    post({ panel: "voice", action: "disconnect" });
  });

  window.addEventListener("resize", postHeight);

  // Render fast, before the first poll returns, so the frame is never blank
  // (blank-frame handling belongs to the host, but we still don't hand it
  // one) and so the ~5s render handshake window is always met. This is the
  // one unconditional "post speaking:false immediately on render" call.
  render({ speaking: false, muted: false, connected: false, exchanges: [] }, true);

  var seed = document.getElementById("initial-state");
  if (seed && seed.textContent) {
    try {
      render(JSON.parse(seed.textContent), false);
    } catch (e) {
      // malformed seed -- keep the empty render above.
    }
  }

  // Poll ./state (relative, so it survives the proxy's directory mapping).
  // The query string (if any -- the fixture harness rides its combination
  // here) is forwarded so the same deterministic sample keeps coming back on
  // every poll. Cadence: 2s while speaking, else 3s -- rescheduled after
  // every response rather than a fixed setInterval, since the interval
  // itself depends on the poll's own result.
  var STATE_URL = "state" + (location.search || "");
  var pollTimer = null;

  function schedulePoll(speaking) {
    if (pollTimer !== null) {
      clearTimeout(pollTimer);
    }
    pollTimer = setTimeout(poll, speaking ? 2000 : 3000);
  }

  function poll() {
    fetch(STATE_URL, { cache: "no-store" })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        render({
          speaking: !!data.speaking,
          muted: data.muted,
          connected: data.connected,
          waiting: data.waiting,
          device: data.device,
          since: data.since,
          exchanges: data.exchanges
        }, false);
        schedulePoll(!!data.speaking);
      })
      .catch(function () {
        // Unreachable is the HOST's state to draw, not the frame's; keep
        // showing the last good render rather than blanking anything.
        schedulePoll(false);
      });
  }

  poll();
}());
