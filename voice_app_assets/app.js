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
//   {panel:"voice", action:"mute"|"unmute"|"disconnect"|"id-on"|"id-off"|
//     "openmic-on"|"openmic-off"}
//   {panel:"line", lineId}
(function () {
  "use strict";

  // Fixture harness (Playwright): every message this frame posts is also
  // recorded here, in order, regardless of panel/host wiring.
  window.__posted = window.__posted || [];

  var frameEl = document.getElementById("frame");
  var devEl = document.getElementById("dev");
  var exchangesEl = document.getElementById("exchanges");
  var openMicBtn = document.getElementById("openMicBtn");
  var muteBtn = document.getElementById("muteBtn");
  var idBtn = document.getElementById("idBtn");
  var disconnectBtn = document.getElementById("disconnectBtn");

  var lastHeight = -1;
  // null = no poll-driven post has happened yet. The very first (hardcoded)
  // render below always posts speaking:false unconditionally -- see render()
  // -- everything after that (the seed render and every subsequent poll)
  // posts only when the value actually changed.
  var lastPostedSpeaking = null;

  // §9/§10.1 agent-mute + open-mic glyphs -- an Echo Dot "puck" seen from
  // above (a circle) with three concentric sound-wave arcs on one side.
  // WAVES_OUT (waves radiating away, right side) is the agent-mute icon: the
  // agent's voice is what leaves the puck. WAVES_IN (waves converging in,
  // left side, mirrored) is the open-mic-after-replies icon: it is a SETTING
  // glyph (the puck listening), never a live "mic is open now" indicator.
  // Both are stroke-only/currentColor/no-fill, matching voice_app_assets/
  // icons/puck-waves-{out,in}.svg verbatim (that pair has no overlay).
  var WAVES_OUT = '<circle cx="7" cy="12" r="4"></circle>' +
    '<path d="M13 8.5a5 5 0 0 1 0 7"></path>' +
    '<path d="M16 6a8.5 8.5 0 0 1 0 12"></path>' +
    '<path d="M19 3.5a12 12 0 0 1 0 17"></path>';
  var WAVES_IN = '<circle cx="17" cy="12" r="4"></circle>' +
    '<path d="M11 8.5a5 5 0 0 0 0 7"></path>' +
    '<path d="M8 6a8.5 8.5 0 0 0 0 12"></path>' +
    '<path d="M5 3.5a12 12 0 0 0 0 17"></path>';
  // Overlay paths (never part of the base icon files): a fixed diagonal
  // slash in the error/danger token for "off", a small check in the ok/
  // success token for "on". Hard-coded to those tokens rather than
  // currentColor -- the state color must not follow the button's own hover/
  // pressed text color the way the base glyph does. tokens.css has no
  // literal --error/--ok name; --magenta is its documented error/disconnect
  // function color and --teal its documented done/ok function color (see
  // --jp-error/--jp-done, --mcp-err/--mcp-ok in tokens.css) -- never --coral,
  // which stays reserved for "waiting".
  var WAVES_SLASH = '<path d="M4 4l16 16" stroke="var(--vfb-off)"></path>';
  var WAVES_CHECK = '<path d="M14.5 16.5l2 2 4-4.5" stroke="var(--vfb-on)"></path>';

  // §10.1 idInRoom toggle glyph -- a small id-badge/shield outline. Off
  // (idInRoom false) draws the same outline plus a diagonal slash; the
  // color split (--dim idle / --text pressed) is handled entirely by CSS
  // (.idbtn / .idbtn[aria-pressed="true"]), not here.
  var ID_BADGE = '<path d="M12 3l6 2.4v4.3c0 4.6-2.6 7.9-6 9.3-3.4-1.4-6-4.7-6-9.3V5.4z"></path>';
  var ID_SLASH = '<path d="M4 4l16 16"></path>';

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

  // The contract is EXCHANGES (a question + the reply/replies that follow
  // it), not raw messages. Pair each "in" with the "out" item(s) up to the
  // next "in": { in, outs: [...] }. An "out" arriving before any "in" (only
  // possible if the server's raw window got truncated mid-pair) still gets
  // a group of its own rather than being dropped.
  function groupExchanges(list) {
    var groups = [];
    var cur = null;
    (list || []).forEach(function (item) {
      if (item.kind === "in") {
        cur = { in: item, outs: [] };
        groups.push(cur);
      } else if (item.kind === "out") {
        if (!cur) {
          cur = { in: null, outs: [] };
          groups.push(cur);
        }
        cur.outs.push(item);
      }
    });
    return groups;
  }

  function renderExchanges(list, waiting) {
    exchangesEl.textContent = "";
    var groups = groupExchanges(list).slice(-3);
    if (groups.length === 0) {
      var empty = document.createElement("li");
      empty.className = "empty-row";
      empty.textContent = "No exchanges yet.";
      exchangesEl.appendChild(empty);
      return;
    }

    var lastIdx = groups.length - 1;

    groups.forEach(function (grp, gi) {
      var li = document.createElement("li");
      li.className = "ex";

      var lastOut = grp.outs.length ? grp.outs[grp.outs.length - 1] : null;

      var lead = document.createElement("span");
      lead.className = "v";
      lead.textContent = (lastOut && lastOut.vid) ? "#" + lastOut.vid : fmtWho(grp.in ? grp.in.who : "voice");
      li.appendChild(lead);

      var q = document.createElement("span");
      q.className = "q";
      q.textContent = grp.in ? (grp.in.text || "") : "";
      if (grp.in && grp.in.text) {
        q.title = grp.in.text;
      }
      li.appendChild(q);

      var a = document.createElement("span");
      a.className = "a";
      if (grp.outs.length) {
        var text = grp.outs.map(function (o) { return o.text || ""; }).join(" ");
        a.textContent = text;
        if (text) {
          a.title = text;
        }
      } else if (waiting && gi === lastIdx) {
        // The ONE waiting marker: coral, only on the most recent exchange,
        // only while state.waiting is true. Never any other row.
        var wt = document.createElement("span");
        wt.className = "wt";
        wt.textContent = "waiting";
        a.appendChild(wt);
      } else {
        // Pending (no reply yet, not the one the host is flagging as
        // waiting-on) -- a plain dim marker, not coral.
        a.textContent = "…";
      }
      li.appendChild(a);

      // The fallback-lead note: plain dim text under the "in" side of an
      // exchange, never amber or coral.
      if (grp.in && grp.in.resolved === "fallback-lead") {
        var note = document.createElement("div");
        note.className = "nt";
        note.textContent = "heard, no seat matched";
        li.appendChild(note);
      }

      // Prefer the last reply's lineId (the most specific "this is what got
      // clicked"); fall back to the question's.
      var lineId = (lastOut && lastOut.lineId) || (grp.in && grp.in.lineId);
      if (lineId) {
        li.tabIndex = 0;
        li.setAttribute("role", "button");
        var pick = makePick(lineId, li);
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
  // It never guesses ahead of that -- no optimistic toggle on click. The
  // glyph is the puck-with-outgoing-waves (the agent's voice leaving the
  // puck); muted overlays a fixed error-token slash across it, so this never
  // again reads as "mute MY mic" the way the old mic glyph did.
  function renderMute(muted, connected) {
    muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
    muteBtn.setAttribute("aria-label", muted ? "Unmute agent voice" : "Mute agent voice");
    muteBtn.title = muted ? "Unmute agent voice" : "Mute agent voice";
    muteBtn.innerHTML = '<svg viewBox="0 0 24 24">' + WAVES_OUT +
      (muted ? WAVES_SLASH : "") + '</svg>';
    muteBtn.disabled = !connected;
    disconnectBtn.disabled = !connected;
  }

  // §9/§10.1 open-mic-after-replies toggle. A SETTING button, not a live
  // "mic is open now" indicator -- its pressed state comes only from
  // data.openMic (state JSON field, default true when absent), never a
  // guess ahead of the click. Glyph is the puck-with-incoming-waves; on
  // overlays a fixed ok-token check, off overlays the same error-token slash
  // renderMute uses.
  function renderOpenMic(openMic, connected) {
    openMicBtn.setAttribute("aria-pressed", openMic ? "true" : "false");
    openMicBtn.setAttribute("aria-label",
      openMic ? "Open mic after replies: on" : "Open mic after replies: off");
    openMicBtn.title = openMicBtn.getAttribute("aria-label");
    openMicBtn.innerHTML = '<svg viewBox="0 0 24 24">' + WAVES_IN +
      (openMic ? WAVES_CHECK : WAVES_SLASH) + '</svg>';
    openMicBtn.disabled = !connected;
  }

  // §10.1 security toggle idInRoom. Same rule as Mute: shows only the last
  // fetched value, never an optimistic guess ahead of the click.
  function renderId(idInRoom, connected) {
    idBtn.setAttribute("aria-pressed", idInRoom ? "true" : "false");
    idBtn.setAttribute("aria-label", idInRoom ? "Voice ID on" : "Voice ID off");
    idBtn.title = idInRoom ? "Voice ID on" : "Voice ID off";
    idBtn.innerHTML = '<svg viewBox="0 0 24 24">' + ID_BADGE +
      (idInRoom ? "" : ID_SLASH) + '</svg>';
    idBtn.disabled = !connected;
  }

  // The CONTENT height (the .fr root -- never the document): .exs's own
  // max-height/overflow-y already keeps the frame's natural layout height
  // at or under the host's 320 clamp with .fr-top always visible; this is
  // belt-and-suspenders on top of that CSS budget, and the 96 floor matches
  // .fr's own min-height.
  function measureHeight() {
    var h = Math.ceil(frameEl.scrollHeight);
    return Math.max(96, Math.min(320, h));
  }

  function postHeight() {
    var px = measureHeight();
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
    renderOpenMic(data.openMic !== false, connected);
    renderId(data.idInRoom !== false, connected);
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

  openMicBtn.addEventListener("click", function () {
    if (openMicBtn.disabled) {
      return;
    }
    var on = openMicBtn.getAttribute("aria-pressed") === "true";
    post({ panel: "voice", action: on ? "openmic-off" : "openmic-on" });
  });

  idBtn.addEventListener("click", function () {
    if (idBtn.disabled) {
      return;
    }
    var on = idBtn.getAttribute("aria-pressed") === "true";
    post({ panel: "voice", action: on ? "id-off" : "id-on" });
  });

  disconnectBtn.addEventListener("click", function () {
    if (disconnectBtn.disabled) {
      return;
    }
    post({ panel: "voice", action: "disconnect" });
  });

  window.addEventListener("resize", postHeight);

  // Catches content-driven size changes (font load reflow, a row's text
  // wrapping differently, etc.) that a window resize wouldn't fire for --
  // on top of the post-after-every-render call already in render().
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(function () {
      postHeight();
    }).observe(frameEl);
  }

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
          idInRoom: data.idInRoom,
          openMic: data.openMic,
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
