// voice_app / app.js -- V2 room voice panel (vk-1826).
//
// Runs inside a sandboxed (opaque-origin) iframe loaded through the
// briefing-table engine's panel proxy. No cookies, no storage, no
// allow-same-origin. Every dynamic value is rendered with textContent,
// never innerHTML. The only communication with the host is postMessage;
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

  var exchangesEl = document.getElementById("exchanges");
  var speakingRow = document.getElementById("speakingRow");
  var muteBtn = document.getElementById("muteBtn");
  var disconnectBtn = document.getElementById("disconnectBtn");

  var lastHeight = -1;

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

  function renderExchanges(list) {
    exchangesEl.textContent = "";
    var items = (list || []).slice(-3);
    if (items.length === 0) {
      var empty = document.createElement("li");
      empty.className = "empty-row";
      empty.textContent = "No exchanges yet.";
      exchangesEl.appendChild(empty);
      return;
    }
    items.forEach(function (ex) {
      var li = document.createElement("li");
      li.className = "exchange";

      var who = document.createElement("div");
      who.className = "who";
      who.textContent = fmtWho(ex.who);
      li.appendChild(who);

      var text = document.createElement("div");
      text.className = "text";
      text.textContent = ex.text || "";
      li.appendChild(text);

      if (ex.vid) {
        var vid = document.createElement("div");
        vid.className = "vid";
        vid.textContent = ex.vid;
        li.appendChild(vid);
      }

      if (ex.lineId) {
        li.tabIndex = 0;
        li.setAttribute("role", "button");
        li.addEventListener("click", function () {
          post({ panel: "line", lineId: ex.lineId });
        });
        li.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            post({ panel: "line", lineId: ex.lineId });
          }
        });
      }

      exchangesEl.appendChild(li);
    });
  }

  function renderSpeaking(speaking) {
    speakingRow.hidden = !speaking;
  }

  // The frame's Mute toggle shows the state the engine/state row last sent.
  // It never guesses ahead of that -- no optimistic toggle on click.
  function renderMute(muted, connected) {
    muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
    muteBtn.textContent = muted ? "◌ Muted" : "Mute";
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

  // No live "speaking" signal exists in the current room-voice state model
  // (room_voice.RoomVoiceState carries channelId/name/lead/muted/waiting/
  // exchanges, not an in-progress-speech flag) -- so this is always false
  // today. The render handshake still fires on every render, honestly
  // reporting "not speaking" rather than skipping the post.
  function render(data) {
    renderSpeaking(!!data.speaking);
    renderMute(!!data.muted, data.connected !== false);
    renderExchanges(data.exchanges);
    postHeight();
    post({ panel: "status", speaking: !!data.speaking });
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
  // one) and so the ~5s render handshake window is always met.
  render({ speaking: false, muted: false, connected: false, exchanges: [] });

  var seed = document.getElementById("initial-state");
  if (seed && seed.textContent) {
    try {
      render(JSON.parse(seed.textContent));
    } catch (e) {
      // malformed seed -- keep the empty render above.
    }
  }

  // Poll ./state (relative, so it survives the proxy's directory mapping)
  // every 3s. The query string (if any -- the fixture harness rides its
  // combination here) is forwarded so the same deterministic sample keeps
  // coming back on every poll.
  var STATE_URL = "state" + (location.search || "");

  function poll() {
    fetch(STATE_URL, { cache: "no-store" })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        render({
          speaking: false,
          muted: data.muted,
          connected: data.connected,
          exchanges: data.exchanges
        });
      })
      .catch(function () {
        // Unreachable is the HOST's state to draw, not the frame's; keep
        // showing the last good render rather than blanking anything.
      });
  }

  poll();
  setInterval(poll, 3000);
}());
