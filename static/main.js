// Voice-to-voice front-end.
// Modes: "ptt" (tap start / tap stop) and "auto" (tap start, auto-stop on pause).
// Live waveform during capture. Mode + voice persisted in localStorage.

const micBtn = document.getElementById("mic");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const player = document.getElementById("player");
const previewPlayer = document.getElementById("previewPlayer");

const voicePickBtn = document.getElementById("voicePick");
const pickName = document.getElementById("pickName");
const pickTags = document.getElementById("pickTags");
const modal = document.getElementById("modal");
const modalClose = document.getElementById("modalClose");
const vlist = document.getElementById("vlist");
const addVoiceBtn = document.getElementById("addVoiceBtn");
const customModal = document.getElementById("customModal");
const customClose = document.getElementById("customClose");
const customName = document.getElementById("customName");
const sampleRec = document.getElementById("sampleRec");
const sampleTimer = document.getElementById("sampleTimer");
const sampleUpload = document.getElementById("sampleUpload");
const sampleInfo = document.getElementById("sampleInfo");
const samplePlayer = document.getElementById("samplePlayer");
const createVoice = document.getElementById("createVoice");
const customStatus = document.getElementById("customStatus");
const modeSeg = document.getElementById("modeSeg");
const personaSelect = document.getElementById("personaSelect");
const cancelBtn = document.getElementById("cancel");
const stopBtn = document.getElementById("stop");
const waveCanvas = document.getElementById("wave");
const waveCtx = waveCanvas.getContext("2d");
const transcriptEl = document.getElementById("transcript");
const emptyHint = document.getElementById("emptyHint");
const draftBox = document.getElementById("draftBox");
const draftText = document.getElementById("draftText");
const sendBtn = document.getElementById("send");
const clearDraftBtn = document.getElementById("clearDraft");

let pendingSegments = []; // Compose-mode queued transcript segments

// ---- voice tuning (persisted) ----
const LS_TUNING = "v2v_tuning";
const TUNING_DEFAULTS = { speed: 1.0, guidance: 2.0, temperature: 0.0, steps: 16 };
let tuning = Object.assign({}, TUNING_DEFAULTS);
try { Object.assign(tuning, JSON.parse(localStorage.getItem(LS_TUNING) || "{}")); } catch (_) {}
// Clamp saved values into the safe slider ranges (snaps back old extreme settings).
const clamp = (v, lo, hi, d) => Math.min(hi, Math.max(lo, isFinite(+v) ? +v : d));
tuning.speed = clamp(tuning.speed, 0.8, 1.3, 1.0);
tuning.guidance = clamp(tuning.guidance, 1.5, 3.5, 2.0);
tuning.temperature = clamp(tuning.temperature, 0.0, 0.6, 0.0);
tuning.steps = clamp(tuning.steps, 16, 48, 16);
localStorage.setItem(LS_TUNING, JSON.stringify(tuning));

const tSpeed = document.getElementById("t_speed");
const tGuidance = document.getElementById("t_guidance");
const tTemperature = document.getElementById("t_temperature");
const tSteps = document.getElementById("t_steps");
const speedVal = document.getElementById("speedVal");
const guidanceVal = document.getElementById("guidanceVal");
const temperatureVal = document.getElementById("temperatureVal");
const stepsVal = document.getElementById("stepsVal");

function applyTuningToUI() {
  tSpeed.value = tuning.speed; tGuidance.value = tuning.guidance;
  tTemperature.value = tuning.temperature; tSteps.value = tuning.steps;
  speedVal.textContent = Number(tuning.speed).toFixed(2);
  guidanceVal.textContent = Number(tuning.guidance).toFixed(1);
  temperatureVal.textContent = Number(tuning.temperature).toFixed(2);
  stepsVal.textContent = tuning.steps;
}
function readTuningFromUI() {
  tuning = {
    speed: parseFloat(tSpeed.value), guidance: parseFloat(tGuidance.value),
    temperature: parseFloat(tTemperature.value), steps: parseInt(tSteps.value, 10),
  };
  localStorage.setItem(LS_TUNING, JSON.stringify(tuning));
  applyTuningToUI();
}
[tSpeed, tGuidance, tTemperature, tSteps].forEach((el) =>
  el.addEventListener("input", readTuningFromUI));
document.getElementById("resetTuning").addEventListener("click", () => {
  tuning = Object.assign({}, TUNING_DEFAULTS);
  localStorage.setItem(LS_TUNING, JSON.stringify(tuning));
  applyTuningToUI();
  log("tuning reset to defaults");
});
const LS_TESTTEXT = "v2v_testtext";
const testText = document.getElementById("testText");
const testBtn = document.getElementById("testBtn");
const savedTest = localStorage.getItem(LS_TESTTEXT);
if (savedTest !== null) testText.value = savedTest;
testText.addEventListener("input", () => localStorage.setItem(LS_TESTTEXT, testText.value));

async function testPhrase() {
  const txt = (testText.value || "").trim();
  if (!txt) return;
  unlockAudio();
  previewPlayer.pause();
  testBtn.disabled = true;
  const orig = testBtn.innerHTML;
  testBtn.innerHTML = "&#8987; Generating…";
  try {
    const url = "/api/preview?voice=" + encodeURIComponent(currentVoice) +
                "&text=" + encodeURIComponent(txt) + tuningQuery();
    const r = await fetch(url);
    if (!r.ok) { log("test phrase failed " + r.status); return; }
    const blob = await r.blob();
    const burl = URL.createObjectURL(blob);
    previewPlayer.src = burl;
    previewPlayer.onended = () => URL.revokeObjectURL(burl);
    await previewPlayer.play();
    log(`test phrase (${currentVoice}, spd ${tuning.speed}/exp ${tuning.guidance}/liv ${tuning.temperature}/q ${tuning.steps})`);
  } catch (e) {
    log("test phrase error: " + e);
  } finally {
    testBtn.disabled = false;
    testBtn.innerHTML = orig;
  }
}
testBtn.addEventListener("click", testPhrase);

function appendTuning(fd) {
  fd.append("speed", tuning.speed);
  fd.append("guidance", tuning.guidance);
  fd.append("temperature", tuning.temperature);
  fd.append("steps", tuning.steps);
}
function tuningQuery() {
  return `&speed=${tuning.speed}&guidance=${tuning.guidance}` +
         `&temperature=${tuning.temperature}&steps=${tuning.steps}`;
}
applyTuningToUI();

// ---- persisted settings ----
const LS_VOICE = "v2v_voice";
const LS_MODE = "v2v_mode";
const LS_PERSONA = "v2v_persona";
let currentVoice = localStorage.getItem(LS_VOICE) || "f_us";
let mode = localStorage.getItem(LS_MODE) || "ptt"; // "ptt" | "auto"
let currentPersona = localStorage.getItem(LS_PERSONA) || "friendly";

// ---- recording state ----
let mediaRecorder = null;
let chunks = [];
let recording = false;
let canceled = false;
let busy = false;
let mimeType = "";
let stream = null;
let voices = [];

// ---- audio analysis (waveform + VAD) ----
let audioCtx = null;
let analyser = null;
let analyserData = null;
let rafId = 0;
let recStartMs = 0;
let speechDetected = false;
let lastVoiceMs = 0;

// VAD tuning (RMS on 0..1 scale from byte time-domain data)
const VOICE_RMS = 0.030;       // above this = speech present
const SILENCE_HANG_MS = 1200;  // auto-stop after this much trailing silence
const MIN_SPEECH_MS = 350;     // require at least this much before auto-stop
const MAX_REC_MS = 20000;      // hard cap
const MIN_HOLD_MS = 350;       // ptt: shorter than this = "too short"

function log(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent += `[${t}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
  console.log(msg);
  sendToServer("log", msg);
}
function setStatus(s) { statusEl.textContent = s; }

// ---------- Transcript ----------
function addTurn(role, text) {
  if (emptyHint) emptyHint.style.display = "none";
  const msg = document.createElement("div");
  msg.className = "msg " + (role === "user" ? "user" : "bot");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : (voiceById(currentVoice)?.label || "Assistant");
  const body = document.createElement("div");
  body.textContent = text;
  msg.appendChild(who);
  msg.appendChild(body);
  transcriptEl.appendChild(msg);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}
function clearTranscript() {
  transcriptEl.querySelectorAll(".msg").forEach((m) => m.remove());
  if (emptyHint) emptyHint.style.display = "";
}

function sendToServer(level, msg) {
  try {
    fetch("/api/clientlog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level, msg: String(msg) }),
      keepalive: true,
    }).catch(() => {});
  } catch (_) {}
}
window.addEventListener("error", (e) =>
  sendToServer("error", `window.error: ${e.message} @ ${e.filename}:${e.lineno}`));
window.addEventListener("unhandledrejection", (e) => {
  const r = e.reason;
  sendToServer("error", "unhandledrejection: " + (r && r.message ? r.message : r));
});

// iOS only lets <audio>.play() run inside a user gesture. Unlock during mic tap.
const SILENT_WAV =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=";
let audioUnlocked = false;
function unlockAudio() {
  if (audioUnlocked) return;
  [player, previewPlayer].forEach((el) => {
    el.src = SILENT_WAV;
    const p = el.play();
    if (p && p.then) p.then(() => el.pause()).catch(() => {});
  });
  audioUnlocked = true;
  log("audio unlocked");
}

function pickMime() {
  const candidates = ["audio/mp4", "audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

// ---------- Mode toggle ----------
function idleMicLabel() {
  if (mode === "compose") return pendingSegments.length ? "Tap to add more" : "Tap to record";
  if (mode === "auto") return "Tap & speak";
  return "Tap to talk";
}
function applyMode() {
  modeSeg.querySelectorAll("button").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  draftBox.classList.toggle("hidden", mode !== "compose");
  if (mode === "compose") renderDraft();
  if (!recording) micBtn.textContent = idleMicLabel();
}

function renderDraft() {
  if (pendingSegments.length) {
    draftText.textContent = pendingSegments.join("  ");
    draftText.classList.remove("draft-empty");
  } else {
    draftText.textContent = "Record a segment — it gets added here. Tap Send when done.";
    draftText.classList.add("draft-empty");
  }
  sendBtn.disabled = pendingSegments.length === 0 || busy;
}
modeSeg.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || recording) return;
  mode = b.dataset.mode;
  localStorage.setItem(LS_MODE, mode);
  applyMode();
  log("mode: " + mode);
});

// ---------- Voices + modal ----------
async function loadVoices() {
  const r = await fetch("/api/voices");
  const data = await r.json();
  voices = data.voices;
  if (!voices.find((v) => v.id === currentVoice)) currentVoice = data.default;
  renderModal();
  updatePickButton();
  log(`loaded ${voices.length} voices`);
}
async function loadPersonalities() {
  try {
    const r = await fetch("/api/personalities");
    const data = await r.json();
    if (!data.personalities.find((p) => p.id === currentPersona)) currentPersona = data.default;
    personaSelect.innerHTML = "";
    const groups = {};
    const order = [];
    data.personalities.forEach((p) => {
      const g = p.group || "Other";
      if (!groups[g]) { groups[g] = []; order.push(g); }
      groups[g].push(p);
    });
    order.forEach((g) => {
      const og = document.createElement("optgroup");
      og.label = g;
      groups[g].forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.label;
        if (p.id === currentPersona) opt.selected = true;
        og.appendChild(opt);
      });
      personaSelect.appendChild(og);
    });
    log(`loaded ${data.personalities.length} personalities (current: ${currentPersona})`);
  } catch (e) {
    log("failed to load personalities: " + e);
  }
}
personaSelect.addEventListener("change", () => {
  currentPersona = personaSelect.value;
  localStorage.setItem(LS_PERSONA, currentPersona);
  log("personality: " + currentPersona + " (applies on next reply)");
});

function voiceById(id) { return voices.find((v) => v.id === id); }
function updatePickButton() {
  const v = voiceById(currentVoice);
  if (!v) return;
  pickName.textContent = v.label;
  pickTags.textContent = v.tags || "";
}
function renderModal() {
  vlist.innerHTML = "";
  voices.forEach((v) => {
    const item = document.createElement("div");
    item.className = "vitem" + (v.id === currentVoice ? " selected" : "");
    item.dataset.id = v.id;

    const play = document.createElement("button");
    play.className = "play";
    play.innerHTML = "&#9654;";
    play.addEventListener("click", (e) => { e.stopPropagation(); previewVoice(v.id, play); });

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<div class="name">${v.label}</div><div class="tags">${v.tags || ""}</div>`;

    const check = document.createElement("div");
    check.className = "check";
    check.textContent = "✓";

    item.appendChild(play);
    item.appendChild(meta);
    item.appendChild(check);
    if (v.custom) {
      const trash = document.createElement("button");
      trash.className = "trash";
      trash.innerHTML = "&#128465;"; // 🗑
      trash.title = "Delete custom voice";
      trash.addEventListener("click", (e) => { e.stopPropagation(); deleteCustomVoice(v); });
      item.appendChild(trash);
    }
    item.addEventListener("click", () => selectVoice(v.id));
    vlist.appendChild(item);
  });
}
function selectVoice(id) {
  currentVoice = id;
  localStorage.setItem(LS_VOICE, id);
  vlist.querySelectorAll(".vitem").forEach((el) =>
    el.classList.toggle("selected", el.dataset.id === id));
  updatePickButton();
  log("voice set: " + (voiceById(id)?.label || id));
  setTimeout(closeModal, 180);
}
let previewToken = 0;
let previewBtnActive = null;
async function previewVoice(id, btn) {
  unlockAudio();
  previewPlayer.pause();
  const myToken = ++previewToken; // invalidates any in-flight preview
  if (previewBtnActive) { previewBtnActive.classList.remove("loading"); previewBtnActive.innerHTML = "&#9654;"; }
  previewBtnActive = btn;
  btn.classList.add("loading");
  btn.innerHTML = "&#8987;";
  try {
    const r = await fetch("/api/preview?voice=" + encodeURIComponent(id) + tuningQuery());
    if (myToken !== previewToken) return; // a newer preview superseded this one
    if (!r.ok) { log("preview failed " + r.status); return; }
    const blob = await r.blob();
    if (myToken !== previewToken) return;
    const url = URL.createObjectURL(blob);
    previewPlayer.src = url;
    previewPlayer.onended = () => URL.revokeObjectURL(url);
    await previewPlayer.play();
  } catch (e) {
    if (myToken === previewToken) log("preview error: " + e);
  } finally {
    if (myToken === previewToken) { btn.classList.remove("loading"); btn.innerHTML = "&#9654;"; }
  }
}
function openModal() { modal.classList.add("open"); }
function closeModal() { modal.classList.remove("open"); previewPlayer.pause(); }
voicePickBtn.addEventListener("click", openModal);
modalClose.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });

async function deleteCustomVoice(v) {
  if (!confirm(`Delete custom voice "${v.label}"?`)) return;
  const fd = new FormData();
  fd.append("voice", v.id);
  await fetch("/api/voices/custom_delete", { method: "POST", body: fd });
  if (currentVoice === v.id) { currentVoice = "f_us"; localStorage.setItem(LS_VOICE, currentVoice); }
  await loadVoices();
  log("deleted custom voice " + v.id);
}

// ---------- Custom voice creator ----------
let sampleBlob = null;
let sampleType = "";
let sampleRecorder = null;
let sampleChunks = [];
let sampleStream = null;
let sampleRecording = false;
let sampleTimerId = 0;
let sampleStartMs = 0;
const SAMPLE_MAX_MS = 30000; // safety cap; user normally taps stop when done reading

// Reading prompts shown above the recorder so the user has something to read.
const READING_PROMPTS = [
  {
    label: "Stella (~15s)",
    text: "Please call Stella. Ask her to bring these things with her from the store: " +
          "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack " +
          "for her brother Bob. We also need a small plastic snake and a big toy frog for the " +
          "kids. She can scoop these things into three red bags, and we will go meet her " +
          "Wednesday at the train station.",
  },
  {
    label: "Rainbow",
    text: "When the sunlight strikes raindrops in the air, they act as a prism and form a " +
          "rainbow. The rainbow is a division of white light into many beautiful colors. These " +
          "take the shape of a long round arch, with its path high above, and its two ends " +
          "apparently beyond the horizon. There is, according to legend, a boiling pot of gold " +
          "at one end. People look, but no one ever finds it.",
  },
  {
    label: "Harvard",
    text: [
      "1. The birch canoe slid on the smooth planks.",
      "2. Glue the sheet to the dark blue background.",
      "3. It's easy to tell the depth of a well.",
      "4. These days a chicken leg is a rare dish.",
      "5. Rice is often served in round bowls.",
      "6. The juice of lemons makes fine punch.",
      "7. The box was thrown beside the parked truck.",
      "8. The hogs were fed chopped corn and garbage.",
      "9. Four hours of steady work faced us.",
      "10. A large size in stockings is hard to sell.",
    ].join("\n"),
  },
];
const readTabs = document.getElementById("readTabs");
const readText = document.getElementById("readText");
function renderReadTabs() {
  readTabs.innerHTML = "";
  READING_PROMPTS.forEach((p, i) => {
    const b = document.createElement("button");
    b.textContent = p.label;
    if (i === 0) b.classList.add("active");
    b.addEventListener("click", () => selectReadTab(i));
    readTabs.appendChild(b);
  });
  selectReadTab(0);
}
function selectReadTab(i) {
  readText.textContent = READING_PROMPTS[i].text;
  readTabs.querySelectorAll("button").forEach((b, j) => b.classList.toggle("active", j === i));
}

function openCustomModal() {
  resetCustomForm();
  renderReadTabs();
  customModal.classList.add("open");
}
function closeCustomModal() {
  stopSampleRecording();
  customModal.classList.remove("open");
  samplePlayer.pause();
}
function resetCustomForm() {
  sampleBlob = null; sampleType = "";
  customName.value = "";
  sampleTimer.textContent = "0.0s";
  sampleInfo.textContent = "No sample yet. Capture 3–10s of clear speech.";
  samplePlayer.classList.add("hidden");
  customStatus.textContent = "";
  createVoice.disabled = true;
}
addVoiceBtn.addEventListener("click", openCustomModal);
customClose.addEventListener("click", closeCustomModal);
customModal.addEventListener("click", (e) => { if (e.target === customModal) closeCustomModal(); });

function showSample(blob, type, sourceLabel) {
  sampleBlob = blob; sampleType = type;
  const url = URL.createObjectURL(blob);
  samplePlayer.src = url;
  samplePlayer.classList.remove("hidden");
  sampleInfo.textContent = `${sourceLabel} — ${(blob.size / 1024).toFixed(0)} KB. Play to check, then Create.`;
  createVoice.disabled = false;
}

function tickTimer() {
  const s = (performance.now() - sampleStartMs) / 1000;
  sampleTimer.textContent = s.toFixed(1) + "s";
  if (s * 1000 >= SAMPLE_MAX_MS) stopSampleRecording();
}

async function startSampleRecording() {
  unlockAudio();
  try {
    sampleStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    customStatus.textContent = "Mic blocked: " + e.name;
    return;
  }
  const mt = pickMime();
  sampleChunks = [];
  sampleRecorder = new MediaRecorder(sampleStream, mt ? { mimeType: mt } : undefined);
  sampleRecorder.ondataavailable = (e) => { if (e.data.size > 0) sampleChunks.push(e.data); };
  sampleRecorder.onstop = () => {
    clearInterval(sampleTimerId);
    sampleRecording = false;
    sampleRec.classList.remove("recording");
    sampleRec.innerHTML = "&#9679; Record sample";
    if (sampleStream) { sampleStream.getTracks().forEach((t) => t.stop()); sampleStream = null; }
    const type = sampleRecorder.mimeType || mt || "audio/mp4";
    const blob = new Blob(sampleChunks, { type });
    if (blob.size > 800) showSample(blob, type, "Recorded");
  };
  sampleRecorder.start(200);
  sampleRecording = true;
  sampleStartMs = performance.now();
  sampleRec.classList.add("recording");
  sampleRec.innerHTML = "&#9632; Stop recording";
  sampleTimer.textContent = "0.0s";
  sampleTimerId = setInterval(tickTimer, 100);
}
function stopSampleRecording() {
  if (sampleRecorder && sampleRecording) { try { sampleRecorder.stop(); } catch (_) {} }
}
sampleRec.addEventListener("click", () => {
  if (!sampleRecording) startSampleRecording(); else stopSampleRecording();
});

sampleUpload.addEventListener("change", () => {
  const f = sampleUpload.files && sampleUpload.files[0];
  if (f) showSample(f, f.type || "audio/*", "Uploaded " + f.name);
});

createVoice.addEventListener("click", async () => {
  if (!sampleBlob) return;
  const name = (customName.value || "").trim() || "My Voice";
  createVoice.disabled = true;
  customStatus.textContent = "Cloning voice… (transcribing + encoding)";
  const ext = sampleType.includes("mp4") ? "mp4" : sampleType.includes("webm") ? "webm"
            : sampleType.includes("wav") ? "wav" : sampleType.includes("mpeg") ? "mp3" : "audio";
  const fd = new FormData();
  fd.append("audio", sampleBlob, "sample." + ext);
  fd.append("name", name);
  try {
    const r = await fetch("/api/voices/custom", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (_) {}
      customStatus.textContent = "Failed: " + detail;
      createVoice.disabled = false;
      return;
    }
    const data = await r.json();
    log(`custom voice created: ${data.voice.label} (heard: "${data.ref_text}")`);
    currentVoice = data.voice.id;
    localStorage.setItem(LS_VOICE, currentVoice);
    await loadVoices();
    closeCustomModal();
  } catch (e) {
    customStatus.textContent = "Network error";
    log("create voice failed: " + e);
    createVoice.disabled = false;
  }
});

// ---------- Waveform + VAD loop ----------
function setupAnalyser() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AC();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyserData = new Uint8Array(analyser.fftSize);
  src.connect(analyser);
}
function computeRms() {
  analyser.getByteTimeDomainData(analyserData);
  let sum = 0;
  for (let i = 0; i < analyserData.length; i++) {
    const v = (analyserData[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / analyserData.length);
}
function drawWave(rms) {
  const w = waveCanvas.width, h = waveCanvas.height;
  waveCtx.clearRect(0, 0, w, h);
  const active = rms > VOICE_RMS;
  waveCtx.lineWidth = 3;
  waveCtx.strokeStyle = active ? "#3fb950" : "#30363d";
  waveCtx.beginPath();
  const slice = w / analyserData.length;
  for (let i = 0; i < analyserData.length; i++) {
    const y = (analyserData[i] / 255) * h;
    const x = i * slice;
    if (i === 0) waveCtx.moveTo(x, y); else waveCtx.lineTo(x, y);
  }
  waveCtx.stroke();
  // level bar at bottom
  waveCtx.fillStyle = active ? "#3fb950" : "#58a6ff";
  waveCtx.fillRect(0, h - 6, Math.min(w, rms * 6 * w), 6);
}
function clearWave() {
  waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
}
function monitor() {
  if (!recording) return;
  const rms = computeRms();
  drawWave(rms);
  const now = performance.now();
  if (rms > VOICE_RMS) { speechDetected = true; lastVoiceMs = now; }
  if (mode === "auto" || mode === "compose") {
    if (speechDetected && (now - recStartMs) > MIN_SPEECH_MS &&
        (now - lastVoiceMs) > SILENCE_HANG_MS) {
      log("auto-stop: pause detected");
      stopRecording();
      return;
    }
    if (now - recStartMs > MAX_REC_MS) { log("auto-stop: max length"); stopRecording(); return; }
  }
  rafId = requestAnimationFrame(monitor);
}

// ---------- Mic / recording ----------
function releaseStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

// iOS Safari records 0 bytes on the 2nd+ use of a reused stream/recorder.
// So we acquire a FRESH stream for every recording.
async function acquireStream() {
  releaseStream();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mimeType = pickMime();
    setupAnalyser();
    log("mic ready, mime=" + (mimeType || "(default)"));
    return true;
  } catch (e) {
    setStatus("Mic blocked - needs HTTPS on iOS");
    log("getUserMedia failed: " + e.name + " " + e.message);
    return false;
  }
}

async function startRecording() {
  if (!(await acquireStream())) return;
  chunks = [];
  const opts = mimeType ? { mimeType } : undefined;
  mediaRecorder = new MediaRecorder(stream, opts);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start(200); // timeslice: flush chunks periodically (helps iOS short clips)
  recording = true;
  canceled = false;
  speechDetected = false;
  recStartMs = performance.now();
  lastVoiceMs = recStartMs;
  micBtn.classList.add("recording");
  micBtn.textContent = mode === "ptt" ? "Tap to send" : "Listening…";
  cancelBtn.classList.remove("hidden");
  setStatus(mode === "ptt" ? "Listening…" : "Listening — pause when done");
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  rafId = requestAnimationFrame(monitor);
}

function stopRecording() {
  if (mediaRecorder && recording) {
    recording = false;
    cancelAnimationFrame(rafId);
    try { mediaRecorder.stop(); } catch (_) {}
  }
}

function cancelRecording() {
  if (!recording) return;
  canceled = true;
  stopRecording();
}

async function onRecordingStop() {
  micBtn.classList.remove("recording");
  cancelBtn.classList.add("hidden");
  clearWave();
  const heldMs = performance.now() - recStartMs;
  const type = mediaRecorder.mimeType || mimeType || "audio/mp4";
  const blob = new Blob(chunks, { type });
  releaseStream(); // free the mic so the next recording gets a fresh stream
  if (canceled) {
    log("recording canceled");
    setStatus("Canceled");
    micBtn.textContent = idleMicLabel();
    return;
  }
  log(`recorded ${(blob.size / 1024).toFixed(1)} KB, held ${Math.round(heldMs)}ms (${type})`);
  if (heldMs < MIN_HOLD_MS || blob.size < 600) {
    setStatus("Too short - hold a bit longer");
    micBtn.textContent = idleMicLabel();
    return;
  }
  if (mode === "compose") {
    await sttSegment(blob, type);
  } else {
    await sendAudio(blob, type);
  }
}

// Compose mode: transcribe one segment and queue it (don't send to the agent yet).
async function sttSegment(blob, type) {
  busy = true;
  micBtn.classList.add("busy");
  micBtn.textContent = "…";
  setStatus("Transcribing…");
  const ext = type.includes("mp4") ? "mp4" : type.includes("webm") ? "webm" : "ogg";
  const fd = new FormData();
  fd.append("audio", blob, "speech." + ext);
  try {
    const r = await fetch("/api/stt", { method: "POST", body: fd });
    if (!r.ok) { setStatus("Transcribe failed"); log("stt " + r.status); return; }
    const data = await r.json();
    const seg = (data.text || "").trim();
    if (seg) {
      pendingSegments.push(seg);
      log("segment queued: " + seg);
      setStatus("Added. Tap mic for more, or Send.");
    } else {
      setStatus("Didn't catch that - try again");
    }
  } catch (e) {
    setStatus("Network error");
    log("stt failed: " + e);
  } finally {
    busy = false;
    micBtn.classList.remove("busy");
    micBtn.textContent = idleMicLabel();
    renderDraft();
  }
}

// Compose mode: send the full appended text to the agent.
async function sendComposed() {
  if (!pendingSegments.length || busy) return;
  const fullText = pendingSegments.join(" ");
  busy = true;
  sendBtn.disabled = true;
  setStatus("Thinking…");
  const fd = new FormData();
  fd.append("text", fullText);
  fd.append("voice", currentVoice);
  fd.append("personality", currentPersona);
  appendTuning(fd);
  try {
    const r = await fetch("/api/send_text", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (_) {}
      setStatus("Error: " + detail);
      log(`send_text ${r.status}: ${detail}`);
      return;
    }
    const reply = decodeURIComponent(r.headers.get("X-Reply") || "");
    const timing = r.headers.get("X-Timing") || "";
    log(`you (composed): ${fullText}`);
    log(`bot: ${reply}`);
    log(`timing ${timing}`);
    addTurn("user", fullText);
    if (reply) addTurn("bot", reply);
    pendingSegments = [];
    const audioBlob = await r.blob();
    const url = URL.createObjectURL(audioBlob);
    player.src = url;
    setStatus("Speaking…");
    stopBtn.classList.remove("hidden");
    player.onended = () => {
      stopBtn.classList.add("hidden");
      URL.revokeObjectURL(url);
      setStatus(idleMicLabel());
    };
    try { await player.play(); }
    catch (pe) { stopBtn.classList.add("hidden"); setStatus("Tap mic area to hear reply"); log(`playback blocked: ${pe.name}`); }
  } catch (e) {
    setStatus("Network error");
    log("send_text failed: " + e);
  } finally {
    busy = false;
    micBtn.textContent = idleMicLabel();
    renderDraft();
  }
}

async function sendAudio(blob, type) {
  busy = true;
  micBtn.classList.add("busy");
  micBtn.textContent = "…";
  setStatus("Thinking…");
  const ext = type.includes("mp4") ? "mp4" : type.includes("webm") ? "webm" : "ogg";
  const fd = new FormData();
  fd.append("audio", blob, "speech." + ext);
  fd.append("voice", currentVoice);
  fd.append("personality", currentPersona);
  appendTuning(fd);

  try {
    const r = await fetch("/api/converse", { method: "POST", body: fd });
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (_) {}
      setStatus(r.status === 422 ? "Didn't catch that - try again" : "Error: " + detail);
      log(`converse ${r.status}: ${detail}`);
      return;
    }
    const transcript = decodeURIComponent(r.headers.get("X-Transcript") || "");
    const reply = decodeURIComponent(r.headers.get("X-Reply") || "");
    const timing = r.headers.get("X-Timing") || "";
    log(`you: ${transcript}`);
    log(`bot: ${reply}`);
    log(`timing ${timing}`);
    if (transcript) addTurn("user", transcript);
    if (reply) addTurn("bot", reply);
    const audioBlob = await r.blob();
    const url = URL.createObjectURL(audioBlob);
    player.src = url;
    setStatus("Speaking…");
    stopBtn.classList.remove("hidden");
    const finishPlayback = () => {
      stopBtn.classList.add("hidden");
      URL.revokeObjectURL(url);
      setStatus(mode === "auto" ? "Tap & speak" : "Tap to talk");
    };
    player.onended = finishPlayback;
    try {
      await player.play();
    } catch (pe) {
      stopBtn.classList.add("hidden");
      setStatus("Tap mic to hear reply");
      log(`playback blocked: ${pe.name} ${pe.message}`);
    }
  } catch (e) {
    setStatus("Network error");
    log("fetch failed: " + e);
  } finally {
    busy = false;
    micBtn.classList.remove("busy");
    if (!recording) micBtn.textContent = mode === "auto" ? "Tap & speak" : "Tap to talk";
  }
}

micBtn.addEventListener("click", () => {
  if (busy) return;
  unlockAudio();
  if (!recording) { startRecording(); } else { stopRecording(); }
});

cancelBtn.addEventListener("click", () => { cancelRecording(); });

function stopPlayback() {
  try { player.pause(); player.currentTime = 0; } catch (_) {}
  stopBtn.classList.add("hidden");
  setStatus(mode === "auto" ? "Tap & speak" : "Tap to talk");
  log("playback stopped");
}
stopBtn.addEventListener("click", stopPlayback);

sendBtn.addEventListener("click", sendComposed);
clearDraftBtn.addEventListener("click", () => {
  pendingSegments = [];
  renderDraft();
  micBtn.textContent = idleMicLabel();
  log("draft cleared");
});

document.getElementById("reset").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  clearTranscript();
  log("conversation reset");
});
async function newSession() {
  await fetch("/api/reset", { method: "POST" });
  clearTranscript();
  pendingSegments = [];
  if (mode === "compose") renderDraft();
  stopPlayback();
  setStatus(idleMicLabel());
  micBtn.textContent = idleMicLabel();
  log("NEW SESSION - history, transcript, and draft cleared");
}
document.getElementById("newSession").addEventListener("click", newSession);
document.getElementById("health").addEventListener("click", async () => {
  const r = await fetch("/api/health");
  log("health: " + JSON.stringify(await r.json()));
});

// ---------- init ----------
applyMode();
loadPersonalities();
loadVoices().then(() => setStatus(mode === "auto" ? "Tap & speak" : "Tap to talk")).catch((e) => {
  setStatus("Failed to load");
  log("init error: " + e);
});
