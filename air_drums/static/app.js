const video = document.querySelector('#video');
const canvas = document.querySelector('#overlay');
const ctx = canvas.getContext('2d');
const status = document.querySelector('#status');
const metrics = document.querySelector('#metrics');
const toggle = document.querySelector('#toggle');
const empty = document.querySelector('#empty');
const delegateSelect = document.querySelector('#delegate');
const responseSlider = document.querySelector('#responsiveness');
const responseValue = document.querySelector('#responseValue');
const showRaw = document.querySelector('#showRaw');
const delegateNote = document.querySelector('#delegateNote');

const POSE_CONNECTIONS = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];
const POSE_COLORS = { shoulder: '#61aaff', elbow: '#43e0b5', wrist: '#f5cb5c' };

let media = null;
let worker = null;
let workerReady = false;
let busy = false;
let generation = 0;
let inFlightGeneration = -1;
let callbackId = null;
let callbackKind = null;
let currentResult = null;
let currentInference = null;
let cameraDescription = '—';
let sentAt = 0;
let resultCount = 0;
let resultWindowStarted = performance.now();
let processedFps = 0;
let displayDelayMs = 0;
const filters = new Map();

function setStatus(text, state = 'idle') {
  status.textContent = text;
  status.dataset.state = state;
}

function setCanvasSize() {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function lowPass(previous, value, weight) {
  return previous + weight * (value - previous);
}

function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuroPoint {
  constructor(point, timestampMs, tuning) {
    this.raw = { x: point.x, y: point.y };
    this.filtered = { x: point.x, y: point.y };
    this.derivative = { x: 0, y: 0 };
    this.timestampMs = timestampMs;
    this.tuning = tuning;
    this.occluded = false;
  }

  update(point, timestampMs) {
    const dt = Math.max(0.001, Math.min(0.1, (timestampMs - this.timestampMs) / 1000));
    this.timestampMs = timestampMs;
    // During partial occlusion, hold the last stable point instead of letting
    // an uncertain model estimate drag the rest of the visible arm.
    if ((point.visibility ?? 1) < 0.35) {
      this.raw = { x: point.x, y: point.y };
      this.derivative = { x: 0, y: 0 };
      this.occluded = true;
      return { ...point, x: this.filtered.x, y: this.filtered.y };
    }
    if (this.occluded) {
      this.raw = { x: point.x, y: point.y };
      this.derivative = { x: 0, y: 0 };
      this.filtered.x = lowPass(this.filtered.x, point.x, alpha(this.tuning.minCutoff, dt));
      this.filtered.y = lowPass(this.filtered.y, point.y, alpha(this.tuning.minCutoff, dt));
      this.occluded = false;
      return { ...point, x: this.filtered.x, y: this.filtered.y };
    }
    const derivativeAlpha = alpha(1, dt);
    const response = Number(responseSlider.value) / 100;
    const beta = this.tuning.beta(response) * (point.visibility ?? 1);
    for (const axis of ['x', 'y']) {
      const derivative = (point[axis] - this.raw[axis]) / dt;
      this.derivative[axis] = lowPass(this.derivative[axis], derivative, derivativeAlpha);
      const cutoff = this.tuning.minCutoff + beta * Math.abs(this.derivative[axis]);
      this.filtered[axis] = lowPass(this.filtered[axis], point[axis], alpha(cutoff, dt));
      this.raw[axis] = point[axis];
    }
    return { ...point, x: this.filtered.x, y: this.filtered.y };
  }
}

function smooth(key, point, timestampMs) {
  if (!point) return null;
  const isWrist = key.endsWith('Wrist');
  const tuning = isWrist
    ? { minCutoff: 0.7, beta: response => 0.005 + response * 0.22 }
    : { minCutoff: 0.45, beta: response => 0.02 + response * 0.07 };
  let filter = filters.get(key);
  if (!filter || timestampMs - filter.timestampMs > 250) {
    filter = new OneEuroPoint(point, timestampMs, tuning);
    filters.set(key, filter);
    return point;
  }
  return filter.update(point, timestampMs);
}

// Filter each new inference exactly once. Rendering runs at display refresh rate,
// which may be faster than inference; filtering there would fake extra samples.
function filterPose(pose, timestampMs) {
  if (!pose) return null;
  return Object.fromEntries(Object.entries(pose).map(([key, point]) => [
    key,
    smooth(key, point, timestampMs),
  ]));
}

function drawDot(point, label, color, width, height, radius = 8) {
  if (!point || point.visibility < 0.18) return;
  const x = point.x * width;
  const y = point.y * height;
  ctx.beginPath();
  ctx.arc(x, y, Math.max(radius, width * 0.006), 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = Math.max(2, width * 0.0025);
  ctx.strokeStyle = '#101313';
  ctx.stroke();
  if (label) {
    ctx.font = `700 ${Math.max(12, width * 0.015)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.lineWidth = Math.max(3, width * 0.004);
    ctx.strokeStyle = '#101313';
    ctx.strokeText(label, x, y - Math.max(12, width * 0.015));
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x, y - Math.max(12, width * 0.015));
  }
}

function drawConnections(points, connections, width, height) {
  ctx.beginPath();
  for (const [from, to] of connections) {
    const start = points[from];
    const end = points[to];
    if (!start || !end || start.visibility < 0.18 || end.visibility < 0.18) continue;
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
  }
  ctx.strokeStyle = '#e2bd5d99';
  ctx.lineWidth = Math.max(4, width * 0.0026);
  ctx.lineCap = 'round';
  ctx.stroke();
}

function drawPose(rawPose, filteredPose, width, height) {
  if (!filteredPose) return;
  const points = new Array(17);
  points[11] = filteredPose.leftShoulder;
  points[12] = filteredPose.rightShoulder;
  points[13] = filteredPose.leftElbow;
  points[14] = filteredPose.rightElbow;
  points[15] = filteredPose.leftWrist;
  points[16] = filteredPose.rightWrist;
  drawConnections(points, POSE_CONNECTIONS, width, height);
  const joints = [
    ['leftShoulder', 'L shoulder', POSE_COLORS.shoulder],
    ['rightShoulder', 'R shoulder', POSE_COLORS.shoulder],
    ['leftElbow', 'L elbow', POSE_COLORS.elbow],
    ['rightElbow', 'R elbow', POSE_COLORS.elbow],
    ['leftWrist', 'L wrist', POSE_COLORS.wrist],
    ['rightWrist', 'R wrist', POSE_COLORS.wrist],
  ];
  for (const [key, label, color] of joints) {
    if (showRaw.checked) drawDot(rawPose?.[key], '', '#ffffff99', width, height, 4);
    drawDot(filteredPose[key], label, color, width, height, 8);
  }
}

function render() {
  setCanvasSize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (currentResult) drawPose(currentResult.pose, currentResult.filteredPose, canvas.width, canvas.height);
  requestAnimationFrame(render);
}

function updateMetrics() {
  const details = currentInference ? `${currentInference.totalMs.toFixed(0)} ms` : '—';
  metrics.textContent = `Camera: ${cameraDescription} · Processed: ${processedFps.toFixed(1)} FPS · Inference: ${details} · Frame delay: ${displayDelayMs ? `${displayDelayMs.toFixed(0)} ms` : '—'}`;
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (message.generation !== generation) return;
  if (message.type === 'ready') {
    workerReady = true;
    filters.clear();
    const fallback = message.fallback ? ' · GPU unavailable; CPU fallback' : '';
    delegateNote.textContent = `Body pose runs in a browser Web Worker on ${message.delegate}. Docker serves its model and WASM locally.${fallback}`;
    setStatus(`Tracking ready · ${message.delegate} delegate`, 'ready');
    return;
  }
  if (message.type === 'delegate-fallback') {
    delegateNote.textContent = 'GPU inference failed; switched to CPU and retried the current frame.';
    return;
  }
  if (message.type === 'fatal') {
    workerReady = false;
    setStatus(message.message, 'error');
    toggle.disabled = false;
    return;
  }
  if (message.generation === inFlightGeneration) busy = false;
  if (message.type === 'detection-error') {
    setStatus(`Tracking error: ${message.message}`, 'error');
    return;
  }
  if (message.type !== 'result') return;

  currentResult = { ...message, filteredPose: filterPose(message.pose, message.timestampMs) };
  currentInference = message.inference;
  displayDelayMs = performance.now() - sentAt;
  resultCount += 1;
  const elapsed = performance.now() - resultWindowStarted;
  if (elapsed >= 1000) {
    processedFps = (resultCount * 1000) / elapsed;
    resultCount = 0;
    resultWindowStarted = performance.now();
  }
  updateMetrics();

  const wristsVisible = Boolean(message.pose?.leftWrist && message.pose?.rightWrist);
  setStatus(wristsVisible ? 'Tracking both wrists' : 'Finding shoulders, elbows, and wrists', wristsVisible ? 'ready' : 'busy');
}

function initializeWorker() {
  generation += 1;
  busy = false;
  workerReady = false;
  inFlightGeneration = generation;
  currentResult = null;
  currentInference = null;
  filters.clear();
  worker.postMessage({
    type: 'initialize',
    generation,
    delegatePreference: delegateSelect.value,
  });
  setStatus('Loading body pose model…', 'busy');
}

async function frameLoop(now, metadata) {
  if (!media) return;
  if (workerReady && !busy && video.readyState >= 2) {
    busy = true;
    inFlightGeneration = generation;
    sentAt = performance.now();
    try {
      const bitmap = await createImageBitmap(video);
      if (!media || !worker || !workerReady) bitmap.close();
      else worker.postMessage({
        type: 'frame',
        generation,
        timestampMs: Math.round(metadata.mediaTime * 1000),
        bitmap,
      }, [bitmap]);
    } catch (error) {
      busy = false;
      setStatus(`Could not prepare camera frame: ${error.message}`, 'error');
    }
  }
  scheduleFrameLoop();
}

function scheduleFrameLoop() {
  if (video.requestVideoFrameCallback) {
    callbackKind = 'video';
    callbackId = video.requestVideoFrameCallback(frameLoop);
  } else {
    callbackKind = 'animation';
    callbackId = requestAnimationFrame(now => frameLoop(now, { mediaTime: video.currentTime }));
  }
}

async function start() {
  toggle.disabled = true;
  setStatus('Requesting camera access…');
  try {
    media = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60, max: 60 } },
    });
    const { width, height, frameRate } = media.getVideoTracks()[0].getSettings();
    cameraDescription = `${width || '?'}×${height || '?'}${frameRate ? ` @ ${frameRate.toFixed(0)} FPS` : ''}`;
    video.srcObject = media;
    await video.play();
    empty.hidden = true;
    worker = new Worker('/tracking-worker.js', { type: 'module' });
    worker.onmessage = handleWorkerMessage;
    worker.onerror = error => setStatus(`Tracking worker failed: ${error.message}`, 'error');
    initializeWorker();
    toggle.textContent = 'Stop camera';
    toggle.disabled = false;
    scheduleFrameLoop();
  } catch (error) {
    stop();
    const reason = error.name === 'NotAllowedError' ? 'Camera permission was denied.' : `Could not start camera: ${error.message}`;
    setStatus(reason, 'error');
    toggle.disabled = false;
  }
}

function stop() {
  if (callbackId !== null) {
    if (callbackKind === 'video' && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(callbackId);
    else cancelAnimationFrame(callbackId);
  }
  callbackId = null;
  callbackKind = null;
  worker?.terminate();
  worker = null;
  workerReady = false;
  busy = false;
  if (media) media.getTracks().forEach(track => track.stop());
  media = null;
  video.srcObject = null;
  empty.hidden = false;
  currentResult = null;
  currentInference = null;
  filters.clear();
  processedFps = 0;
  displayDelayMs = 0;
  cameraDescription = '—';
  resultCount = 0;
  metrics.textContent = 'Camera: — · Processed: — · Inference: — · Frame delay: —';
  toggle.textContent = 'Start camera';
  toggle.disabled = false;
  setStatus('Camera is off');
}

toggle.addEventListener('click', () => media ? stop() : start());
delegateSelect.addEventListener('change', () => { if (worker && media) initializeWorker(); });
responseSlider.addEventListener('input', () => { responseValue.textContent = `${responseSlider.value}%`; });
render();
