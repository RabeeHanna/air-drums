import { StrokeDetector } from './stroke-detector.mjs';
import { isWithinBoundary, moveBoundary, resizeBoundary } from './hit-boundary.mjs';
import { comparePredictions, TargetPredictor } from './target-predictor.mjs';
import { estimateHandPoint } from './pose-hand-point.mjs';
import { LocalAudioPlayer } from './local-audio.mjs';
import { CandidateObserver } from './candidate-observer.mjs';

const video = document.querySelector('#video');
const canvas = document.querySelector('#overlay');
const ctx = canvas.getContext('2d');
const status = document.querySelector('#status');
const strokeReadout = document.querySelector('#strokeReadout');
const predictionReadout = document.querySelector('#predictionReadout');
const metrics = document.querySelector('#metrics');
const toggle = document.querySelector('#toggle');
const empty = document.querySelector('#empty');
const delegateSelect = document.querySelector('#delegate');
const responseSlider = document.querySelector('#responsiveness');
const responseValue = document.querySelector('#responseValue');
const detectorControls = Object.fromEntries(['left', 'right'].map(hand => {
  const prefix = hand === 'left' ? 'left' : 'right';
  return [hand, {
    downThreshold: { slider: document.querySelector(`#${prefix}DownThreshold`), value: document.querySelector(`#${prefix}DownThresholdValue`) },
    minTravel: { slider: document.querySelector(`#${prefix}MinTravel`), value: document.querySelector(`#${prefix}MinTravelValue`) },
    impactDrop: { slider: document.querySelector(`#${prefix}ImpactDrop`), value: document.querySelector(`#${prefix}ImpactDropValue`) },
  }];
}));
const resetStrokeSettingsButton = document.querySelector('#resetStrokeSettings');
const restrictHitsToZones = document.querySelector('#restrictHitsToZones');
const showRaw = document.querySelector('#showRaw');
const showHandEstimate = document.querySelector('#showHandEstimate');
const handExtensionSlider = document.querySelector('#handExtension');
const handExtensionValue = document.querySelector('#handExtensionValue');
const inferenceSizeSelect = document.querySelector('#inferenceSize');
const audioEnabled = document.querySelector('#audioEnabled');
const audioVolumeSlider = document.querySelector('#audioVolume');
const audioVolumeValue = document.querySelector('#audioVolumeValue');
const audioStatus = document.querySelector('#audioStatus');
const delegateNote = document.querySelector('#delegateNote');
const view = document.querySelector('#view');
const layoutEdit = document.querySelector('#layoutEdit');
const layoutSave = document.querySelector('#layoutSave');
const layoutReset = document.querySelector('#layoutReset');
const layoutHelp = document.querySelector('#layoutHelp');
const zoneSelect = document.querySelector('#zoneSelect');
const boundaryEditButton = document.querySelector('#boundaryEdit');
const calibrationZoneSelect = document.querySelector('#calibrationZone');
const calibrationHandSelect = document.querySelector('#calibrationHand');
const promptTrialButton = document.querySelector('#promptTrial');
const markMissedButton = document.querySelector('#markMissed');
const markTrackingLossButton = document.querySelector('#markTrackingLoss');
const cancelTrialButton = document.querySelector('#cancelTrial');
const calibrationStatus = document.querySelector('#calibrationStatus');
const calibrationSummary = document.querySelector('#calibrationSummary');
const autoAdjustToggle = document.querySelector('#autoAdjust');
const adjustmentReadout = document.querySelector('#adjustmentReadout');
const revertAdjustmentButton = document.querySelector('#revertAdjustment');
const markFalseHitButton = document.querySelector('#markFalseHit');

const POSE_COLORS = { elbow: '#43e0b5', wrist: '#f5cb5c', handEstimate: '#ff9e64' };
const DETECTOR_DEFAULTS = { downThreshold: 0.55, minTravel: 0.05, impactDrop: 0.62 };
const DETECTOR_LIMITS = { downThreshold: [0.2, 1.5], minTravel: [0.015, 0.12], impactDrop: [0.35, 0.9] };

function loadAppSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem('airDrums.appSettings') || '{}');
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch {
    return {};
  }
}

const appSettings = loadAppSettings();

function loadDetectorSettings() {
  try {
    const saved = appSettings.strokeSettings ?? JSON.parse(localStorage.getItem('airDrums.strokeSettings') || '{}');
    return Object.fromEntries(['left', 'right'].map(hand => {
      // Upgrade the previous shared setting format by copying it to each hand.
      const source = saved[hand] && typeof saved[hand] === 'object' ? saved[hand] : saved;
      const settings = Object.fromEntries(Object.entries(DETECTOR_DEFAULTS).map(([key, fallback]) => {
        const [minimum, maximum] = DETECTOR_LIMITS[key];
        const value = Number(source[key]);
        return [key, Number.isFinite(value) ? Math.max(minimum, Math.min(maximum, value)) : fallback];
      }));
      return [hand, settings];
    }));
  } catch {
    return { left: { ...DETECTOR_DEFAULTS }, right: { ...DETECTOR_DEFAULTS } };
  }
}

const detectorSettings = loadDetectorSettings();
function loadZoneRestriction() {
  if (typeof appSettings.restrictHitsToZones === 'boolean') return appSettings.restrictHitsToZones;
  try { return localStorage.getItem('airDrums.restrictHitsToZones') !== 'false'; } catch { return true; }
}
restrictHitsToZones.checked = loadZoneRestriction();
const audioPlayer = new LocalAudioPlayer(message => { audioStatus.textContent = message; });

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
let kitLayout = null;
let defaultLayout = null;
let selectedZoneId = null;
let dragging = null;
let editingLayout = false;
let layoutDirty = false;
let editingBoundary = false;
const filters = new Map();
const strokeDetectors = new Map([
  ['left', new StrokeDetector('left', detectorSettings.left)],
  ['right', new StrokeDetector('right', detectorSettings.right)],
]);
const targetPredictors = new Map([
  ['left', new TargetPredictor()],
  ['right', new TargetPredictor()],
]);
const candidateObservers = new Map([
  ['left', new CandidateObserver()],
  ['right', new CandidateObserver()],
]);
const visibleHits = new Map();
let totalHits = 0;
let ignoredHits = 0;
let outsideBoundaryHits = 0;
let outsidePadHits = 0;
let detectedStrokes = 0;
let lastPredictionComparison = '';
let strokeSessionId = null;
let strokeWriteQueue = Promise.resolve();
let activeCalibrationTrial = null;
let calibrationTrials = [];
let lastDetectedStroke = null;
let lastAdjustment = null;

function saveAppSettings() {
  try {
    const settings = {
      delegate: delegateSelect.value,
      responsiveness: Number(responseSlider.value),
      inferenceWidth: inferenceSizeSelect.value,
      showRaw: showRaw.checked,
      showHandEstimate: showHandEstimate.checked,
      handExtension: Number(handExtensionSlider.value),
      audioEnabled: audioEnabled.checked,
      audioVolume: Number(audioVolumeSlider.value),
      restrictHitsToZones: restrictHitsToZones.checked,
      autoAdjust: autoAdjustToggle.checked,
      calibrationZone: calibrationZoneSelect.value,
      calibrationHand: calibrationHandSelect.value,
      selectedZoneId,
      strokeSettings: detectorSettings,
    };
    localStorage.setItem('airDrums.appSettings', JSON.stringify(settings));
  } catch {
    // All controls still apply for this page if browser storage is unavailable.
  }
}

function queueCalibrationTrial(trial) {
  if (!strokeSessionId) return;
  const sessionId = strokeSessionId;
  strokeWriteQueue = strokeWriteQueue.then(() => fetch(`/api/sessions/${sessionId}/calibration-trials`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(trial),
  })).then(response => {
    if (!response.ok) throw new Error(`Calibration record failed (${response.status})`);
  }).catch(error => {
    calibrationStatus.textContent = `Calibration record could not be saved: ${error.message}`;
  });
}

function renderCalibrationSummary() {
  const counts = Object.fromEntries(['correct', 'wrong_zone', 'outside_area', 'outside_zone', 'miss', 'tracking_loss', 'false_hit'].map(key => [key, 0]));
  const mismatches = new Map();
  let missesWithCandidate = 0;
  for (const trial of calibrationTrials) counts[trial.outcome] = (counts[trial.outcome] ?? 0) + 1;
  for (const trial of calibrationTrials) {
    if (trial.outcome === 'miss' && (trial.peakSpeed !== null || trial.travel !== null)) missesWithCandidate += 1;
    if (trial.outcome !== 'wrong_zone') continue;
    const intended = kitLayout?.zones.find(zone => zone.id === trial.intendedZoneId)?.label ?? trial.intendedZoneId;
    const actual = kitLayout?.zones.find(zone => zone.id === trial.actualZoneId)?.label ?? 'unknown';
    const key = `${intended} → ${actual}`;
    mismatches.set(key, (mismatches.get(key) ?? 0) + 1);
  }
  const zoneSummary = [...mismatches].map(([pair, count]) => `${pair} (${count})`).join(', ');
  const handSummary = ['left', 'right'].map(hand => {
    const handTrials = calibrationTrials.filter(trial => trial.hand === hand);
    const hits = handTrials.filter(trial => trial.outcome === 'correct').length;
    const misses = handTrials.filter(trial => trial.outcome === 'miss').length;
    const falseHits = handTrials.filter(trial => trial.outcome === 'false_hit').length;
    return `${hand === 'left' ? 'L' : 'R'} ${hits} hits / ${misses} misses / ${falseHits} false`;
  }).join(' · ');
  calibrationSummary.textContent = `Trials ${calibrationTrials.length} · Correct ${counts.correct} · Wrong drum ${counts.wrong_zone}${zoneSummary ? ` [${zoneSummary}]` : ''} · Outside rectangle ${counts.outside_area} · Outside pads ${counts.outside_zone} · Missed ${counts.miss} (${missesWithCandidate} with candidate motion, ${counts.miss - missesWithCandidate} without) · Tracking lost ${counts.tracking_loss} · False hits ${counts.false_hit} · ${handSummary}`;
  markFalseHitButton.disabled = !lastDetectedStroke;
}

function postCalibrationTrial(trial) {
  calibrationTrials.push(trial);
  queueCalibrationTrial(trial);
  renderCalibrationSummary();
}

function armCalibrationTrial() {
  if (!strokeSessionId) {
    calibrationStatus.textContent = 'Start the camera before arming a prompted tap.';
    return;
  }
  activeCalibrationTrial = { intendedZoneId: calibrationZoneSelect.value, hand: calibrationHandSelect.value, candidate: null };
  calibrationStatus.textContent = `Ready: tap ${calibrationZoneSelect.selectedOptions[0].text} with your ${calibrationHandSelect.value} hand. The next detected stroke will be compared with that drum.`;
  promptTrialButton.disabled = true;
  markMissedButton.disabled = false;
  markTrackingLossButton.disabled = false;
  cancelTrialButton.hidden = false;
}

function finishPromptedTrial(outcome, stroke = null, candidate = activeCalibrationTrial?.candidate) {
  if (!activeCalibrationTrial) return;
  const promptedHand = activeCalibrationTrial.hand;
  const trial = {
    timestamp: new Date().toISOString(),
    intendedZoneId: activeCalibrationTrial.intendedZoneId,
    outcome,
    actualZoneId: stroke?.actualZoneId ?? null,
    insideHitArea: stroke?.insideHitArea ?? null,
    insideDrumZone: stroke?.insideDrumZone ?? null,
    hand: stroke?.hand ?? candidate?.hand ?? promptedHand,
    peakSpeed: stroke?.peakSpeed ?? candidate?.peakSpeed ?? null,
    travel: stroke?.travel ?? candidate?.travel ?? null,
  };
  postCalibrationTrial(trial);
  const intended = kitLayout?.zones.find(zone => zone.id === trial.intendedZoneId)?.label ?? trial.intendedZoneId;
  const actual = kitLayout?.zones.find(zone => zone.id === trial.actualZoneId)?.label;
  calibrationStatus.textContent = outcome === 'correct' ? `Correct · ${intended}`
    : outcome === 'wrong_zone' ? `Zone mismatch · aimed ${intended}, detected ${actual ?? 'unknown'}`
    : outcome === 'outside_area' ? `Stroke detected for ${intended}, outside the hit rectangle`
      : outcome === 'outside_zone' ? `Stroke detected for ${intended}, outside all drum pads`
        : outcome === 'tracking_loss' ? `Tracking lost during tap for ${intended}` : `Miss recorded for ${intended}`;
  activeCalibrationTrial = null;
  for (const observer of candidateObservers.values()) observer.reset();
  promptTrialButton.disabled = false;
  markMissedButton.disabled = true;
  markTrackingLossButton.disabled = true;
  cancelTrialButton.hidden = true;
  if (outcome === 'miss' && autoAdjustToggle.checked && candidate) applyAutomaticAdjustment('miss', candidate, candidate.hand ?? promptedHand);
}

function applyAutomaticAdjustment(kind, measurement, hand) {
  if (!hand || !detectorSettings[hand]) return;
  const settings = detectorSettings[hand];
  const before = { ...settings };
  let field = null;
  let direction = 0;
  if (kind === 'miss') {
    if (measurement.peakSpeed < settings.downThreshold && measurement.peakSpeed >= settings.downThreshold * 0.45) {
      field = 'downThreshold'; direction = -0.05;
    } else if (measurement.travel < settings.minTravel && measurement.travel >= settings.minTravel * 0.45) {
      field = 'minTravel'; direction = -0.005;
    }
  } else {
    if (measurement.travel < settings.minTravel * 1.35) {
      field = 'minTravel'; direction = 0.005;
    } else if (measurement.peakSpeed < settings.downThreshold * 1.35) {
      field = 'downThreshold'; direction = 0.05;
    }
  }
  if (!field) {
    adjustmentReadout.textContent = 'No sensitivity change: this result did not identify a near-threshold stroke.';
    return;
  }
  const [minimum, maximum] = DETECTOR_LIMITS[field];
  const precision = field === 'downThreshold' ? 2 : 3;
  const next = Math.max(minimum, Math.min(maximum, Number((settings[field] + direction).toFixed(precision))));
  if (next === settings[field]) {
    adjustmentReadout.textContent = 'Sensitivity is already at its allowed limit; no change applied.';
    return;
  }
  lastAdjustment = { hand, before };
  detectorControls[hand][field].slider.value = String(next);
  updateDetectorSettings(hand);
  adjustmentReadout.textContent = `${hand === 'left' ? 'Left' : 'Right'} hand · ${field === 'downThreshold' ? 'Speed threshold' : 'Minimum wrist travel'} ${before[field].toFixed(precision)} → ${next.toFixed(precision)} after a marked ${kind === 'miss' ? 'near-miss' : 'false hit'}.`;
  revertAdjustmentButton.hidden = false;
}

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
    const labelY = y - Math.max(12, width * 0.015);
    ctx.font = `700 ${Math.max(12, width * 0.015)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.lineWidth = Math.max(3, width * 0.004);
    ctx.strokeStyle = '#101313';
    ctx.save();
    ctx.translate(x, labelY);
    ctx.scale(-1, 1);
    ctx.strokeText(label, 0, 0);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
}

function drawPose(rawPose, filteredPose, width, height) {
  if (!filteredPose) return;
  const aspectRatio = width / height;
  const extension = Number(handExtensionSlider.value) / 100;
  for (const hand of ['left', 'right']) {
    const elbow = filteredPose[`${hand}Elbow`];
    const wrist = filteredPose[`${hand}Wrist`];
    if (!wrist) continue;
    if (elbow) {
      ctx.beginPath();
      ctx.moveTo(elbow.x * width, elbow.y * height);
      ctx.lineTo(wrist.x * width, wrist.y * height);
      ctx.strokeStyle = '#43e0b5cc';
      ctx.lineWidth = Math.max(3, width * 0.0035);
      ctx.lineCap = 'round';
      ctx.stroke();
      if (showRaw.checked) drawDot(rawPose?.[`${hand}Elbow`], '', '#ffffff99', width, height, 4);
      drawDot(elbow, '', POSE_COLORS.elbow, width, height, 5);
    }
    if (showRaw.checked) drawDot(rawPose?.[`${hand}Wrist`], '', '#ffffff99', width, height, 4);
    drawDot(wrist, '', POSE_COLORS.wrist, width, height, 6);
    if (!showHandEstimate.checked) continue;
    const endpoint = estimateHandPoint(filteredPose, hand, extension, aspectRatio);
    if (!endpoint) continue;
    ctx.beginPath();
    ctx.moveTo(wrist.x * width, wrist.y * height);
    ctx.lineTo(endpoint.x * width, endpoint.y * height);
    ctx.strokeStyle = '#ff9e64cc';
    ctx.lineWidth = Math.max(3, width * 0.0035);
    ctx.lineCap = 'round';
    ctx.stroke();
    drawDot(endpoint, `${hand === 'left' ? 'L' : 'R'} hit point`, POSE_COLORS.handEstimate, width, height, 7);
  }
}

function drawHitFeedback(width, height) {
  const now = performance.now();
  for (const [hand, hit] of visibleHits) {
    if (now > hit.until) {
      visibleHits.delete(hand);
      continue;
    }
    const x = hit.point.x * width;
    const y = hit.point.y * height;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, Math.max(22, width * 0.028), 0, Math.PI * 2);
    ctx.fillStyle = '#f5cb5c';
    ctx.globalAlpha = 0.28;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = Math.max(3, width * 0.004);
    ctx.strokeStyle = '#fff2b8';
    ctx.stroke();
    ctx.font = `900 ${Math.max(17, width * 0.024)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.lineWidth = Math.max(4, width * 0.005);
    ctx.strokeStyle = '#101313';
    ctx.translate(x, y - Math.max(28, width * 0.034));
    ctx.scale(-1, 1);
    ctx.strokeText('HIT', 0, 0);
    ctx.fillStyle = '#fff';
    ctx.fillText('HIT', 0, 0);
    ctx.restore();
  }
}

function logStroke(stroke) {
  if (!strokeSessionId) return;
  // Serialize compact stroke events; frame processing never waits here.
  const sessionId = strokeSessionId;
  strokeWriteQueue = strokeWriteQueue.then(() => fetch(`/api/sessions/${sessionId}/strokes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(stroke),
  })).catch(() => {});
}

function detectStrokes(pose, timestampMs) {
  const left = pose?.leftShoulder;
  const right = pose?.rightShoulder;
  const shoulderScale = left && right && Math.min(left.visibility ?? 1, right.visibility ?? 1) >= 0.35
    ? Math.hypot((left.x - right.x) * (canvas.width / canvas.height), left.y - right.y)
    : null;
  const aspectRatio = canvas.width / canvas.height;
  for (const hand of ['left', 'right']) {
    const point = pose?.[`${hand}Wrist`];
    const detector = strokeDetectors.get(hand);
    const observer = candidateObservers.get(hand);
    const predictor = targetPredictors.get(hand);
    const candidate = activeCalibrationTrial
      ? observer.update(point, timestampMs, shoulderScale, aspectRatio, Math.min(detector.downThreshold * 0.5, 0.15))
      : null;
    if (candidate) {
      candidate.hand = hand;
      if (activeCalibrationTrial) activeCalibrationTrial.candidate = candidate;
    }
    const priorPrediction = predictor.predictions;
    const result = detector.update(point, timestampMs, shoulderScale, aspectRatio);
    const wristPrediction = result.event ? priorPrediction : predictor.update(point, timestampMs, shoulderScale, aspectRatio, {
      active: result.state === 'downswing',
      peakDownSpeed: detector.peakDownSpeed,
      impactDrop: detector.impactDrop,
    });
    const estimatedPoint = estimateHandPoint(
      currentResult?.filteredPose,
      hand,
      Number(handExtensionSlider.value) / 100,
      aspectRatio,
    );
    const prediction = result.event ? wristPrediction : wristPrediction && estimatedPoint ? {
      ...wristPrediction,
      fixed: wristPrediction.fixed ? { ...wristPrediction.fixed, x: wristPrediction.fixed.x + estimatedPoint.x - point.x, y: wristPrediction.fixed.y + estimatedPoint.y - point.y } : null,
      deceleration: wristPrediction.deceleration ? { ...wristPrediction.deceleration, x: wristPrediction.deceleration.x + estimatedPoint.x - point.x, y: wristPrediction.deceleration.y + estimatedPoint.y - point.y } : null,
    } : null;
    if (!result.event) predictor.predictions = prediction;
    if (result.event) {
      observer.reset();
      detectedStrokes += 1;
      const hitPoint = estimatedPoint;
      result.event.x = hitPoint?.x ?? null;
      result.event.y = hitPoint?.y ?? null;
      result.event.insideHitArea = hitPoint ? isInsideStrikeBoundary(hitPoint) : false;
      const containedZone = hitPoint ? drumZoneAt(hitPoint) : null;
      result.event.insideDrumZone = Boolean(containedZone);
      result.event.actualZoneId = containedZone?.id ?? (hitPoint ? nearestZone(hitPoint)?.id ?? null : null);
      result.event.predictions = hitPoint ? comparePredictions(
        prediction || predictor.predictions,
        result.event,
        canvas.width,
        canvas.height,
        kitLayout?.zones ?? [],
      ) : null;
      const modelSummaries = [['fixed', '120ms'], ['deceleration', 'turn']].map(([model, label]) => {
        const item = result.event.predictions?.[model];
        if (!item) return `${label} unavailable`;
        const predictedZone = kitLayout?.zones.find(zone => zone.id === item.zoneId)?.label ?? '—';
        const actualZone = kitLayout?.zones.find(zone => zone.id === result.event.actualZoneId)?.label ?? '—';
        return `${label} ${predictedZone} → ${actualZone}: ${item.positionErrorPx.toFixed(0)}px, ${item.timingErrorMs >= 0 ? '+' : ''}${item.timingErrorMs.toFixed(0)}ms`;
      });
      lastPredictionComparison = `${hand === 'left' ? 'L' : 'R'} impact · ${modelSummaries.join(' · ')}`;
      const acceptedByZone = !restrictHitsToZones.checked || result.event.insideDrumZone;
      if (result.event.insideHitArea && acceptedByZone) {
        totalHits += 1;
        audioPlayer.play(result.event.actualZoneId);
        visibleHits.set(hand, { point: { x: hitPoint.x, y: hitPoint.y }, until: performance.now() + 280 });
        requestRender();
      } else {
        ignoredHits += 1;
      }
      if (!result.event.insideHitArea) outsideBoundaryHits += 1;
      else if (!result.event.insideDrumZone) outsidePadHits += 1;
      logStroke(result.event);
      lastDetectedStroke = result.event;
      markFalseHitButton.disabled = false;
      if (activeCalibrationTrial) {
        const outcome = !result.event.insideHitArea ? 'outside_area'
          : restrictHitsToZones.checked && !result.event.insideDrumZone ? 'outside_zone'
            : result.event.actualZoneId === activeCalibrationTrial.intendedZoneId ? 'correct' : 'wrong_zone';
        finishPromptedTrial(outcome, result.event);
      }
      predictor.reset();
    } else if (!prediction && result.state !== 'downswing') {
      predictor.predictions = null;
    }
  }
  updatePredictionReadout();
  strokeReadout.textContent = `Strokes: L ${strokeDetectors.get('left').state} · R ${strokeDetectors.get('right').state} · Detected ${detectedStrokes} · Hits ${totalHits} · Outside rectangle ${outsideBoundaryHits} · Outside pads ${outsidePadHits}`;
}

function updateDetectorSettings(hand) {
  const settings = detectorSettings[hand];
  for (const key of Object.keys(DETECTOR_DEFAULTS)) {
    settings[key] = Number(detectorControls[hand][key].slider.value);
  }
  detectorControls[hand].downThreshold.value.textContent = settings.downThreshold.toFixed(2);
  detectorControls[hand].minTravel.value.textContent = settings.minTravel.toFixed(3);
  detectorControls[hand].impactDrop.value.textContent = settings.impactDrop.toFixed(2);
  Object.assign(strokeDetectors.get(hand), settings);
  saveAppSettings();
}

function saveZoneRestriction() {
  saveAppSettings();
}

function initializeDetectorSettings() {
  for (const hand of ['left', 'right']) {
    for (const key of Object.keys(DETECTOR_DEFAULTS)) detectorControls[hand][key].slider.value = detectorSettings[hand][key];
    updateDetectorSettings(hand);
  }
}

function resetDetectorSettings() {
  for (const hand of ['left', 'right']) Object.assign(detectorSettings[hand], DETECTOR_DEFAULTS);
  initializeDetectorSettings();
}

function cancelCalibrationTrial(message = 'Prompt cancelled.') {
  activeCalibrationTrial = null;
  for (const observer of candidateObservers.values()) observer.reset();
  promptTrialButton.disabled = false;
  markMissedButton.disabled = true;
  markTrackingLossButton.disabled = true;
  cancelTrialButton.hidden = true;
  calibrationStatus.textContent = message;
}

function revertAutomaticAdjustment() {
  if (!lastAdjustment) return;
  const { hand, before } = lastAdjustment;
  for (const key of Object.keys(DETECTOR_DEFAULTS)) detectorControls[hand][key].slider.value = String(before[key]);
  updateDetectorSettings(hand);
  lastAdjustment = null;
  adjustmentReadout.textContent = `Last ${hand} hand automatic sensitivity adjustment reverted.`;
  revertAdjustmentButton.hidden = true;
}

function markLastStrokeFalseHit() {
  if (!lastDetectedStroke) return;
  const stroke = lastDetectedStroke;
  postCalibrationTrial({
    timestamp: new Date().toISOString(), outcome: 'false_hit', intendedZoneId: null,
    actualZoneId: stroke.actualZoneId, insideHitArea: stroke.insideHitArea,
    insideDrumZone: stroke.insideDrumZone,
    hand: stroke.hand, peakSpeed: stroke.peakSpeed, travel: stroke.travel,
  });
  calibrationStatus.textContent = 'Last detected stroke marked as a false hit.';
  lastDetectedStroke = null;
  markFalseHitButton.disabled = true;
  if (autoAdjustToggle.checked) applyAutomaticAdjustment('false_hit', stroke, stroke.hand);
}

function loadVisualSettings() {
  try {
    const inferenceWidth = appSettings.inferenceWidth ?? localStorage.getItem('airDrums.inferenceWidth');
    if (['480', '640'].includes(String(inferenceWidth))) inferenceSizeSelect.value = String(inferenceWidth);
    const storedExtension = appSettings.handExtension ?? localStorage.getItem('airDrums.handExtension');
    const extension = Number(storedExtension);
    if (storedExtension !== null && Number.isFinite(extension)) {
      handExtensionSlider.value = Math.max(0, Math.min(70, extension));
    }
    const oldShowHandEstimate = localStorage.getItem('airDrums.showHandEstimate');
    showHandEstimate.checked = typeof appSettings.showHandEstimate === 'boolean'
      ? appSettings.showHandEstimate : oldShowHandEstimate !== 'false';
    showRaw.checked = typeof appSettings.showRaw === 'boolean' ? appSettings.showRaw : false;
    audioEnabled.checked = typeof appSettings.audioEnabled === 'boolean' ? appSettings.audioEnabled : audioEnabled.checked;
    const audioVolume = Number(appSettings.audioVolume ?? audioVolumeSlider.value);
    if (Number.isFinite(audioVolume)) audioVolumeSlider.value = Math.max(0, Math.min(100, audioVolume));
    if (['auto', 'CPU'].includes(appSettings.delegate)) delegateSelect.value = appSettings.delegate;
    if (Number.isFinite(Number(appSettings.responsiveness))) {
      responseSlider.value = Math.max(0, Math.min(100, Number(appSettings.responsiveness)));
    }
    autoAdjustToggle.checked = appSettings.autoAdjust === true;
    const zoneOptions = [...calibrationZoneSelect.options].map(option => option.value);
    if (zoneOptions.includes(appSettings.calibrationZone)) calibrationZoneSelect.value = appSettings.calibrationZone;
    if (['left', 'right'].includes(appSettings.calibrationHand)) calibrationHandSelect.value = appSettings.calibrationHand;
  } catch {
    // Keep the built-in defaults when browser storage is unavailable.
  }
  responseValue.textContent = `${responseSlider.value}%`;
  audioVolumeValue.textContent = `${audioVolumeSlider.value}%`;
  updateHandExtensionLabel();
}

function updateHandExtensionLabel() {
  handExtensionValue.textContent = `${(Number(handExtensionSlider.value) / 100).toFixed(2)}× shoulder width`;
  requestRender();
}

function updatePredictionReadout() {
  const active = [];
  for (const [hand, predictor] of targetPredictors) {
    if (!predictor.predictions) continue;
    const summaries = [['120ms', predictor.predictions.fixed], ['turn', predictor.predictions.deceleration]]
      .map(([name, prediction]) => {
        const zone = prediction ? nearestZone(prediction)?.label ?? '—' : 'unavailable';
        const eta = prediction ? `${prediction.timeToImpactMs.toFixed(0)}ms` : '—';
        return `${name}: ${zone} @ ${eta}`;
      });
    active.push(`${hand === 'left' ? 'L' : 'R'} ${summaries.join(' · ')}`);
  }
  predictionReadout.textContent = active.length
    ? `Prediction · ${active.join(' | ')}`
    : lastPredictionComparison || 'Prediction · move into a downswing to see estimates';
}

function isInsideStrikeBoundary(point) {
  return isWithinBoundary(point, kitLayout?.strikeBoundary);
}

async function openStrokeSession() {
  strokeWriteQueue = Promise.resolve();
  const response = await fetch('/api/sessions', { method: 'POST' });
  if (!response.ok) throw new Error(`Recording setup failed (${response.status})`);
  strokeSessionId = (await response.json()).id;
  calibrationTrials = [];
  lastDetectedStroke = null;
  renderCalibrationSummary();
  calibrationStatus.textContent = 'Choose a drum and prompt a tap when ready.';
}

function closeStrokeSession() {
  const sessionId = strokeSessionId;
  strokeSessionId = null;
  if (sessionId) {
    strokeWriteQueue = strokeWriteQueue
      .then(() => fetch(`/api/sessions/${sessionId}/finish`, { method: 'POST' }))
      .catch(() => {});
  }
}

function drawKitLayout(width, height) {
  if (!kitLayout) return;
  const shortSide = Math.min(width, height);
  for (const zone of kitLayout.zones) {
    const x = zone.x * width;
    const y = zone.y * height;
    const rx = zone.width * shortSide / 2;
    const ry = zone.height * shortSide / 2;
    const selected = zone.id === selectedZoneId;
    const highlighted = [...targetPredictors.values()].some(predictor =>
      predictor.predictions && [predictor.predictions.fixed, predictor.predictions.deceleration]
        .some(prediction => prediction && nearestZone(prediction)?.id === zone.id));
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.globalAlpha = editingLayout ? 0.28 : 0.22;
    ctx.fillStyle = zone.color;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.setLineDash(editingLayout && selected ? [8, 5] : []);
    ctx.lineWidth = highlighted ? 5 : selected && editingLayout ? 4 : 3;
    ctx.strokeStyle = highlighted ? '#fff' : zone.color;
    ctx.stroke();
    ctx.setLineDash([]);
    if (editingLayout && selected) drawResizeBox(zone, width, height);
    ctx.font = `700 ${Math.max(12, width * 0.018)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.lineWidth = Math.max(3, width * 0.004);
    ctx.strokeStyle = '#101313';
    ctx.save();
    ctx.translate(x, y - ry - Math.max(8, width * 0.012));
    ctx.scale(-1, 1);
    ctx.strokeText(zone.label, 0, 0);
    ctx.fillStyle = '#fff';
    ctx.fillText(zone.label, 0, 0);
    ctx.restore();
  }
}

function drawStrikeBoundary(width, height) {
  if (!editingBoundary || !kitLayout?.strikeBoundary) return;
  const boundary = kitLayout.strikeBoundary;
  const left = boundary.left * width;
  const top = boundary.top * height;
  const right = boundary.right * width;
  const bottom = boundary.bottom * height;
  const cx = (left + right) / 2;
  const cy = (top + bottom) / 2;
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.fillStyle = '#ffffff';
  ctx.globalAlpha = 0.08;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 4;
  ctx.setLineDash([12, 7]);
  ctx.strokeStyle = '#ffffff';
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = `700 ${Math.max(12, width * 0.016)}px system-ui`;
  ctx.textAlign = 'center';
  ctx.lineWidth = Math.max(3, width * 0.004);
  ctx.strokeStyle = '#101313';
  ctx.save();
  ctx.translate(cx, top - Math.max(10, width * 0.012));
  ctx.scale(-1, 1);
  ctx.strokeText('HIT AREA', 0, 0);
  ctx.fillStyle = '#fff';
  ctx.fillText('HIT AREA', 0, 0);
  ctx.restore();
  if (editingBoundary) drawBoundaryHandles(boundary, width, height);
  ctx.restore();
}

function drawBoundaryHandles(boundary, width, height) {
  const size = Math.max(12, width * 0.018);
  const corners = [
    [boundary.left, boundary.top], [boundary.right, boundary.top],
    [boundary.right, boundary.bottom], [boundary.left, boundary.bottom],
  ];
  for (const [x, y] of corners) {
    const screenX = (1 - x) * width;
    ctx.fillStyle = '#fff';
    ctx.fillRect(screenX - size / 2, y * height - size / 2, size, size);
    ctx.strokeStyle = '#172019';
    ctx.lineWidth = 2;
    ctx.strokeRect(screenX - size / 2, y * height - size / 2, size, size);
  }
}

function nearestZone(point) {
  if (!kitLayout || !point) return null;
  const width = canvas.width;
  const height = canvas.height;
  const shortSide = Math.min(width, height);
  return kitLayout.zones.map(zone => {
    const dx = (point.x - zone.x) * width / (zone.width * shortSide / 2);
    const dy = (point.y - zone.y) * height / (zone.height * shortSide / 2);
    return { zone, distance: dx * dx + dy * dy };
  }).sort((a, b) => a.distance - b.distance)[0]?.zone || null;
}

function drumZoneAt(point) {
  if (!kitLayout || !point) return null;
  const width = canvas.width;
  const height = canvas.height;
  const shortSide = Math.min(width, height);
  return kitLayout.zones.map(zone => {
    const dx = (point.x - zone.x) * width / (zone.width * shortSide / 2);
    const dy = (point.y - zone.y) * height / (zone.height * shortSide / 2);
    return { zone, distance: dx * dx + dy * dy };
  }).filter(item => item.distance <= 1).sort((a, b) => a.distance - b.distance)[0]?.zone ?? null;
}

function drawTargetPredictions(width, height) {
  for (const [hand, predictor] of targetPredictors) {
    if (!predictor.predictions) continue;
    for (const [prediction, color, label] of [
      [predictor.predictions.fixed, '#55e8e0', '120'],
      [predictor.predictions.deceleration, '#ff80d5', 'TURN'],
    ]) {
      if (!prediction) continue;
      const x = prediction.x * width;
      const y = prediction.y * height;
      const wrist = currentResult?.filteredPose?.[`${hand}Wrist`];
      ctx.save();
      ctx.beginPath();
      ctx.moveTo((wrist?.x ?? prediction.x) * width, (wrist?.y ?? prediction.y) * height);
      ctx.lineTo(x, y);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, width * 0.003);
      ctx.setLineDash([7, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(10, width * 0.014), 0, Math.PI * 2);
      ctx.fillStyle = `${color}55`;
      ctx.fill();
      ctx.lineWidth = Math.max(3, width * 0.0035);
      ctx.strokeStyle = color;
      ctx.stroke();
      const zone = nearestZone(prediction);
      ctx.font = `800 ${Math.max(11, width * 0.014)}px system-ui`;
      ctx.textAlign = 'center';
      ctx.lineWidth = Math.max(3, width * 0.004);
      ctx.strokeStyle = '#101313';
      ctx.save();
      ctx.translate(x, y - Math.max(14, width * 0.018));
      ctx.scale(-1, 1);
      const text = `${hand === 'left' ? 'L' : 'R'} ${label} ${prediction.timeToImpactMs.toFixed(0)}ms · ${zone?.label ?? '—'}`;
      ctx.strokeText(text, 0, 0);
      ctx.fillStyle = color;
      ctx.fillText(text, 0, 0);
      ctx.restore();
      ctx.restore();
    }
  }
}

function zoneBounds(zone, width = canvas.width, height = canvas.height) {
  const rx = zone.width * Math.min(width, height) / 2;
  const ry = zone.height * Math.min(width, height) / 2;
  return { left: zone.x * width - rx, right: zone.x * width + rx, top: zone.y * height - ry, bottom: zone.y * height + ry };
}

function resizeHandlePoints(zone) {
  const b = zoneBounds(zone);
  const mx = (b.left + b.right) / 2;
  const my = (b.top + b.bottom) / 2;
  return { nw: [b.left, b.top], n: [mx, b.top], ne: [b.right, b.top], e: [b.right, my], se: [b.right, b.bottom], s: [mx, b.bottom], sw: [b.left, b.bottom], w: [b.left, my] };
}

function drawResizeBox(zone, width, height) {
  const bounds = zoneBounds(zone, width, height);
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.strokeRect(bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);
  ctx.setLineDash([]);
  const size = Math.max(7, width * 0.009);
  for (const [x, y] of Object.values(resizeHandlePoints(zone))) {
    ctx.fillStyle = '#fff';
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
    ctx.strokeStyle = '#172019';
    ctx.strokeRect(x - size / 2, y - size / 2, size, size);
  }
  ctx.restore();
}

function render() {
  setCanvasSize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawKitLayout(canvas.width, canvas.height);
  drawStrikeBoundary(canvas.width, canvas.height);
  drawTargetPredictions(canvas.width, canvas.height);
  if (currentResult) {
    drawPose(currentResult.pose, currentResult.filteredPose, canvas.width, canvas.height);
    drawHitFeedback(canvas.width, canvas.height);
  }
}

let renderFrameId = null;
function requestRender() {
  if (renderFrameId !== null) return;
  renderFrameId = requestAnimationFrame(() => {
    renderFrameId = null;
    render();
    if (visibleHits.size) requestRender();
  });
}

function cloneLayout(layout) {
  return JSON.parse(JSON.stringify(layout));
}

function setLayoutDirty(dirty) {
  layoutDirty = dirty;
  layoutSave.disabled = !dirty;
  layoutHelp.textContent = dirty
    ? 'Unsaved kit changes. Save kit to keep these positions.'
    : 'Kit layout loaded.';
  requestRender();
}

function updateZoneControls() {
  zoneSelect.disabled = !kitLayout;
  if (selectedZoneId) zoneSelect.value = selectedZoneId;
}

async function loadKitLayout() {
  try {
    const [layoutResponse, defaultsResponse] = await Promise.all([
      fetch('/api/kit-layout'),
      fetch('/api/kit-layout/defaults'),
    ]);
    if (!layoutResponse.ok || !defaultsResponse.ok) throw new Error('Could not load kit layout settings');
    kitLayout = await layoutResponse.json();
    defaultLayout = await defaultsResponse.json();
    selectedZoneId = kitLayout.zones.some(zone => zone.id === appSettings.selectedZoneId)
      ? appSettings.selectedZoneId : kitLayout.zones[0]?.id || null;
    updateZoneControls();
    saveAppSettings();
    setLayoutDirty(false);
    requestRender();
  } catch (error) {
    layoutHelp.textContent = `Kit layout unavailable: ${error.message}`;
    layoutSave.disabled = true;
  }
}

function pointFromPointer(event) {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.max(rect.width / canvas.width, rect.height / canvas.height);
  const displayedWidth = canvas.width * scale;
  const displayedHeight = canvas.height * scale;
  const offsetX = (rect.width - displayedWidth) / 2;
  const offsetY = (rect.height - displayedHeight) / 2;
  const displayX = (event.clientX - rect.left - offsetX) / displayedWidth;
  const displayY = (event.clientY - rect.top - offsetY) / displayedHeight;
  // The preview is mirrored; convert the pointer position back to raw camera
  // coordinates so the saved zones line up with MediaPipe's landmarks.
  return { x: 1 - displayX, y: displayY };
}

function zoneAt(point) {
  if (!kitLayout) return null;
  const width = canvas.width;
  const height = canvas.height;
  const shortSide = Math.min(width, height);
  return kitLayout.zones.map(zone => {
    const dx = (point.x - zone.x) * width / (zone.width * shortSide / 2);
    const dy = (point.y - zone.y) * height / (zone.height * shortSide / 2);
    return { zone, distance: dx * dx + dy * dy };
  }).filter(({ distance }) => distance <= 1).sort((a, b) => a.distance - b.distance)[0]?.zone || null;
}

function moveRegion(region, point) {
  const shortSide = Math.min(canvas.width, canvas.height);
  const marginX = (region.width * shortSide / 2) / canvas.width;
  const marginY = (region.height * shortSide / 2) / canvas.height;
  region.x = Math.max(marginX, Math.min(1 - marginX, point.x));
  region.y = Math.max(marginY, Math.min(1 - marginY, point.y));
  setLayoutDirty(true);
}

function resizeRegion(region, drag, point, limits) {
  const shortSide = Math.min(canvas.width, canvas.height);
  const dx = (point.x - drag.point.x) * canvas.width;
  const dy = (point.y - drag.point.y) * canvas.height;
  let { x, y, width, height } = drag;
  if (drag.handle.includes('e')) { width += 2 * dx / shortSide; x += dx / (2 * canvas.width); }
  if (drag.handle.includes('w')) { width -= 2 * dx / shortSide; x += dx / (2 * canvas.width); }
  if (drag.handle.includes('s')) { height += 2 * dy / shortSide; y += dy / (2 * canvas.height); }
  if (drag.handle.includes('n')) { height -= 2 * dy / shortSide; y += dy / (2 * canvas.height); }
  region.width = Math.max(limits.minWidth, Math.min(limits.maxWidth, width));
  region.height = Math.max(limits.minHeight, Math.min(limits.maxHeight, height));
  region.x = Math.max(region.width * shortSide / (2 * canvas.width), Math.min(1 - region.width * shortSide / (2 * canvas.width), x));
  region.y = Math.max(region.height * shortSide / (2 * canvas.height), Math.min(1 - region.height * shortSide / (2 * canvas.height), y));
  setLayoutDirty(true);
}

function resizeHandleAt(point, region) {
  const handleRadius = Math.max(10 / canvas.width, 0.012);
  return Object.entries(resizeHandlePoints(region)).find(([, [x, y]]) => {
    const pointX = 1 - point.x;
    const handleX = 1 - x / canvas.width;
    return Math.hypot((pointX - handleX) * canvas.width, (point.y - y / canvas.height) * canvas.height)
      <= Math.max(12, canvas.width * handleRadius);
  })?.[0];
}

function beginZoneEdit(event) {
  if (editingBoundary) {
    beginBoundaryEdit(event);
    return;
  }
  if (!editingLayout) return;
  const point = pointFromPointer(event);
  const zone = kitLayout?.zones.find(item => item.id === selectedZoneId);
  const handle = zone ? resizeHandleAt(point, zone) : null;
  if (zone && handle) {
    dragging = { mode: 'resize', region: zone, handle, point, x: zone.x, y: zone.y, width: zone.width, height: zone.height };
    canvas.setPointerCapture(event.pointerId);
    requestRender();
    return;
  }
  const hit = zoneAt(point);
  // Clicking a selected overlap edits the selected region; the selector can
  // explicitly switch to another region occupying the same camera space.
  const selectedHit = zone && (() => {
    const dx = (point.x - zone.x) * canvas.width / (zone.width * Math.min(canvas.width, canvas.height) / 2);
    const dy = (point.y - zone.y) * canvas.height / (zone.height * Math.min(canvas.width, canvas.height) / 2);
    return dx * dx + dy * dy <= 1;
  })();
  const target = selectedHit ? zone : hit;
  if (!target) return;
  selectedZoneId = target.id;
  dragging = { mode: 'move', region: target };
  canvas.setPointerCapture(event.pointerId);
  updateZoneControls();
  requestRender();
}

function beginBoundaryEdit(event) {
  const boundary = kitLayout?.strikeBoundary;
  if (!boundary) return;
  const point = pointFromPointer(event);
  const handle = boundaryCornerAt(point, boundary);
  if (handle) dragging = { mode: 'resizeBoundary', corner: handle, startPoint: point, startBoundary: { ...boundary }, region: boundary };
  else if (isWithinBoundary(point, boundary)) dragging = { mode: 'moveBoundary', startPoint: point, startBoundary: { ...boundary }, region: boundary };
  else return;
  canvas.setPointerCapture(event.pointerId);
  requestRender();
}

function continueZoneEdit(event) {
  if ((!editingLayout && !editingBoundary) || !dragging) return;
  const point = pointFromPointer(event);
  if (dragging.mode === 'moveBoundary') {
    Object.assign(dragging.region, moveBoundary(
      dragging.startBoundary,
      point.x - dragging.startPoint.x,
      point.y - dragging.startPoint.y,
    ));
    setLayoutDirty(true);
    return;
  }
  if (dragging.mode === 'resizeBoundary') {
    Object.assign(dragging.region, resizeBoundary(
      dragging.startBoundary,
      dragging.corner,
      point.x - dragging.startPoint.x,
      point.y - dragging.startPoint.y,
    ));
    setLayoutDirty(true);
    return;
  }
  if (dragging.mode === 'move') {
    moveRegion(dragging.region, point);
    return;
  }
  const limits = dragging.region === kitLayout.strikeBoundary
    ? { minWidth: 0.08, maxWidth: 0.90, minHeight: 0.08, maxHeight: 0.70 }
    : { minWidth: 0.06, maxWidth: 0.70, minHeight: 0.06, maxHeight: 0.50 };
  resizeRegion(dragging.region, dragging, point, limits);
}

function boundaryCornerAt(point, boundary) {
  const pointX = 1 - point.x;
  const tolerance = Math.max(14, canvas.width * 0.022);
  const corners = {
    nw: [1 - boundary.left, boundary.top],
    ne: [1 - boundary.right, boundary.top],
    se: [1 - boundary.right, boundary.bottom],
    sw: [1 - boundary.left, boundary.bottom],
  };
  return Object.entries(corners).find(([, [x, y]]) =>
    Math.hypot((pointX - x) * canvas.width, (point.y - y) * canvas.height) <= tolerance,
  )?.[0];
}

function endZoneEdit(event) {
  if (dragging && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  if (dragging?.region && dragging.region !== kitLayout?.strikeBoundary) saveAppSettings();
  dragging = null;
}

function setEditingLayout(editing) {
  editingBoundary = false;
  boundaryEditButton.setAttribute('aria-pressed', 'false');
  boundaryEditButton.textContent = 'Edit hit rectangle';
  editingLayout = editing;
  view.classList.toggle('editing', editingLayout || editingBoundary);
  layoutEdit.setAttribute('aria-pressed', String(editing));
  layoutEdit.textContent = editing ? 'Done editing' : 'Edit layout';
  layoutHelp.textContent = editing
    ? 'Drag an oval to move it. Drag one of its eight bounding-box handles to resize it. Overlap is allowed.'
    : layoutDirty ? 'Unsaved kit changes. Save kit to keep these positions.' : 'Kit layout loaded.';
  updateZoneControls();
  requestRender();
}

function setEditingBoundary(editing) {
  editingLayout = false;
  layoutEdit.setAttribute('aria-pressed', 'false');
  layoutEdit.textContent = 'Edit layout';
  editingBoundary = editing;
  view.classList.toggle('editing', editingLayout || editingBoundary);
  boundaryEditButton.setAttribute('aria-pressed', String(editing));
  boundaryEditButton.textContent = editing ? 'Done editing rectangle' : 'Edit hit rectangle';
  layoutHelp.textContent = editing
    ? 'Drag inside the rectangle to move it. Drag a corner to resize it.'
    : layoutDirty ? 'Unsaved kit or hit-area changes. Save kit to keep them.' : 'Kit layout loaded.';
  requestRender();
}

async function saveKitLayout() {
  layoutSave.disabled = true;
  layoutHelp.textContent = 'Saving kit layout…';
  try {
    const response = await fetch('/api/kit-layout', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(kitLayout),
    });
    if (!response.ok) throw new Error(await response.text());
    kitLayout = await response.json();
    setLayoutDirty(false);
    updateZoneControls();
  } catch (error) {
    layoutSave.disabled = false;
    layoutHelp.textContent = `Could not save kit layout: ${error.message}`;
  }
}

function updateMetrics() {
  const details = currentInference ? `${currentInference.totalMs.toFixed(0)} ms` : '—';
  const input = currentInference ? `${currentInference.width}×${currentInference.height}` : '—';
  metrics.textContent = `Camera: ${cameraDescription} · Input: ${input} · Processed: ${processedFps.toFixed(1)} FPS · Inference: ${details} · Frame delay: ${displayDelayMs ? `${displayDelayMs.toFixed(0)} ms` : '—'}`;
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
  detectStrokes(message.pose, message.timestampMs);
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
  requestRender();

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
  for (const detector of strokeDetectors.values()) detector.reset();
  for (const predictor of targetPredictors.values()) predictor.reset();
  visibleHits.clear();
  totalHits = 0;
  ignoredHits = 0;
  outsideBoundaryHits = 0;
  outsidePadHits = 0;
  detectedStrokes = 0;
  lastPredictionComparison = '';
  strokeReadout.textContent = 'Strokes: L idle · R idle · Detected 0 · Hits 0 · Outside rectangle 0 · Outside pads 0';
  updatePredictionReadout();
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
      const bitmap = await createInferenceBitmap();
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

async function createInferenceBitmap() {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const maxWidth = Number(inferenceSizeSelect.value);
  const maxHeight = maxWidth * 0.75;
  const scale = Math.min(1, maxWidth / sourceWidth, maxHeight / sourceHeight);
  if (scale >= 0.999) return createImageBitmap(video);
  return createImageBitmap(video, {
    resizeWidth: Math.max(1, Math.round(sourceWidth * scale)),
    resizeHeight: Math.max(1, Math.round(sourceHeight * scale)),
    resizeQuality: 'low',
  });
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
  audioPlayer.enabled = audioEnabled.checked;
  if (audioPlayer.enabled) audioPlayer.activate();
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
    await openStrokeSession();
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
  if (activeCalibrationTrial) finishPromptedTrial('tracking_loss');
  closeStrokeSession();
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
  for (const detector of strokeDetectors.values()) detector.reset();
  for (const observer of candidateObservers.values()) observer.reset();
  for (const predictor of targetPredictors.values()) predictor.reset();
  lastPredictionComparison = '';
  visibleHits.clear();
  totalHits = 0;
  detectedStrokes = 0;
  ignoredHits = 0;
  outsideBoundaryHits = 0;
  outsidePadHits = 0;
  strokeReadout.textContent = 'Strokes: L idle · R idle · Detected 0 · Hits 0 · Outside rectangle 0 · Outside pads 0';
  updatePredictionReadout();
  processedFps = 0;
  displayDelayMs = 0;
  cameraDescription = '—';
  resultCount = 0;
  metrics.textContent = 'Camera: — · Input: — · Processed: — · Inference: — · Frame delay: —';
  toggle.textContent = 'Start camera';
  toggle.disabled = false;
  setStatus('Camera is off');
  lastDetectedStroke = null;
  renderCalibrationSummary();
  calibrationStatus.textContent = 'Start the camera to run prompted session calibration.';
}

toggle.addEventListener('click', () => media ? stop() : start());
promptTrialButton.addEventListener('click', armCalibrationTrial);
markMissedButton.addEventListener('click', () => finishPromptedTrial('miss'));
markTrackingLossButton.addEventListener('click', () => finishPromptedTrial('tracking_loss'));
cancelTrialButton.addEventListener('click', () => cancelCalibrationTrial());
markFalseHitButton.addEventListener('click', markLastStrokeFalseHit);
revertAdjustmentButton.addEventListener('click', revertAutomaticAdjustment);
delegateSelect.addEventListener('change', () => {
  saveAppSettings();
  if (worker && media) initializeWorker();
});
responseSlider.addEventListener('input', () => { responseValue.textContent = `${responseSlider.value}%`; saveAppSettings(); });
audioEnabled.addEventListener('change', () => {
  audioPlayer.enabled = audioEnabled.checked;
  if (audioPlayer.enabled) audioPlayer.activate();
  saveAppSettings();
});
audioVolumeSlider.addEventListener('input', () => {
  const volume = Number(audioVolumeSlider.value) / 100;
  audioPlayer.setVolume(volume);
  audioVolumeValue.textContent = `${audioVolumeSlider.value}%`;
  saveAppSettings();
});
for (const hand of ['left', 'right']) {
  for (const key of Object.keys(DETECTOR_DEFAULTS)) {
    detectorControls[hand][key].slider.addEventListener('input', () => updateDetectorSettings(hand));
  }
}
resetStrokeSettingsButton.addEventListener('click', resetDetectorSettings);
restrictHitsToZones.addEventListener('change', saveZoneRestriction);
autoAdjustToggle.addEventListener('change', saveAppSettings);
calibrationZoneSelect.addEventListener('change', saveAppSettings);
calibrationHandSelect.addEventListener('change', saveAppSettings);
loadVisualSettings();
initializeDetectorSettings();
markMissedButton.disabled = true;
markTrackingLossButton.disabled = true;
markFalseHitButton.disabled = true;
renderCalibrationSummary();
audioPlayer.enabled = audioEnabled.checked;
audioPlayer.setVolume(Number(audioVolumeSlider.value) / 100);
audioPlayer.prepare();
handExtensionSlider.addEventListener('input', () => {
  updateHandExtensionLabel();
  saveAppSettings();
});
showHandEstimate.addEventListener('change', () => {
  requestRender();
  saveAppSettings();
});
inferenceSizeSelect.addEventListener('change', () => {
  saveAppSettings();
});
layoutEdit.addEventListener('click', () => setEditingLayout(!editingLayout));
boundaryEditButton.addEventListener('click', () => setEditingBoundary(!editingBoundary));
layoutSave.addEventListener('click', saveKitLayout);
layoutReset.addEventListener('click', () => {
  if (!defaultLayout) return;
  kitLayout = cloneLayout(defaultLayout);
  selectedZoneId = kitLayout.zones[0]?.id || null;
  setEditingBoundary(false);
  updateZoneControls();
  saveAppSettings();
  setLayoutDirty(true);
});
zoneSelect.addEventListener('change', () => {
  selectedZoneId = zoneSelect.value;
  updateZoneControls();
  saveAppSettings();
  requestRender();
});
showRaw.addEventListener('change', () => { requestRender(); saveAppSettings(); });
canvas.addEventListener('pointerdown', beginZoneEdit);
canvas.addEventListener('pointermove', continueZoneEdit);
canvas.addEventListener('pointerup', endZoneEdit);
canvas.addEventListener('pointercancel', endZoneEdit);
loadKitLayout();
requestRender();
