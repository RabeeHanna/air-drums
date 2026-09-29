// DOM-independent short-horizon predictions from raw wrist samples.
const SAMPLE_WINDOW = 5;
const FIXED_HORIZON_MS = 120;
const MAX_TURN_HORIZON_MS = 500;

function linearSlope(samples, valueFor, timeOrigin) {
  const xs = samples.map(sample => (sample.timestampMs - timeOrigin) / 1000);
  const ys = samples.map(valueFor);
  const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
  const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;
  const denominator = xs.reduce((sum, value) => sum + (value - meanX) ** 2, 0);
  if (denominator <= 0) return 0;
  return xs.reduce((sum, value, index) => sum + (value - meanX) * (ys[index] - meanY), 0) / denominator;
}

export class TargetPredictor {
  constructor() {
    this.samples = [];
    this.predictions = null;
  }

  reset() {
    this.samples = [];
    this.predictions = null;
  }

  update(point, timestampMs, bodyScale, aspectRatio, options = {}) {
    if (!point || (point.visibility ?? 1) < 0.35 || !Number.isFinite(bodyScale) || bodyScale <= 0) {
      this.reset();
      return null;
    }
    this.samples.push({
      x: point.x,
      y: point.y,
      timestampMs,
      bodyScale,
    });
    if (this.samples.length > SAMPLE_WINDOW) this.samples.shift();

    if (!options.active) {
      this.predictions = null;
      return null;
    }
    if (this.samples.length < 2) {
      this.predictions = null;
      return null;
    }

    const samples = this.samples;
    const origin = samples[0].timestampMs;
    const latest = samples.at(-1);
    const scale = Math.max(samples.reduce((sum, sample) => sum + sample.bodyScale, 0) / samples.length, 0.04);
    const vx = linearSlope(samples, sample => sample.x * aspectRatio, origin) / scale;
    const vy = linearSlope(samples, sample => sample.y, origin) / scale;
    const fixedSeconds = FIXED_HORIZON_MS / 1000;
    const fixed = {
      x: latest.x + vx * fixedSeconds * scale / aspectRatio,
      y: latest.y + vy * fixedSeconds * scale,
      timeToImpactMs: FIXED_HORIZON_MS,
    };

    let deceleration = null;
    if (samples.length >= 3 && Number.isFinite(options.peakDownSpeed) && options.peakDownSpeed > 0) {
      const velocities = [];
      for (let index = 1; index < samples.length; index += 1) {
        const before = samples[index - 1];
        const after = samples[index];
        const dt = (after.timestampMs - before.timestampMs) / 1000;
        if (dt <= 0) continue;
        velocities.push({
          timestampMs: (before.timestampMs + after.timestampMs) / 2,
          value: (after.y - before.y) / dt / Math.max((before.bodyScale + after.bodyScale) / 2, 0.04),
        });
      }
      if (velocities.length >= 2) {
        const acceleration = linearSlope(velocities, sample => sample.value, velocities[0].timestampMs);
        const threshold = options.peakDownSpeed * (options.impactDrop ?? 0.62);
        const seconds = (threshold - vy) / acceleration;
        if (acceleration < -0.05 && seconds > 0 && seconds <= MAX_TURN_HORIZON_MS / 1000) {
          deceleration = {
            x: latest.x + vx * seconds * scale / aspectRatio,
            y: latest.y + (vy * seconds + 0.5 * acceleration * seconds ** 2) * scale,
            timeToImpactMs: seconds * 1000,
          };
        }
      }
    }

    this.predictions = { fixed, deceleration, timestampMs };
    return this.predictions;
  }
}

export function comparePredictions(predictions, impact, width, height, zones) {
  if (!predictions) return null;
  const nearest = point => zones.map(zone => {
    const dx = (point.x - zone.x) * width / (zone.width * Math.min(width, height) / 2);
    const dy = (point.y - zone.y) * height / (zone.height * Math.min(width, height) / 2);
    return { zone, distance: dx * dx + dy * dy };
  }).sort((a, b) => a.distance - b.distance)[0]?.zone ?? null;

  const compare = prediction => {
    if (!prediction || !Number.isFinite(prediction.x) || !Number.isFinite(prediction.y)) return null;
    const zone = nearest(prediction);
    return {
      x: prediction.x,
      y: prediction.y,
      timeToImpactMs: prediction.timeToImpactMs,
      predictedImpactTimestampMs: predictions.timestampMs + prediction.timeToImpactMs,
      zoneId: zone?.id ?? null,
      positionErrorPx: Math.hypot((prediction.x - impact.x) * width, (prediction.y - impact.y) * height),
      timingErrorMs: prediction.timeToImpactMs - (impact.cameraTimestampMs - predictions.timestampMs),
    };
  };

  return { fixed: compare(predictions.fixed), deceleration: compare(predictions.deceleration) };
}
