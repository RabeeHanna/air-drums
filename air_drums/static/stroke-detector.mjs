// Per-wrist, body-scale-normalized strike detector. All times are monotonic
// camera timestamps in milliseconds; it has no DOM or audio dependencies.
export function isWithinBoundary(point, boundary, width, height) {
  if (!point || !boundary || !(width > 0) || !(height > 0)) return false;
  const halfWidth = boundary.width * Math.min(width, height) / 2;
  const halfHeight = boundary.height * Math.min(width, height) / 2;
  const dx = (point.x - boundary.x) * width;
  const dy = (point.y - boundary.y) * height;
  return Math.abs(dx) <= halfWidth && Math.abs(dy) <= halfHeight;
}

export class StrokeDetector {
  constructor(hand, options = {}) {
    this.hand = hand;
    this.downThreshold = options.downThreshold ?? 0.55; // shoulder widths / second
    this.minTravel = options.minTravel ?? 0.05; // shoulder widths
    this.impactDrop = options.impactDrop ?? 0.62;
    this.state = 'idle';
    this.previous = null;
    this.pendingDown = null;
    this.resetStroke();
  }

  resetStroke() {
    this.startY = 0;
    this.startAt = 0;
    this.peakDownSpeed = 0;
    this.impactAt = 0;
    this.impactY = 0;
    this.travel = 0;
  }

  reset() {
    this.state = 'idle';
    this.previous = null;
    this.pendingDown = null;
    this.resetStroke();
  }

  update(point, timestampMs, bodyScale, aspectRatio = 1) {
    if (!point || (point.visibility ?? 1) < 0.35 || !Number.isFinite(bodyScale) || bodyScale <= 0) {
      this.reset();
      return { state: this.state, event: null, motion: null };
    }
    if (!this.previous) {
      this.previous = { x: point.x, y: point.y, timestampMs, bodyScale };
      return { state: this.state, event: null, motion: null };
    }

    const prior = this.previous;
    const dt = Math.max(0.001, Math.min(0.1, (timestampMs - prior.timestampMs) / 1000));
    const scale = Math.max(0.5 * bodyScale + 0.5 * this.previous.bodyScale, 0.04);
    const vx = (point.x - prior.x) * aspectRatio / dt / scale;
    const vy = (point.y - prior.y) / dt / scale;
    const speed = Math.hypot(vx, vy);
    this.previous = { x: point.x, y: point.y, timestampMs, bodyScale };

    let event = null;
    if (this.state === 'idle') {
      if (vy >= this.downThreshold && speed >= this.downThreshold) {
        if (this.pendingDown && timestampMs - this.pendingDown.lastAt <= 100) {
          this.pendingDown.count += 1;
          this.pendingDown.peak = Math.max(this.pendingDown.peak, vy);
          this.pendingDown.lastAt = timestampMs;
        } else {
          this.pendingDown = {
            startY: prior.y,
            startAt: prior.timestampMs,
            count: 1,
            peak: vy,
            lastAt: timestampMs,
          };
        }
        const segmentTravel = Math.max(0, (point.y - prior.y) / scale);
        // Two successive downward samples reject most raw landmark jitter.
        // A clearly fast segment is allowed through immediately for quick hits.
        if (this.pendingDown.count >= 2 || (vy >= 5.5 && segmentTravel >= 0.12)) {
          this.state = 'downswing';
          this.startY = this.pendingDown.startY;
          this.startAt = this.pendingDown.startAt;
          this.peakDownSpeed = this.pendingDown.peak;
          this.travel = Math.max(0, (point.y - this.startY) / scale);
          this.pendingDown = null;
        }
      } else {
        this.pendingDown = null;
      }
    } else if (this.state === 'downswing') {
      this.travel = Math.max(this.travel, (point.y - this.startY) / scale);
      this.peakDownSpeed = Math.max(this.peakDownSpeed, vy);
      const impactTurn = vy <= 0 || vy <= this.peakDownSpeed * this.impactDrop;
      if (this.travel >= this.minTravel && impactTurn && this.peakDownSpeed >= this.downThreshold) {
        this.state = 'impact';
        this.impactAt = timestampMs;
        this.impactY = point.y;
        event = {
          hand: this.hand,
          timestamp: new Date().toISOString(),
          cameraTimestampMs: timestampMs,
          durationMs: Math.max(0, Math.round(timestampMs - this.startAt)),
          peakSpeed: Number(this.peakDownSpeed.toFixed(3)),
          travel: Number(this.travel.toFixed(3)),
          x: point.x,
          y: point.y,
        };
      }
    } else if (this.state === 'impact') {
      if (vy < -0.25 || timestampMs - this.impactAt >= 45) {
        this.state = 'rebound';
        if (vy < -0.25 && (this.impactY - point.y) / scale >= this.minTravel * 0.35) {
          this.state = 'idle';
          this.resetStroke();
        }
      }
    } else if (this.state === 'rebound') {
      if ((vy < -0.25 && (this.impactY - point.y) / scale >= this.minTravel * 0.35)
          || timestampMs - this.impactAt >= 350) {
        this.state = 'idle';
        this.resetStroke();
      }
    }

    return {
      state: this.state,
      event,
      motion: { vx: Number(vx.toFixed(2)), vy: Number(vy.toFixed(2)), speed: Number(speed.toFixed(2)), travel: Number(this.travel.toFixed(3)) },
    };
  }
}
