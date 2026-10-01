// Small, low-cost wrist motion observer for prompted calibration only.
// It records plausible downward gestures without changing strike detection.
export class CandidateObserver {
  constructor() {
    this.reset();
  }

  reset() {
    this.previous = null;
    this.active = false;
    this.startY = 0;
    this.startAt = 0;
    this.peakSpeed = 0;
    this.travel = 0;
  }

  update(point, timestampMs, bodyScale, aspectRatio = 1, speedFloor = 0.15) {
    if (!point || (point.visibility ?? 1) < 0.35 || !Number.isFinite(bodyScale) || bodyScale <= 0) {
      this.reset();
      return null;
    }
    if (!this.previous) {
      this.previous = { y: point.y, x: point.x, timestampMs, bodyScale };
      return null;
    }
    const prior = this.previous;
    const dt = Math.max(0.001, Math.min(0.1, (timestampMs - prior.timestampMs) / 1000));
    const scale = Math.max(0.5 * bodyScale + 0.5 * prior.bodyScale, 0.04);
    const vy = (point.y - prior.y) / dt / scale;
    const vx = (point.x - prior.x) * aspectRatio / dt / scale;
    const speed = Math.hypot(vx, vy);
    this.previous = { y: point.y, x: point.x, timestampMs, bodyScale };

    if (!this.active && vy >= speedFloor) {
      this.active = true;
      this.startY = prior.y;
      this.startAt = prior.timestampMs;
      this.peakSpeed = vy;
      this.travel = Math.max(0, (point.y - prior.y) / scale);
      return null;
    }
    if (!this.active) return null;

    this.peakSpeed = Math.max(this.peakSpeed, vy);
    this.travel = Math.max(this.travel, (point.y - this.startY) / scale);
    if (vy <= 0 || timestampMs - this.startAt >= 450) {
      const candidate = this.travel >= 0.012 && this.peakSpeed >= 0.15
        ? { peakSpeed: Number(this.peakSpeed.toFixed(3)), travel: Number(this.travel.toFixed(3)) }
        : null;
      this.active = false;
      this.startY = 0;
      this.startAt = 0;
      this.peakSpeed = 0;
      this.travel = 0;
      return candidate;
    }
    return null;
  }
}
