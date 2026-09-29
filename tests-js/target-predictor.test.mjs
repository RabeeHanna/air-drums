import test from 'node:test';
import assert from 'node:assert/strict';
import { comparePredictions, TargetPredictor } from '../air_drums/static/target-predictor.mjs';

function sample(predictor, time, y, options = {}) {
  return predictor.update({ x: 0.5, y, visibility: 1 }, time, 0.25, 4, {
    active: true,
    peakDownSpeed: 8,
    impactDrop: 0.62,
    ...options,
  });
}

test('fixed-horizon estimate projects a straight downswing by 120 ms', () => {
  const predictor = new TargetPredictor();
  sample(predictor, 0, 0.3);
  const result = sample(predictor, 33, 0.333);
  assert.equal(result.fixed.timeToImpactMs, 120);
  assert.ok(result.fixed.y > 0.333);
  assert.equal(result.deceleration, null);
});

test('deceleration model estimates the impact-turn threshold from recent samples', () => {
  const predictor = new TargetPredictor();
  const speeds = [6, 5, 4, 3, 2];
  let y = 0.2;
  let result = null;
  for (let index = 0; index < speeds.length; index += 1) {
    if (index > 0) y += speeds[index - 1] * 0.25 * 0.033;
    result = sample(predictor, index * 33, y, { peakDownSpeed: 5.5 });
  }
  assert.ok(result.deceleration);
  assert.ok(result.deceleration.timeToImpactMs > 0);
  assert.ok(result.deceleration.timeToImpactMs <= 500);
});

test('noisy samples remain bounded by the five-sample window and invalid tracking clears estimates', () => {
  const predictor = new TargetPredictor();
  const ys = [0.3, 0.315, 0.329, 0.346, 0.359, 0.376, 0.39];
  let result;
  ys.forEach((y, index) => { result = sample(predictor, index * 33, y); });
  assert.equal(predictor.samples.length, 5);
  assert.ok(Number.isFinite(result.fixed.x));
  assert.ok(Number.isFinite(result.fixed.y));
  assert.equal(predictor.update(null, 231, 0.25, 4), null);
  assert.equal(predictor.predictions, null);
  assert.equal(predictor.samples.length, 0);
});

test('inactive movement clears the displayed estimate before a later downswing', () => {
  const predictor = new TargetPredictor();
  sample(predictor, 0, 0.3);
  sample(predictor, 33, 0.32);
  assert.ok(predictor.predictions);
  predictor.update({ x: 0.5, y: 0.32, visibility: 1 }, 66, 0.25, 4, { active: false });
  assert.equal(predictor.predictions, null);
});

test('prediction comparison picks nearest overlapping zone and keeps raw mirrored camera coordinates', () => {
  const zones = [
    { id: 'snare', label: 'Snare', x: 0.5, y: 0.6, width: 0.3, height: 0.2 },
    { id: 'hi_hat', label: 'Hi-hat', x: 0.75, y: 0.6, width: 0.3, height: 0.2 },
  ];
  const result = comparePredictions({
    timestampMs: 100,
    fixed: { x: 0.75, y: 0.6, timeToImpactMs: 120 },
    deceleration: null,
  }, { x: 0.7, y: 0.58, cameraTimestampMs: 180 }, 640, 480, zones);
  assert.equal(result.fixed.zoneId, 'hi_hat');
  assert.ok(result.fixed.positionErrorPx > 0);
  assert.equal(result.fixed.timingErrorMs, 40);
  assert.equal(result.deceleration, null);
});

test('a point outside all zones still resolves to the closest zone', () => {
  const zones = [
    { id: 'snare', x: 0.5, y: 0.5, width: 0.2, height: 0.2 },
    { id: 'ride', x: 0.9, y: 0.9, width: 0.2, height: 0.2 },
  ];
  const result = comparePredictions({
    timestampMs: 0,
    fixed: { x: 0.05, y: 0.05, timeToImpactMs: 120 },
    deceleration: null,
  }, { x: 0.05, y: 0.05, cameraTimestampMs: 120 }, 640, 480, zones);
  assert.equal(result.fixed.zoneId, 'snare');
});
