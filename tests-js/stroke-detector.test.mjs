import test from 'node:test';
import assert from 'node:assert/strict';
import { StrokeDetector, isWithinBoundary } from '../air_drums/static/stroke-detector.mjs';

function feed(detector, samples) {
  return samples.map(([time, y]) => detector.update({ x: 0.5, y, visibility: 1 }, time, 0.25, 4));
}

test('recognizes downswing, impact, and rebound and emits one hit', () => {
  const detector = new StrokeDetector('left');
  const results = feed(detector, [
    [0, 0.40], [33, 0.42], [66, 0.45], [99, 0.49],
    [132, 0.53], [165, 0.55], [198, 0.54], [265, 0.52], [298, 0.50], [331, 0.43],
  ]);
  assert.equal(results[1].state, 'idle');
  assert.equal(results[2].state, 'downswing');
  const hitIndex = results.findIndex(result => result.event);
  assert.ok(hitIndex >= 0);
  assert.equal(results[hitIndex].event.hand, 'left');
  assert.equal(results[hitIndex].state, 'impact');
  assert.equal(results.filter(result => result.event).length, 1);
  assert.equal(results.at(-1).state, 'idle');
});

test('registers a quick two-frame stroke including its first fast movement', () => {
  const detector = new StrokeDetector('right');
  const results = feed(detector, [[0, 0.40], [17, 0.47], [34, 0.42]]);
  const hit = results.find(result => result.event);
  assert.equal(hit?.event.hand, 'right');
  assert.ok(hit.event.peakSpeed > 10);
  assert.ok(hit.event.travel > 0.05);
});

test('ignores small motion and stationary landmark jitter', () => {
  const detector = new StrokeDetector('right');
  const results = feed(detector, [[0, 0.50], [33, 0.503], [66, 0.499], [99, 0.502], [132, 0.50]]);
  assert.equal(results.some(result => result.event), false);
  assert.equal(results.at(-1).state, 'idle');
});

test('rejects a single raw-point jump that immediately returns', () => {
  const detector = new StrokeDetector('right');
  const results = feed(detector, [[0, 0.40], [33, 0.43], [66, 0.40], [99, 0.40]]);
  assert.equal(results.some(result => result.event), false);
});

test('resets on lost wrist tracking instead of carrying stale motion forward', () => {
  const detector = new StrokeDetector('right');
  feed(detector, [[0, 0.40], [33, 0.42], [66, 0.45]]);
  assert.equal(detector.state, 'downswing');
  assert.equal(detector.update(null, 66, 0.25).state, 'idle');
});

test('requires a completed minimum-travel downswing before reporting impact', () => {
  const detector = new StrokeDetector('left');
  const results = feed(detector, [[0, 0.40], [33, 0.405], [66, 0.40], [99, 0.395]]);
  assert.equal(results.some(result => result.event), false);
});

test('accepts points inside an oval hit boundary and rejects points outside either axis', () => {
  const boundary = { x: 0.5, y: 0.55, width: 0.4, height: 0.2 };
  assert.equal(isWithinBoundary({ x: 0.5, y: 0.65 }, boundary, 640, 480), true);
  assert.equal(isWithinBoundary({ x: 0.5, y: 0.66 }, boundary, 640, 480), false);
  assert.equal(isWithinBoundary({ x: 0.95, y: 0.55 }, boundary, 640, 480), false);
});
