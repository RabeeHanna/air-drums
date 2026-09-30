import test from 'node:test';
import assert from 'node:assert/strict';
import { indexFingertip } from '../air_drums/static/hit-point.mjs';

test('uses a visible index fingertip as the hit point', () => {
  const tip = { x: 0.42, y: 0.73, visibility: 0.9 };
  assert.equal(indexFingertip({ leftIndex: tip }, 'left'), tip);
});

test('rejects an occluded, missing, malformed, or out-of-frame fingertip', () => {
  assert.equal(indexFingertip({ leftIndex: { x: 0.4, y: 0.7, visibility: 0.2 } }, 'left'), null);
  assert.equal(indexFingertip({}, 'right'), null);
  assert.equal(indexFingertip({ rightIndex: { x: NaN, y: 0.7 } }, 'right'), null);
  assert.equal(indexFingertip({ rightIndex: { x: 1.2, y: 0.7 } }, 'right'), null);
});
