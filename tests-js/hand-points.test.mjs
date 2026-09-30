import test from 'node:test';
import assert from 'node:assert/strict';
import { mapIndexTips } from '../air_drums/static/hand-points.mjs';

function handAt(x, tipX) {
  const landmarks = Array.from({ length: 21 }, () => ({ x, y: 0.5, z: 0 }));
  landmarks[8] = { x: tipX, y: 0.25, z: -0.1 };
  return landmarks;
}

test('maps actual hand index-finger tips to the matching pose wrists', () => {
  const leftTip = { x: 0.2, y: 0.25, z: -0.1, visibility: 0.94 };
  const rightTip = { x: 0.8, y: 0.25, z: -0.1, visibility: 0.88 };
  const result = mapIndexTips({
    landmarks: [handAt(0.15, leftTip.x), handAt(0.85, rightTip.x)],
    handedness: [[{ score: 0.94 }], [{ score: 0.88 }]],
  }, {
    leftWrist: { x: 0.16, y: 0.5, visibility: 0.9 },
    rightWrist: { x: 0.84, y: 0.5, visibility: 0.9 },
  });

  assert.deepEqual(result.leftIndex, leftTip);
  assert.deepEqual(result.rightIndex, rightTip);
});

test('matches hands by wrist proximity when detector ordering is reversed', () => {
  const result = mapIndexTips({
    landmarks: [handAt(0.82, 0.8), handAt(0.18, 0.2)],
    handedness: [[{ score: 0.9 }], [{ score: 0.9 }]],
  }, {
    leftWrist: { x: 0.17, y: 0.5, visibility: 0.9 },
    rightWrist: { x: 0.83, y: 0.5, visibility: 0.9 },
  });

  assert.equal(result.leftIndex.x, 0.2);
  assert.equal(result.rightIndex.x, 0.8);
});
