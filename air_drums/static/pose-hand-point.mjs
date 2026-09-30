function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function visible(point, minimumVisibility = 0.35) {
  return Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y)
    && (point.visibility ?? 1) >= minimumVisibility);
}

// Estimate the hand end by extending the forearm direction by a fraction of
// shoulder width. The wrist remains the fallback when elbow direction is lost.
export function estimateHandPoint(pose, hand, shoulderWidthFraction = 0.35, aspectRatio = 4 / 3) {
  if (hand !== 'left' && hand !== 'right') return null;
  const wrist = pose?.[`${hand}Wrist`];
  if (!visible(wrist)) return null;

  const elbow = pose?.[`${hand}Elbow`];
  const leftShoulder = pose?.leftShoulder;
  const rightShoulder = pose?.rightShoulder;
  if (!visible(elbow, 0.18) || !visible(leftShoulder) || !visible(rightShoulder)) return wrist;

  const directionX = (wrist.x - elbow.x) * aspectRatio;
  const directionY = wrist.y - elbow.y;
  const directionLength = Math.hypot(directionX, directionY);
  const shoulderWidth = Math.hypot(
    (leftShoulder.x - rightShoulder.x) * aspectRatio,
    leftShoulder.y - rightShoulder.y,
  );
  if (directionLength < 0.005 || shoulderWidth < 0.02) return wrist;

  const extension = clamp(shoulderWidthFraction, 0, 0.7) * shoulderWidth;
  return {
    x: clamp(wrist.x + (directionX / directionLength) * extension / aspectRatio, 0, 1),
    y: clamp(wrist.y + (directionY / directionLength) * extension, 0, 1),
    z: wrist.z ?? 0,
    visibility: Math.min(wrist.visibility ?? 1, elbow.visibility ?? 1),
  };
}
