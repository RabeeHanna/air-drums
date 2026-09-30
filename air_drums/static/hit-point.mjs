// Hand Landmarker tips are attached to the corresponding Pose wrists by the worker.
export function indexFingertip(pose, hand, minimumVisibility = 0.35) {
  if (hand !== 'left' && hand !== 'right') return null;
  const point = pose?.[`${hand}Index`];
  if (!point || (point.visibility ?? 1) < minimumVisibility) return null;
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return null;
  return point;
}
