export function isWithinBoundary(point, boundary) {
  return Boolean(point && boundary
    && point.x >= boundary.left && point.x <= boundary.right
    && point.y >= boundary.top && point.y <= boundary.bottom);
}

export function moveBoundary(boundary, dx, dy) {
  const width = boundary.right - boundary.left;
  const height = boundary.bottom - boundary.top;
  const left = clamp(boundary.left + dx, 0, 1 - width);
  const top = clamp(boundary.top + dy, 0, 1 - height);
  return { left, top, right: left + width, bottom: top + height };
}

export function resizeBoundary(boundary, corner, dx, dy, minimumSize = 0.08) {
  const next = { ...boundary };
  const minX = corner.includes('w') ? Math.max(0, boundary.right - 0.96) : boundary.left + minimumSize;
  const maxX = corner.includes('w') ? boundary.right - minimumSize : Math.min(1, boundary.left + 0.96);
  const minY = corner.includes('n') ? Math.max(0, boundary.bottom - 0.96) : boundary.top + minimumSize;
  const maxY = corner.includes('n') ? boundary.bottom - minimumSize : Math.min(1, boundary.top + 0.96);
  if (corner.includes('w')) next.left = clamp(boundary.left + dx, minX, maxX);
  if (corner.includes('e')) next.right = clamp(boundary.right + dx, minX, maxX);
  if (corner.includes('n')) next.top = clamp(boundary.top + dy, minY, maxY);
  if (corner.includes('s')) next.bottom = clamp(boundary.bottom + dy, minY, maxY);
  return next;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}
