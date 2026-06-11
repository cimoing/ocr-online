// Pure 2D geometry used by the detection post-process and the editor.
// No DOM here — everything is unit-testable.

// Andrew monotone chain convex hull. Input/output: [[x,y], ...].
export function convexHull(points) {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length <= 2) return pts;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// Rotating calipers: smallest-area oriented rectangle enclosing the hull.
// Returns 4 corners (consistent winding, arbitrary start).
export function minAreaRect(hull) {
  if (hull.length === 1) return [hull[0], hull[0], hull[0], hull[0]];
  if (hull.length === 2) return [hull[0], hull[1], hull[1], hull[0]];
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i];
    const [x2, y2] = hull[(i + 1) % hull.length];
    const len = Math.hypot(x2 - x1, y2 - y1);
    if (!len) continue;
    const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    for (const [px, py] of hull) {
      const u = px * ux + py * uy;
      const v = py * ux - px * uy; // projection on the perpendicular (-uy, ux)
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const area = (maxU - minU) * (maxV - minV);
    if (!best || area < best.area) best = { area, ux, uy, minU, maxU, minV, maxV };
  }
  const { ux, uy, minU, maxU, minV, maxV } = best;
  const corner = (u, v) => [u * ux - v * uy, u * uy + v * ux];
  return [corner(minU, minV), corner(maxU, minV), corner(maxU, maxV), corner(minU, maxV)];
}

// PaddleOCR get_mini_boxes ordering: [tl, tr, br, bl], tl→tr = reading axis.
export function orderCorners(rect) {
  const pts = rect.slice().sort((a, b) => a[0] - b[0]);
  const tl = pts[0][1] <= pts[1][1] ? pts[0] : pts[1];
  const bl = pts[0][1] <= pts[1][1] ? pts[1] : pts[0];
  const tr = pts[2][1] <= pts[3][1] ? pts[2] : pts[3];
  const br = pts[2][1] <= pts[3][1] ? pts[3] : pts[2];
  return [tl, tr, br, bl];
}

// Axis-aligned box -> 4-point polygon [tl,tr,br,bl].
export function rectPoly(b) {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
}

// Sort {box:[x0,y0,..]} entries top-to-bottom (10px bands), then left-to-right.
export function sortReadingOrder(entries) {
  entries.sort(
    (a, b) => (Math.round(a.box[1] / 10) - Math.round(b.box[1] / 10)) || a.box[0] - b.box[0],
  );
}

// Normalize a dragged rectangle: any corner order in, clamped-to-image
// [x0,y0,x1,y1] with a minimum size out. Used by the fine-tune editor.
export function normRect(x0, y0, x1, y1, W, H, minBox) {
  let a = Math.min(x0, x1);
  let b = Math.min(y0, y1);
  let c = Math.max(x0, x1);
  let d = Math.max(y0, y1);
  a = Math.max(0, Math.min(a, W - minBox));
  b = Math.max(0, Math.min(b, H - minBox));
  c = Math.min(W, Math.max(c, a + minBox));
  d = Math.min(H, Math.max(d, b + minBox));
  return [a, b, c, d];
}
