// DB (Differentiable Binarization) post-process, mirroring PaddleOCR's
// DBPostProcess: connected components on the binarized probability map ->
// convex hull -> min-area rect -> unclip expansion.
// Pure typed-array math — no DOM — so it can be unit-tested.
import { convexHull, minAreaRect, orderCorners } from "./geometry.js";

export const DB_DEFAULTS = {
  thresh: 0.3, // binarization threshold on the prob map
  boxThresh: 0.6, // min mean prob over a component to keep it
  unclip: 1.5, // polygon expansion ratio
  minSize: 3, // reject boxes whose short side is below this (map px)
};

// prob: Float32Array prob map [mh*mw]; sx/sy scale map px -> original px;
// W/H clamp to the original image. Returns [{poly, box, score}] where poly is
// [tl,tr,br,bl] in original px and box is its axis-aligned envelope (used by
// the fine-tune editor).
export function dbPostprocess(prob, mh, mw, sx, sy, W, H, opts = {}) {
  const { thresh, boxThresh, unclip, minSize } = { ...DB_DEFAULTS, ...opts };
  const bin = new Uint8Array(mh * mw);
  for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > thresh ? 1 : 0;
  const visited = new Uint8Array(mh * mw);
  const stack = new Int32Array(mh * mw);
  // per-row extremes of the current component; enough for its convex hull
  const rowMin = new Int32Array(mh).fill(-1);
  const rowMax = new Int32Array(mh);
  const rows = [];
  const out = [];
  for (let start = 0; start < bin.length; start++) {
    if (!bin[start] || visited[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    visited[start] = 1;
    let sum = 0, cnt = 0;
    rows.length = 0;
    while (sp > 0) {
      const idx = stack[--sp];
      const x = idx % mw, y = (idx - x) / mw;
      sum += prob[idx];
      cnt++;
      if (rowMin[y] < 0) { rowMin[y] = x; rowMax[y] = x; rows.push(y); }
      else if (x < rowMin[y]) rowMin[y] = x;
      else if (x > rowMax[y]) rowMax[y] = x;
      if (x + 1 < mw && bin[idx + 1] && !visited[idx + 1]) { visited[idx + 1] = 1; stack[sp++] = idx + 1; }
      if (x - 1 >= 0 && bin[idx - 1] && !visited[idx - 1]) { visited[idx - 1] = 1; stack[sp++] = idx - 1; }
      if (y + 1 < mh && bin[idx + mw] && !visited[idx + mw]) { visited[idx + mw] = 1; stack[sp++] = idx + mw; }
      if (y - 1 >= 0 && bin[idx - mw] && !visited[idx - mw]) { visited[idx - mw] = 1; stack[sp++] = idx - mw; }
    }
    const score = sum / cnt;
    const pts = [];
    for (const y of rows) {
      pts.push([rowMin[y], y], [rowMax[y], y]);
      rowMin[y] = -1;
    }
    if (score < boxThresh) continue;
    const rect = orderCorners(minAreaRect(convexHull(pts)));
    const w = Math.hypot(rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]);
    const h = Math.hypot(rect[3][0] - rect[0][0], rect[3][1] - rect[0][1]);
    if (Math.min(w, h) < minSize) continue;
    // unclip: offset each side outward by area*ratio/perimeter (Vatti offset
    // of a rectangle == growing both half-extents by d)
    const d = (w * h * unclip) / (2 * (w + h));
    const cx = (rect[0][0] + rect[2][0]) / 2;
    const cy = (rect[0][1] + rect[2][1]) / 2;
    const ux = (rect[1][0] - rect[0][0]) / w, uy = (rect[1][1] - rect[0][1]) / w;
    const vx = (rect[3][0] - rect[0][0]) / h, vy = (rect[3][1] - rect[0][1]) / h;
    const hw = w / 2 + d, hh = h / 2 + d;
    let poly = [
      [cx - ux * hw - vx * hh, cy - uy * hw - vy * hh],
      [cx + ux * hw - vx * hh, cy + uy * hw - vy * hh],
      [cx + ux * hw + vx * hh, cy + uy * hw + vy * hh],
      [cx - ux * hw + vx * hh, cy - uy * hw + vy * hh],
    ];
    // map-space -> original px, clamped to the image
    poly = poly.map(([x, y]) => [
      Math.max(0, Math.min(x * sx, W)),
      Math.max(0, Math.min(y * sy, H)),
    ]);
    const xs = poly.map((p) => p[0]);
    const ys = poly.map((p) => p[1]);
    out.push({
      poly,
      box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      score,
    });
  }
  return out;
}
