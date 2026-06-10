// In-browser PP-OCRv5 (mobile) via onnxruntime-web.
//
// Everything runs client-side: det.onnx + rec.onnx are fetched once and cached
// by the browser; there is no OCR API round-trip.
//
// Exposes window.LocalOCR with a stable {width,height,items[]} result shape so
// the Vue overlay / copy / fine-tune UI can consume it directly.
import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs?v=1";

// Single-threaded wasm: avoids the SharedArrayBuffer / COOP+COEP requirement
// while still using SIMD. `proxy` moves session.run into a worker so heavy
// inference never blocks the UI thread.
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = true;

const MODELS = {
  det: new URL("./models/det.onnx", import.meta.url).href,
  rec: new URL("./models/rec.onnx", import.meta.url).href,
  cls: new URL("./models/cls.onnx", import.meta.url).href,
  keys: new URL("./models/ppocrv5_keys.txt", import.meta.url).href,
};

// --- detection: mirrors PaddleOCR DetResizeForTest + NormalizeImage ---
const DET_LIMIT_DEFAULT = 960; // resize long side to this (DetResizeForTest)
const DET_MEAN = [0.485, 0.456, 0.406]; // applied per channel to B,G,R order
const DET_STD = [0.229, 0.224, 0.225];
// DBPostProcess thresholds — PaddleOCR defaults. Geometry now matches the
// reference pipeline (min-area rect + polygon unclip), so the conservative
// values that compensated for the old axis-aligned approximation are gone.
const DET_THRESH = 0.3;
const DET_BOX_THRESH = 0.6;
const DET_UNCLIP = 1.5;
const DET_MIN_SIZE = 3; // reject boxes whose short side is below this (map px)

const REC_H = 48; // recognition input height (image_shape [3,48,*])
// Engine-level floor only rejects garbage; the UI filters further with a
// user-adjustable threshold so borderline-but-correct lines aren't lost.
const REC_SCORE_FLOOR = 0.3;

// --- text-line orientation classifier (optional, models/cls.onnx) ---
// PP-LCNet_x0_25_textline_ori: plain resize to 160x80, RGB + ImageNet
// normalization, softmax over ['0_degree','180_degree'] (per its inference.yml).
const CLS_W = 160;
const CLS_H = 80;
// PaddleX flips on bare argmax; a mild margin avoids flipping upright lines
// on coin-toss outputs while still catching genuinely inverted text.
const CLS_THRESH = 0.6;

let _sessions = null; // { det, rec, cls|null }
let _classes = null; // ['<blank>', ...18383 keys, ' ']  (len 18385)
let _initPromise = null;
let _onProgress = () => {};

// ---------- model loading ----------
async function fetchBuffer(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`下载失败 ${label}: HTTP ${resp.status}`);
  const total = +resp.headers.get("content-length") || 0;
  if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    _onProgress(`${label} ${(got / 1048576).toFixed(1)}/${(total / 1048576).toFixed(1)}MB`);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function init() {
  if (_initPromise) return _initPromise;
  _initPromise = (async () => {
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    _onProgress("下载检测模型…");
    const detBuf = await fetchBuffer(MODELS.det, "检测模型");
    _onProgress("下载识别模型…");
    const recBuf = await fetchBuffer(MODELS.rec, "识别模型");
    _onProgress("加载字典…");
    const keysTxt = await (await fetch(MODELS.keys)).text();
    const keys = keysTxt.replace(/\n$/, "").split("\n");
    _classes = ["<blank>", ...keys, " "]; // CTC: blank + dict + space
    _onProgress("初始化推理引擎…");
    const det = await ort.InferenceSession.create(detBuf, opts);
    const rec = await ort.InferenceSession.create(recBuf, opts);
    // Orientation classifier is optional: older deployments may not ship it.
    let cls = null;
    try {
      const clsBuf = await fetchBuffer(MODELS.cls, "方向分类模型");
      cls = await ort.InferenceSession.create(clsBuf, opts);
    } catch (_) {
      console.warn("cls.onnx 不可用，跳过文本方向分类（180° 翻转图片将识别失败）");
    }
    _sessions = { det, rec, cls };
    _onProgress("");
    return _sessions;
  })();
  return _initPromise;
}

function runSession(sess, tensor) {
  return sess.run({ [sess.inputNames[0]]: tensor });
}

// ---------- shared canvas scratch ----------
const _cv = document.createElement("canvas");
const _ctx = _cv.getContext("2d", { willReadFrequently: true });
function drawScaled(source, sx, sy, sw, sh, dw, dh) {
  _cv.width = dw;
  _cv.height = dh;
  _ctx.clearRect(0, 0, dw, dh);
  _ctx.drawImage(source, sx, sy, sw, sh, 0, 0, dw, dh);
  return _ctx.getImageData(0, 0, dw, dh).data; // RGBA
}

// dedicated scratch for rotated line crops (recPreprocess reuses _cv)
const _cropCv = document.createElement("canvas");
const _cropCtx = _cropCv.getContext("2d", { willReadFrequently: true });

// ---------- detection ----------
function detPreprocess(source, W, H, limit) {
  const ratio = limit / Math.max(W, H); // long side -> limit (up or down)
  const nw = Math.max(32, Math.round((W * ratio) / 32) * 32);
  const nh = Math.max(32, Math.round((H * ratio) / 32) * 32);
  const rgba = drawScaled(source, 0, 0, W, H, nw, nh);
  const data = new Float32Array(3 * nh * nw);
  const plane = nh * nw;
  for (let i = 0, p = 0; p < plane; p++, i += 4) {
    const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
    // channel order B,G,R with mean/std as in NormalizeImage
    data[p] = (b / 255 - DET_MEAN[0]) / DET_STD[0];
    data[plane + p] = (g / 255 - DET_MEAN[1]) / DET_STD[1];
    data[2 * plane + p] = (r / 255 - DET_MEAN[2]) / DET_STD[2];
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, nh, nw]), nw, nh };
}

// ---------- geometry ----------
// Andrew monotone chain convex hull. Input/output: [[x,y], ...].
function convexHull(points) {
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
function minAreaRect(hull) {
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
function orderCorners(rect) {
  const pts = rect.slice().sort((a, b) => a[0] - b[0]);
  const tl = pts[0][1] <= pts[1][1] ? pts[0] : pts[1];
  const bl = pts[0][1] <= pts[1][1] ? pts[1] : pts[0];
  const tr = pts[2][1] <= pts[3][1] ? pts[2] : pts[3];
  const br = pts[2][1] <= pts[3][1] ? pts[3] : pts[2];
  return [tl, tr, br, bl];
}

// DB post-process, mirroring PaddleOCR DBPostProcess: connected components on
// the binarized prob map -> convex hull -> min-area rect -> unclip expansion.
// Returns [{poly, box, score}] in ORIGINAL px; poly is [tl,tr,br,bl], box is
// the axis-aligned [x0,y0,x1,y1] envelope used by the fine-tune editor.
function dbPostprocess(prob, mh, mw, sx, sy, W, H) {
  const bin = new Uint8Array(mh * mw);
  for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > DET_THRESH ? 1 : 0;
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
    if (score < DET_BOX_THRESH) continue;
    const rect = orderCorners(minAreaRect(convexHull(pts)));
    const w = Math.hypot(rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]);
    const h = Math.hypot(rect[3][0] - rect[0][0], rect[3][1] - rect[0][1]);
    if (Math.min(w, h) < DET_MIN_SIZE) continue;
    // unclip: offset each side outward by area*ratio/perimeter (Vatti offset
    // of a rectangle == growing both half-extents by d)
    const d = (w * h * DET_UNCLIP) / (2 * (w + h));
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

// ---------- rotated line crop ----------
// Equivalent of PaddleOCR get_rotate_crop_image. det polys are parallelograms
// (axis-scaled rotated rects), so an affine map is exact: dest (0,0)<-tl,
// (w,0)<-tr, (0,h)<-bl. Vertical lines (h >= 1.5w) are rotated 90° CCW like
// np.rot90 so the recognizer reads them horizontally.
function cropPoly(source, poly) {
  const [p0, p1, p2, p3] = poly;
  const w = Math.max(1, Math.round(Math.max(
    Math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
    Math.hypot(p2[0] - p3[0], p2[1] - p3[1]),
  )));
  const h = Math.max(1, Math.round(Math.max(
    Math.hypot(p3[0] - p0[0], p3[1] - p0[1]),
    Math.hypot(p2[0] - p1[0], p2[1] - p1[1]),
  )));
  _cropCv.width = w;
  _cropCv.height = h;
  const mA = (p1[0] - p0[0]) / w, mB = (p1[1] - p0[1]) / w;
  const mC = (p3[0] - p0[0]) / h, mD = (p3[1] - p0[1]) / h;
  const det = mA * mD - mC * mB;
  if (!det) return null;
  _cropCtx.setTransform(
    mD / det, -mB / det, -mC / det, mA / det,
    (mC * p0[1] - mD * p0[0]) / det,
    (mB * p0[0] - mA * p0[1]) / det,
  );
  _cropCtx.drawImage(source, 0, 0);
  _cropCtx.setTransform(1, 0, 0, 1, 0, 0);
  if (h >= 1.5 * w) {
    const o = document.createElement("canvas");
    o.width = h;
    o.height = w;
    const ctx = o.getContext("2d");
    ctx.translate(0, w);
    ctx.rotate(-Math.PI / 2);
    ctx.drawImage(_cropCv, 0, 0);
    return o;
  }
  return _cropCv;
}

// ---------- orientation classifier ----------
function clsPreprocess(canvas) {
  const rgba = drawScaled(canvas, 0, 0, canvas.width, canvas.height, CLS_W, CLS_H);
  const data = new Float32Array(3 * CLS_H * CLS_W);
  const plane = CLS_H * CLS_W;
  for (let i = 0, p = 0; p < plane; p++, i += 4) {
    // RGB order with ImageNet stats (classification heads differ from det/rec)
    data[p] = (rgba[i] / 255 - DET_MEAN[0]) / DET_STD[0];
    data[plane + p] = (rgba[i + 1] / 255 - DET_MEAN[1]) / DET_STD[1];
    data[2 * plane + p] = (rgba[i + 2] / 255 - DET_MEAN[2]) / DET_STD[2];
  }
  return new ort.Tensor("float32", data, [1, 3, CLS_H, CLS_W]);
}

async function clsMaybeFlip(canvas) {
  if (!_sessions.cls) return canvas;
  const out = await runSession(_sessions.cls, clsPreprocess(canvas));
  const o = out[_sessions.cls.outputNames[0]].data; // softmax over ['0','180']
  if (o[1] > o[0] && o[1] >= CLS_THRESH) {
    const f = document.createElement("canvas");
    f.width = canvas.width;
    f.height = canvas.height;
    const ctx = f.getContext("2d");
    ctx.translate(f.width, f.height);
    ctx.rotate(Math.PI);
    ctx.drawImage(canvas, 0, 0);
    return f;
  }
  return canvas;
}

// ---------- recognition ----------
function recPreprocess(canvas) {
  const rw = Math.max(1, Math.round((REC_H * canvas.width) / canvas.height));
  const rgba = drawScaled(canvas, 0, 0, canvas.width, canvas.height, rw, REC_H);
  const data = new Float32Array(3 * REC_H * rw);
  const plane = REC_H * rw;
  for (let i = 0, p = 0; p < plane; p++, i += 4) {
    // BGR, normalized to [-1,1]:  (v/255 - 0.5) / 0.5
    data[p] = rgba[i + 2] / 127.5 - 1; // B
    data[plane + p] = rgba[i + 1] / 127.5 - 1; // G
    data[2 * plane + p] = rgba[i] / 127.5 - 1; // R
  }
  return new ort.Tensor("float32", data, [1, 3, REC_H, rw]);
}

function ctcDecode(probs, T, C) {
  let prev = -1;
  let text = "";
  let scoreSum = 0, scoreN = 0;
  for (let t = 0; t < T; t++) {
    const off = t * C;
    let best = 0, bestVal = probs[off];
    for (let c = 1; c < C; c++) {
      const v = probs[off + c];
      if (v > bestVal) { bestVal = v; best = c; }
    }
    if (best !== prev && best !== 0) {
      text += _classes[best];
      scoreSum += bestVal; // rec output is already softmax prob -> use directly
      scoreN++;
    }
    prev = best;
  }
  return { text, score: scoreN ? scoreSum / scoreN : 0 };
}

async function recognizeLine(source, poly) {
  const crop = cropPoly(source, poly);
  if (!crop || crop.width < 2 || crop.height < 2) return { text: "", score: 0 };
  const oriented = await clsMaybeFlip(crop);
  const tensor = recPreprocess(oriented);
  const out = await runSession(_sessions.rec, tensor);
  const o = out[_sessions.rec.outputNames[0]];
  const [, T, C] = o.dims; // [1, T, 18385]
  return ctcDecode(o.data, T, C);
}

// ---------- public API ----------
function rectPoly(b) {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
}

function sortReadingOrder(entries) {
  entries.sort(
    (a, b) => (Math.round(a.box[1] / 10) - Math.round(b.box[1] / 10)) || a.box[0] - b.box[0],
  );
}

// Detect + recognize a full image. `source` is an HTMLImageElement / canvas;
// W,H are its natural (original) pixel size.
async function recognizeFull(source, W, H, opts = {}) {
  await init();
  const detLimit = +opts.detLimit || DET_LIMIT_DEFAULT;
  const t0 = performance.now();
  const { tensor, nw, nh } = detPreprocess(source, W, H, detLimit);
  const t1 = performance.now();
  const detOut = await runSession(_sessions.det, tensor);
  const probT = detOut[_sessions.det.outputNames[0]];
  const [, , mh, mw] = probT.dims;
  const t2 = performance.now();
  const found = dbPostprocess(probT.data, mh, mw, W / nw, H / nh, W, H);
  sortReadingOrder(found);
  const tDb = performance.now(); // detection inference (t2-t1) | DB post-process (tDb-t2)
  const items = [];
  for (const f of found) {
    const { text, score } = await recognizeLine(source, f.poly);
    if (!text || score < REC_SCORE_FLOOR) continue;
    items.push({ text, score, poly: f.poly, box: f.box });
  }
  const t3 = performance.now();
  return {
    width: W,
    height: H,
    items,
    timings: {
      // non-overlapping buckets: preprocess + inference(det+cls+rec) + postprocess(DB) = total
      preprocess_ms: round1(t1 - t0),
      inference_ms: round1((t2 - t1) + (t3 - tDb)),
      postprocess_ms: round1(tDb - t2),
      client_total_ms: round1(t3 - t0),
    },
  };
}

// Recognize one pre-cropped region (fine-tune feature): det+rec inside it,
// join lines in reading order.
async function recognizeRegion(source, W, H) {
  await init();
  const t0 = performance.now();
  const { tensor, nw, nh } = detPreprocess(source, W, H, DET_LIMIT_DEFAULT);
  const detOut = await runSession(_sessions.det, tensor);
  const probT = detOut[_sessions.det.outputNames[0]];
  const [, , mh, mw] = probT.dims;
  let found = dbPostprocess(probT.data, mh, mw, W / nw, H / nh, W, H);
  // if detection finds nothing in a hand-drawn region, fall back to the whole box
  if (!found.length) found = [{ poly: rectPoly([0, 0, W, H]), box: [0, 0, W, H], score: 1 }];
  sortReadingOrder(found);
  const parts = [];
  const scores = [];
  for (const f of found) {
    const { text, score } = await recognizeLine(source, f.poly);
    if (text) { parts.push(text); scores.push(score); }
  }
  return {
    text: parts.join(" "),
    score: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null,
    timings: { region_ms: round1(performance.now() - t0) },
  };
}

const round1 = (x) => Math.round(x * 10) / 10;

window.LocalOCR = {
  init,
  recognizeFull,
  recognizeRegion,
  isReady: () => !!_sessions,
  onProgress: (cb) => { _onProgress = cb || (() => {}); },
};
