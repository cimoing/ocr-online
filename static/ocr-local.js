// In-browser PP-OCRv5 (mobile) via onnxruntime-web.
//
// Everything runs client-side: det.onnx + rec.onnx are fetched once and cached
// by the browser; there is no OCR API round-trip.
//
// Exposes window.LocalOCR with a stable {width,height,items[]} result shape so
// the Vue overlay / copy / fine-tune UI can consume it directly.
import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs?v=1";

// Single-threaded wasm: avoids the SharedArrayBuffer / COOP+COEP requirement
// while still using SIMD. (Threads are a P3 optimization behind cross-origin
// isolation headers.)
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1;

const MODELS = {
  det: new URL("./models/det.onnx", import.meta.url).href,
  rec: new URL("./models/rec.onnx", import.meta.url).href,
  keys: new URL("./models/ppocrv5_keys.txt", import.meta.url).href,
};

// --- detection: must mirror app/ocr_service.py + inference.yml exactly ---
const DET_LIMIT = 960; // resize long side to this (DetResizeForTest)
const DET_MEAN = [0.485, 0.456, 0.406]; // applied per channel to B,G,R order
const DET_STD = [0.229, 0.224, 0.225];
// DBPostProcess thresholds — the app's tuned values (precision on clean print)
const DET_THRESH = 0.4;
const DET_BOX_THRESH = 0.7;
const DET_UNCLIP = 1.2;
const DET_MIN_SIZE = 3;

const REC_H = 48; // recognition input height (image_shape [3,48,*])
const MIN_REC_SCORE = 0.85; // drop low-confidence reads (matches recognize())

let _sessions = null; // { det, rec }
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
    _sessions = { det, rec };
    _onProgress("");
    return _sessions;
  })();
  return _initPromise;
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

// ---------- detection ----------
function detPreprocess(source, W, H) {
  const ratio = DET_LIMIT / Math.max(W, H); // long side -> 960 (up or down)
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

// Axis-aligned DB post-process via BFS flood-fill (mirrors db_postprocess in
// scripts/onnx_verify.py). Returns boxes [x0,y0,x1,y1,score] in ORIGINAL px.
function dbPostprocess(prob, mh, mw, sx, sy) {
  const bin = new Uint8Array(mh * mw);
  for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > DET_THRESH ? 1 : 0;
  const visited = new Uint8Array(mh * mw);
  const stack = new Int32Array(mh * mw);
  const boxes = [];
  for (let start = 0; start < bin.length; start++) {
    if (!bin[start] || visited[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    visited[start] = 1;
    let minx = mw, maxx = 0, miny = mh, maxy = 0, sum = 0, cnt = 0;
    while (sp > 0) {
      const idx = stack[--sp];
      const x = idx % mw, y = (idx - x) / mw;
      sum += prob[idx]; cnt++;
      if (x < minx) minx = x;
      if (x > maxx) maxx = x;
      if (y < miny) miny = y;
      if (y > maxy) maxy = y;
      if (x + 1 < mw && bin[idx + 1] && !visited[idx + 1]) { visited[idx + 1] = 1; stack[sp++] = idx + 1; }
      if (x - 1 >= 0 && bin[idx - 1] && !visited[idx - 1]) { visited[idx - 1] = 1; stack[sp++] = idx - 1; }
      if (y + 1 < mh && bin[idx + mw] && !visited[idx + mw]) { visited[idx + mw] = 1; stack[sp++] = idx + mw; }
      if (y - 1 >= 0 && bin[idx - mw] && !visited[idx - mw]) { visited[idx - mw] = 1; stack[sp++] = idx - mw; }
    }
    if (cnt < DET_MIN_SIZE) continue;
    const score = sum / cnt;
    if (score < DET_BOX_THRESH) continue;
    const x0 = minx, y0 = miny, x1 = maxx + 1, y1 = maxy + 1;
    const w = x1 - x0, h = y1 - y0;
    const per = 2 * (w + h);
    const d = per ? (w * h * DET_UNCLIP) / per : 0; // unclip dilation
    boxes.push([(x0 - d) * sx, (y0 - d) * sy, (x1 + d) * sx, (y1 + d) * sy, score]);
  }
  return boxes;
}

// ---------- recognition ----------
function recPreprocess(source, sx, sy, sw, sh) {
  const rw = Math.max(1, Math.round((REC_H * sw) / sh));
  const rgba = drawScaled(source, sx, sy, sw, sh, rw, REC_H);
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

async function recognizeLine(source, box) {
  const [x0, y0, x1, y1] = box;
  const sw = x1 - x0, sh = y1 - y0;
  if (sw < 2 || sh < 2) return { text: "", score: 0 };
  const tensor = recPreprocess(source, x0, y0, sw, sh);
  const out = await _sessions.rec.run({ x: tensor });
  const o = out[_sessions.rec.outputNames[0]];
  const [, T, C] = o.dims; // [1, T, 18385]
  return ctcDecode(o.data, T, C);
}

// ---------- public API ----------
function rectPoly(b) {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
}

// Detect + recognize a full image. `source` is an HTMLImageElement / canvas;
// W,H are its natural (original) pixel size.
async function recognizeFull(source, W, H) {
  await init();
  const t0 = performance.now();
  const { tensor, nw, nh } = detPreprocess(source, W, H);
  const t1 = performance.now();
  const detOut = await _sessions.det.run({ x: tensor });
  const probT = detOut[_sessions.det.outputNames[0]];
  const [, , mh, mw] = probT.dims;
  const t2 = performance.now();
  const boxes = dbPostprocess(probT.data, mh, mw, W / nw, H / nh);
  // clamp to image and sort in reading order (top, then left)
  for (const b of boxes) {
    b[0] = Math.max(0, Math.min(b[0], W));
    b[1] = Math.max(0, Math.min(b[1], H));
    b[2] = Math.max(0, Math.min(b[2], W));
    b[3] = Math.max(0, Math.min(b[3], H));
  }
  boxes.sort((a, b) => (Math.round(a[1] / 10) - Math.round(b[1] / 10)) || a[0] - b[0]);
  const tDb = performance.now(); // detection inference (t2-t1) | DB post-process (tDb-t2)
  const items = [];
  for (const b of boxes) {
    const { text, score } = await recognizeLine(source, b);
    if (!text || score < MIN_REC_SCORE) continue;
    const box = [b[0], b[1], b[2], b[3]];
    items.push({ text, score, poly: rectPoly(box), box });
  }
  const t3 = performance.now();
  return {
    width: W,
    height: H,
    items,
    timings: {
      // non-overlapping buckets: preprocess + inference(det+rec) + postprocess(DB) = total
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
  const { tensor, nw, nh } = detPreprocess(source, W, H);
  const detOut = await _sessions.det.run({ x: tensor });
  const probT = detOut[_sessions.det.outputNames[0]];
  const [, , mh, mw] = probT.dims;
  let boxes = dbPostprocess(probT.data, mh, mw, W / nw, H / nh);
  // if detection finds nothing in a hand-drawn region, fall back to the whole box
  if (!boxes.length) boxes = [[0, 0, W, H, 1]];
  boxes.sort((a, b) => (Math.round(a[1] / 10) - Math.round(b[1] / 10)) || a[0] - b[0]);
  const parts = [];
  const scores = [];
  for (const b of boxes) {
    const { text, score } = await recognizeLine(source, b);
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
