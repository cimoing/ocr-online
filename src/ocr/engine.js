// In-browser PP-OCRv5 (mobile) via onnxruntime-web.
//
// Everything runs client-side: det/rec/cls ONNX models are fetched once,
// persisted in Cache Storage, and executed in wasm; there is no OCR API
// round-trip. Pure math lives in geometry.js / db.js / ctc.js so it can be
// unit-tested without a DOM; this module owns canvases, ort and model I/O.
import { ctcDecode } from "./ctc.js";
import { dbPostprocess } from "./db.js";
import { rectPoly, sortReadingOrder } from "./geometry.js";

const BASE = import.meta.env.BASE_URL;
// ort stays a runtime (vendored) dependency rather than a bundled one: the
// wasm bundle resolves its proxy worker + .wasm relative to its own URL, so
// loading it from vendor/ort/ keeps that machinery self-contained. The URL is
// built at runtime so vite's import analysis leaves the dynamic import native
// (public-dir files must not enter the module graph). `bust` forces a fresh
// module instance — ort caches a failed backend init per instance, so the
// proxy-less retry needs a new one.
const ortUrl = (bust) =>
  new URL(`${BASE}vendor/ort/ort.wasm.bundle.min.mjs?${bust}`, location.href).href;

const MODELS = {
  det: `${BASE}models/det.onnx`,
  rec: `${BASE}models/rec.onnx`,
  cls: `${BASE}models/cls.onnx`,
  keys: `${BASE}models/ppocrv5_keys.txt`,
};

// --- detection: mirrors PaddleOCR DetResizeForTest + NormalizeImage ---
const DET_LIMIT_DEFAULT = 960; // resize long side to this (DetResizeForTest)
const DET_MEAN = [0.485, 0.456, 0.406]; // applied per channel to B,G,R order
const DET_STD = [0.229, 0.224, 0.225];

const REC_H = 48; // recognition input height (image_shape [3,48,*])
const REC_BATCH = 8; // crops per rec run, sorted by width and zero-padded
// Engine-level floor only rejects garbage; the UI filters further with a
// user-adjustable threshold so borderline-but-correct lines aren't lost.
const REC_SCORE_FLOOR = 0.3;

// --- text-line orientation classifier (optional, models/cls.onnx) ---
// PP-LCNet_x0_25_textline_ori: plain resize to 160x80, RGB + ImageNet
// normalization, softmax over ['0_degree','180_degree'] (per its inference.yml).
const CLS_W = 160;
const CLS_H = 80;
const CLS_BATCH = 8;
// PaddleX flips on bare argmax; a mild margin avoids flipping upright lines
// on coin-toss outputs while still catching genuinely inverted text.
const CLS_THRESH = 0.6;

let _ort = null;
let _sessions = null; // { det, rec, cls|null }
let _classes = null; // ['<blank>', ...18383 keys, ' ']  (len 18385)
let _initPromise = null;
let _onProgress = () => {};

// ---------- model loading ----------
// Bump when any entry in MODELS (or the ort wasm) changes content, otherwise
// returning visitors keep using the previously cached bytes forever.
const ASSET_CACHE = "ppocr-assets-v1";

async function openAssetCache() {
  try {
    if (typeof caches === "undefined") return null;
    return await caches.open(ASSET_CACHE);
  } catch (_) {
    return null; // non-secure context / storage disabled
  }
}

async function fetchBuffer(url, label) {
  const cache = await openAssetCache();
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        _onProgress(`${label}（本地缓存）`);
        return new Uint8Array(await hit.arrayBuffer());
      }
    } catch (_) {}
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`下载失败 ${label}: HTTP ${resp.status}`);
  const total = +resp.headers.get("content-length") || 0;
  let out;
  if (!resp.body || !total) {
    out = new Uint8Array(await resp.arrayBuffer());
  } else {
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
    out = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
  }
  if (cache) {
    try { await cache.put(url, new Response(out)); } catch (_) {}
  }
  return out;
}

// ort's proxy spawns `new Worker(<bundle url>, {type:"module"})`. Probe that
// exact capability with the small wasm glue module first: embedded/locked-down
// browsers that only allow blob workers fire `error` almost immediately, and
// booting with proxy enabled there would hang the whole init.
function probeModuleWorker() {
  return new Promise((resolve) => {
    let w;
    const done = (ok) => {
      try { if (w) w.terminate(); } catch (_) {}
      resolve(ok);
    };
    try {
      w = new Worker(
        new URL(`${BASE}vendor/ort/ort-wasm-simd-threaded.mjs`, location.href),
        { type: "module" },
      );
      w.addEventListener("error", () => done(false));
      setTimeout(() => done(true), 1200); // no quick error -> workers usable
    } catch (_) {
      done(false);
    }
  });
}

const withTimeout = (promise, ms, label) =>
  Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(label)), ms)),
  ]);

// Load a fresh ort module instance and build all sessions on it.
// `proxy` moves session.run into a worker so heavy inference never blocks the
// UI thread. Threads need SharedArrayBuffer, i.e. crossOriginIsolated — true
// behind `vite preview` headers and on Pages via coi-serviceworker; when
// isolation is missing we degrade to single-threaded SIMD wasm.
async function boot(proxy, bust, bufs) {
  const ort = await import(/* @vite-ignore */ ortUrl(bust));
  ort.env.wasm.wasmPaths = new URL(`${BASE}vendor/ort/`, location.href).href;
  ort.env.wasm.numThreads =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated
      ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1))
      : 1;
  ort.env.wasm.proxy = proxy;
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const det = await ort.InferenceSession.create(bufs.det, opts);
  const rec = await ort.InferenceSession.create(bufs.rec, opts);
  // Orientation classifier is optional: older deployments may not ship it.
  let cls = null;
  if (bufs.cls) {
    try {
      cls = await ort.InferenceSession.create(bufs.cls, opts);
    } catch (_) {
      console.warn("cls.onnx 加载失败，跳过文本方向分类（180° 翻转图片将识别失败）");
    }
  }
  _ort = ort;
  _sessions = { det, rec, cls };
}

export function init() {
  if (_initPromise) return _initPromise;
  _initPromise = (async () => {
    _onProgress("下载检测模型…");
    const det = await fetchBuffer(MODELS.det, "检测模型");
    _onProgress("下载识别模型…");
    const rec = await fetchBuffer(MODELS.rec, "识别模型");
    _onProgress("加载字典…");
    const keysBuf = await fetchBuffer(MODELS.keys, "字典");
    const keys = new TextDecoder().decode(keysBuf).replace(/\n$/, "").split("\n");
    _classes = ["<blank>", ...keys, " "]; // CTC: blank + dict + space
    let cls = null;
    try {
      cls = await fetchBuffer(MODELS.cls, "方向分类模型");
    } catch (_) {
      console.warn("cls.onnx 不可用，跳过文本方向分类");
    }
    const bufs = { det, rec, cls };
    _onProgress("初始化推理引擎…");
    const proxy = await probeModuleWorker();
    try {
      if (!proxy) throw new Error("module worker 探测失败");
      const proxyBoot = boot(true, "v=1", bufs);
      proxyBoot.catch(() => {}); // raced loser must not surface as unhandled
      await withTimeout(proxyBoot, 20000, "ort proxy 初始化超时");
    } catch (err) {
      // Inference still works on the main thread (UI freezes during runs) —
      // strictly better than failing outright.
      console.warn("ort proxy worker 不可用，降级为主线程推理：", err && err.message ? err.message : err);
      await boot(false, "v=1&noproxy", bufs);
    }
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
  return _ctx.getImageData(0, 0, dw, dh).data; // RGBA (fresh copy)
}

// dedicated scratch for rotated line crops (drawScaled reuses _cv)
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
  return { tensor: new _ort.Tensor("float32", data, [1, 3, nh, nw]), nw, nh };
}

async function runDet(source, W, H, limit) {
  const { tensor, nw, nh } = detPreprocess(source, W, H, limit);
  const detOut = await runSession(_sessions.det, tensor);
  const probT = detOut[_sessions.det.outputNames[0]];
  const [, , mh, mw] = probT.dims;
  return { probT, mh, mw, nw, nh };
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

// ---------- batched line recognition (cls -> rec) ----------
// 180° rotation of an RGBA buffer == reversing its pixel order; doing it on
// the already-resized rec input avoids keeping per-line crop canvases alive.
function flip180(rgba) {
  for (let a = 0, b = rgba.length - 4; a < b; a += 4, b -= 4) {
    for (let k = 0; k < 4; k++) {
      const t = rgba[a + k];
      rgba[a + k] = rgba[b + k];
      rgba[b + k] = t;
    }
  }
}

// Run cls on every line (chunked); flag lines the classifier wants flipped.
async function clsFlags(lines) {
  const flags = new Array(lines.length).fill(false);
  if (!_sessions.cls) return flags;
  const todo = lines.map((l, i) => (l ? i : -1)).filter((i) => i >= 0);
  const plane = CLS_H * CLS_W;
  for (let off = 0; off < todo.length; off += CLS_BATCH) {
    const chunk = todo.slice(off, off + CLS_BATCH);
    const run = async (idxs) => {
      const n = idxs.length;
      const data = new Float32Array(n * 3 * plane);
      idxs.forEach((lineIdx, j) => {
        const rgba = lines[lineIdx].clsRgba;
        const base = j * 3 * plane;
        for (let i = 0, p = 0; p < plane; p++, i += 4) {
          // RGB order with ImageNet stats (classification heads differ from det/rec)
          data[base + p] = (rgba[i] / 255 - DET_MEAN[0]) / DET_STD[0];
          data[base + plane + p] = (rgba[i + 1] / 255 - DET_MEAN[1]) / DET_STD[1];
          data[base + 2 * plane + p] = (rgba[i + 2] / 255 - DET_MEAN[2]) / DET_STD[2];
        }
      });
      const out = await runSession(_sessions.cls, new _ort.Tensor("float32", data, [n, 3, CLS_H, CLS_W]));
      const o = out[_sessions.cls.outputNames[0]].data; // [n,2] softmax ['0','180']
      idxs.forEach((lineIdx, j) => {
        const p0 = o[j * 2], p180 = o[j * 2 + 1];
        if (p180 > p0 && p180 >= CLS_THRESH) flags[lineIdx] = true;
      });
    };
    try {
      await run(chunk);
    } catch (err) {
      if (chunk.length === 1) throw err;
      for (const i of chunk) await run([i]); // model may reject batches
    }
  }
  return flags;
}

// Recognize every poly of `source` in one pass: crop+resize all lines, batch
// the orientation classifier, then batch rec with width-sorted zero-padded
// chunks (PaddleOCR's rec_batch_num strategy). Returns results aligned with
// `polys`; failed/degenerate crops yield {text:'', score:0}.
async function recognizeLines(source, polys) {
  const results = polys.map(() => ({ text: "", score: 0 }));
  const lines = polys.map((poly) => {
    const crop = cropPoly(source, poly);
    if (!crop || crop.width < 2 || crop.height < 2) return null;
    const rw = Math.max(1, Math.round((REC_H * crop.width) / crop.height));
    return {
      rw,
      recRgba: drawScaled(crop, 0, 0, crop.width, crop.height, rw, REC_H),
      clsRgba: _sessions.cls
        ? drawScaled(crop, 0, 0, crop.width, crop.height, CLS_W, CLS_H)
        : null,
    };
  });

  const flags = await clsFlags(lines);
  flags.forEach((f, i) => { if (f) flip180(lines[i].recRgba); });

  const order = lines
    .map((l, i) => (l ? i : -1))
    .filter((i) => i >= 0)
    .sort((a, b) => lines[a].rw - lines[b].rw);
  let done = 0;
  for (let off = 0; off < order.length; off += REC_BATCH) {
    const chunk = order.slice(off, off + REC_BATCH);
    const run = async (idxs) => {
      const maxW = Math.max(...idxs.map((i) => lines[i].rw));
      const plane = REC_H * maxW;
      const data = new Float32Array(idxs.length * 3 * plane); // zeros = padding
      idxs.forEach((lineIdx, j) => {
        const { rw, recRgba } = lines[lineIdx];
        const base = j * 3 * plane;
        for (let y = 0; y < REC_H; y++) {
          for (let x = 0; x < rw; x++) {
            const i = (y * rw + x) * 4;
            const p = y * maxW + x;
            // BGR, normalized to [-1,1]:  (v/255 - 0.5) / 0.5
            data[base + p] = recRgba[i + 2] / 127.5 - 1;
            data[base + plane + p] = recRgba[i + 1] / 127.5 - 1;
            data[base + 2 * plane + p] = recRgba[i] / 127.5 - 1;
          }
        }
      });
      const out = await runSession(
        _sessions.rec,
        new _ort.Tensor("float32", data, [idxs.length, 3, REC_H, maxW]),
      );
      const o = out[_sessions.rec.outputNames[0]];
      const [, T, C] = o.dims; // [n, T, 18385]
      idxs.forEach((lineIdx, j) => {
        results[lineIdx] = ctcDecode(o.data, j * T * C, T, C, _classes);
      });
    };
    try {
      await run(chunk);
    } catch (err) {
      if (chunk.length === 1) throw err;
      for (const i of chunk) await run([i]); // model may reject batches
    }
    done += chunk.length;
    _onProgress(`识别 ${done}/${order.length} 行…`);
  }
  _onProgress("");
  return results;
}

// ---------- public API ----------
// Detect + recognize a full image. `source` is an HTMLImageElement / canvas;
// W,H are its natural (original) pixel size.
export async function recognizeFull(source, W, H, opts = {}) {
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
  const texts = await recognizeLines(source, found.map((f) => f.poly));
  const items = [];
  found.forEach((f, i) => {
    const { text, score } = texts[i];
    if (!text || score < REC_SCORE_FLOOR) return;
    items.push({ text, score, poly: f.poly, box: f.box });
  });
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
export async function recognizeRegion(source, W, H) {
  await init();
  const t0 = performance.now();
  const { probT, mh, mw, nw, nh } = await runDet(source, W, H, DET_LIMIT_DEFAULT);
  let found = dbPostprocess(probT.data, mh, mw, W / nw, H / nh, W, H);
  // if detection finds nothing in a hand-drawn region, fall back to the whole box
  if (!found.length) found = [{ poly: rectPoly([0, 0, W, H]), box: [0, 0, W, H], score: 1 }];
  sortReadingOrder(found);
  const texts = await recognizeLines(source, found.map((f) => f.poly));
  const parts = texts.filter((t) => t.text);
  return {
    text: parts.map((t) => t.text).join(" "),
    score: parts.length ? parts.reduce((a, t) => a + t.score, 0) / parts.length : null,
    timings: { region_ms: round1(performance.now() - t0) },
  };
}

const round1 = (x) => Math.round(x * 10) / 10;

export function isReady() {
  return !!_sessions;
}

export function onProgress(cb) {
  _onProgress = cb || (() => {});
}
