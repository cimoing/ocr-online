import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ppu =
  "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/";
const ort = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

const assets = {
  "static/models/det.onnx": [
    ppu + "detection/PP-OCRv5_mobile_det_infer.onnx",
    "d7fe3ea74652890722c0f4d02458b7261d9f5ae6c92904d05707c9eb155c7924",
  ],
  "static/models/rec.onnx": [
    ppu + "recognition/PP-OCRv5_mobile_rec_infer.onnx",
    "d253c3cbee6e507828a5271a30ab0ec8ae7c2a99d0cc8e6f844fe380809d22b3",
  ],
  "static/models/cls.onnx": [
    ppu + "correction/PP-LCNet_x0_25_textline_ori.onnx",
    "44dd0033f5215447fdce5f9333883c155806aebbb6bd00964d4a9b20e05d44b9",
  ],
  "static/vendor/ort/ort-wasm-simd-threaded.wasm": [
    ort + "ort-wasm-simd-threaded.wasm",
    "040d52ce5066707a10d45cb9500c35e70a9c2fb33c4fb63428da9ae45b956b97",
  ],
  "static/vendor/ort/ort.wasm.bundle.min.mjs": [
    ort + "ort.wasm.bundle.min.mjs",
    "1e3491b976ffaf231ac5bd48fb17516ce31d5f1a47c67a63b6bbb868655483bd",
  ],
  "static/vendor/ort/ort-wasm-simd-threaded.mjs": [
    ort + "ort-wasm-simd-threaded.mjs",
    "2de262ca1fe2d6e0ef9236bf77632fa01de232ce7b6a33071c37637ee53f4669",
  ],
};

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function readIfValid(path, want) {
  if (!existsSync(path)) return false;
  return sha256(await readFile(path)) === want;
}

let ok = true;
for (const [rel, [url, want]] of Object.entries(assets)) {
  const dst = resolve(root, rel);
  if (await readIfValid(dst, want)) {
    console.log(`[skip] ${rel} (present, sha256 ok)`);
    continue;
  }

  console.log(`[get ] ${rel} <- ${url}`);
  await mkdir(dirname(dst), { recursive: true });
  try {
    const response = await fetch(url, { headers: { "User-Agent": "node" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const got = sha256(bytes);
    if (got !== want) {
      throw new Error(`SHA-256 mismatch: got ${got}, expected ${want}`);
    }
    await writeFile(dst, bytes);
    console.log(`       saved ${bytes.length} bytes, sha256 ok`);
  } catch (err) {
    ok = false;
    console.error(`       FAILED: ${err.message || err}`);
  }
}

console.log(ok ? "\nAll assets ready." : "\nSome assets failed. See above.");
process.exit(ok ? 0 : 1);
