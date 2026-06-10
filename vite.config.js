import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// COOP/COEP make crossOriginIsolated true so onnxruntime-web can use
// multi-threaded wasm. Production (GitHub Pages) gets the same effect from
// static/coi-serviceworker.min.js since Pages can't set response headers.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: "/ocr-online/",
  plugins: [vue()],
  publicDir: "static",
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
