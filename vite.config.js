import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// COOP/COEP make crossOriginIsolated true so onnxruntime-web can use
// multi-threaded wasm; CORP mirrors what coi-serviceworker injects so every
// resource stays loadable from worker contexts under require-corp.
// Only `vite preview` (built dist) sends them: production GitHub Pages gets
// the same effect from static/coi-serviceworker.min.js, while plain dev stays
// header-free — embedded/automation browsers that enforce COEP worker-script
// blocking without ever granting isolation would otherwise break inference,
// and dev gains nothing from threads anyway.
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "cross-origin",
};

export default defineConfig({
  base: "/ocr-online/",
  plugins: [vue()],
  publicDir: "static",
  preview: { headers: isolationHeaders },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
