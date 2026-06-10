<template>
  <header class="topbar">
    <div class="brand">
      <span class="logo">OCR</span>
      <div>
        <h1>PP-OCRv5 浏览器 OCR</h1>
        <p class="sub">拖入 / 粘贴 / 选择图片 · Vue 静态前端本地推理</p>
      </div>
    </div>
    <div class="controls">
      <span class="mode-pill" title="det + rec 全流程在浏览器本地运行，不调用服务端识别接口">本地·浏览器</span>
      <select
        v-model.number="detLimit"
        class="det-limit"
        :disabled="busy"
        title="检测分辨率：更高对小字/大图更准，但更慢"
      >
        <option :value="960">标准 960</option>
        <option :value="1280">高清 1280</option>
        <option :value="1536">超清 1536</option>
      </select>
      <button class="btn primary" type="button" :disabled="busy || !currentBlob" @click="recognize">识别</button>
      <button class="btn" type="button" :disabled="!hasItems" @click="copyAll">复制全部</button>
      <button
        class="btn"
        type="button"
        :disabled="!hasItems"
        title="按住时隐藏识别文字、显示原图，松开恢复，用于对比"
        @mousedown="startPeek"
        @touchstart.prevent="startPeek"
      >
        按住对比原图
      </button>
      <button
        class="btn"
        type="button"
        :disabled="!current"
        :class="{ active: editMode }"
        :aria-pressed="editMode ? 'true' : 'false'"
        title="微调文本区域：拖拽手柄改大小 · 拖空白处画新框 · ✕ 或 Delete 删除 · 改动后自动重新识别该区域"
        @click="toggleEdit"
      >
        微调区域
      </button>
      <button class="btn ghost" type="button" :disabled="!currentBlob" @click="clearAll">清空</button>
    </div>
  </header>

  <main
    class="stage"
    :class="{ dragover }"
    @click="stagePick"
    @dragenter.prevent="onDragEnter"
    @dragover.prevent="onDragEnter"
    @dragleave="onDragLeave"
    @drop.prevent="onDrop"
  >
    <div v-if="!currentBlob" class="empty">
      <div class="empty-inner">
        <div class="empty-icon">IMG</div>
        <p class="empty-title">拖入图片到此处</p>
        <p class="empty-hint">
          或 粘贴 (Ctrl+V) · 或
          <button class="link" type="button" @click.stop="pickFile">点击选择文件</button>
        </p>
      </div>
    </div>

    <div v-show="currentBlob" class="canvas">
      <img ref="imgEl" :src="currentURL" alt="待识别图片" @load="onImageLoad" />
      <div
        ref="overlay"
        class="overlay"
        :class="{ peek, edit: editMode }"
        :style="{ '--mask': String(mask / 100) }"
        @click="onOverlayClick"
        @mousedown="onOverlayMouseDown"
        @mouseover="onOverlayMouseOver"
        @mousemove="onOverlayMouseMove"
        @mouseout="onOverlayMouseOut"
      >
        <div
          v-for="item in visibleItems"
          :key="item.id"
          class="ocr-box"
          :class="{ selected: item.id === selectedId, recognizing: item.id === recognizingId }"
          :data-id="item.id"
          :data-score="item.score ?? undefined"
          :data-text="item.text || ''"
          :style="boxStyle(item)"
        >
          <span class="txt">{{ item.text }}</span>
          <template v-if="editMode">
            <button class="del-btn" type="button" title="删除该区域">×</button>
            <div
              v-for="dir in handleDirs"
              :key="dir"
              class="handle"
              :class="'h-' + dir"
              :data-dir="dir"
            ></div>
          </template>
        </div>
      </div>
    </div>
  </main>

  <footer class="statusbar">
    <div class="bar-row">
      <span class="status" :class="{ err: statusError }">{{ status }}</span>
      <span v-if="hiddenCount" class="hidden-note" title="低于置信过滤阈值的行已隐藏，调低阈值可显示">
        已隐藏 {{ hiddenCount }} 行低置信
      </span>
      <label class="mask-ctl" title="低于该置信度的识别行将被隐藏（不参与复制/导出）">
        置信过滤 {{ minScorePct }}%
        <input type="range" min="0" max="100" v-model.number="minScorePct" />
      </label>
      <label class="mask-ctl" title="调节文字遮罩不透明度：高=用识别文字盖住原图，低=透出原图对照">
        文字遮罩
        <input type="range" min="0" max="100" v-model.number="mask" />
      </label>
    </div>
    <div v-if="timingChips.length" class="timings">
      <span v-for="chip in timingChips" :key="chip.key" class="chip" :class="chip.cls">
        <i>{{ chip.key }}</i>{{ chip.value }}
      </span>
    </div>
  </footer>

  <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileChange" />
  <div v-show="toastVisible" class="toast" :class="{ show: toastVisible }">{{ toastText }}</div>
  <div v-show="busy" class="spinner-overlay">
    <div class="spinner"></div>
    <span>{{ spinnerText }}</span>
  </div>
  <div
    v-show="scoreTip.visible"
    ref="scoreTipEl"
    class="ocr-tip"
    :style="{ left: scoreTip.left + 'px', top: scoreTip.top + 'px' }"
  >
    {{ scoreTip.text }}
  </div>
</template>

<script>
import { nextTick } from "vue";

const HANDLE_DIRS = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const MIN_BOX = 4;
const DRAG_THRESHOLD = 3;

function rectPoly(b) {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
}

function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

export default {
  name: "App",
  data() {
    return {
      currentBlob: null,
      currentURL: "",
      current: null,
      busy: false,
      status: "就绪 — 等待图片",
      statusError: false,
      spinnerText: "识别中…",
      toastText: "",
      toastVisible: false,
      toastTimer: null,
      mask: 90,
      minScorePct: 60,
      detLimit: 960,
      timingChips: [],
      editMode: false,
      selectedId: null,
      nextId: 1,
      dragging: false,
      dragover: false,
      peek: false,
      recognizingId: null,
      scoreTip: { visible: false, text: "", left: 0, top: 0 },
      handleDirs: HANDLE_DIRS,
      ocrModulePromise: null,
      resizeTimer: null,
    };
  },
  computed: {
    items() {
      return this.current?.items || [];
    },
    // User-drawn / re-recognized boxes (pinned) always show; the rest must
    // clear the confidence slider. Hidden lines are excluded from copy/export.
    visibleItems() {
      const min = this.minScorePct / 100;
      return this.items.filter((it) => it.pinned || it.score == null || it.score >= min);
    },
    hiddenCount() {
      return this.items.length - this.visibleItems.length;
    },
    hasItems() {
      return this.visibleItems.length > 0;
    },
  },
  updated() {
    this.fitText();
  },
  mounted() {
    window.addEventListener("paste", this.onPaste);
    window.addEventListener("resize", this.onResize);
    window.addEventListener("mouseup", this.endPeek);
    window.addEventListener("touchend", this.endPeek);
    window.addEventListener("blur", this.endPeek);
    window.addEventListener("keydown", this.onKeyDown);
  },
  beforeUnmount() {
    window.removeEventListener("paste", this.onPaste);
    window.removeEventListener("resize", this.onResize);
    window.removeEventListener("mouseup", this.endPeek);
    window.removeEventListener("touchend", this.endPeek);
    window.removeEventListener("blur", this.endPeek);
    window.removeEventListener("keydown", this.onKeyDown);
    if (this.currentURL) URL.revokeObjectURL(this.currentURL);
  },
  methods: {
    setStatus(msg, isErr = false) {
      this.status = msg;
      this.statusError = isErr;
    },
    showToast(msg) {
      this.toastText = msg;
      this.toastVisible = true;
      clearTimeout(this.toastTimer);
      this.toastTimer = setTimeout(() => {
        this.toastVisible = false;
      }, 1400);
    },
    pickFile() {
      this.$refs.fileInput.click();
    },
    stagePick() {
      if (!this.currentBlob) this.pickFile();
    },
    onFileChange(e) {
      const file = e.target.files?.[0];
      if (file) this.loadImage(file);
      e.target.value = "";
    },
    loadImage(blob) {
      if (!blob?.type?.startsWith("image/")) {
        this.showToast("不是图片文件");
        return;
      }
      if (this.currentURL) URL.revokeObjectURL(this.currentURL);
      this.currentBlob = blob;
      this.currentURL = URL.createObjectURL(blob);
      this.current = null;
      this.timingChips = [];
      this.editMode = false;
      this.selectedId = null;
      this.peek = false;
    },
    onImageLoad() {
      const img = this.$refs.imgEl;
      this.current = { width: img.naturalWidth, height: img.naturalHeight, items: [] };
      this.setStatus(`已载入 ${img.naturalWidth}×${img.naturalHeight} · 点击「识别」`);
      this.fitText();
    },
    clearAll() {
      if (this.currentURL) URL.revokeObjectURL(this.currentURL);
      this.currentBlob = null;
      this.currentURL = "";
      this.current = null;
      this.busy = false;
      this.editMode = false;
      this.selectedId = null;
      this.timingChips = [];
      this.peek = false;
      this.hideTip();
      this.setStatus("就绪 — 等待图片");
    },
    async loadLocalOCR() {
      if (!this.ocrModulePromise) {
        this.ocrModulePromise = new Promise((resolve, reject) => {
          if (window.LocalOCR) {
            resolve();
            return;
          }

          const existing = document.querySelector('script[data-ocr-local="true"]');
          if (existing) {
            existing.addEventListener("load", resolve, { once: true });
            existing.addEventListener("error", () => reject(new Error("本地推理模块加载失败")), { once: true });
            return;
          }

          const script = document.createElement("script");
          script.type = "module";
          script.src = `${import.meta.env.BASE_URL}ocr-local.js?v=4`;
          script.dataset.ocrLocal = "true";
          script.addEventListener("load", resolve, { once: true });
          script.addEventListener("error", () => reject(new Error("本地推理模块加载失败")), { once: true });
          document.head.appendChild(script);
        });
      }
      await this.ocrModulePromise;
      if (!window.LocalOCR) throw new Error("本地推理模块未加载，请刷新页面");
      return window.LocalOCR;
    },
    ensureItemIds(items) {
      for (const item of items) {
        if (item.id == null) item.id = this.nextId++;
      }
    },
    async recognize() {
      if (!this.currentBlob || this.busy) return;
      const img = this.$refs.imgEl;
      this.busy = true;
      this.spinnerText = "识别中…（本地·浏览器）";
      this.setStatus("识别中… 本地模型首次加载可能较慢");
      const tStart = performance.now();
      try {
        const localOCR = await this.loadLocalOCR();
        localOCR.onProgress((s) => {
          if (s) this.setStatus(`本地模型 · ${s}`);
        });
        const data = await localOCR.recognizeFull(img, img.naturalWidth, img.naturalHeight, {
          detLimit: this.detLimit,
        });
        data.model = "local";
        data.count = data.items.length;
        this.ensureItemIds(data.items);
        this.current = data;
        this.selectedId = null;
        await nextTick();
        const tDone = performance.now();
        this.setStatus(`本地·浏览器 · ${data.count} 行 · ${data.width}×${data.height}`);
        this.renderTimings(data.timings, tDone - tStart);
      } catch (err) {
        this.setStatus(`识别失败：${err.message || err}`, true);
        this.showToast("识别失败");
      } finally {
        this.busy = false;
      }
    },
    renderTimings(t, endToEnd) {
      t = t || {};
      this.timingChips = [
        { key: "预处理", value: fmtMs(t.preprocess_ms), cls: "" },
        { key: "推理", value: fmtMs(t.inference_ms), cls: "hot" },
        { key: "后处理", value: fmtMs(t.postprocess_ms), cls: "" },
        { key: "总计", value: fmtMs(endToEnd), cls: "total" },
      ];
    },
    displayScale() {
      const img = this.$refs.imgEl;
      return this.current?.width && img?.clientWidth ? img.clientWidth / this.current.width : 1;
    },
    boxGeom(item) {
      const scale = this.displayScale();
      if (this.editMode || !item.poly) {
        const b = item.box;
        if (!b) return null;
        const w = (b[2] - b[0]) * scale;
        const h = (b[3] - b[1]) * scale;
        if (w < 1 || h < 1) return null;
        return { left: b[0] * scale, top: b[1] * scale, w, h, angle: 0 };
      }
      const [p0, p1, , p3] = item.poly;
      const w = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * scale;
      const h = Math.hypot(p3[0] - p0[0], p3[1] - p0[1]) * scale;
      if (w < 1 || h < 1) return null;
      return {
        left: p0[0] * scale,
        top: p0[1] * scale,
        w,
        h,
        angle: Math.atan2(p1[1] - p0[1], p1[0] - p0[0]),
      };
    },
    boxStyle(item) {
      const g = this.boxGeom(item);
      if (!g) return { display: "none" };
      return {
        left: `${g.left}px`,
        top: `${g.top}px`,
        width: `${g.w}px`,
        height: `${g.h}px`,
        transform: g.angle ? `rotate(${g.angle}rad)` : "",
        "--font-size": `${g.h * 0.86}px`,
      };
    },
    fitText() {
      nextTick(() => {
        const overlay = this.$refs.overlay;
        if (!overlay) return;
        for (const box of overlay.querySelectorAll(".ocr-box")) {
          const span = box.querySelector(".txt");
          if (!span) continue;
          span.style.transform = "";
          const natW = span.scrollWidth;
          if (natW > 0) span.style.transform = `scaleX(${(box.clientWidth / natW).toFixed(4)})`;
        }
      });
    },
    async copyText(text) {
      try {
        await navigator.clipboard.writeText(text);
      } catch (_) {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
        } catch (__) {}
        document.body.removeChild(ta);
      }
    },
    async copyAll() {
      if (!this.hasItems) return;
      await this.copyText(this.visibleItems.map((i) => i.text).join("\n"));
      this.showToast(`已复制 ${this.visibleItems.length} 行`);
    },
    onOverlayClick(e) {
      if (this.editMode) {
        const del = e.target.closest(".del-btn");
        if (del) {
          const box = del.closest(".ocr-box");
          if (box) this.deleteBox(box.dataset.id);
        }
        return;
      }
      const box = e.target.closest(".ocr-box");
      if (!box) return;
      const sel = window.getSelection();
      if (sel && String(sel).length > 0) return;
      this.copyText(box.dataset.text || "");
      this.showToast("已复制该行");
    },
    moveTip(x, y) {
      const tip = this.$refs.scoreTipEl;
      const pad = 12;
      let left = x + pad;
      let top = y + pad;
      const width = tip?.offsetWidth || 120;
      const height = tip?.offsetHeight || 28;
      if (left + width > window.innerWidth - 4) left = x - pad - width;
      if (top + height > window.innerHeight - 4) top = y - pad - height;
      this.scoreTip.left = left;
      this.scoreTip.top = top;
    },
    hideTip() {
      this.scoreTip.visible = false;
    },
    onOverlayMouseOver(e) {
      if (this.dragging) return;
      const box = e.target.closest(".ocr-box");
      if (!box || box.dataset.score === undefined) return;
      this.scoreTip.text = `置信度 ${(parseFloat(box.dataset.score) * 100).toFixed(1)}%`;
      this.scoreTip.visible = true;
      this.moveTip(e.clientX, e.clientY);
    },
    onOverlayMouseMove(e) {
      if (this.scoreTip.visible) this.moveTip(e.clientX, e.clientY);
    },
    onOverlayMouseOut(e) {
      const box = e.target.closest(".ocr-box");
      if (box && !(e.relatedTarget && box.contains(e.relatedTarget))) this.hideTip();
    },
    overlayScale() {
      return this.displayScale();
    },
    toOrig(clientX, clientY) {
      const r = this.$refs.overlay.getBoundingClientRect();
      const s = this.overlayScale();
      return [(clientX - r.left) / s, (clientY - r.top) / s];
    },
    itemById(id) {
      return this.items.find((it) => String(it.id) === String(id));
    },
    normRect(x0, y0, x1, y1) {
      let a = Math.min(x0, x1);
      let b = Math.min(y0, y1);
      let c = Math.max(x0, x1);
      let d = Math.max(y0, y1);
      const W = this.current.width;
      const H = this.current.height;
      a = Math.max(0, Math.min(a, W - MIN_BOX));
      b = Math.max(0, Math.min(b, H - MIN_BOX));
      c = Math.min(W, Math.max(c, a + MIN_BOX));
      d = Math.min(H, Math.max(d, b + MIN_BOX));
      return [a, b, c, d];
    },
    toggleEdit() {
      this.editMode = !this.editMode;
      this.selectedId = null;
      this.dragging = false;
      this.hideTip();
      if (this.editMode) {
        this.setStatus("微调模式：拖手柄改大小 · 拖框体移动 · 拖空白处画新框 · ✕/Delete 删除 · 改动后自动重识别");
      } else {
        this.setStatus("已退出微调模式");
      }
    },
    selectBox(id) {
      this.selectedId = id;
    },
    deleteBox(id) {
      const index = this.items.findIndex((it) => String(it.id) === String(id));
      if (index < 0) return;
      this.items.splice(index, 1);
      if (String(this.selectedId) === String(id)) this.selectedId = null;
      this.showToast("已删除该区域");
    },
    onOverlayMouseDown(e) {
      if (!this.editMode || e.button !== 0) return;
      if (e.target.closest(".del-btn")) return;
      const handle = e.target.closest(".handle");
      const box = e.target.closest(".ocr-box");
      if (handle && box) this.startResize(e, box, handle.dataset.dir);
      else if (box) this.startMove(e, box);
      else if (e.target === this.$refs.overlay) this.startDraw(e);
    },
    beginGesture(onMove, onUp) {
      this.dragging = true;
      this.hideTip();
      const move = (ev) => onMove(ev);
      const up = (ev) => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        this.dragging = false;
        onUp(ev);
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    },
    startResize(e, boxEl, dir) {
      e.preventDefault();
      const item = this.itemById(boxEl.dataset.id);
      if (!item) return;
      this.selectBox(item.id);
      const start = item.box.slice();
      const [ox, oy] = this.toOrig(e.clientX, e.clientY);
      this.beginGesture(
        (ev) => {
          const [mx, my] = this.toOrig(ev.clientX, ev.clientY);
          const dx = mx - ox;
          const dy = my - oy;
          let [x0, y0, x1, y1] = start;
          if (dir.includes("w")) x0 = start[0] + dx;
          if (dir.includes("e")) x1 = start[2] + dx;
          if (dir.includes("n")) y0 = start[1] + dy;
          if (dir.includes("s")) y1 = start[3] + dy;
          item.box = this.normRect(x0, y0, x1, y1);
        },
        () => {
          item.poly = rectPoly(item.box);
          this.reRecognize(item);
        },
      );
    },
    startMove(e, boxEl) {
      e.preventDefault();
      const item = this.itemById(boxEl.dataset.id);
      if (!item) return;
      const start = item.box.slice();
      const [ox, oy] = this.toOrig(e.clientX, e.clientY);
      const s = this.overlayScale();
      let moved = false;
      this.beginGesture(
        (ev) => {
          const [mx, my] = this.toOrig(ev.clientX, ev.clientY);
          const dx = mx - ox;
          const dy = my - oy;
          if (!moved && Math.hypot(dx, dy) * s < DRAG_THRESHOLD) return;
          moved = true;
          const w = start[2] - start[0];
          const h = start[3] - start[1];
          const nx = Math.max(0, Math.min(start[0] + dx, this.current.width - w));
          const ny = Math.max(0, Math.min(start[1] + dy, this.current.height - h));
          item.box = [nx, ny, nx + w, ny + h];
        },
        () => {
          if (moved) {
            item.poly = rectPoly(item.box);
            this.reRecognize(item);
          } else {
            this.selectBox(item.id);
          }
        },
      );
    },
    startDraw(e) {
      e.preventDefault();
      this.selectBox(null);
      const r = this.$refs.overlay.getBoundingClientRect();
      const sx = e.clientX - r.left;
      const sy = e.clientY - r.top;
      const rubber = document.createElement("div");
      rubber.className = "rubber";
      this.$refs.overlay.appendChild(rubber);
      this.beginGesture(
        (ev) => {
          const cx = ev.clientX - r.left;
          const cy = ev.clientY - r.top;
          rubber.style.left = `${Math.min(sx, cx)}px`;
          rubber.style.top = `${Math.min(sy, cy)}px`;
          rubber.style.width = `${Math.abs(cx - sx)}px`;
          rubber.style.height = `${Math.abs(cy - sy)}px`;
        },
        (ev) => {
          rubber.remove();
          const cx = ev.clientX - r.left;
          const cy = ev.clientY - r.top;
          if ((cx - sx) * (cx - sx) + (cy - sy) * (cy - sy) < 36) return;
          const s = this.overlayScale();
          const rect = this.normRect(sx / s, sy / s, cx / s, cy / s);
          const item = { id: this.nextId++, box: rect, poly: rectPoly(rect), text: "", score: null };
          this.items.push(item);
          this.selectBox(item.id);
          this.reRecognize(item);
        },
      );
    },
    cropCanvas(x, y, w, h) {
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(w));
      c.height = Math.max(1, Math.round(h));
      c.getContext("2d").drawImage(this.$refs.imgEl, x, y, w, h, 0, 0, c.width, c.height);
      return c;
    },
    async reRecognize(item) {
      const [x0, y0, x1, y1] = item.box;
      const w = x1 - x0;
      const h = y1 - y0;
      if (w < 2 || h < 2) return;
      this.recognizingId = item.id;
      try {
        const localOCR = await this.loadLocalOCR();
        const c = this.cropCanvas(x0, y0, w, h);
        const data = await localOCR.recognizeRegion(c, c.width, c.height);
        item.text = data.text || "";
        item.score = data.score;
        item.pinned = true; // user-adjusted region: never hide behind the slider
        this.showToast(item.text ? `已识别：${item.text.slice(0, 24)}` : "该区域未识别到文字");
      } catch (err) {
        this.showToast(`区域识别失败：${err.message || err}`);
      } finally {
        this.recognizingId = null;
      }
    },
    onKeyDown(e) {
      if (!this.editMode || this.selectedId == null) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        this.deleteBox(this.selectedId);
      }
    },
    onDragEnter(e) {
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files")) {
        this.dragover = true;
      }
    },
    onDragLeave(e) {
      if (e.relatedTarget === null) this.dragover = false;
    },
    onDrop(e) {
      this.dragover = false;
      const file = e.dataTransfer?.files?.[0];
      if (file) this.loadImage(file);
    },
    onPaste(e) {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const it of items) {
        if (it.type?.startsWith("image/")) {
          const file = it.getAsFile();
          if (file) {
            this.loadImage(file);
            e.preventDefault();
          }
          break;
        }
      }
    },
    onResize() {
      clearTimeout(this.resizeTimer);
      this.resizeTimer = setTimeout(() => this.fitText(), 120);
    },
    startPeek(e) {
      if (!this.hasItems) return;
      if (e?.cancelable) e.preventDefault();
      this.peek = true;
    },
    endPeek() {
      this.peek = false;
    },
  },
};
</script>
