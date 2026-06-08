(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const stage = $("stage");
  const empty = $("empty");
  const canvas = $("canvas");
  const img = $("img");
  const overlay = $("overlay");
  const fileInput = $("fileInput");
  const pickBtn = $("pickBtn");
  const recognizeBtn = $("recognizeBtn");
  const copyBtn = $("copyBtn");
  const clearBtn = $("clearBtn");
  const statusEl = $("status");
  const toast = $("toast");
  const spinner = $("spinner");
  const spinnerText = $("spinnerText");
  const maskRange = $("maskRange");
  const modelSwitch = $("modelSwitch");
  const peekBtn = $("peekBtn");
  const timingsEl = $("timings");

  let currentModel = "mobile";
  let currentBlob = null;
  let currentURL = null;
  let current = null; // last /api/ocr response
  let busy = false;

  const modelLabel = (m) => (m === "server" ? "服务端 server" : "移动端 mobile");

  function setStatus(msg, isErr) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("err", !!isErr);
  }

  let toastTimer = null;
  function showToast(msg) {
    toast.textContent = msg;
    toast.hidden = false;
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => (toast.hidden = true), 220);
    }, 1400);
  }

  function setBusy(b) {
    busy = b;
    spinner.hidden = !b;
    recognizeBtn.disabled = b || !currentBlob;
  }

  // ---------- image loading ----------
  function loadImage(blob) {
    if (!blob) return;
    if (!(blob.type && blob.type.startsWith("image/"))) {
      showToast("不是图片文件");
      return;
    }
    currentBlob = blob;
    if (currentURL) URL.revokeObjectURL(currentURL);
    currentURL = URL.createObjectURL(blob);
    current = null;
    overlay.innerHTML = "";
    timingsEl.hidden = true;
    timingsEl.innerHTML = "";
    img.onload = () => {
      empty.hidden = true;
      canvas.hidden = false;
      recognizeBtn.disabled = false;
      clearBtn.disabled = false;
      copyBtn.disabled = true;
      peekBtn.disabled = true;
      setStatus(`已载入 ${img.naturalWidth}×${img.naturalHeight} · 点击「识别」`);
    };
    img.src = currentURL;
  }

  function clearAll() {
    currentBlob = null;
    current = null;
    if (currentURL) {
      URL.revokeObjectURL(currentURL);
      currentURL = null;
    }
    img.removeAttribute("src");
    overlay.innerHTML = "";
    canvas.hidden = true;
    empty.hidden = false;
    recognizeBtn.disabled = true;
    copyBtn.disabled = true;
    peekBtn.disabled = true;
    clearBtn.disabled = true;
    overlay.classList.remove("peek");
    timingsEl.hidden = true;
    timingsEl.innerHTML = "";
    setStatus("就绪 — 等待图片");
  }

  // ---------- recognize ----------
  async function recognize() {
    if (!currentBlob || busy) return;
    setBusy(true);
    spinnerText.textContent = `识别中…（${modelLabel(currentModel)}）`;
    setStatus(`识别中… ${modelLabel(currentModel)}（首次加载该模型可能较慢）`);
    const fd = new FormData();
    fd.append("image", currentBlob, currentBlob.name || "image.png");
    fd.append("model", currentModel);
    const tStart = performance.now();
    try {
      const r = await fetch("/api/ocr", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      const tResp = performance.now();
      current = data;
      render();
      const tDone = performance.now();
      copyBtn.disabled = data.count === 0;
      peekBtn.disabled = data.count === 0;
      setStatus(
        `${modelLabel(data.model)} · ${data.count} 行 · ${data.width}×${data.height}`
      );
      renderTimings(data.timings, tResp - tStart, tDone - tResp, tDone - tStart);
    } catch (e) {
      setStatus("识别失败：" + e.message, true);
      showToast("识别失败");
    } finally {
      setBusy(false);
    }
  }

  // ---------- render OCR overlay (positioned to match the image) ----------
  function render() {
    overlay.innerHTML = "";
    if (!current || !current.items || !current.width) return;
    const scale = img.clientWidth / current.width; // displayed px / original px
    const frag = document.createDocumentFragment();
    const toFit = [];

    for (const item of current.items) {
      const p = item.poly;
      if (!p || p.length < 4) continue;
      const p0 = p[0], p1 = p[1], p3 = p[3];
      const wpx = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * scale;
      const hpx = Math.hypot(p3[0] - p0[0], p3[1] - p0[1]) * scale;
      if (wpx < 1 || hpx < 1) continue;
      const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);

      const box = document.createElement("div");
      box.className = "ocr-box";
      box.style.left = p0[0] * scale + "px";
      box.style.top = p0[1] * scale + "px";
      box.style.width = wpx + "px";
      box.style.height = hpx + "px";
      box.style.transform = `rotate(${angle}rad)`;
      box.dataset.text = item.text;
      if (item.score != null)
        box.title = `${item.text}\n置信度 ${(item.score * 100).toFixed(1)}%`;

      const span = document.createElement("span");
      span.className = "txt";
      span.textContent = item.text;
      span.style.fontSize = hpx * 0.86 + "px";
      box.appendChild(span);

      frag.appendChild(box);
      toFit.push([span, wpx]);
    }
    overlay.appendChild(frag);

    // Horizontally squeeze/stretch each line to exactly fill its box width,
    // so the rendered text lines up with the original characters.
    for (const [span, wpx] of toFit) {
      const natW = span.scrollWidth;
      if (natW > 0) span.style.transform = `scaleX(${(wpx / natW).toFixed(4)})`;
    }
  }

  // ---------- per-stage timing chips ----------
  function fmtMs(ms) {
    if (ms == null || isNaN(ms)) return "—";
    return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";
  }

  function renderTimings(t, roundTrip, renderMs, endToEnd) {
    t = t || {};
    const serverTotal = t.server_total_ms;
    const network =
      serverTotal != null ? Math.max(0, roundTrip - serverTotal) : roundTrip;
    const chips = [];
    // one-time model load + graph warmup, shown only on the first (cold) run
    if (t.load_ms != null && t.load_ms >= 1)
      chips.push(["模型加载", t.load_ms, "load"]);
    chips.push(["解码", t.decode_ms, ""]);
    chips.push(["预处理", t.preprocess_ms, ""]);
    chips.push(["推理", t.inference_ms, "hot"]);
    chips.push(["后处理", t.postprocess_ms, ""]);
    chips.push(["服务端", serverTotal, ""]);
    chips.push(["网络", network, ""]);
    chips.push(["渲染", renderMs, ""]);
    chips.push(["总计", endToEnd, "total"]);
    timingsEl.innerHTML = chips
      .map(([k, v, cls]) => `<span class="chip ${cls}"><i>${k}</i>${fmtMs(v)}</span>`)
      .join("");
    timingsEl.hidden = false;
  }

  // ---------- clipboard ----------
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (__) {}
      document.body.removeChild(ta);
    }
  }
  async function copyAll() {
    if (!current || !current.items || !current.items.length) return;
    await copyText(current.items.map((i) => i.text).join("\n"));
    showToast(`已复制 ${current.items.length} 行`);
  }

  // ---------- events ----------
  empty.addEventListener("click", () => fileInput.click());
  pickBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) loadImage(fileInput.files[0]);
    fileInput.value = "";
  });
  recognizeBtn.addEventListener("click", recognize);
  copyBtn.addEventListener("click", copyAll);
  clearBtn.addEventListener("click", clearAll);

  // click a box to copy that single line (unless the user is selecting text)
  overlay.addEventListener("click", (e) => {
    const box = e.target.closest(".ocr-box");
    if (!box) return;
    const sel = window.getSelection();
    if (sel && String(sel).length > 0) return;
    copyText(box.dataset.text || "");
    showToast("已复制该行");
  });

  // model switch — auto re-run on an existing result for easy comparison
  modelSwitch.addEventListener("click", (e) => {
    const b = e.target.closest(".seg");
    if (!b) return;
    const m = b.dataset.model;
    if (m === currentModel) return;
    currentModel = m;
    for (const c of modelSwitch.children) c.classList.toggle("active", c === b);
    if (current) recognize();
    else if (currentBlob) setStatus(`已切换到 ${modelLabel(m)} · 点击「识别」`);
  });

  // mask opacity slider
  function applyMask() {
    overlay.style.setProperty("--mask", (maskRange.value / 100).toString());
  }
  maskRange.addEventListener("input", applyMask);
  applyMask();

  // drag & drop anywhere
  ["dragenter", "dragover"].forEach((ev) =>
    window.addEventListener(ev, (e) => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files")) {
        e.preventDefault();
        stage.classList.add("dragover");
      }
    })
  );
  window.addEventListener("dragleave", (e) => {
    if (e.relatedTarget === null) stage.classList.remove("dragover");
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    stage.classList.remove("dragover");
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) loadImage(f);
  });

  // paste image from clipboard
  window.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const it of items) {
      if (it.type && it.type.startsWith("image/")) {
        const f = it.getAsFile();
        if (f) {
          loadImage(f);
          e.preventDefault();
          break;
        }
      }
    }
  });

  // re-position overlay when the displayed image size changes
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => current && render(), 120);
  });

  // ---------- 按住对比原图 (hold to hide the overlay and reveal the image) ----------
  const startPeek = (e) => {
    if (peekBtn.disabled) return;
    if (e && e.cancelable) e.preventDefault();
    overlay.classList.add("peek");
  };
  const endPeek = () => overlay.classList.remove("peek");
  peekBtn.addEventListener("mousedown", startPeek);
  peekBtn.addEventListener("touchstart", startPeek, { passive: false });
  // End on release anywhere (so dragging off the button mid-hold keeps peeking)
  // or when the window loses focus while held.
  window.addEventListener("mouseup", endPeek);
  window.addEventListener("touchend", endPeek);
  window.addEventListener("blur", endPeek);
})();
