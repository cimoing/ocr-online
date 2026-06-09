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
  const editBtn = $("editBtn");
  const timingsEl = $("timings");

  let currentModel = "mobile";
  let currentBlob = null;
  let currentURL = null;
  let current = null; // last /api/ocr response ({width,height,items})
  let busy = false;

  // ---- fine-tune region editing state ----
  let editMode = false;
  let selectedId = null; // id of the currently selected box (edit mode)
  let nextId = 1; // monotonic id source for boxes
  let dragging = false; // suppress hover tooltip mid-gesture

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
      // a freshly loaded image starts with an empty, editable region set so the
      // user can draw boxes even before running full-image OCR.
      current = { width: img.naturalWidth, height: img.naturalHeight, items: [] };
      setEditMode(false);
      editBtn.disabled = false;
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
    setEditMode(false);
    editBtn.disabled = true;
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
      selectedId = null;
      render();
      const tDone = performance.now();
      editBtn.disabled = false;
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
      if (item.id == null) item.id = nextId++;
      const g = boxGeom(item, scale);
      if (!g) continue;

      const box = document.createElement("div");
      box.className = "ocr-box";
      box.dataset.id = item.id;
      box.dataset.text = item.text || "";
      if (item.score != null) box.dataset.score = item.score;
      box.style.left = g.left + "px";
      box.style.top = g.top + "px";
      box.style.width = g.w + "px";
      box.style.height = g.h + "px";
      box.style.transform = g.angle ? `rotate(${g.angle}rad)` : "";

      const span = document.createElement("span");
      span.className = "txt";
      span.textContent = item.text || "";
      span.style.fontSize = g.h * 0.86 + "px";
      box.appendChild(span);

      if (editMode) {
        if (item.id === selectedId) box.classList.add("selected");
        addEditAffordances(box);
      }

      frag.appendChild(box);
      toFit.push([span, g.w]);
    }
    overlay.appendChild(frag);

    // Horizontally squeeze/stretch each line to exactly fill its box width,
    // so the rendered text lines up with the original characters.
    for (const [span, wpx] of toFit) {
      const natW = span.scrollWidth;
      if (natW > 0) span.style.transform = `scaleX(${(wpx / natW).toFixed(4)})`;
    }
  }

  // View mode: position from the (possibly rotated) detection polygon, exactly
  // as before. Edit mode: position from the axis-aligned box so resize handles
  // and drawing stay simple (no rotation math).
  function boxGeom(item, scale) {
    if (editMode || !item.poly) {
      const b = item.box;
      if (!b) return null;
      const w = (b[2] - b[0]) * scale;
      const h = (b[3] - b[1]) * scale;
      if (w < 1 || h < 1) return null;
      return { left: b[0] * scale, top: b[1] * scale, w, h, angle: 0 };
    }
    const p = item.poly;
    if (!p || p.length < 4) return null;
    const p0 = p[0], p1 = p[1], p3 = p[3];
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

  // click a box to copy that single line (view mode). In edit mode a click only
  // handles the delete (✕) button — selecting/moving happens on mousedown.
  overlay.addEventListener("click", (e) => {
    if (editMode) {
      const del = e.target.closest(".del-btn");
      if (del) {
        const b = del.closest(".ocr-box");
        if (b) deleteBox(b.dataset.id);
      }
      return;
    }
    const box = e.target.closest(".ocr-box");
    if (!box) return;
    const sel = window.getSelection();
    if (sel && String(sel).length > 0) return;
    copyText(box.dataset.text || "");
    showToast("已复制该行");
  });

  // hover a box to show its recognition confidence in a floating tooltip
  const scoreTip = document.createElement("div");
  scoreTip.className = "ocr-tip";
  scoreTip.hidden = true;
  document.body.appendChild(scoreTip);

  const moveTip = (x, y) => {
    const pad = 12;
    let left = x + pad;
    let top = y + pad;
    if (left + scoreTip.offsetWidth > window.innerWidth - 4)
      left = x - pad - scoreTip.offsetWidth;
    if (top + scoreTip.offsetHeight > window.innerHeight - 4)
      top = y - pad - scoreTip.offsetHeight;
    scoreTip.style.left = left + "px";
    scoreTip.style.top = top + "px";
  };
  const hideTip = () => { scoreTip.hidden = true; };

  overlay.addEventListener("mouseover", (e) => {
    if (dragging) return;
    const box = e.target.closest(".ocr-box");
    if (!box || box.dataset.score === undefined) return;
    scoreTip.textContent = `置信度 ${(parseFloat(box.dataset.score) * 100).toFixed(1)}%`;
    scoreTip.hidden = false;
    moveTip(e.clientX, e.clientY);
  });
  overlay.addEventListener("mousemove", (e) => {
    if (!scoreTip.hidden) moveTip(e.clientX, e.clientY);
  });
  overlay.addEventListener("mouseout", (e) => {
    const box = e.target.closest(".ocr-box");
    if (box && !(e.relatedTarget && box.contains(e.relatedTarget))) hideTip();
  });

  // ============================================================
  //  Fine-tune text regions (edit mode)
  // ============================================================
  const HANDLE_DIRS = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
  const MIN_BOX = 4; // smallest region in original px
  const DRAG_THRESHOLD = 3; // screen px before a body-drag counts as a move

  function overlayScale() {
    return current && current.width ? img.clientWidth / current.width : 1;
  }
  // viewport client coords -> original-image pixel coords
  function toOrig(clientX, clientY) {
    const r = overlay.getBoundingClientRect();
    const s = overlayScale();
    return [(clientX - r.left) / s, (clientY - r.top) / s];
  }
  function itemById(id) {
    return current && current.items
      ? current.items.find((it) => String(it.id) === String(id))
      : null;
  }
  function boxElById(id) {
    return overlay.querySelector(`.ocr-box[data-id="${id}"]`);
  }
  function rectPoly(b) {
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
  }
  // order corners, enforce a minimum size, clamp inside the image
  function normRect(x0, y0, x1, y1) {
    let a = Math.min(x0, x1), b = Math.min(y0, y1);
    let c = Math.max(x0, x1), d = Math.max(y0, y1);
    const W = current.width, H = current.height;
    a = Math.max(0, Math.min(a, W - MIN_BOX));
    b = Math.max(0, Math.min(b, H - MIN_BOX));
    c = Math.min(W, Math.max(c, a + MIN_BOX));
    d = Math.min(H, Math.max(d, b + MIN_BOX));
    return [a, b, c, d];
  }
  function applyBoxStyle(boxEl, item) {
    const s = overlayScale();
    const b = item.box;
    boxEl.style.left = b[0] * s + "px";
    boxEl.style.top = b[1] * s + "px";
    boxEl.style.width = (b[2] - b[0]) * s + "px";
    boxEl.style.height = (b[3] - b[1]) * s + "px";
  }
  function updateButtons() {
    const n = current && current.items ? current.items.length : 0;
    copyBtn.disabled = n === 0;
    peekBtn.disabled = n === 0;
  }

  function setEditMode(on) {
    editMode = !!on;
    selectedId = null;
    dragging = false;
    hideTip();
    overlay.classList.toggle("edit", editMode);
    editBtn.classList.toggle("active", editMode);
    editBtn.setAttribute("aria-pressed", editMode ? "true" : "false");
    if (current) render();
    if (editMode)
      setStatus(
        "微调模式：拖手柄改大小 · 拖框体移动 · 拖空白处画新框 · ✕/Delete 删除 · 改动后自动重识别"
      );
  }

  function addEditAffordances(box) {
    const del = document.createElement("button");
    del.type = "button";
    del.className = "del-btn";
    del.textContent = "×";
    del.title = "删除该区域";
    box.appendChild(del);
    for (const d of HANDLE_DIRS) {
      const h = document.createElement("div");
      h.className = "handle h-" + d;
      h.dataset.dir = d;
      box.appendChild(h);
    }
  }

  function selectBox(id) {
    selectedId = id;
    for (const el of overlay.querySelectorAll(".ocr-box"))
      el.classList.toggle("selected", el.dataset.id === String(id));
  }

  function deleteBox(id) {
    if (!current || !current.items) return;
    const i = current.items.findIndex((it) => String(it.id) === String(id));
    if (i < 0) return;
    current.items.splice(i, 1);
    if (String(selectedId) === String(id)) selectedId = null;
    render();
    updateButtons();
    showToast("已删除该区域");
  }

  // one mousedown dispatcher decides draw / resize / move
  overlay.addEventListener("mousedown", (e) => {
    if (!editMode || e.button !== 0) return;
    if (e.target.closest(".del-btn")) return; // handled on click
    const handle = e.target.closest(".handle");
    const boxEl = e.target.closest(".ocr-box");
    if (handle && boxEl) startResize(e, boxEl, handle.dataset.dir);
    else if (boxEl) startMove(e, boxEl);
    else if (e.target === overlay) startDraw(e);
  });

  function beginGesture(onMove, onUp) {
    dragging = true;
    hideTip();
    const move = (ev) => onMove(ev);
    const up = (ev) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      dragging = false;
      onUp(ev);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function startResize(e, boxEl, dir) {
    e.preventDefault();
    const item = itemById(boxEl.dataset.id);
    if (!item) return;
    selectBox(item.id);
    const start = item.box.slice();
    const [ox, oy] = toOrig(e.clientX, e.clientY);
    beginGesture(
      (ev) => {
        const [mx, my] = toOrig(ev.clientX, ev.clientY);
        const dx = mx - ox, dy = my - oy;
        let [x0, y0, x1, y1] = start;
        if (dir.includes("w")) x0 = start[0] + dx;
        if (dir.includes("e")) x1 = start[2] + dx;
        if (dir.includes("n")) y0 = start[1] + dy;
        if (dir.includes("s")) y1 = start[3] + dy;
        item.box = normRect(x0, y0, x1, y1);
        applyBoxStyle(boxEl, item);
      },
      () => {
        item.poly = rectPoly(item.box);
        render();
        reRecognize(item);
      }
    );
  }

  function startMove(e, boxEl) {
    e.preventDefault();
    const item = itemById(boxEl.dataset.id);
    if (!item) return;
    const start = item.box.slice();
    const [ox, oy] = toOrig(e.clientX, e.clientY);
    const s = overlayScale();
    let moved = false;
    beginGesture(
      (ev) => {
        const [mx, my] = toOrig(ev.clientX, ev.clientY);
        const dx = mx - ox, dy = my - oy;
        if (!moved && Math.hypot(dx, dy) * s < DRAG_THRESHOLD) return;
        moved = true;
        const w = start[2] - start[0], h = start[3] - start[1];
        const nx = Math.max(0, Math.min(start[0] + dx, current.width - w));
        const ny = Math.max(0, Math.min(start[1] + dy, current.height - h));
        item.box = [nx, ny, nx + w, ny + h];
        applyBoxStyle(boxEl, item);
      },
      () => {
        if (moved) {
          item.poly = rectPoly(item.box);
          render();
          reRecognize(item);
        } else {
          selectBox(item.id);
        }
      }
    );
  }

  function startDraw(e) {
    e.preventDefault();
    selectBox(null);
    const r = overlay.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    const rubber = document.createElement("div");
    rubber.className = "rubber";
    overlay.appendChild(rubber);
    beginGesture(
      (ev) => {
        const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
        rubber.style.left = Math.min(sx, cx) + "px";
        rubber.style.top = Math.min(sy, cy) + "px";
        rubber.style.width = Math.abs(cx - sx) + "px";
        rubber.style.height = Math.abs(cy - sy) + "px";
      },
      (ev) => {
        rubber.remove();
        const cx = ev.clientX - r.left, cy = ev.clientY - r.top;
        // a stray click (no real drag) should not create a box
        if ((cx - sx) * (cx - sx) + (cy - sy) * (cy - sy) < 36) return;
        const s = overlayScale();
        const rect = normRect(sx / s, sy / s, cx / s, cy / s);
        const item = {
          id: nextId++,
          box: rect,
          poly: rectPoly(rect),
          text: "",
          score: null,
        };
        current.items.push(item);
        render();
        updateButtons();
        selectBox(item.id);
        reRecognize(item);
      }
    );
  }

  // crop the region from the original-resolution image, then re-OCR just it
  function cropBlob(x, y, w, h) {
    return new Promise((resolve, reject) => {
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(w));
      c.height = Math.max(1, Math.round(h));
      try {
        c.getContext("2d").drawImage(img, x, y, w, h, 0, 0, c.width, c.height);
      } catch (err) {
        reject(err);
        return;
      }
      c.toBlob((b) => (b ? resolve(b) : reject(new Error("crop failed"))), "image/png");
    });
  }

  async function reRecognize(item) {
    const [x0, y0, x1, y1] = item.box;
    const w = x1 - x0, h = y1 - y0;
    if (w < 2 || h < 2) return;
    const el = boxElById(item.id);
    if (el) el.classList.add("recognizing");
    try {
      const blob = await cropBlob(x0, y0, w, h);
      const fd = new FormData();
      fd.append("image", blob, "region.png");
      fd.append("model", currentModel);
      const r = await fetch("/api/ocr_region", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      item.text = data.text || "";
      item.score = data.score;
      render();
      updateButtons();
      showToast(item.text ? `已识别：${item.text.slice(0, 24)}` : "该区域未识别到文字");
    } catch (err) {
      const e2 = boxElById(item.id);
      if (e2) e2.classList.remove("recognizing");
      showToast("区域识别失败：" + (err.message || err));
    }
  }

  editBtn.addEventListener("click", () => {
    if (editBtn.disabled) return;
    setEditMode(!editMode);
    if (!editMode) setStatus("已退出微调模式");
  });

  // Delete / Backspace removes the selected region (edit mode only)
  window.addEventListener("keydown", (e) => {
    if (!editMode || selectedId == null) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable))
      return;
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      deleteBox(selectedId);
    }
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
