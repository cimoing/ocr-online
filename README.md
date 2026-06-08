# PP-OCRv5 在线 OCR

基于 **PaddleOCR 3.x（PP-OCRv5）** 的在线 OCR 演示服务：

- 🔁 一键切换 **服务端模型（server）** 与 **移动端模型（mobile）**，对比识别效果与速度（切换后自动重跑）
- 📥 支持 **拖入 / 粘贴 (Ctrl+V) / 点击选择** 图片
- 🎯 识别结果 **按原图坐标叠加渲染** 在图片上，文字 **可选中、可复制**，位置与原图保持一致
- 🖱️ 点击单行复制该行，「复制全部」复制整页文本
- ⏱️ 展示 **各环节耗时**（解码 / 预处理 / 推理 / 后处理 / 网络 / 渲染）与 **总耗时**；首次调用某模型还会单列一次性「模型加载」耗时
- 👁️ **「按住对比原图」**：按住时隐藏识别文字、显示原图，松开恢复，便于核对识别效果

## 架构

```
ppocr/
├── app/
│   ├── main.py          # FastAPI：托管前端静态页 + POST /api/ocr
│   └── ocr_service.py   # PaddleOCR 封装，缓存 server / mobile 两套 PP-OCRv5 管线
├── static/
│   ├── index.html       # 单页前端
│   ├── style.css
│   └── app.js           # 拖拽/粘贴、模型切换、识别框叠加渲染、复制
├── requirements.txt
└── run.py               # 启动器：python run.py
```

后端用 FastAPI 同时托管前端与 API；前端是零构建的原生单页，无需 Node。OCR 为 CPU 推理（本机 GTX 960 算力 5.2 低于现代 Paddle GPU 轮子要求，移动端模型 CPU 很快，服务端模型 CPU 亦可接受）。

| 类型 | 服务端 server | 移动端 mobile |
|---|---|---|
| 检测 | `PP-OCRv5_server_det` | `PP-OCRv5_mobile_det` |
| 识别 | `PP-OCRv5_server_rec` | `PP-OCRv5_mobile_rec` |

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

> 首次调用某个模型时会自动下载对应权重（缓存在 `~/.paddlex/`），稍等片刻即可。

## 运行

```powershell
.venv\Scripts\python run.py
# 或： .venv\Scripts\uvicorn app.main:app --port 8000
```

打开 <http://127.0.0.1:8000> ，拖入或粘贴一张图片，选择模型后点击「识别」。

## API

`POST /api/ocr` （`multipart/form-data`）

| 字段 | 说明 |
|---|---|
| `image` | 图片文件 |
| `model` | `mobile`（默认）或 `server` |

响应：

```jsonc
{
  "model": "mobile",
  "width": 1200, "height": 800,   // 原图像素尺寸
  "count": 12,
  "timings": {                    // 各环节耗时（毫秒）
    "load_ms": 0,                 // 模型加载 + 预热，仅首次调用该模型 > 0
    "decode_ms": 3,               // 解码上传图片
    "preprocess_ms": 12,          // 转 RGB / 缩放 / 转 BGR
    "inference_ms": 310,          // 检测 + 识别推理
    "postprocess_ms": 2,          // 整理结果
    "server_total_ms": 327        // 服务端合计（含 load_ms）
  },
  "items": [
    {
      "text": "识别出的文本",
      "score": 0.987,
      "poly": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],  // 四点多边形（原图像素坐标）
      "box":  [x_min, y_min, x_max, y_max]
    }
  ]
}
```

前端在此基础上叠加 **网络**（往返 − 服务端）、**渲染** 与 **总计**（端到端）耗时，展示在底部状态栏。

坐标均为 **原图像素坐标**；前端按 `显示宽度 / 原图宽度` 缩放，并用多边形的旋转角与宽高把每行文字精确叠加到对应位置。

## 说明

- 大图在送入 OCR 前会被等比缩放到最长边 ≤ 2048px，再把坐标映射回原图空间（见 `app/ocr_service.py` 的 `MAX_SIDE`）。
- 为保持轻量，默认关闭文档方向分类 / 矫正 / 文本行方向分类（`use_doc_orientation_classify` 等），只跑检测 + 识别两个模型。
- 首次调用某模型时会加载权重并用 `examples/sample.png` **预热**（触发 Paddle 计算图编译），因此首个请求较慢并单列「模型加载」耗时；之后即为热态性能。
- CPU 推理强制 `enable_mkldnn=False`：Paddle 3.x 的 oneDNN + PIR 执行器在 PP-OCRv5 上会崩溃（`ConvertPirAttribute2RuntimeAttribute`），改用原生 CPU 内核规避。
