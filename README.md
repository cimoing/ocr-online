# PP-OCRv5 浏览器 OCR

基于 **Vue 3 + Vite + onnxruntime-web** 的纯前端 OCR 应用。构建后产物是静态文件，识别全流程在浏览器本地运行，不再包含服务端识别 API，也不依赖 Python/FastAPI/PaddleOCR 运行服务。

## 功能

- 拖入 / 粘贴 / 选择图片
- PP-OCRv5 mobile det + rec 通过 ONNX/WASM 在浏览器本地推理
- 识别结果按原图坐标叠加渲染，文字可选中、可复制
- 点击单行复制，「复制全部」复制整页文本
- 微调区域：拖拽/缩放/新增/删除文本框，并对区域本地重识别
- 展示预处理、推理、后处理与总耗时

## 结构

```text
ppocr/
├── index.html
├── package.json
├── vite.config.js
├── src/
│   ├── App.vue          # Vue 前端应用
│   └── main.js
├── static/
│   ├── style.css
│   ├── ocr-local.js     # 浏览器侧 det/rec 前后处理 + onnxruntime-web 调用
│   ├── models/          # det.onnx / rec.onnx / ppocrv5_keys.txt
│   └── vendor/ort/      # onnxruntime-web WASM 运行时
└── scripts/
    └── fetch-browser-assets.mjs
```

`static/` 是 Vite 的 public 目录，构建时会复制到 `dist/` 根目录；因此运行 `dist/` 时模型路径为 `models/*`、运行时路径为 `vendor/ort/*`。

## 安装与构建

```powershell
npm install
npm run fetch-assets
npm run build
```

构建完成后，`dist/` 就是可部署的静态站点。可以用任意静态服务器托管，例如：

```powershell
npm run preview
```

也可以把 `dist/` 部署到 Nginx、静态文件服务器、对象存储或前端托管平台。

## 本地开发

```powershell
npm run dev
```

打开 Vite 输出的本地地址后，拖入图片并点击「识别」即可。

## 资源说明

- 模型：`PP-OCRv5_mobile_det` 和 `PP-OCRv5_mobile_rec` 的 ONNX 版本
- 运行时：`onnxruntime-web` WASM 单线程运行时
- 字典：`static/models/ppocrv5_keys.txt`

大文件默认不纳入 Git；`npm run fetch-assets` 会下载并校验 SHA-256。浏览器首次加载模型和 WASM 会较慢，之后可由浏览器缓存加速。
