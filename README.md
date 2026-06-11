# PP-OCRv5 浏览器 OCR

基于 **Vue 3 + Vite + onnxruntime-web** 的纯前端 OCR 应用。构建后产物是静态文件，识别全流程在浏览器本地运行：图片不上传任何服务器，也不依赖 Python/FastAPI/PaddleOCR 运行服务。

## 功能

- 拖入 / 粘贴 / 选择图片
- PP-OCRv5 mobile det + rec 通过 ONNX/WASM 在浏览器本地推理
- 标准 DB 后处理（连通域 → minAreaRect → unclip）：支持**倾斜文本**，叠加框带角度渲染
- PP-LCNet textline 方向分类：**180° 翻转的图片**也能正确识别
- 检测分辨率档位（960 / 1280 / 1536），大图小字可调高
- 置信度过滤滑块：低置信行实时隐藏/恢复，不必重新识别
- 识别结果按原图坐标叠加渲染，文字可选中、可复制；点击单行复制，「复制全部」复制整页
- 导出 TXT / JSON（JSON 含坐标、置信度、四点框）
- 微调区域：拖拽/缩放/新增/删除文本框并对区域重识别（pointer events，支持触屏）
- rec/cls 按宽度分桶批量推理；模型持久缓存在 Cache Storage（回访秒载）
- 推理在 ort proxy worker 中运行不阻塞 UI；crossOriginIsolated 时自动启用多线程 WASM；worker 受限的环境自动降级主线程

## 结构

```text
ppocr/
├── index.html               # 注册 coi-serviceworker + meta/favicon
├── package.json
├── vite.config.js           # preview 模式带 COOP/COEP/CORP 头
├── src/
│   ├── main.js
│   ├── App.vue              # UI：叠加渲染 / 复制导出 / 微调编辑
│   └── ocr/
│       ├── engine.js        # ort 加载、模型 I/O、批量推理管线（DOM 侧）
│       ├── db.js            # DB 后处理（纯函数）
│       ├── geometry.js      # 凸包 / minAreaRect / 排序（纯函数）
│       ├── ctc.js           # CTC 解码（纯函数）
│       └── *.test.js        # vitest 单测
├── static/                  # Vite public 目录，原样复制进 dist/
│   ├── style.css
│   ├── coi-serviceworker.min.js  # GitHub Pages 上注入 COOP/COEP 启用多线程
│   ├── models/              # det/rec/cls.onnx + ppocrv5_keys.txt
│   └── vendor/ort/          # onnxruntime-web WASM 运行时
└── scripts/
    └── fetch-browser-assets.mjs  # 下载并 SHA-256 校验模型与运行时
```

## 安装与构建

```powershell
npm install
npm run fetch-assets
npm run build
```

构建完成后 `dist/` 即可部署到任意静态托管。本地预览（带跨域隔离头，可验证多线程）：

```powershell
npm run preview
```

## 测试

```powershell
npm test
```

vitest 覆盖检测后处理、几何与 CTC 解码等纯函数模块（无需浏览器/模型）。

## GitHub Pages

仓库内置工作流 `.github/workflows/deploy-pages.yml`，推送到 `master` 或 `main` 后自动：

1. 安装依赖并运行单测
2. 下载并校验 OCR 模型与 WASM 运行时（带 Actions 缓存）
3. `npm run build` 并发布 `dist/` 到 GitHub Pages

访问地址：`https://cimoing.github.io/ocr-online/`。仓库 Settings → Pages 需选择 **GitHub Actions** 作为发布来源。

GitHub Pages 无法自定义响应头，多线程所需的 COOP/COEP 由 `coi-serviceworker` 在首次访问时注入（会自动刷新一次页面）。

## 本地开发

```powershell
npm run dev
```

打开 Vite 输出的本地地址后，拖入图片并点击「识别」。dev 模式不带隔离头（部分内嵌浏览器对 COEP+worker 的实现不完整），推理为单线程。

## 资源说明

- 检测/识别：`PP-OCRv5_mobile_det` / `PP-OCRv5_mobile_rec` 的 ONNX 版本
- 方向分类：`PP-LCNet_x0_25_textline_ori`（输入 160×80，输出 0°/180°）
- 运行时：`onnxruntime-web` WASM（SIMD，隔离环境下多线程）
- 字典：`static/models/ppocrv5_keys.txt`（已提交，CTC blank + 18383 字 + 空格）

大文件不入 Git；`npm run fetch-assets` 下载并校验 SHA-256。浏览器首次加载约 34MB（模型 + WASM），之后由 Cache Storage 持久缓存，回访近乎即时。
