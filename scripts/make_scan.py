"""Generate a representative high-DPI document scan (A4 @ ~200 DPI, ~50 lines)
for speed benchmarking."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "examples" / "scan.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


W, H = 1654, 2339  # A4 at ~200 DPI
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

d.text((120, 90), "PP-OCRv5 高清扫描件识别基准", fill="black", font=_font(48))

bodies = [
    "本文档用于评估高清扫描件场景下的检测与识别性能。",
    "PP-OCRv5 mobile ONNX runs directly in the browser.",
    "识别结果按原图坐标叠加渲染，文字可选中、可复制。",
    "数字与符号 0123456789 +-*/=%@#& （括号）《书名号》。",
    "浏览器端模型完成检测与识别，无需服务端 OCR。",
    "The quick brown fox jumps over the lazy dog. 1234567890",
    "仅面向高清扫描件，不支持手写、拍照与旋转文本。",
    "检测默认将长边缩放到 960 像素后再做推理。",
]

f_body = _font(30)
y = 200
for i in range(50):
    d.text((120, y), f"{i + 1:02d}. {bodies[i % len(bodies)]}", fill="black", font=f_body)
    y += 42

img.save(OUT)
print(f"wrote {OUT}  ({img.width}x{img.height})")
