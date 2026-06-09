"""Generate a sample image with mixed CN/EN text for OCR smoke-testing."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "examples" / "sample.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        r"C:\Windows\Fonts\msyh.ttc",   # Microsoft YaHei
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


img = Image.new("RGB", (820, 360), "white")
d = ImageDraw.Draw(img)
lines = [
    ("PP-OCRv5 在线 OCR 测试", 40),
    ("浏览器端 PP-OCRv5 识别", 34),
    ("Hello, PaddleOCR! 1234567890", 32),
    ("识别结果按位置渲染，可复制。", 30),
]
y = 36
for text, size in lines:
    d.text((40, y), text, fill="black", font=_font(size))
    y += size + 28

img.save(OUT)
print(f"wrote {OUT}  ({img.width}x{img.height})")
