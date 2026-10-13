"""测试用的素材生成与公共 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# TIFF 的压缩方式存在 tag 259 里。Pillow 的 img.info["compression"] 会把 LZW
# 报成 "raw"（解压后确实是 raw 像素），判断真实压缩必须直接读 tag。
TIFF_TAG_COMPRESSION = 259
TIFF_NONE = 1
TIFF_LZW = 5
TIFF_JPEG = 7
TIFF_ADOBE_DEFLATE = 8

# PNG 的 DPI 存在 pHYs 块里，单位是"像素/米"，300 dpi 无法被整数 ppm 精确表示
# （11811 ppm 读回是 299.9994 dpi）。这是 PNG 格式的固有精度上限，Photoshop 也一样。
PNG_DPI_TOLERANCE = 0.01


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    return out


def make_raster(path: Path, size: tuple[int, int] = (1000, 800), *, dpi=None, mode="RGB") -> Path:
    """生成一张纯色（或 16 位渐变）测试图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "I;16":
        import numpy as np

        ramp = np.linspace(0, 65535, size[0] * size[1], dtype=np.uint16)
        image = Image.fromarray(ramp.reshape(size[1], size[0]))
    else:
        image = Image.new(mode, size, (200, 30, 60) if mode == "RGB" else 128)
    kwargs = {"dpi": dpi} if dpi else {}
    image.save(path, **kwargs)
    return path


def make_photo(path: Path, subject: tuple[int, int, int, int], size=(1000, 800)) -> Path:
    """生成一张"深色台面 + 浅色试样"的模拟实验照片。"""
    from PIL import ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (60, 70, 80))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = subject
    draw.rectangle([left, top, right - 1, bottom - 1], fill=(235, 235, 230))
    image.save(path, dpi=(72, 72))
    return path


def read_dpi(path: Path) -> tuple[float, float]:
    with Image.open(path) as img:
        dpi = img.info.get("dpi")
    assert dpi is not None, f"{path.name} 未写入 DPI 元数据"
    return float(dpi[0]), float(dpi[1])


def read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def read_tiff_compression(path: Path) -> int:
    with Image.open(path) as img:
        return int(img.tag_v2.get(TIFF_TAG_COMPRESSION, TIFF_NONE))


__all__ = [
    "PNG_DPI_TOLERANCE",
    "ROOT",
    "TIFF_ADOBE_DEFLATE",
    "TIFF_JPEG",
    "TIFF_LZW",
    "TIFF_NONE",
    "TIFF_TAG_COMPRESSION",
    "make_photo",
    "make_raster",
    "read_dpi",
    "read_size",
    "read_tiff_compression",
]
