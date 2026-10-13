"""位图载入与重采样：把磁盘上的图片变成干净的 PIL 位图，再缩放成目标尺寸。

这里只处理栅格图（PNG / JPEG / TIFF / BMP / WebP …）。裁剪器不渲染 PDF 或
幻灯片 —— 那类需求请先用「图片转换器」导出成图片，再拿过来裁。

载入时会把各种麻烦的模式统一掉（透明通道合成白底、索引色调色板转 RGB、
CMYK 转 RGB、16 位保持高位深），这样后续的裁剪与编码只需面对少数几种模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 600_000_000

# 这些是 16 位灰度/彩色模式，重采样与编码需要特殊照顾
SIXTEEN_BIT_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I"}


@dataclass
class LoadedImage:
    """已从磁盘完整载入的栅格图。"""

    path: Path
    image: Image.Image
    warnings: list[str] = field(default_factory=list)

    @property
    def is_16bit(self) -> bool:
        return self.image.mode in SIXTEEN_BIT_MODES

    @property
    def pixel_size(self) -> tuple[int, int]:
        return self.image.size


def load_raster(path: Path | str) -> LoadedImage:
    """读入栅格图，必要时合成白底。"""
    path = Path(path)
    warnings: list[str] = []
    img = Image.open(path)
    try:
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"图片数据损坏或不完整：{path.name} —— {exc}") from exc

    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
        img = img.copy()
        warnings.append("多帧文件仅取第 1 帧")

    mode = img.mode
    if mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.convert("RGBA")
        background.paste(alpha, mask=alpha.split()[-1])
        img = background
        warnings.append("源图含透明通道，已合成到白色背景")
    elif mode == "P":
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
        warnings.append("索引色调色板图已转换为 RGB")
    elif mode == "CMYK":
        img = img.convert("RGB")
        warnings.append("CMYK 源图已转换为 RGB（多数期刊电子版要求 RGB）")
    elif mode == "1":
        img = img.convert("L")
    elif mode in ("L", "RGB", "I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        pass
    else:
        img = img.convert("RGB")
        warnings.append(f"源图色彩模式 {mode} 已转换为 RGB")

    return LoadedImage(path=path, image=img, warnings=warnings)


def _resample_filter() -> int:
    return Image.Resampling.LANCZOS


def resize_raster(image: Image.Image, target_px: tuple[int, int]) -> tuple[Image.Image, list[str]]:
    """把栅格图重采样到目标像素尺寸。

    Pillow 对 16 位模式的重采样支持有限，这里统一先升成 float32 再插值，
    避免高位深图像被静默截断到 8 位。
    """
    warnings: list[str] = []
    if image.size == tuple(target_px):
        return image, warnings

    if image.mode in SIXTEEN_BIT_MODES:
        arr = np.asarray(image)
        if arr.dtype != np.uint16:
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0, 65535).astype(np.uint16)
        float_img = Image.fromarray(arr.astype(np.float32), mode="F")
        resized = float_img.resize(tuple(target_px), _resample_filter())
        out = np.clip(np.asarray(resized), 0, 65535).round().astype(np.uint16)
        # 不传 mode=，让 Pillow 从 uint16 自动推断为 I;16（显式 mode= 已被弃用）
        return Image.fromarray(out), warnings

    return image.resize(tuple(target_px), _resample_filter()), warnings


__all__ = [
    "LoadedImage",
    "SIXTEEN_BIT_MODES",
    "load_raster",
    "resize_raster",
]
