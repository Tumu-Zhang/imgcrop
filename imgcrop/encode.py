"""编码层：把位图写成目标格式，并写出正确的 DPI 元数据。

1.1.0 瘦身定案（D6、D11，见《功能瘦身与补足讨论稿.md》）：
- **格式跟随源**：JPG→JPG、PNG→PNG（保透明）、TIFF→PNG，其余→JPG。
  质量是内部常量（95），不设 UI —— "输出工程"链路已整体移除。
- **DPI 跟随源图元数据，缺省 300**。纯元数据，不影响像素。

实现细节：
- JPEG 的 optimize=True 会让 libjpeg 走二次扫描，遇到没有真实文件描述符
  的内存流（BytesIO）会在稍大的图上直接失败（"Suspension not allowed here"）。
  所以体积测量统一走有 fileno 的临时文件，而不是 BytesIO。
- RGBA 编码成 JPEG 时先把透明区合成白底（直接 convert("RGB") 会变黑）。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from imgcrop.model import format_bytes

Image.MAX_IMAGE_PIXELS = 600_000_000

# 输出 JPEG 的固定质量（D6 定案：不设 UI，保持内部常量）
JPEG_QUALITY = 95

# 源图没有 DPI 元数据时写入的缺省值（D11 定案）
DEFAULT_DPI = 300

# 视作"JPEG 家族"的源后缀 -> 输出仍是 JPEG
_JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".jfif"}


def resolve_output_format(source_suffix: str) -> str:
    """按 D6 定案从源文件后缀解析输出格式名："png" / "jpeg"。

    JPG→JPG、PNG→PNG（保透明）、TIFF→PNG；其余→JPG。
    返回 Pillow 的格式名（"PNG" / "JPEG"），同时决定输出扩展名与编码路径。
    """
    suffix = source_suffix.lower().strip()
    if suffix == ".png" or suffix in (".tif", ".tiff"):
        return "PNG"
    return "JPEG"


def extension_for_format(fmt: str) -> str:
    """输出格式的文件扩展名。"""
    return ".png" if fmt == "PNG" else ".jpg"


def resolve_source_dpi(dpi: tuple[float, float] | None) -> int:
    """D11：DPI 元数据跟随源图；缺省 300。"""
    if dpi:
        try:
            x, y = float(dpi[0]), float(dpi[1])
        except (TypeError, ValueError, IndexError):
            return DEFAULT_DPI
        if x > 0 and abs(x - y) < 0.01 * max(x, y):
            return max(1, int(round(x)))
        if x > 0:
            return max(1, int(round(x)))
    return DEFAULT_DPI


@dataclass
class EncodeResult:
    path: Path
    num_bytes: int
    fmt: str = ""
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        head = format_bytes(self.num_bytes)
        return f"{head} / {self.fmt}" if self.fmt else head


# ----------------------------------------------------------------------
# 模式适配
# ----------------------------------------------------------------------
def _flatten_rgba(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """把带透明通道的图合成到白底（供 JPEG 等不支持 alpha 的格式）。"""
    if image.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.convert("RGBA")
        background.paste(alpha, mask=alpha.split()[-1])
        return background, ["透明通道已合成到白色背景以适配 JPEG"]
    if image.mode == "P":
        rgba = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background, ["调色板图已合成到白色背景以适配 JPEG"]
    return image, []


def _prepare_for_format(image: Image.Image, fmt: str) -> tuple[Image.Image, list[str]]:
    """把位图调整成目标格式能承载的模式。"""
    warnings: list[str] = []
    mode = image.mode

    if fmt == "JPEG":
        if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
            # JPEG 不支持高位深，按 PS 的"16 位 -> 8 位"惯例取高字节
            arr = np.asarray(image)
            if arr.dtype != np.uint16:
                arr = np.clip(arr, 0, 65535).astype(np.uint16)
            image = Image.fromarray((arr >> 8).astype(np.uint8), mode="L")
            warnings.append("16 位灰度已按高字节降为 8 位以适配 JPEG")
        elif mode == "I":
            arr = np.clip(np.asarray(image), 0, 65535).astype(np.uint16)
            image = Image.fromarray((arr >> 8).astype(np.uint8), mode="L")
            warnings.append("32 位整型图已降为 8 位灰度以适配 JPEG")
        elif mode == "F":
            arr = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="L")
            warnings.append("浮点图已降为 8 位灰度以适配 JPEG")
        elif mode not in ("L", "RGB"):
            image, flat_warnings = _flatten_rgba(image)
            warnings.extend(flat_warnings)
            if image.mode == "CMYK":
                image = image.convert("RGB")
                warnings.append("CMYK 已转换为 RGB 以适配 JPEG")
        return image, warnings

    # PNG：保留透明通道（D6：PNG→PNG 保透明），只处理它装不下的模式
    if mode in ("I", "F"):
        arr = np.asarray(image)
        if mode == "F":
            arr = np.clip(arr * 255.0, 0, 255)
        image = Image.fromarray(arr.astype(np.uint8), mode="L")
        warnings.append("高位深/浮点图已降为 8 位以适配 PNG")
    elif mode == "CMYK":
        image = image.convert("RGB")
        warnings.append("CMYK 已转换为 RGB 以适配 PNG")
    return image, warnings


# ----------------------------------------------------------------------
# 单次编码
# ----------------------------------------------------------------------
def _save_to(
    image: Image.Image,
    target: str | Path,
    fmt: str,
    dpi: int,
) -> None:
    dpi_tuple = (dpi, dpi)

    if fmt == "JPEG":
        _save_jpeg(image, target, dpi=dpi_tuple)
    elif fmt == "PNG":
        image.save(
            target,
            format="PNG",
            dpi=dpi_tuple,
            compress_level=9,
            optimize=True,
        )
    else:  # pragma: no cover - resolve_output_format 已穷举
        raise ValueError(f"未知输出格式：{fmt}")


def _reset_target(target: str | Path) -> None:
    """重写同一个 file object 前把它清空，避免上一次尝试的残留字节。"""
    if hasattr(target, "seek"):
        target.seek(0)
        if hasattr(target, "truncate"):
            target.truncate(0)


def _save_jpeg(
    image: Image.Image, target: str | Path, *, dpi: tuple[int, int]
) -> None:
    """保存 JPEG，必要时放弃 Huffman 表优化。

    Pillow 给 libjpeg 的输出缓冲是按像素数估算的；遇到高熵图像（显微图、噪声多的图）
    配合 subsampling=0 的高质量输出，真实码流会超出该估算，libjpeg 于是要求 suspend，
    而 Pillow 的编码路径不允许，直接抛 "Suspension not allowed here" /
    "broken data stream when writing image file"。
    optimize 只是重排 Huffman 表、属无损优化，放弃它只会让文件略大，画质完全不变，
    所以这里自动回退，而不是把一个本该成功的转换报成失败。
    """
    common = dict(
        format="JPEG",
        quality=JPEG_QUALITY,
        dpi=dpi,
        subsampling=0,      # 4:4:4，保住细线条与彩色文字的边缘
        progressive=False,
    )
    try:
        image.save(target, optimize=True, **common)
    except OSError:
        _reset_target(target)
        image.save(target, optimize=False, **common)


# ----------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------
def encode_to_file(image: Image.Image, path: Path, fmt: str, *, dpi: int = DEFAULT_DPI) -> EncodeResult:
    """编码并落盘。fmt 是 "PNG" / "JPEG"（resolve_output_format 的返回值）。"""
    path = Path(path)
    prepared, warnings = _prepare_for_format(image, fmt)

    path.parent.mkdir(parents=True, exist_ok=True)
    _save_to(prepared, path, fmt, dpi)

    return EncodeResult(
        path=path,
        num_bytes=path.stat().st_size,
        fmt=fmt,
        warnings=warnings,
    )


def encode_image(image: Image.Image, fmt: str, *, dpi: int = DEFAULT_DPI) -> tuple[bytes, EncodeResult]:
    """编码到内存，返回 (字节, 结果元信息)。供需要二次处理的调用方使用。"""
    prepared, warnings = _prepare_for_format(image, fmt)

    with tempfile.TemporaryFile() as tmp:
        _save_to(prepared, tmp, fmt, dpi)
        tmp.seek(0)
        data = tmp.read()

    return data, EncodeResult(
        path=Path(),
        num_bytes=len(data),
        fmt=fmt,
        warnings=warnings,
    )


__all__ = [
    "DEFAULT_DPI",
    "EncodeResult",
    "JPEG_QUALITY",
    "encode_image",
    "encode_to_file",
    "extension_for_format",
    "resolve_output_format",
    "resolve_source_dpi",
]
