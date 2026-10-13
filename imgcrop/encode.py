"""编码层：把位图写成目标格式，并写出正确的 DPI 元数据。

体积上限的处理原则：能无损达标就无损；无损实在达不到，才在用户允许的前提下
退到有损，并且一定回传警告，绝不静默降质。

实现细节：JPEG 的 optimize=True 会让 libjpeg 走二次扫描，遇到没有真实文件描述符
的内存流（BytesIO）会在稍大的图上直接失败（"Suspension not allowed here"）。
所以体积测量统一走有 fileno 的临时文件，而不是 BytesIO。
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
from PIL import Image

from imgcrop.model import (
    OutputFormat,
    OutputSpec,
    TiffCompression,
    format_bytes,
)

Image.MAX_IMAGE_PIXELS = 600_000_000

# 体积上限搜索的兜底质量，低于它就没必要继续试了
MIN_JPEG_QUALITY = 5

# 规格层的压缩名 -> Pillow 认识的 TIFF 压缩名。
# Pillow 对无法识别的名字不会报错，而是直接落成未压缩（tag 259 = 1），必须显式映射。
PILLOW_TIFF_COMPRESSION = {
    TiffCompression.LZW.value: "tiff_lzw",
    TiffCompression.DEFLATE.value: "tiff_adobe_deflate",
    TiffCompression.NONE.value: "raw",
    TiffCompression.JPEG.value: "jpeg",
}


@dataclass
class EncodeResult:
    path: Path
    num_bytes: int
    quality_used: int | None = None
    compression_used: str | None = None
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [format_bytes(self.num_bytes)]
        if self.quality_used is not None:
            parts.append(f"quality={self.quality_used}")
        if self.compression_used:
            parts.append(self.compression_used)
        return " / ".join(parts)


class VolumeLimitUnreachable(RuntimeError):
    """即使压到最低档也无法满足体积上限。"""


# ----------------------------------------------------------------------
# 模式适配
# ----------------------------------------------------------------------
def _prepare_for_format(image: Image.Image, fmt: OutputFormat) -> tuple[Image.Image, list[str]]:
    """把位图调整成目标格式能承载的模式。"""
    warnings: list[str] = []
    mode = image.mode

    if fmt is OutputFormat.JPEG:
        if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
            # JPEG 不支持高位深，按 PS 的"16 位 -> 8 位"惯例取高字节
            arr = np.asarray(image)
            if arr.dtype != np.uint16:
                arr = np.clip(arr, 0, 65535).astype(np.uint16)
            image = Image.fromarray((arr >> 8).astype(np.uint8), mode="L")
            warnings.append("16 位灰度已按高字节降为 8 位以适配 JPEG（等同 PS 的 8 位/通道转换）")
        elif mode == "I":
            arr = np.clip(np.asarray(image), 0, 65535).astype(np.uint16)
            image = Image.fromarray((arr >> 8).astype(np.uint8), mode="L")
            warnings.append("32 位整型图已降为 8 位灰度以适配 JPEG")
        elif mode == "F":
            arr = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="L")
            warnings.append("浮点图已降为 8 位灰度以适配 JPEG")
        elif mode not in ("L", "RGB", "CMYK"):
            image = image.convert("RGB")
            warnings.append(f"色彩模式 {mode} 已转换为 RGB 以适配 JPEG")
        return image, warnings

    if fmt is OutputFormat.PDF:
        if mode not in ("L", "RGB", "CMYK"):
            if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
                arr = np.asarray(image)
                image = Image.fromarray((arr >> 8).astype(np.uint8), mode="L")
                warnings.append("16 位灰度已降为 8 位以适配 PDF")
            else:
                image = image.convert("RGB")
        return image, warnings

    if fmt is OutputFormat.PNG:
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

    # TIFF 是科研场景里最宽容的容器，保留原模式
    return image, warnings


# ----------------------------------------------------------------------
# 单次编码 / 体积测量
# ----------------------------------------------------------------------
def _save_to(
    image: Image.Image,
    target: str | Path | IO[bytes],
    fmt: OutputFormat,
    dpi: int,
    *,
    quality: int,
    compression: str,
) -> None:
    dpi_tuple = (dpi, dpi)

    if fmt is OutputFormat.JPEG:
        _save_jpeg(image, target, quality=quality, dpi=dpi_tuple)
    elif fmt is OutputFormat.PNG:
        image.save(
            target,
            format="PNG",
            dpi=dpi_tuple,
            compress_level=9,
            optimize=True,
        )
    elif fmt is OutputFormat.TIFF:
        params: dict = {
            "dpi": dpi_tuple,
            "compression": PILLOW_TIFF_COMPRESSION.get(compression, compression),
        }
        if compression == "jpeg":
            params["quality"] = quality
        image.save(target, format="TIFF", **params)
    elif fmt is OutputFormat.PDF:
        image.save(target, format="PDF", resolution=float(dpi), quality=quality)
    else:  # pragma: no cover - OutputFormat 已穷举
        raise ValueError(f"未知输出格式：{fmt}")


def _reset_target(target: str | Path | IO[bytes]) -> None:
    """重写同一个 file object 前把它清空，避免上一次尝试的残留字节。"""
    if hasattr(target, "seek"):
        target.seek(0)
        if hasattr(target, "truncate"):
            target.truncate(0)


def _save_jpeg(
    image: Image.Image, target: str | Path | IO[bytes], *, quality: int, dpi: tuple[int, int]
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
        quality=quality,
        dpi=dpi,
        subsampling=0,      # 4:4:4，保住细线条与彩色文字的边缘
        progressive=False,
    )
    try:
        image.save(target, optimize=True, **common)
    except OSError:
        _reset_target(target)
        image.save(target, optimize=False, **common)


def _measure(
    image: Image.Image, fmt: OutputFormat, dpi: int, *, quality: int, compression: str
) -> int:
    """编码一次并返回字节数，不保留结果。"""
    with tempfile.TemporaryFile() as tmp:
        _save_to(image, tmp, fmt, dpi, quality=quality, compression=compression)
        tmp.seek(0, 2)  # 定位到末尾：编码过程中可能有 seek，以文件真实长度为准
        return tmp.tell()


def _search_quality(
    measure: Callable[[int], int], limit: int, start_quality: int
) -> int | None:
    """二分搜索满足体积上限的最高质量；返回 None 表示连最低质量都达不到。"""
    if measure(MIN_JPEG_QUALITY) > limit:
        return None

    best = MIN_JPEG_QUALITY
    lo, hi = MIN_JPEG_QUALITY, start_quality
    while lo <= hi:
        mid = (lo + hi) // 2
        if measure(mid) <= limit:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _decide_params(
    prepared: Image.Image, spec: OutputSpec, warnings: list[str]
) -> tuple[int, str]:
    """决定最终的 (quality, compression)，并保证满足体积上限或给出警告。

    没有设置体积上限时完全不做额外编码，直接返回用户设定的参数。
    """
    fmt = spec.fmt
    limit = spec.max_bytes
    quality = spec.jpeg_quality
    compression = spec.tiff_compression.value

    if limit is None:
        return quality, compression

    size = _measure(prepared, fmt, spec.dpi, quality=quality, compression=compression)
    if size <= limit:
        return quality, compression

    if fmt is OutputFormat.TIFF:
        # 先在两个无损档位之间尝试，都不行才考虑有损
        for candidate in ("lzw", "deflate"):
            if candidate == compression:
                continue
            if _measure(prepared, fmt, spec.dpi, quality=quality, compression=candidate) <= limit:
                warnings.append(
                    f"为满足体积上限，TIFF 压缩由 {compression} 改用 {candidate}（仍为无损）"
                )
                return quality, candidate

        if not spec.lossy_fallback:
            raise VolumeLimitUnreachable(
                f"TIFF 无损压缩后为 {format_bytes(size)}，超过上限 {format_bytes(limit)}；"
                "请降低 DPI 或输出尺寸（或允许有损压缩兜底）"
            )

        chosen = _search_quality(
            lambda q: _measure(prepared, fmt, spec.dpi, quality=q, compression="jpeg"),
            limit,
            quality,
        )
        if chosen is None:
            floor_size = _measure(
                prepared, fmt, spec.dpi, quality=MIN_JPEG_QUALITY, compression="jpeg"
            )
            raise VolumeLimitUnreachable(
                f"TIFF 即使使用有损 JPEG 压缩（质量 {MIN_JPEG_QUALITY}）仍为 "
                f"{format_bytes(floor_size)}，超过上限 {format_bytes(limit)}；"
                "请降低 DPI 或输出尺寸"
            )
        warnings.append(
            f"为满足体积上限 {format_bytes(limit)}，TIFF 已改用有损 JPEG 压缩（quality={chosen}）。"
            "如需完全无损，请降低 DPI、缩小尺寸，或提高体积上限。"
        )
        return chosen, "jpeg"

    if fmt in (OutputFormat.JPEG, OutputFormat.PDF):
        chosen = _search_quality(
            lambda q: _measure(prepared, fmt, spec.dpi, quality=q, compression=compression),
            limit,
            quality,
        )
        if chosen is None:
            floor_size = _measure(
                prepared, fmt, spec.dpi, quality=MIN_JPEG_QUALITY, compression=compression
            )
            raise VolumeLimitUnreachable(
                f"即使质量降到 {MIN_JPEG_QUALITY}，体积仍为 {format_bytes(floor_size)}，"
                f"超过上限 {format_bytes(limit)}；请降低 DPI 或输出尺寸"
            )
        warnings.append(f"为满足体积上限 {format_bytes(limit)}，质量由 {quality} 调整为 {chosen}")
        return chosen, compression

    # PNG：无损格式，压不动就只能如实报告，不做静默降级
    if not spec.lossy_fallback:
        raise VolumeLimitUnreachable(
            f"PNG 为无损格式，最小 {format_bytes(size)}，超过上限 {format_bytes(limit)}；"
            "请降低 DPI、缩小尺寸，或改用 JPEG"
        )
    warnings.append(
        f"PNG 为无损格式，无法压到体积上限 {format_bytes(limit)} 之下（当前 {format_bytes(size)}）。"
        "建议降低 DPI/尺寸或改用 JPEG；已按无损输出。"
    )
    return quality, compression


def _quality_for_report(fmt: OutputFormat, quality: int, compression: str) -> int | None:
    if fmt in (OutputFormat.JPEG, OutputFormat.PDF):
        return quality
    if compression == "jpeg":
        return quality
    return None


# ----------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------
def encode_to_file(image: Image.Image, path: Path, spec: OutputSpec) -> EncodeResult:
    """编码并落盘。"""
    path = Path(path)
    prepared, warnings = _prepare_for_format(image, spec.fmt)
    quality, compression = _decide_params(prepared, spec, warnings)

    path.parent.mkdir(parents=True, exist_ok=True)
    _save_to(prepared, path, spec.fmt, spec.dpi, quality=quality, compression=compression)

    return EncodeResult(
        path=path,
        num_bytes=path.stat().st_size,
        quality_used=_quality_for_report(spec.fmt, quality, compression),
        compression_used=compression if spec.fmt is OutputFormat.TIFF else None,
        warnings=warnings,
    )


def encode_image(image: Image.Image, spec: OutputSpec) -> tuple[bytes, EncodeResult]:
    """编码到内存，返回 (字节, 结果元信息)。供需要二次处理的调用方使用。"""
    prepared, warnings = _prepare_for_format(image, spec.fmt)
    quality, compression = _decide_params(prepared, spec, warnings)

    with tempfile.TemporaryFile() as tmp:
        _save_to(prepared, tmp, spec.fmt, spec.dpi, quality=quality, compression=compression)
        tmp.seek(0)
        data = tmp.read()

    return data, EncodeResult(
        path=Path(),
        num_bytes=len(data),
        quality_used=_quality_for_report(spec.fmt, quality, compression),
        compression_used=compression if spec.fmt is OutputFormat.TIFF else None,
        warnings=warnings,
    )


__all__ = [
    "EncodeResult",
    "MIN_JPEG_QUALITY",
    "PILLOW_TIFF_COMPRESSION",
    "VolumeLimitUnreachable",
    "encode_image",
    "encode_to_file",
]
