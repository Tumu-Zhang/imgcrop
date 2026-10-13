"""输出规格的数据模型与几何解析。

这个模块是整个工具的地基，只做一件事：把"我想要的输出"翻译成
"确定的像素尺寸 + 确定的 DPI"。

关键概念
--------
源宽高比 (aspect ratio, AR)
    纯几何量，等于 源像素宽/源像素高（栅格图）或 页面宽/页面高（矢量页）。
    它与 DPI 无关，因此尺寸拟合判定完全只依赖 AR。

两条互不干扰的链路
    1. 物理模式：给 定物理尺寸(cm/mm/inch) + DPI  ->  像素 = 尺寸(inch) * DPI
    2. 像素模式：给 定像素尺寸 + DPI              ->  物理尺寸 = 像素 / DPI

    两者最终都归约成同一个 TargetGeometry，下游渲染层不需要关心用户用了哪种模式。

等比 (keep_aspect) 的三种情形
    - 只给宽  -> 高 = 宽 / AR，输出宽精确等于所填值
    - 只给高  -> 宽 = 高 * AR，输出高精确等于所填值
    - 宽高都给 -> 在目标框内等比内接(fit)，不会超出框，也不会拉伸变形
    - 关闭等比且宽高都给 -> 精确拉伸到目标框（允许变形，等同 PS 取消"保持长宽比"）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

MM_PER_INCH = 25.4


class SizeMode(str, Enum):
    """尺寸输入方式。"""

    PHYSICAL = "physical"
    PIXELS = "pixels"


class Unit(str, Enum):
    """物理长度单位。"""

    MM = "mm"
    CM = "cm"
    INCH = "inch"

    @property
    def per_inch(self) -> float:
        """1 英寸等于多少个该单位。"""
        return {Unit.MM: 25.4, Unit.CM: 2.54, Unit.INCH: 1.0}[self]

    @property
    def label(self) -> str:
        return {"mm": "毫米", "cm": "厘米", "inch": "英寸"}[self.value]


class OutputFormat(str, Enum):
    """支持的输出格式。"""

    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"
    PDF = "pdf"

    @property
    def extension(self) -> str:
        return {OutputFormat.TIFF: ".tif", OutputFormat.PNG: ".png",
                OutputFormat.JPEG: ".jpg", OutputFormat.PDF: ".pdf"}[self]

    @property
    def is_lossless(self) -> bool:
        return self in (OutputFormat.TIFF, OutputFormat.PNG)

    @property
    def label(self) -> str:
        return {OutputFormat.TIFF: "TIFF（无损，投稿首选）",
                OutputFormat.PNG: "PNG（无损，带透明通道）",
                OutputFormat.JPEG: "JPEG（有损，体积小）",
                OutputFormat.PDF: "PDF（单页图，矢量阅读器可缩放）"}[self]


class TiffCompression(str, Enum):
    """TIFF 压缩方式。体积上限优先时会自动从无损档退到有损档。"""

    LZW = "lzw"          # 无损，投稿最通用
    DEFLATE = "deflate"  # 无损，压缩率通常略高于 LZW
    NONE = "none"        # 不压缩
    JPEG = "jpeg"        # 有损，仅在必须压进体积上限时使用


class ConflictPolicy(str, Enum):
    """输出文件重名时的处理方式。"""

    OVERWRITE = "overwrite"
    RENAME = "rename"
    SKIP = "skip"


class SpecError(ValueError):
    """输出规格本身不合法（例如宽高都没填）。"""


@dataclass(frozen=True)
class TargetGeometry:
    """解析后的最终几何：物理尺寸 + DPI，像素尺寸由两者算出。"""

    width_in: float
    height_in: float
    dpi: int

    @property
    def pixel_size(self) -> tuple[int, int]:
        """返回 (宽像素, 高像素)，至少 1 像素。"""
        return (
            max(1, int(round(self.width_in * self.dpi))),
            max(1, int(round(self.height_in * self.dpi))),
        )

    @property
    def aspect(self) -> float:
        return self.width_in / self.height_in

    def describe(self) -> str:
        w, h = self.pixel_size
        return f"{w} x {h} px @ {self.dpi} dpi"


@dataclass
class OutputSpec:
    """一次转换的全部输出要求。所有输入源的设置共用同一份规格。"""

    fmt: OutputFormat = OutputFormat.TIFF
    size_mode: SizeMode = SizeMode.PHYSICAL

    # --- 尺寸：物理模式 ---
    phys_unit: Unit = Unit.CM
    phys_width: float | None = None
    phys_height: float | None = None

    # --- 尺寸：像素模式 ---
    px_width: int | None = None
    px_height: int | None = None

    dpi: int = 300
    keep_aspect: bool = True
    allow_upscale: bool = True

    # --- 编码细节 ---
    tiff_compression: TiffCompression = TiffCompression.LZW
    jpeg_quality: int = 95
    max_bytes: int | None = None  # 单张图的体积上限，None 表示不限
    # 无损格式（TIFF/PNG）在无损档位压不进体积上限时，是否允许自动改为有损。
    # 无论开关如何，实际发生有损降级时都会回传警告，不会静默降质。
    lossy_fallback: bool = True

    # --- 输出落盘 ---
    output_dir: Path | None = None
    name_template: str = "{stem}"  # 可用 {stem} {index} {ext}
    on_conflict: ConflictPolicy = ConflictPolicy.RENAME

    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 尺寸解析
    # ------------------------------------------------------------------
    def target_box_inches(self) -> tuple[float | None, float | None]:
        """把用户填写的尺寸统一换算成英寸。未填的一边返回 None。"""
        if self.size_mode is SizeMode.PHYSICAL:
            w = self.phys_width / self.phys_unit.per_inch if self.phys_width else None
            h = self.phys_height / self.phys_unit.per_inch if self.phys_height else None
            return w, h
        w = self.px_width / self.dpi if self.px_width else None
        h = self.px_height / self.dpi if self.px_height else None
        return w, h

    def pixel_box(self) -> tuple[int | None, int | None]:
        """用户直接指定的像素框（仅像素模式有意义），否则返回由物理尺寸推出的像素。"""
        if self.size_mode is SizeMode.PIXELS:
            return self.px_width, self.px_height
        w_in, h_in = self.target_box_inches()
        return (
            int(round(w_in * self.dpi)) if w_in else None,
            int(round(h_in * self.dpi)) if h_in else None,
        )

    def resolve(self, src_aspect: float | None) -> TargetGeometry:
        return resolve_geometry(self, src_aspect)

    def validate(self) -> None:
        if self.dpi <= 0:
            raise SpecError("DPI 必须为正整数")
        if self.size_mode is SizeMode.PHYSICAL:
            if not self.phys_width and not self.phys_height:
                raise SpecError("请至少填写宽度或高度")
            for value in (self.phys_width, self.phys_height):
                if value is not None and value <= 0:
                    raise SpecError("物理尺寸必须大于 0")
            if not self.keep_aspect and not (self.phys_width and self.phys_height):
                raise SpecError("取消等比时必须同时填写宽度和高度")
        else:
            if not self.px_width and not self.px_height:
                raise SpecError("请至少填写宽度或高度")
            for value in (self.px_width, self.px_height):
                if value is not None and value <= 0:
                    raise SpecError("像素尺寸必须大于 0")
            if not self.keep_aspect and not (self.px_width and self.px_height):
                raise SpecError("取消等比时必须同时填写宽度和高度")
        if not (1 <= self.jpeg_quality <= 100):
            raise SpecError("JPEG 质量需在 1-100 之间")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise SpecError("体积上限必须大于 0")


def resolve_geometry(spec: OutputSpec, src_aspect: float | None) -> TargetGeometry:
    """把规格与源宽高比合成为确定的 TargetGeometry。

    src_aspect 为 None 时表示源是矢量页但调用方尚未提供页面比例；
    此时只允许单边定尺寸，双边定尺寸会按填写的框直接输出。
    """
    spec.validate()
    box_w, box_h = spec.target_box_inches()

    if not spec.keep_aspect:
        # 精确拉伸：宽高必须都给（validate 已保证）
        assert box_w is not None and box_h is not None
        return TargetGeometry(box_w, box_h, spec.dpi)

    if src_aspect is None or src_aspect <= 0:
        if box_w is None or box_h is None:
            raise SpecError(
                "无法确定源图宽高比：请只填一边尺寸，或提供可读取的源文件"
            )
        return TargetGeometry(box_w, box_h, spec.dpi)

    if box_w is None and box_h is None:  # pragma: no cover - validate 已拦截
        raise SpecError("请至少填写宽度或高度")

    if box_h is None:
        # 只给宽：宽度精确
        return TargetGeometry(box_w, box_w / src_aspect, spec.dpi)

    if box_w is None:
        # 只给高：高度精确
        return TargetGeometry(box_h * src_aspect, box_h, spec.dpi)

    # 宽高都给：在框内等比内接
    if src_aspect >= box_w / box_h:
        # 源更扁 -> 宽度顶满
        return TargetGeometry(box_w, box_w / src_aspect, spec.dpi)
    return TargetGeometry(box_h * src_aspect, box_h, spec.dpi)


def fit_within(target: TargetGeometry, src_pixels: tuple[int, int], *, allow_upscale: bool
               ) -> tuple[TargetGeometry, list[str]]:
    """把目标几何限制在源像素能力内（用于"不放大"）。

    返回可能被收缩后的几何，以及需要提示给用户的警告。
    """
    warnings: list[str] = []
    if allow_upscale:
        return target, warnings

    src_w, src_h = src_pixels
    if src_w <= 0 or src_h <= 0:
        return target, warnings

    dst_w, dst_h = target.pixel_size
    scale = min(dst_w / src_w, dst_h / src_h)
    if scale <= 1.0:
        # 目标是缩小或等大，不属于"放大"，照常执行
        return target, warnings

    # scale > 1：目标像素多于源像素，此时放大只会产生插值假细节，按源像素输出

    warnings.append(
        f"源图为 {src_w}x{src_h} px，低于目标 {dst_w}x{dst_h} px；"
        f"已按原始像素输出（未放大），实际物理尺寸约为 "
        f"{src_w / target.dpi:.2f} x {src_h / target.dpi:.2f} 英寸"
    )
    return (
        TargetGeometry(src_w / target.dpi, src_h / target.dpi, target.dpi),
        warnings,
    )


def format_bytes(num: int) -> str:
    """人类可读的体积。"""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"  # pragma: no cover


def inches_to_unit(value_in: float, unit: Unit) -> float:
    return value_in * unit.per_inch


def round_half_up(value: float) -> int:
    """四舍五入到整数，用于像素换算显示（避免 math 的银行家舍入）。"""
    return int(math.floor(value + 0.5))
