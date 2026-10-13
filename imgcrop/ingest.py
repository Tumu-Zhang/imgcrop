"""输入探测：读出图片的像素尺寸、DPI 与需要注意的地方。

这一层只读文件头、不解码像素，所以界面把一整个文件夹拖进来也能立刻有反馈。
裁剪器只认栅格图；遇到 PDF / PPT / RAW 这类文件，直接给出"先转成图片"的
明确指引，而不是静默跳过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# 裁剪器认的图片后缀（含几个常被忽略的：JFIF / JP2 / PPM / TGA / DIB）
IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".tif", ".tiff", ".bmp",
    ".gif", ".webp", ".jp2", ".ppm", ".pgm", ".tga", ".dib",
}

# 不认的格式里，这几类最常见，给出可操作的下一步而不是一句"不支持"
NEEDS_CONVERSION = {
    ".pdf": "PDF 请先用「图片转换器」导出成 PNG/TIFF 再裁剪",
    ".svg": "矢量图请先导出成 PNG 再裁剪",
    ".ppt": "幻灯片请先导出成 PNG 再裁剪",
    ".pptx": "幻灯片请先导出成 PNG 再裁剪",
    ".eps": "EPS 需要 Ghostscript，或先在 Illustrator 中另存为 PNG",
    ".ai": "Illustrator 文件请先导出为 PNG",
    ".cdr": "CorelDRAW 文件请先导出为 PNG 或 TIFF",
    ".psd": "PSD 请先另存为 TIFF 或 PNG（避免只读合并层的歧义）",
    ".raw": "RAW 请先导出为 TIFF",
    ".cr2": "RAW 请先导出为 TIFF",
    ".nef": "RAW 请先导出为 TIFF",
    ".arw": "RAW 请先导出为 TIFF",
}

MAX_RASTER_PIXELS = 600_000_000  # PIL 解压炸弹阈值，允许超大科研图


class SourceKind(str, Enum):
    """输入类别：裁剪器只有"图片"和"不是图片"两种。"""

    IMAGE = "image"
    UNSUPPORTED = "unsupported"


@dataclass
class SourceInfo:
    """一个输入文件的可展示元信息。"""

    path: Path
    kind: SourceKind
    pixel_size: tuple[int, int] | None = None
    dpi: tuple[float, float] | None = None
    frames: int = 1
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def aspect(self) -> float | None:
        if not self.pixel_size or not self.pixel_size[1]:
            return None
        return self.pixel_size[0] / self.pixel_size[1]

    @property
    def kind_label(self) -> str:
        return {SourceKind.IMAGE: "图片", SourceKind.UNSUPPORTED: "不支持"}[self.kind]

    @property
    def source_dpi_text(self) -> str:
        if self.dpi is None:
            return "无 DPI 信息"
        x, y = self.dpi
        if abs(x - y) < 0.01:
            return f"{x:.0f} dpi"
        return f"{x:.0f} x {y:.0f} dpi"

    def describe_source(self) -> str:
        if self.kind is SourceKind.IMAGE and self.pixel_size:
            return f"{self.pixel_size[0]} x {self.pixel_size[1]} px / {self.source_dpi_text}"
        return "—"


def classify(path: Path | str) -> SourceKind:
    return SourceKind.IMAGE if Path(path).suffix.lower() in IMAGE_SUFFIXES else SourceKind.UNSUPPORTED


def is_image(path: Path | str) -> bool:
    return classify(path) is SourceKind.IMAGE


def probe(path: Path | str) -> SourceInfo:
    """探测单个文件。永不抛异常，失败信息放在 SourceInfo.error 里。"""
    path = Path(path)
    kind = classify(path)

    if not path.exists():
        return SourceInfo(path, SourceKind.UNSUPPORTED, error="文件不存在")
    if kind is SourceKind.UNSUPPORTED:
        suffix = path.suffix.lower()
        message = f"不支持的文件类型：{suffix or '（无扩展名）'}"
        hint = NEEDS_CONVERSION.get(suffix)
        if hint:
            message += f"。{hint}"
        return SourceInfo(path, SourceKind.UNSUPPORTED, error=message)

    info = SourceInfo(path, SourceKind.IMAGE)
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_RASTER_PIXELS
    try:
        with Image.open(path) as img:
            info.pixel_size = img.size
            dpi = img.info.get("dpi")
            if dpi:
                try:
                    x, y = float(dpi[0]), float(dpi[1])
                except (TypeError, ValueError, IndexError):
                    x = y = 0.0
                # 有些工具会写 dpi=(0, 0)；那不是可用信息，按"无 DPI"处理，
                # 否则界面上会出现"0 dpi"这种误导性显示。
                if x > 0 and y > 0:
                    info.dpi = (x, y)
            info.frames = getattr(img, "n_frames", 1)
            if info.frames > 1:
                info.notes.append(f"多帧文件，仅处理第 1 帧（共 {info.frames} 帧）")
                info.frames = 1
            if img.mode in ("P", "LA", "RGBA"):
                info.notes.append(f"含透明通道（{img.mode}），输出时会合成白底")
            if info.dpi is None:
                info.notes.append("源文件无 DPI 元数据，将直接按像素重采样")
    except Exception as exc:  # noqa: BLE001 - 探测失败不应中断批量流程
        return SourceInfo(path, SourceKind.IMAGE, error=f"无法读取：{exc}")
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return info


def gather(paths) -> list[Path]:
    """把"文件 + 目录"的混合输入展开成一份图片清单（目录递归，按名称排序）。"""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*") if p.is_file() and is_image(p)))
        else:
            found.append(path)

    # 去重但保持顺序：同一文件既被单独拖入、又被所在目录覆盖时只处理一次
    seen: set[str] = set()
    unique: list[Path] = []
    for path in found:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


__all__ = [
    "IMAGE_SUFFIXES",
    "MAX_RASTER_PIXELS",
    "NEEDS_CONVERSION",
    "SourceInfo",
    "SourceKind",
    "UnidentifiedImageError",
    "classify",
    "gather",
    "is_image",
    "probe",
]
