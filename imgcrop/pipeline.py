"""裁剪编排：把一批照片按各自的裁剪方案裁好，再按"所见即所得"输出。

管线本身不关心"主体框从哪来"—— 自动分割、点选、手动拖框、锁定复用
都只是产出一个 Box，于是全自动与人工干预走的是同一条代码路径。这也意味着
界面与命令行可以共享同一条输出链路。

1.1.0 瘦身定案（D5/D6/D11）：输出不再做任何重采样 ——
    成品像素 = 裁剪窗在当前旋转角下的原始像素，1:1 落盘；
    格式跟随源文件（JPG→JPG、PNG→PNG 保透明、TIFF→PNG、其余→JPG），
    DPI 元数据跟随源图（缺省 300）。
"输出出现了 N 种尺寸"因此有了新含义：尺寸由每张图的取景框决定，
批内不一致通常意味着取景框不一致 —— 这正是复核时想知道的事。

逐文件流式处理，不需要先把整批读进来。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from imgcrop import crop, encode, ingest, render
from imgcrop.crop import CropSpec
from imgcrop.encode import EncodeResult
from imgcrop.geometry import Box
from imgcrop.model import ConflictPolicy, OutputSpec, SpecError

ProgressCallback = Callable[["CropProgress"], None]
CancelCheck = Callable[[], bool]


@dataclass
class CropPlan:
    """一张图的裁剪方案：规格 +（可选的）主体掩膜。

    掩膜只在做倾斜校正时才需要：自动分割与点选都能给出掩膜，手动拖框给不了，
    所以它不是必需项 —— 缺掩膜时只是不做倾斜校正。
    """

    spec: CropSpec
    mask: object | None = None


CropProvider = Callable[[Path], "CropPlan | CropSpec | None"]


def _no_crop(_source: Path) -> CropPlan | None:
    """默认策略：不裁剪，原图直接按源格式落盘。"""
    return None


def _as_plan(value: CropPlan | CropSpec | None) -> CropPlan | None:
    """允许直接把 CropSpec 交给管线，省掉包一层的噪音。"""
    if value is None:
        return None
    return value if isinstance(value, CropPlan) else CropPlan(spec=value)


# ----------------------------------------------------------------------
# 进度与结果
# ----------------------------------------------------------------------
@dataclass
class CropProgress:
    completed: int
    total: int
    current: str


@dataclass
class CropResult:
    """一张图的裁剪输出结果（成功或失败）。"""

    source: Path
    out_path: Path | None
    ok: bool
    out_size: tuple[int, int] | None = None
    """成品像素尺寸 = 裁剪窗尺寸（1:1，无重采样）。"""
    num_bytes: int = 0
    encode: EncodeResult | None = None
    box: Box | None = None
    """实际生效的裁剪窗（工作图坐标系）；未取景时为 None。"""

    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed: float = 0.0

    @property
    def summary(self) -> str:
        if not self.ok:
            return f"失败：{self.error}"
        parts = []
        if self.out_size:
            parts.append(f"{self.out_size[0]} x {self.out_size[1]} px")
        if self.encode:
            parts.append(self.encode.describe())
        return " / ".join(parts)


@dataclass
class CropReport:
    results: list[CropResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.warnings)

    @property
    def success(self) -> bool:
        return self.fail_count == 0 and self.ok_count > 0

    def pixel_sizes(self) -> set[tuple[int, int]]:
        """本批输出的像素尺寸集合。多于一种就说明批内取景还不完全一致。"""
        return {r.out_size for r in self.results if r.ok and r.out_size}


# ----------------------------------------------------------------------
# 输出路径
# ----------------------------------------------------------------------
def default_output_dir(source_dir: Path) -> Path:
    """不指定输出目录时，落在源文件同级的 cropped 子目录，避免污染原始素材。"""
    return Path(source_dir) / "cropped"


def _resolve_output_dir(spec: OutputSpec, source: Path) -> Path:
    return Path(spec.output_dir) if spec.output_dir else default_output_dir(source.parent)


def build_output_name(spec: OutputSpec, source: Path, extension: str) -> str:
    """按命名模板与**输出格式**的扩展名拼出文件名。

    扩展名由编码层按源格式解析（D6 跟随源），不进模板变量 —— 模板里的
    {stem} 是唯一变量，写 {index} 这类未知占位符直接报 SpecError。
    """
    template = spec.name_template or "{stem}"
    try:
        base = template.format(stem=source.stem)
    except (KeyError, IndexError, ValueError) as exc:
        raise SpecError(f"命名模板不合法：{template} —— {exc}") from exc
    return f"{base}{extension}"


def resolve_conflict(path: Path, policy: ConflictPolicy) -> Path | None:
    """按冲突策略决定最终路径；SKIP 时返回 None。"""
    if not path.exists():
        return path
    if policy is ConflictPolicy.OVERWRITE:
        return path
    if policy is ConflictPolicy.SKIP:
        return None
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


# ----------------------------------------------------------------------
# 裁剪器
# ----------------------------------------------------------------------
class Cropper:
    """按一份 OutputSpec 批量裁剪输出。

    可复用的长生命周期对象：模型会话由外部持有（见 segment.py 的懒加载），
    这里只负责"一张一张地裁、编码、落盘"。
    """

    def __init__(
        self,
        spec: OutputSpec,
        *,
        crop_provider: CropProvider | None = None,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> None:
        self.spec = spec
        self.crop_provider: CropProvider = crop_provider or _no_crop
        self.progress = progress
        self.should_cancel = should_cancel or (lambda: False)

    # -- 公共入口 -------------------------------------------------------
    def run(self, paths: Iterable[Path | str]) -> CropReport:
        sources = [Path(p) for p in paths]
        report = CropReport()
        total = len(sources)
        for index, source in enumerate(sources):
            if self.should_cancel():
                break
            self._emit(index, total, source.name)
            report.results.append(self.crop_one(source))
        return report

    def crop_one(self, source: Path) -> CropResult:
        started = time.perf_counter()
        info = ingest.probe(source)
        if info.error:
            return CropResult(
                source=source, out_path=None, ok=False, error=info.error,
                elapsed=time.perf_counter() - started,
            )

        try:
            self.spec.validate()
        except SpecError as exc:
            return CropResult(
                source=source, out_path=None, ok=False,
                error=f"输出规格有误：{exc}",
                elapsed=time.perf_counter() - started,
            )

        try:
            result = self._crop_image(source, info)
        except SpecError as exc:
            return CropResult(
                source=source, out_path=None, ok=False, error=str(exc),
                elapsed=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - 单张失败不应中断整批
            return CropResult(
                source=source, out_path=None, ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed=time.perf_counter() - started,
            )

        result.elapsed = time.perf_counter() - started
        return result

    # -- 单张实现 -------------------------------------------------------
    def _crop_image(self, source: Path, info: ingest.SourceInfo) -> CropResult:
        loaded = render.load_raster(source)
        plan = _as_plan(self.crop_provider(source))
        image = loaded.image
        crop_warnings: list[str] = []
        box: Box | None = None

        if plan is not None and plan.spec.enabled:
            outcome = crop.apply_crop(image, plan.spec, plan.mask)
            image = outcome.image
            box = outcome.box
            crop_warnings = list(outcome.warnings)

        # 格式跟随源（D6）、DPI 跟随源（D11）—— 都是编码层的内部决策
        fmt = encode.resolve_output_format(source.suffix)
        dpi = encode.resolve_source_dpi(info.dpi)

        out_path = self._plan_output(source, fmt)
        if out_path is None:
            return CropResult(
                source=source, out_path=None, ok=False, box=box,
                error="同名文件已存在，按「跳过」策略未输出",
                warnings=loaded.warnings + crop_warnings,
            )

        encoded = encode.encode_to_file(image, out_path, fmt, dpi=dpi)
        return CropResult(
            source=source, out_path=out_path, ok=True, box=box,
            out_size=image.size, num_bytes=encoded.num_bytes, encode=encoded,
            warnings=loaded.warnings + crop_warnings + encoded.warnings,
        )

    # -- 内部工具 -------------------------------------------------------
    def _plan_output(self, source: Path, fmt: str) -> Path | None:
        out_dir = _resolve_output_dir(self.spec, source)
        extension = encode.extension_for_format(fmt)
        name = build_output_name(self.spec, source, extension)
        return resolve_conflict(out_dir / name, self.spec.on_conflict)

    def _emit(self, index: int, total: int, current: str) -> None:
        if self.progress:
            self.progress(CropProgress(index, total, current))


def crop_batch(
    paths: Iterable[Path | str],
    spec: OutputSpec,
    *,
    crop_provider: CropProvider | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> CropReport:
    """一次性裁剪的便捷入口。"""
    return Cropper(
        spec,
        crop_provider=crop_provider,
        progress=progress,
        should_cancel=should_cancel,
    ).run(paths)


__all__ = [
    "CropPlan",
    "CropProgress",
    "CropReport",
    "CropResult",
    "Cropper",
    "build_output_name",
    "crop_batch",
    "default_output_dir",
    "resolve_conflict",
]
