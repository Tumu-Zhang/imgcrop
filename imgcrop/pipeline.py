"""裁剪编排：把一批照片按各自的裁剪方案裁好，再按统一规格输出。

管线本身不关心"主体框从哪来"—— 自动分割、点选兜底、手动拖框、锁定复用
都只是产出一个 Box，于是全自动与人工干预走的是同一条代码路径。这也意味着
界面与命令行可以共享同一条输出链路：裁完直接按目标像素/DPI/格式出图。

逐文件流式处理，不需要先把整批读进来：批内一致性由「主体归一化取景 +
固定输出像素」共同保证（见 crop.py 的说明）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from imgcrop import crop, encode, ingest, render
from imgcrop.crop import CropSpec
from imgcrop.encode import EncodeResult, VolumeLimitUnreachable
from imgcrop.geometry import Box
from imgcrop.model import (
    ConflictPolicy,
    OutputSpec,
    SpecError,
    TargetGeometry,
    fit_within,
)

ProgressCallback = Callable[["CropProgress"], None]
CancelCheck = Callable[[], bool]


@dataclass
class CropPlan:
    """一张图的裁剪方案：规格 +（可选的）主体掩膜。

    掩膜只在做倾斜校正时才需要：自动分割与点选都能给出掩膜，手动拖框给不了，
    所以它不是必需项 —— 缺掩膜时只是不做倾斜校正。
    """

    spec: CropSpec
    mask: np.ndarray | None = None


CropProvider = Callable[[Path], "CropPlan | CropSpec | None"]


def _no_crop(_source: Path) -> CropPlan | None:
    """默认策略：不裁剪，只按输出规格重采样。"""
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
    geometry: TargetGeometry | None = None
    num_bytes: int = 0
    encode: EncodeResult | None = None
    box: Box | None = None
    """实际生效的裁剪窗（原图坐标系）；未裁剪时为 None。"""

    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed: float = 0.0

    @property
    def summary(self) -> str:
        if not self.ok:
            return f"失败：{self.error}"
        parts = []
        if self.geometry:
            parts.append(self.geometry.describe())
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
        """本批输出的像素尺寸集合。多于一种就说明批内还不完全一致。"""
        return {r.geometry.pixel_size for r in self.results if r.ok and r.geometry}


# ----------------------------------------------------------------------
# 输出路径
# ----------------------------------------------------------------------
def default_output_dir(source_dir: Path) -> Path:
    """不指定输出目录时，落在源文件同级的 cropped 子目录，避免污染原始素材。"""
    return Path(source_dir) / "cropped"


def _resolve_output_dir(spec: OutputSpec, source: Path) -> Path:
    return Path(spec.output_dir) if spec.output_dir else default_output_dir(source.parent)


def build_output_name(spec: OutputSpec, source: Path) -> str:
    template = spec.name_template or "{stem}"
    try:
        base = template.format(stem=source.stem)
    except (KeyError, IndexError, ValueError) as exc:
        raise SpecError(f"命名模板不合法：{template} —— {exc}") from exc
    return f"{base}{spec.fmt.extension}"


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
            result = self._crop_image(source)
        except (VolumeLimitUnreachable, SpecError) as exc:
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
    def _crop_image(self, source: Path) -> CropResult:
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

        # 裁剪改变了图的尺寸，所以宽高比要按裁剪后的结果来解析 ——
        # 这保证"裁完直接按投稿规格出图"是按裁剪结果的形状算的。
        aspect = image.size[0] / image.size[1]
        geometry = self.spec.resolve(aspect)
        geometry, scale_warnings = fit_within(
            geometry, image.size, allow_upscale=self.spec.allow_upscale
        )

        image, resize_warnings = render.resize_raster(image, geometry.pixel_size)

        out_path = self._plan_output(source)
        if out_path is None:
            return CropResult(
                source=source, out_path=None, ok=False, box=box,
                error="同名文件已存在，按「跳过」策略未输出",
                warnings=loaded.warnings + crop_warnings,
            )

        encoded = encode.encode_to_file(image, out_path, self.spec)
        return CropResult(
            source=source, out_path=out_path, ok=True, box=box,
            geometry=geometry, num_bytes=encoded.num_bytes, encode=encoded,
            warnings=loaded.warnings + crop_warnings + scale_warnings
            + resize_warnings + encoded.warnings,
        )

    def _apply_crop(self, image: Image.Image, plan: CropPlan | None) -> tuple[Image.Image, Box | None, list[str]]:
        if plan is None or not plan.spec.enabled:
            return image, None, []
        outcome = crop.apply_crop(image, plan.spec, plan.mask)
        return outcome.image, outcome.box, list(outcome.warnings)

    # -- 内部工具 -------------------------------------------------------
    def _plan_output(self, source: Path) -> Path | None:
        out_dir = _resolve_output_dir(self.spec, source)
        name = build_output_name(self.spec, source)
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
