"""裁剪链路：把"主体在哪里"变成"一张裁好的图"。

本模块坐在几何内核（geometry）与输出管线（pipeline / encode）之间：
geometry 只算坐标，本模块负责按规格取出图像区域，之后交给 model.OutputSpec
去决定最终像素尺寸、DPI 与格式。

主体框从哪来
------------
本模块不关心主体的框是怎么得到的 —— 自动分割、点选、手动框选、锁定复用
都只是产出同一个 Box。于是"人工兜底"和"全自动"走完全相同的代码路径，
点选只是换了一个框的来源，不是另一条流程。

批内一致
--------
一致性由两件事共同保证，都不需要预先扫描整批：

1. 取景用 frame_for_subject：画面尺寸 = 主体尺寸 x (1 + 2 x 边距)，
   所以主体占比只由参数决定，与拍摄远近无关。
2. 输出分辨率由 OutputSpec 的像素模式固定，与源图大小无关。

因此本模块可以保持逐文件流式处理，不必等整批都读进来。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image

from imgcrop.geometry import (
    Box,
    Frame,
    frame_for_subject,
    mask_angle,
    mask_to_box,
    order_quad,
    pad_to_frame,
    rectify_quad,
    rotate_box,
    rotate_image,
    tilt_is_meaningful,
)

# 默认留白：主体四周各留 8%，即画面长边 = 主体长边 x 1.16。
# 这个值偏紧是有意的 —— 留白太大会让主体在成品里显得小，投稿时白占版面。
DEFAULT_MARGIN = 0.08

# 界面上的比例预设。None 表示跟随主体自身形状（"原始比例"）。
ASPECT_PRESETS: tuple[tuple[str, float | None], ...] = (
    ("原始比例", None),
    ("1:1", 1.0),
    ("4:3", 4.0 / 3.0),
    ("3:2", 3.0 / 2.0),
    ("16:9", 16.0 / 9.0),
    ("3:4", 3.0 / 4.0),
    ("2:3", 2.0 / 3.0),
    ("9:16", 9.0 / 16.0),
)


class CropMode(str, Enum):
    """裁剪方式。"""

    NONE = "none"
    """不裁剪，原图原样进入后续的尺寸/格式处理。"""

    SUBJECT = "subject"
    """按主体取景：以主体框为基准，按 margin 与 aspect 扩张出裁剪窗。"""

    FREE = "free"
    """自由框选：给什么框就裁什么框，不再加边距（用户直接拖出来的框）。"""

    QUAD = "quad"
    """四点透视矫正：把任意四边形拉成正视矩形。"""


@dataclass
class CropSpec:
    """一次裁剪的规格。

    注意本对象是"可序列化的用户意图"，不含掩膜等运行期中间产物 ——
    掩膜通过 apply_crop 的参数传入。这样规格可以直接记忆到配置文件里。
    """

    mode: CropMode = CropMode.NONE

    subject: Box | None = None
    """主体框（原图坐标系）。SUBJECT 用它取景；FREE 直接拿它当裁剪窗。"""

    quad: Sequence[tuple[float, float]] | None = None
    """手动四点（原图坐标系，任意顺序）。仅 QUAD 模式使用。"""

    margin: float = DEFAULT_MARGIN
    """相对主体尺寸的留白比例，0.08 即每边 8%。"""

    aspect: float | None = None
    """目标宽高比；None 表示跟随主体形状。"""

    quad_size: tuple[int, int] | None = None
    """四点矫正的输出尺寸；None 表示按四边形边长自动估算。"""

    rectify_tilt: bool = False
    """是否按主体长轴自动纠正倾斜（需要提供掩膜）。"""

    rotate_deg: float = 0.0
    """手动旋转角（度，正 = 逆时针）。

    与 rectify_tilt **叠加**：勾了自动转正就先按主体长轴转正，再叠这个手动
    角度微调。两者都作用在"画面"上 —— 转完之后仍然按轴对齐的矩形取景，
    所以成品永远是正的。
    """

    box_in_rotated_frame: bool = False
    """subject（或 quad）是否已经是"转正后画布坐标系"里的框。

    自动分割、点选、命令行给的都是**原图坐标**（默认 False）—— apply_crop
    会把框跟着画面一起转。界面里用户是在已经转正的画面上摆框的（True）：
    那个框本身就属于旋转后的坐标系，再转一次就错了。

    这件事必须由调用方说清楚，不能靠猜：轴对齐的框在旋转下不可精确往返
    （转过去再转回来只会变大），猜错一次，预览和导出就对不上了。
    """

    background: tuple[int, int, int] = (255, 255, 255)
    """取景窗越出原图边界时的补边颜色。"""

    origin: str = ""
    """主体框的来源说明（自动 / 点选 / 手动 / 锁定），仅用于日志与界面提示。"""

    warnings: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.mode is not CropMode.NONE

    def validate(self) -> None:
        """规格自检。错误用 SpecError 抛出，交给管线按"单文件失败"处理。"""
        from imgcrop.model import SpecError

        if not self.enabled:
            return
        if self.margin < 0:
            raise SpecError("裁剪边距不能为负")
        if self.aspect is not None and self.aspect <= 0:
            raise SpecError("裁剪比例必须为正")
        if not math.isfinite(self.rotate_deg):
            raise SpecError("旋转角度必须是有限数值")
        if abs(self.rotate_deg) > 180.0:
            raise SpecError("旋转角度请控制在 ±180° 以内")
        if self.mode in (CropMode.SUBJECT, CropMode.FREE):
            if self.subject is None or self.subject.is_empty:
                raise SpecError("尚未确定主体范围，无法裁剪")
        elif self.mode is CropMode.QUAD:
            if not self.quad or len(self.quad) != 4:
                raise SpecError("四点矫正需要恰好 4 个点")
            try:
                order_quad(self.quad)  # 顺便校验四点不退化
            except ValueError as exc:
                raise SpecError(f"四点矫正的角点无效：{exc}") from exc

    def describe(self) -> str:
        if not self.enabled:
            return "不裁剪"
        if self.mode is CropMode.QUAD:
            size = f"{self.quad_size[0]}x{self.quad_size[1]}" if self.quad_size else "按边长"
            return f"四点透视矫正（输出 {size}）"
        share = 1.0 / (1.0 + 2.0 * self.margin)
        ratio = "跟随主体" if self.aspect is None else f"{self.aspect:.3f}"
        head = "自由框选" if self.mode is CropMode.FREE else f"按主体取景（主体占 {share:.0%}）"
        if abs(self.rotate_deg) > 1e-6:
            head += f"，旋转 {self.rotate_deg:+.1f}°"
        return f"{head}，比例 {ratio}"


@dataclass
class CropOutcome:
    """裁剪结果。"""

    image: Image.Image
    box: Box | None
    """实际生效的裁剪窗（已平移到原图坐标系）。QUAD 模式为 None。"""

    exact_scale: bool
    """是否严格满足"主体长边占画面长边 = 1/(1+2*margin)"。"""

    warnings: list[str] = field(default_factory=list)


def fit_mask(mask: np.ndarray | Image.Image, size: tuple[int, int]) -> np.ndarray:
    """把掩膜统一成"指定尺寸的布尔数组"。

    掩膜通常来自模型输入分辨率（1024 见方的 letterbox 画布），而裁剪的坐标
    一律在原图坐标系里 —— 少这一步缩放，倾斜校正就会整体错位。

    **掩膜必须先换回原图空间**（`segment.Proposal.source_mask()` /
    `PointHit.source_mask()`，或 `geometry.Letterbox.to_source_mask`）。
    这里只做等比缩放，不知道 letterbox 的填充区；直接喂模型输入空间的
    掩膜，填充区会参与缩放，宽高比悬殊的图上角度会算错（20 度变 10 度）。
    """
    if isinstance(mask, Image.Image):
        picture = mask.convert("L")
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.dtype == np.bool_:
            picture = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
        else:
            if arr.dtype.kind == "f":
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            picture = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    if picture.size != tuple(size):
        picture = picture.resize(tuple(size), Image.Resampling.BILINEAR)
    return np.asarray(picture) > 127


def planned_box(
    subject: Box,
    *,
    margin: float = DEFAULT_MARGIN,
    aspect: float | None = None,
) -> Box:
    """算出 SUBJECT 模式最终会裁出的窗口（原图坐标系）。

    界面要"所见即所得"就必须先知道这个框 —— 与 apply_crop 内部用的是同一个
    frame_for_subject，所以预览和成品不会各算一套。
    """
    return frame_for_subject(subject, margin=margin, aspect=aspect).box


def normalize_angle(angle: float) -> float:
    """折到 (-180, 180]，免得自动与手动角度加起来绕圈。"""
    return (angle + 180.0) % 360.0 - 180.0


def planned_rotation(
    spec: CropSpec,
    mask: np.ndarray | Image.Image | None = None,
) -> tuple[float, float, list[str]]:
    """这次裁剪会用到的角度：返回 (自动转正角, 手动角度, 提示)。

    两者相加就是实际旋转画面的角度。界面也必须走这个函数 —— 否则"勾了自动
    转正"在预览里看不出来，而导出时图却转了，"预览里看到的 = 导出的"这条
    约定就断了。

    mask 必须是**原图空间**的掩膜（见 segment.Proposal.source_mask）。
    """
    warnings: list[str] = []
    auto = 0.0
    if spec.rectify_tilt and mask is not None and spec.mode is CropMode.SUBJECT:
        measured = mask_angle(mask)
        mask_box = mask_to_box(mask)
        if mask_box is not None and tilt_is_meaningful(mask_box, measured):
            auto = measured
        elif mask_box is not None:
            warnings.append(f"检测到倾斜 {measured:+.1f}°，但主体朝向不明确，未自动旋转")
    return auto, float(spec.rotate_deg), warnings


def _rotation_note(auto_angle: float, manual_angle: float) -> str:
    """把"转了多少、为什么转"写清楚 —— 任何自动改动都要说明白。"""
    if abs(auto_angle) > 1e-6 and abs(manual_angle) > 1e-6:
        total = normalize_angle(auto_angle + manual_angle)
        return (
            f"已旋转 {total:+.1f}°"
            f"（自动转正 {auto_angle:+.1f}° + 手动 {manual_angle:+.1f}°）"
        )
    if abs(auto_angle) > 1e-6:
        return f"已按主体长轴自动转正 {auto_angle:+.1f}°"
    return f"已按手动角度旋转 {manual_angle:+.1f}°"


def apply_crop(
    image: Image.Image,
    spec: CropSpec,
    mask: np.ndarray | Image.Image | None = None,
) -> CropOutcome:
    """按规格裁剪，返回新图。

    image 不会被修改（PIL 的 crop/transform 都返回新对象），调用方可以放心
    保留原图 —— "原图零改动"是这个工具的一贯约定。

    mask 是可选的主体掩膜（分辨率不限，内部会缩放到原图）。只有在做倾斜
    校正时才需要它：手动框选和自由框选没有掩膜，也就没有可靠的朝向信息。
    """
    if not spec.enabled:
        return CropOutcome(image=image, box=None, exact_scale=True)

    spec.validate()
    warnings: list[str] = []

    # 掩膜可能来自模型输入分辨率（1024 见方的 letterbox 画布），而这里的所有
    # 坐标都在原图坐标系里 —— 先把它拉到原图尺寸，否则倾斜校正会整体错位。
    if mask is not None:
        mask = fit_mask(mask, image.size)

    # ---- 四点透视：一条完全独立的路径，不走主体取景 ----
    if spec.mode is CropMode.QUAD:
        assert spec.quad is not None
        out = rectify_quad(image, spec.quad, spec.quad_size)
        warnings.append("已按四点做透视矫正")
        return CropOutcome(image=out, box=None, exact_scale=True, warnings=warnings)

    subject = spec.subject
    assert subject is not None

    # ---- 旋转：自动转正（按主体长轴）与手动角度叠加，作用在"画面"上 ----
    # 转完之后仍然按轴对齐矩形取景，所以成品永远是正的。
    auto_angle, manual_angle, rotation_warnings = planned_rotation(spec, mask)
    warnings.extend(rotation_warnings)
    total_angle = normalize_angle(auto_angle + manual_angle)

    work_image = image
    if abs(total_angle) > 1e-6:
        work_image = rotate_image(image, total_angle, spec.background)
        # 框跟着画面一起转 —— 但界面传来的框已经在转正后的画面上摆好了
        # （box_in_rotated_frame），再转一次就错位了。
        if not spec.box_in_rotated_frame:
            subject = rotate_box(subject, image.size, work_image.size, total_angle)
        warnings.append(_rotation_note(auto_angle, manual_angle))

    # ---- 取景 ----
    if spec.mode is CropMode.FREE:
        frame = Frame(box=subject, exact_scale=True)
    else:
        frame = frame_for_subject(subject, margin=spec.margin, aspect=spec.aspect)
        if not frame.exact_scale:
            warnings.append(
                "主体形状与所选比例冲突，已放大画面保证主体完整，"
                "这张图的主体占比会比同批略小"
            )

    # 归一化取景下裁剪窗可能越出原图，补背景而不是把窗缩回来 ——
    # 缩回来会破坏批内一致的主体占比，补背景只是多一圈留白。
    if frame.needs_padding(work_image.size):
        work_image, shifted = pad_to_frame(work_image, frame.box, spec.background)
        warnings.append("主体贴近画面边缘，已补背景留白以保持主体占比一致")
    else:
        shifted = frame.box

    cropped = work_image.crop((shifted.x1, shifted.y1, shifted.x2, shifted.y2))
    return CropOutcome(
        image=cropped,
        box=shifted,
        exact_scale=frame.exact_scale,
        warnings=warnings,
    )


__all__ = [
    "ASPECT_PRESETS",
    "DEFAULT_MARGIN",
    "CropMode",
    "CropOutcome",
    "CropSpec",
    "apply_crop",
    "fit_mask",
    "normalize_angle",
    "planned_box",
    "planned_rotation",
]
