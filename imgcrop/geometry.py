"""裁剪几何内核：从"主体在哪里"到"裁剪窗在哪里"的全部坐标运算。

这个模块是裁剪功能的地基，只做纯几何 —— 不加载模型、不读磁盘、不碰界面。
所有函数都是坐标与矩形进、坐标与矩形出（少数几个直接作用于 PIL 图像），
因此可以脱离模型和界面单独测试。

与 model.py 的分工
------------------
model.py 负责"输出多大、多少 dpi"（投稿规格）；
本模块负责"从原图里取哪一块、怎么摆正"。两者在管线里是前后关系。

关键概念
--------
Box
    轴对齐矩形，左闭右开（x2/y2 不含在内），全模块统一用这一种表示，
    避免 (x, y, w, h) 四元组在不同函数里换序出错。

Letterbox
    模型吃固定边长的正方形输入，原图要等比缩放后居中贴进去。于是
    "模型输出的坐标"与原图坐标之间差一个缩放加一个平移。这个映射必须
    双向可逆 —— 掩膜贴回原图时的整体偏移是这类功能最常见的 bug。

主体归一化取景 (subject normalization)
    批量照片若只按"固定百分比扩边"，成品里主体的大小会随拍摄距离变化。
    这里改为以主体为基准取景：画面长边 = 主体长边 x (1 + 2 x 边距)。
    同批照片只要主体检测正确，成品中的主体占比与留白就自然一致，
    与拍摄远近无关 —— 这是"批内一致"的实现方式。

四点透视里 Pillow 的坑
    Image.Transform.QUAD 要求的顺序是 左上、左下、右下、右上（非直观序），
    传错会得到一张静默垂直翻转的图。对外接口一律用直观的
    左上、右上、右下、左下，重排在 rectify_quad 内部完成，并由单测锁死。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = [
    "Box",
    "Frame",
    "Letterbox",
    "frame_for_subject",
    "mask_angle",
    "mask_to_box",
    "order_quad",
    "pad_to_frame",
    "quad_output_size",
    "rectify_quad",
    "rotate_box",
    "rotate_image",
    "tilt_is_meaningful",
]

Point = tuple[float, float]

# Pillow 的 QUAD 变换要求"左上、左下、右下、右上"这一非直观顺序。
# 这里把它集中定义一次，其余代码只用直观顺序。
_QUAD_ORDER = (0, 3, 2, 1)


# ======================================================================
# 矩形
# ======================================================================
@dataclass(frozen=True)
class Box:
    """轴对齐矩形，左闭右开：x1/y1 含在内，x2/y2 不含。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def center(self) -> Point:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def aspect(self) -> float:
        """宽高比；高度为 0 时返回 0 而不是抛异常。"""
        return self.width / self.height if self.height else 0.0

    @property
    def elongations(self) -> float:
        """长短边之比，>=1。用于判断主体是否有明确朝向。"""
        return self.long_side / self.short_side if self.short_side else 0.0

    def clipped(self, width: int, height: int) -> Box:
        """裁进 [0, width) x [0, height) 范围内，可能得到空框。"""
        return Box(
            max(0, self.x1), max(0, self.y1),
            min(width, self.x2), min(height, self.y2),
        )

    def expanded(self, margin: float) -> Box:
        """按自身宽高的 margin 比例向四周外扩（0.08 即每边 8%）。"""
        if margin <= 0 or self.is_empty:
            return self
        dx = max(1, round(self.width * margin))
        dy = max(1, round(self.height * margin))
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def union(self, other: Box) -> Box:
        return Box(
            min(self.x1, other.x1), min(self.y1, other.y1),
            max(self.x2, other.x2), max(self.y2, other.y2),
        )

    def intersection(self, other: Box) -> Box:
        return Box(
            max(self.x1, other.x1), max(self.y1, other.y1),
            min(self.x2, other.x2), min(self.y2, other.y2),
        )

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"({self.x1},{self.y1})-({self.x2},{self.y2}) {self.width}x{self.height}"


def _centered_box(cx: float, cy: float, width: float, height: float) -> Box:
    """以 (cx, cy) 为中心、给定浮点宽高，取整成整数矩形（宽高至少 1）。"""
    w = max(1, round(width))
    h = max(1, round(height))
    x1 = round(cx - width / 2.0)
    y1 = round(cy - height / 2.0)
    return Box(x1, y1, x1 + w, y1 + h)


# ======================================================================
# 取景：主体归一化（批内一致性的核心）
# ======================================================================
@dataclass(frozen=True)
class Frame:
    """一次取景的结果。

    box
        裁剪窗。**可能超出原图边界** —— 这不是错误，而是保持批内尺度一致的
        必要代价：越界部分在裁剪时补背景，而不是把窗缩回来。
    exact_scale
        True 表示主体长边严格等于画面长边 x (1 + 2 x 边距)，
        即该图与同批其它图的主体占比完全一致。
        False 表示主体形状与所选比例冲突，为了不让主体被切掉而放大了画面，
        这张图的主体占比会比同批略小。
    """

    box: Box
    exact_scale: bool

    @property
    def overflows_source(self) -> bool:
        """裁剪窗是否超出原图边界（需要补背景）。"""
        return self.box.x1 < 0 or self.box.y1 < 0

    def needs_padding(self, source_size: tuple[int, int]) -> bool:
        width, height = source_size
        return self.box.x1 < 0 or self.box.y1 < 0 or self.box.x2 > width or self.box.y2 > height


def frame_for_subject(
    subject: Box,
    margin: float = 0.08,
    aspect: float | None = None,
) -> Frame:
    """以主体为基准算裁剪窗 —— 批内一致性靠的就是这个函数。

    margin
        相对主体尺寸的留白比例。0.08 表示主体四周各留 8%，
        即画面长边 = 主体长边 x 1.16。整批照片用同一个 margin，
        成品里的主体大小与留白就自动对齐，与拍摄距离无关。

    aspect
        目标宽高比（宽/高）。None 表示跟随主体自身形状 —— 此时
        exact_scale 恒为 True。

    主体形状与 aspect 冲突时（例如把 16:9 的框套在方形主体上），
    以"主体完整"优先：放大画面并标记 exact_scale=False，绝不切掉主体。
    """
    if subject.is_empty:
        raise ValueError("主体框为空，无法取景")
    if margin < 0:
        raise ValueError("边距比例不能为负")

    scale = 1.0 + 2.0 * margin
    cx, cy = subject.center

    if aspect is None:
        return Frame(_centered_box(cx, cy, subject.width * scale, subject.height * scale), True)

    if aspect <= 0:
        raise ValueError("宽高比必须为正")

    # 画面按主体长边定尺度，另一条边由 aspect 推出
    long_target = subject.long_side * scale
    if aspect >= 1.0:
        width, height = long_target, long_target / aspect
    else:
        width, height = long_target * aspect, long_target

    # 主体的两条边都必须完整装进画面；装不下就放大画面
    exact = True
    if width < subject.width * scale:
        width = subject.width * scale
        height = width / aspect
        exact = False
    if height < subject.height * scale:
        height = subject.height * scale
        width = height * aspect
        exact = False

    return Frame(_centered_box(cx, cy, width, height), exact)


# ======================================================================
# 模型坐标映射
# ======================================================================
@dataclass(frozen=True)
class Letterbox:
    """等比缩放 + 居中填充的坐标映射（模型输入 <-> 原图）。

    模型吃 N x N 的正方形输入，原图要等比缩放后居中贴入。掩膜回到原图时
    必须做逆向映射，否则整体偏移 —— 这类功能最常见的 bug 就在这里。
    """

    scale: float
    """原图 -> 模型输入 的缩放系数。"""

    pad_x: float
    pad_y: float
    """缩放后的原图在模型输入里的左上角偏移（像素）。"""

    src_size: tuple[int, int]
    dst_size: tuple[int, int]

    @classmethod
    def fit(
        cls,
        src_width: int,
        src_height: int,
        dst_width: int,
        dst_height: int,
    ) -> Letterbox:
        if src_width <= 0 or src_height <= 0:
            raise ValueError("源尺寸必须为正")
        scale = min(dst_width / src_width, dst_height / src_height)
        return cls(
            scale=scale,
            pad_x=(dst_width - src_width * scale) / 2.0,
            pad_y=(dst_height - src_height * scale) / 2.0,
            src_size=(src_width, src_height),
            dst_size=(dst_width, dst_height),
        )

    def to_source(self, x: float, y: float) -> Point:
        """模型输入坐标 -> 原图坐标。"""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale

    def to_model(self, x: float, y: float) -> Point:
        """原图坐标 -> 模型输入坐标。"""
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y

    def to_source_box(self, box: Box, clip: bool = True) -> Box:
        """模型输入上的矩形 -> 原图坐标。越界部分默认裁掉。"""
        x1, y1 = self.to_source(box.x1, box.y1)
        x2, y2 = self.to_source(box.x2, box.y2)
        out = Box(round(x1), round(y1), round(x2), round(y2))
        return out.clipped(*self.src_size) if clip else out

    def to_model_box(self, box: Box, clip: bool = True) -> Box:
        """原图上的矩形 -> 模型输入坐标。"""
        x1, y1 = self.to_model(box.x1, box.y1)
        x2, y2 = self.to_model(box.x2, box.y2)
        out = Box(round(x1), round(y1), round(x2), round(y2))
        return out.clipped(*self.dst_size) if clip else out

    # -- 掩膜 -----------------------------------------------------------
    def content_bounds(self, shape: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
        """图像内容（不含填充）在模型输入里的范围，返回 (左, 上, 右, 下)。

        填充区里的掩膜值没有任何意义 —— 它既不是主体，也不该参与任何
        坐标换算。取内容区一律走这里，不要自己按比例算。
        """
        if shape is None:
            height, width = int(self.dst_size[1]), int(self.dst_size[0])
        else:
            height, width = int(shape[0]), int(shape[1])

        left = min(max(0, int(round(self.pad_x))), width)
        top = min(max(0, int(round(self.pad_y))), height)
        right = min(width, left + max(1, int(round(self.src_size[0] * self.scale))))
        bottom = min(height, top + max(1, int(round(self.src_size[1] * self.scale))))
        return left, top, right, bottom

    def to_source_mask(
        self,
        mask: np.ndarray | Image.Image,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """模型输入空间的掩膜 -> 原图尺寸的布尔掩膜。

        必须"先裁内容区、再等比缩放"，不能把整块模型输入直接 resize 回原图：
        填充区会让后者在宽高比悬殊的图上变成非等比拉伸，掩膜里的角度直接
        失真（20 度的倾斜会被算成 10 度），倾斜校正于是越修越歪。
        """
        hit = _as_bool_mask(mask, threshold)
        if hit.shape == (int(self.src_size[1]), int(self.src_size[0])):
            return hit  # 已经在原图空间，直接可用

        left, top, right, bottom = self.content_bounds(hit.shape)
        content = hit[top:bottom, left:right]
        if content.size == 0:
            return np.zeros((int(self.src_size[1]), int(self.src_size[0])), dtype=bool)

        picture = Image.fromarray((content.astype(np.uint8) * 255), mode="L")
        picture = picture.resize(
            (int(self.src_size[0]), int(self.src_size[1])), Image.Resampling.BILINEAR
        )
        return np.asarray(picture) > 127


def _as_bool_mask(mask: np.ndarray | Image.Image, threshold: float = 0.5) -> np.ndarray:
    """把各种形态的掩膜统一成布尔数组。"""
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
        return arr >= threshold * 255
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"掩膜应为二维，实际是 {arr.ndim} 维")
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype.kind == "f":
        return arr >= threshold
    return arr > 0


def mask_to_box(mask: np.ndarray | Image.Image, threshold: float = 0.5) -> Box | None:
    """从掩膜取紧致包围盒；掩膜里没有主体时返回 None。

    注意：若掩膜尺寸是模型输入尺寸（例如 1024x1024），
    请先用 Letterbox.to_source_box 换算回原图坐标再使用。
    """
    hit = _as_bool_mask(mask, threshold)
    if not hit.any():
        return None
    rows = np.flatnonzero(hit.any(axis=1))
    cols = np.flatnonzero(hit.any(axis=0))
    return Box(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


# ======================================================================
# 四点透视矫正
# ======================================================================
def order_quad(points: Sequence[Point]) -> list[Point]:
    """把四个点整理成 左上、右上、右下、左下 的直观顺序。

    顺序错了会得到镜像或交叉的结果，所以入口一律先归一化顺序。
    """
    if len(points) != 4:
        raise ValueError(f"需要 4 个点，收到 {len(points)} 个")
    pts = [(float(x), float(y)) for x, y in points]
    s = [x + y for x, y in pts]
    d = [x - y for x, y in pts]
    top_left = pts[int(np.argmin(s))]
    bottom_right = pts[int(np.argmax(s))]
    top_right = pts[int(np.argmax(d))]
    bottom_left = pts[int(np.argmin(d))]
    # 极端情况下（例如退化成一条线）可能有重复点，去重后再补
    ordered = [top_left, top_right, bottom_right, bottom_left]
    if len(set(ordered)) != 4:
        raise ValueError("四个角点不构成有效四边形（存在重合点）")
    return ordered


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def quad_output_size(quad: Sequence[Point]) -> tuple[int, int]:
    """按四边形的实际边长估输出尺寸，避免把目标拉宽或压扁。

    取对边中的较长者，这样矩形的"歪"不会让结果被裁掉一条边。
    """
    tl, tr, br, bl = order_quad(quad)
    width = max(_dist(tl, tr), _dist(bl, br))
    height = max(_dist(tl, bl), _dist(tr, br))
    return max(1, round(width)), max(1, round(height))


def rectify_quad(
    image: Image.Image,
    quad: Sequence[Point],
    out_size: tuple[int, int] | None = None,
    resample: int = Image.Resampling.BICUBIC,
) -> Image.Image:
    """四点透视矫正：把任意四边形拉成正视矩形。

    纯 Pillow 实现（Image.Transform.QUAD），不需要 OpenCV。
    quad 用直观顺序（左上、右上、右下、左下）传入，内部重排成
    Pillow 要求的顺序。

    输出尺寸默认按边长估算；给定时按给定尺寸输出（可用于统一分辨率）。
    """
    tl, tr, br, bl = order_quad(quad)
    if out_size is None:
        out_size = quad_output_size([tl, tr, br, bl])
    corners = [tl, tr, br, bl]
    data = [c for index in _QUAD_ORDER for c in corners[index]]
    return image.transform(out_size, Image.Transform.QUAD, data, resample=resample)


# ======================================================================
# 倾斜
# ======================================================================
def mask_angle(mask: np.ndarray | Image.Image, threshold: float = 0.5) -> float:
    """用 PCA 长轴估计主体的倾斜角（度），范围 (-45, 45]。

    正值表示长轴从水平方向逆时针偏转。长轴无方向，所以结果折到
    半圈之内 —— 否则同一个物体旋转 180 度会给出两个"正确"答案。
    """
    hit = _as_bool_mask(mask, threshold)
    ys, xs = np.nonzero(hit)
    if xs.size < 2:
        return 0.0
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    cov = np.array([[float((x * x).mean()), float((x * y).mean())],
                    [float((x * y).mean()), float((y * y).mean())]])
    _, eigvecs = np.linalg.eigh(cov)
    vx, vy = eigvecs[:, -1]  # 最大特征值方向 = 长轴
    angle = math.degrees(math.atan2(vy, vx))
    angle = (angle + 90.0) % 180.0 - 90.0  # -> (-90, 90]
    if angle > 45.0:
        angle -= 90.0
    elif angle <= -45.0:
        angle += 90.0
    return angle


def tilt_is_meaningful(
    subject: Box,
    angle_deg: float,
    min_elongation: float = 1.5,
    max_angle: float = 15.0,
) -> bool:
    """判断这个倾斜是否值得自动纠正。

    球形、方形这类没有明确朝向的主体，PCA 长轴本身就不稳定，
    强行旋转只会把它转歪。所以只在主体明显细长、且角度落在一个
    可信区间内时才认为"值得转"。
    """
    if abs(angle_deg) < 0.5 or abs(angle_deg) > max_angle:
        return False
    return subject.elongations >= min_elongation


def rotate_image(
    image: Image.Image,
    angle_deg: float,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """旋转图像并自动扩展画布（Pillow 约定：正角度为逆时针）。"""
    fill = background if image.mode == "RGB" else 255 if image.mode == "L" else background
    return image.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fill,
    )


def rotate_box(
    box: Box,
    source_size: tuple[int, int],
    rotated_size: tuple[int, int],
    angle_deg: float,
) -> Box:
    """矩形经过"绕中心旋转 + 换画布"之后的轴对齐包围盒。

    与 rotate_image 是同一个变换：图像绕自身中心旋转，结果摆到新画布
    （expand 出来的那张）的正中。所以 `rotate_box(subject, 原图尺寸,
    旋转后尺寸, 角度)` 回答的就是"画面转正之后，主体跑到哪里去了"。

    两个必须知道的点：

    1. 旋转后的轴对齐包围盒**一定不小于**原框（斜着放的矩形要更大才能
       装下），这是几何必然，不是计算误差 —— 要按主体取景时，留白是
       围着这个更大的框算的，所以"转正后主体看着小一点"是正常的；
    2. 因此这个变换**不可逆**（转过去再转回来会变大）。要保留原图坐标
       的那一份，别拿转过的框反推。
    """
    width, height = source_size
    target_width, target_height = rotated_size
    center_x, center_y = width / 2.0, height / 2.0
    radians = math.radians(angle_deg)
    cos, sin = math.cos(radians), math.sin(radians)
    shift_x = (target_width - width) / 2.0
    shift_y = (target_height - height) / 2.0

    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((box.x1, box.y1), (box.x2, box.y1), (box.x1, box.y2), (box.x2, box.y2)):
        offset_x, offset_y = x - center_x, y - center_y
        xs.append(cos * offset_x + sin * offset_y + center_x + shift_x)
        ys.append(-sin * offset_x + cos * offset_y + center_y + shift_y)
    return Box(round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))


# ======================================================================
# 画布扩展
# ======================================================================
def pad_to_frame(
    image: Image.Image,
    box: Box,
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[Image.Image, Box]:
    """把画布扩到能容纳 box，返回扩展后的图与平移后的 box。

    归一化取景下裁剪窗可能超出原图边界。这时宁可补背景，也不要把窗缩回来 ——
    缩回来会破坏批内一致的封主体占比，而补背景只是多一圈留白。
    """
    width, height = image.size
    if box.x1 >= 0 and box.y1 >= 0 and box.x2 <= width and box.y2 <= height:
        return image, box

    pad_left = max(0, -box.x1)
    pad_top = max(0, -box.y1)
    pad_right = max(0, box.x2 - width)
    pad_bottom = max(0, box.y2 - height)
    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return image, box

    fill = background if image.mode == "RGB" else 255 if image.mode == "L" else background
    canvas = Image.new(
        image.mode,
        (width + pad_left + pad_right, height + pad_top + pad_bottom),
        fill,
    )
    canvas.paste(image, (pad_left, pad_top))
    return canvas, Box(box.x1 + pad_left, box.y1 + pad_top, box.x2 + pad_left, box.y2 + pad_top)
