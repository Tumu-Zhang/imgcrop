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

def min_area_rect_corners(points: np.ndarray) -> tuple[Point, ...]:
    """包围点集的最小面积矩形的四个角点（直观顺序 左上/右上/右下/左下）。

    quad_from_mask 的退化路径：掩膜不是四边形（圆形、不规则）时，
    凸包逼近给不出 4 顶点，就退回"最小面积包围矩形"——拉平结果至少
    是一个规矩的矩形，不会歪。
    """
    angle = _min_area_rect_angle(points)
    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    along = points[:, 0] * cos + points[:, 1] * sin
    across = -points[:, 0] * sin + points[:, 1] * cos
    u0, u1 = float(along.min()), float(along.max())
    v0, v1 = float(across.min()), float(across.max())
    corners = [
        (u * cos - v * sin, u * sin + v * cos)
        for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
    ]
    return tuple(corners)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链凸包（纯 Python，边界采样点规模下毫秒级）。

    输入重复点会被去重；返回按逆时针顺序的顶点（首尾不重复）。
    """
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _dp_simplify(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Douglas-Peucker 开放折线逼近：保留偏离弦超过 eps 的点。"""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        best_dist, best_index = -1.0, start
        for i in range(start + 1, end):
            px, py = points[i]
            if length < 1e-12:
                dist = math.hypot(px - ax, py - ay)
            else:
                dist = abs(dy * (px - ax) - dx * (py - ay)) / length
            if dist > best_dist:
                best_dist, best_index = dist, i
        if best_dist > eps:
            keep[best_index] = True
            stack.append((start, best_index))
            stack.append((best_index, end))
    return [p for p, k in zip(points, keep) if k]


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """鞋带公式算多边形面积（顶点按顺序）。"""
    total = 0.0
    count = len(points)
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def approx_quad(points: np.ndarray) -> tuple[Point, ...] | None:
    """凸包 → 恰好 4 顶点且贴合的四边形；给不出就返回 None。

    做法：把凸包环从一条近似直径切成两段，各自做 D-P 逼近；eps 从粗到细
    扫描，第一次得到"两段合计 4 个顶点（共享端点只算一次）"就检查贴合度
    —— 四边形面积必须 ≥ 凸包面积的 92%。这条阈值挡住"圆形内接四边形"
    这类切掉边缘的假四边形（圆的内接四边形只有凸包面积的六成多），
    证件、纸张这类边界清晰的矩形物则轻松过关。
    """
    hull = convex_hull(points)
    count = len(hull)
    if count < 4:
        return None
    hull_list = [tuple(p) for p in hull]

    # 近似直径端点：从顶点 0 出发最远的那个点。用它把闭合环切成两段。
    offset = hull - hull[0]
    far = int(np.argmax((offset * offset).sum(axis=1)))
    if far == 0:
        return None
    first = hull_list[: far + 1]
    second = hull_list[far:] + [hull_list[0]]

    hull_area = _polygon_area(hull_list)
    perimeter = sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(hull_list, hull_list[1:] + hull_list[:1])
    )
    eps = perimeter * 0.05
    while eps >= perimeter * 0.002:
        simplified = _dp_simplify(first, eps)[:-1] + _dp_simplify(second, eps)[:-1]
        if len(simplified) == 4:
            if hull_area <= 0 or _polygon_area(simplified) < hull_area * 0.92:
                return None  # 四边形切掉了太多边缘：不是矩形物，交给包围矩形
            try:
                return tuple(order_quad(simplified))
            except ValueError:
                return None
        if len(simplified) > 4:
            # 顶点数随 eps 减小只会增多：已经跳过 4，再扫也不会回来
            break
        eps *= 0.7  # 顶点还不够（eps 太粗）：继续细化
    return None


def quad_from_mask(
    mask: np.ndarray | Image.Image, threshold: float = 0.5
) -> tuple[Point, ...] | None:
    """从主体掩膜自动估计透视拉平的四角（原图坐标，直观顺序）。

    透视矫正的"自动打底"（UI-007）：边界清晰的矩形物（证件、纸张、画框）
    走凸包四边形逼近，角点准；形状不规则时退化到最小面积包围矩形，
    保证总能给出一个规矩的四边形。掩膜里没有主体时返回 None。

    掩膜必须已经在**原图空间**（segment.Proposal.source_mask() 等）。
    """
    hit = _as_bool_mask(mask, threshold)
    if not hit.any():
        return None
    points = _boundary_samples(hit, max_points=4096)
    if len(points) < 4:
        return None

    quad = approx_quad(points)
    if quad is None:
        quad = min_area_rect_corners(points)
    quad = tuple(order_quad(quad))

    # 面积保护：太小（退化/噪声）的四边形没有拉平价值
    tl, tr, br, bl = quad
    width = max(_dist(tl, tr), _dist(bl, br))
    height = max(_dist(tl, bl), _dist(tr, br))
    if width < 4 or height < 4:
        return None
    return quad


__all__ = [
    "Box",
    "Frame",
    "Letterbox",
    "approx_quad",
    "convex_hull",
    "frame_for_subject",
    "map_box_by_homography",
    "map_point_by_homography",
    "mask_angle",
    "mask_to_box",
    "min_area_rect_corners",
    "order_quad",
    "pad_to_frame",
    "quad_from_mask",
    "quad_homography",
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


def quad_homography(quad: Sequence[Point], out_size: tuple[int, int]) -> np.ndarray:
    """源图四边形 -> 输出矩形的**正向**单应（8 个系数，h33 = 1）。

    PIL 的 QUAD 变换是"输出像素从源图采样"的反向映射；要把源图上的坐标
    （主体框、提示点）映射到拉平后的图上（界面在拉平结果上继续取景），
    需要正向单应。用四组对应点（源四角 -> 输出矩形四角）解 8 元线性方程组
    （标准 find_coeffs，numpy 纯实现，无新依赖）。
    """
    tl, tr, br, bl = order_quad(quad)
    width, height = out_size
    source = [tl, tr, br, bl]
    target = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]

    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    return coeffs


def map_point_by_homography(coeffs: np.ndarray, x: float, y: float) -> Point:
    """用 quad_homography 的系数把源图坐标映射到拉平图坐标。"""
    denominator = coeffs[6] * x + coeffs[7] * y + 1.0
    if abs(denominator) < 1e-12:  # pragma: no cover - 拉平图上不会出现无穷远
        raise ValueError("单应映射分母为零：四边形退化")
    u = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / denominator
    v = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / denominator
    return float(u), float(v)


def map_box_by_homography(coeffs: np.ndarray, box: Box) -> Box:
    """源图矩形 -> 拉平图上的轴对齐包围盒。

    与 rotate_box 同一条纪律：矩形的单应像不一定是矩形，轴对齐包围盒
    **一定不小于**真实像，且不可逆 —— 要保留坐标的那一份，别拿映射过的
    框反推。
    """
    corners = [
        map_point_by_homography(coeffs, float(x), float(y))
        for x, y in ((box.x1, box.y1), (box.x2, box.y1), (box.x1, box.y2), (box.x2, box.y2))
    ]
    xs = [round(u) for u, _ in corners]
    ys = [round(v) for _, v in corners]
    return Box(min(xs), min(ys), max(xs), max(ys))


# ======================================================================
# 倾斜
# ======================================================================
def mask_angle(mask: np.ndarray | Image.Image, threshold: float = 0.5) -> float:
    """用最小面积矩形拟合估计主体的倾斜角（度），范围 (-45, 45]。

    1.1.0 升级（D10 定案）：旧实现用 PCA 长轴，对细长物体（笔、尺子）好用，
    对块状物体（证件、书本）偏弱 —— PCA 的轴向在近方形上不稳定。这里改为
    **最小面积矩形拟合**（numpy 纯实现，等价 cv2.minAreaRect）：
    扫描旋转角，找包围掩膜面积最小的那个方向，证件/文档类估角明显更稳。

    正值表示长轴从水平方向逆时针偏转。轴向无方向（矩形转 90° 面积不变），
    所以结果折到半圈之内 —— 否则同一个物体旋转 180 度会给出两个"正确"答案。

    实现：用每行/列的边界像素当支撑点（数量远小于全部前景点），对候选角度
    网格做向量化投影，取投影包围盒面积最小者，再在最优角附近细化。
    """
    hit = _as_bool_mask(mask, threshold)
    points = _boundary_samples(hit)
    if len(points) < 2:
        return 0.0
    angle = _min_area_rect_angle(points)
    if angle > 45.0:
        angle -= 90.0
    elif angle <= -45.0:
        angle += 90.0
    return angle


def _boundary_samples(hit: np.ndarray, max_points: int = 2048) -> np.ndarray:
    """提取掩膜的边界采样点 (N, 2)，供最小面积矩形扫描。

    最小面积矩形的支撑点一定落在凸包上，而每行/每列的最左最右、最上最下
    像素必然在凸包上 —— 取这些点（隔行/隔列采样封顶）既不丢精度又够快。
    """
    if not hit.any():
        return np.zeros((0, 2), dtype=np.float64)

    points: list[tuple[float, float]] = []
    row_indices = np.flatnonzero(hit.any(axis=1))
    if row_indices.size:
        step = max(1, row_indices.size // max(1, max_points // 2))
        for y in row_indices[::step]:
            cols = np.flatnonzero(hit[y])
            points.append((float(cols[0]), float(y)))
            points.append((float(cols[-1]), float(y)))

    col_indices = np.flatnonzero(hit.any(axis=0))
    if col_indices.size:
        step = max(1, col_indices.size // max(1, max_points // 2))
        for x in col_indices[::step]:
            rows = np.flatnonzero(hit[:, x])
            points.append((float(x), float(rows[0])))
            points.append((float(x), float(rows[-1])))

    return np.asarray(points, dtype=np.float64)


def _min_area_rect_angle(points: np.ndarray) -> float:
    """扫描旋转角，返回包围点集面积最小的方向角，范围 (-90, 90]。

    对每个候选角 θ，把点投影到 (cosθ, sinθ) 与其法向，两个方向的跨度之积
    就是该角度下包围矩形的面积。先 0.5° 粗扫，再在最优角 ±1° 内按 0.05°
    细化 —— 精度足够倾斜校正用，计算量对 numpy 是毫秒级。
    """
    coarse = np.arange(-90.0, 90.0, 0.5)
    best = _best_angle(points, coarse)
    fine = np.arange(best - 1.0, best + 1.0, 0.05)
    return float(_best_angle(points, fine))


def _best_angle(points: np.ndarray, angles: np.ndarray) -> float:
    """返回 angles 里包围盒面积最小的角度。"""
    xs = points[:, 0]
    ys = points[:, 1]
    radians = np.deg2rad(angles)
    cos = np.cos(radians)[:, None]  # (A, 1)
    sin = np.sin(radians)[:, None]

    along = xs[None, :] * cos + ys[None, :] * sin          # 沿 θ 方向的投影
    across = -xs[None, :] * sin + ys[None, :] * cos        # 垂直方向的投影
    areas = (along.max(axis=1) - along.min(axis=1)) * (
        across.max(axis=1) - across.min(axis=1)
    )
    return float(angles[int(np.argmin(areas))])


def tilt_is_meaningful(
    subject: Box,
    angle_deg: float,
    min_elongation: float = 1.5,
    max_angle: float = 15.0,
) -> bool:
    """判断这个倾斜是否值得自动纠正。

    球形、方形这类没有明确朝向的主体，最小面积矩形本身就不稳定，
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
    fill = _frame_fill(image.mode, background)
    return image.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=fill,
    )


def _frame_fill(mode: str, background: tuple[int, int, int]):
    """取景/旋转补边的填充色：透明图补透明白，灰度图补灰阶，其余补背景色。"""
    if mode == "RGBA":
        return (*background, 0)
    if mode == "L":
        return 255
    if mode in ("I", "I;16", "F"):
        return 0
    return background


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

    fill = _frame_fill(image.mode, background)
    canvas = Image.new(
        image.mode,
        (width + pad_left + pad_right, height + pad_top + pad_bottom),
        fill,
    )
    canvas.paste(image, (pad_left, pad_top))
    return canvas, Box(box.x1 + pad_left, box.y1 + pad_top, box.x2 + pad_left, box.y2 + pad_top)
