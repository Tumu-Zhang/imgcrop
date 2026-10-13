"""掩膜区域分析：从一张显著性/分割掩膜里找出候选主体框。

这个模块不需要模型、不碰磁盘、也不依赖 OpenCV —— 连通域用"行内连续段 +
并查集"自己实现。之所以不直接用 cv2.connectedComponentsWithStats，是因为
打包时刻意不带 cv2（省几十 MB），而这里需要的功能只是它的一小部分。

连通性用 8-连通
    细长且斜置的物体（斜放的试管、倾斜的标尺）在掩膜里往往只有一两个像素宽。
    4-连通会把这种斜线切成碎块，外接框就只框住其中一小段 —— 所以这里按
    8-连通合并：宁可多连一点，也不要切碎主体。

为什么要多个候选
----------------
单一阈值往往把细长主体截断，或把背景噪声并进来。换几个阈值重跑，候选
之间通常正好覆盖"偏紧 / 适中 / 偏松"三种取法，界面里就能直接按键切换，
而不必让用户手动拖框。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageFilter

from imgcrop.geometry import Box

__all__ = ["candidates", "cleanup", "connected_boxes", "similar"]


def connected_boxes(mask: np.ndarray, *, min_area: int = 0) -> list[Box]:
    """求连通域的外接框，按像素面积降序。

    8-连通语义：行内连续段本身左右相连，相邻两行的列区间只要有重叠或相接，
    就视为同一连通域。实现是标准的两遍扫描（这里用"切段 + 并查集"表达第二遍），
    复杂度与连续段数量成正比，1024x1024 的典型掩膜只需几毫秒。
    """
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"掩膜应为二维，实际是 {binary.ndim} 维")
    if binary.dtype != np.bool_:
        binary = binary > 0

    height = binary.shape[0]

    # 第一遍：把每一行切成连续段（run）
    runs: list[tuple[int, int, int]] = []  # (行, 起点, 终点不含)
    for row_index in range(height):
        row = binary[row_index]
        if not row.any():
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(edges[0::2].tolist(), edges[1::2].tolist()):
            runs.append((row_index, start, end))

    if not runs:
        return []

    parent = list(range(len(runs)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:  # 路径压缩
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_row: dict[int, list[int]] = {}
    for index, (row_index, _, _) in enumerate(runs):
        by_row.setdefault(row_index, []).append(index)

    for row_index, indices in by_row.items():
        above = by_row.get(row_index - 1)
        if not above:
            continue
        for current in indices:
            _, cur_start, cur_end = runs[current]
            for previous in above:
                _, prev_start, prev_end = runs[previous]
                # 相接也算连通（8-连通）：一像素宽的斜线不会被切成碎块
                if cur_start <= prev_end and prev_start <= cur_end:
                    union(current, previous)

    boxes: dict[int, Box] = {}
    areas: dict[int, int] = {}
    for index, (row_index, start, end) in enumerate(runs):
        root = find(index)
        areas[root] = areas.get(root, 0) + (end - start)
        current = boxes.get(root)
        if current is None:
            boxes[root] = Box(start, row_index, end, row_index + 1)
        else:
            boxes[root] = Box(
                min(current.x1, start), min(current.y1, row_index),
                max(current.x2, end), max(current.y2, row_index + 1),
            )

    ranked = [(areas[root], box) for root, box in boxes.items() if areas[root] >= min_area]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in ranked]


def cleanup(
    mask: np.ndarray,
    *,
    open_radius: int = 0,
    close_radius: int = 0,
) -> np.ndarray:
    """形态学开/闭运算，用 Pillow 的 Min/MaxFilter（C 实现，比 numpy 手写快）。

    开运算（先腐蚀后膨胀）去掉孤立噪点；闭运算（先膨胀后腐蚀）补上主体内部
    的小洞。半径按模型输入分辨率（通常 1024）来给，2 像素足以去掉毛刺而
    不伤及主体轮廓。
    """
    result = np.asarray(mask)
    if result.dtype != np.bool_:
        result = result > 0
    if open_radius <= 0 and close_radius <= 0:
        return result

    picture = Image.fromarray((result.astype(np.uint8) * 255), mode="L")
    if open_radius > 0:
        size = open_radius * 2 + 1
        picture = picture.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    if close_radius > 0:
        size = close_radius * 2 + 1
        picture = picture.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(picture) > 127


def similar(a: Box, b: Box, tolerance: float = 0.85) -> bool:
    """两个框是否基本指同一个东西（用交并比判断）。"""
    overlap = a.intersection(b)
    if overlap.is_empty:
        return False
    union_area = a.area + b.area - overlap.area
    return union_area > 0 and overlap.area / union_area >= tolerance


def candidates(
    probabilities: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.7, 0.5, 0.3),
    top_n: int = 4,
    min_area_ratio: float = 0.002,
    clean: bool = True,
) -> list[Box]:
    """多阈值扫描，给出若干个可能的"主体"框。

    顺序是**从紧到松**：先给最高阈值（最贴合主体的那一档）里最大的框，
    再一档一档放宽；只有与已给出的框明显不同（IoU < 0.85）的才继续往后排。
    所以 boxes[0] 是最紧的取法，往后依次更宽松 —— 界面上的"候选"下拉与
    1-4 快捷键就是这个顺序，默认取第一项。

    反过来的顺序（面积降序）会让默认候选恒等于"最松阈值下最大的连通域"，
    低阈值带出的背景伪影会被排到最前面。

    probabilities 是模型的显著性输出（0..1，模型输入分辨率）。模型输出常见
    形态是 (1,1,H,W) 或 (1,H,W)，这里统一压平，调用方不必自己 reshape。

    min_area_ratio 用来滤掉针尖大的误检：小于画面千分之二的一律不算候选。
    """
    grid = np.asarray(probabilities, dtype=np.float32)
    if grid.ndim > 2:
        grid = grid.squeeze()  # 去掉 batch / channel 维度
    if grid.ndim != 2:
        raise ValueError(f"掩膜应为二维（可含大小为 1 的前置维度），实际是 {grid.ndim} 维")

    area_floor = max(1, int(grid.size * min_area_ratio))
    kept: list[Box] = []

    for threshold in thresholds:
        binary = grid >= threshold
        if clean:
            binary = cleanup(binary, open_radius=2, close_radius=2)
        # connected_boxes 内部按像素面积降序：同一档里先给面积大的
        for box in connected_boxes(binary, min_area=area_floor):
            if any(similar(box, existing) for existing in kept):
                continue
            kept.append(box)
            if len(kept) >= top_n:
                return kept
    return kept
