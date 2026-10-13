"""掩膜区域分析的单元测试。

连通域是自己实现的（为了不打包 cv2），所以这里要把它测扎实 ——
它错了会直接导致候选框错位，而且不会崩。
"""

from __future__ import annotations

import numpy as np
import pytest

from imgcrop.geometry import Box
from imgcrop.regions import candidates, cleanup, connected_boxes, similar


def blank(height: int = 100, width: int = 100) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


class TestConnectedBoxes:
    def test_single_square(self):
        mask = blank()
        mask[20:50, 30:70] = True
        assert connected_boxes(mask) == [Box(30, 20, 70, 50)]

    def test_two_components_sorted_by_area_desc(self):
        mask = blank()
        mask[10:20, 10:20] = True   # 100 px
        mask[40:70, 40:80] = True   # 1200 px
        boxes = connected_boxes(mask)
        assert len(boxes) == 2
        assert boxes[0] == Box(40, 40, 80, 70)
        assert boxes[1] == Box(10, 10, 20, 20)

    def test_empty_mask(self):
        assert connected_boxes(blank()) == []

    def test_corner_touch_merges_under_eight_connectivity(self):
        """只有角碰到也算同一块。

        这是为细长斜置物体（斜放的试管、倾斜的标尺）准备的 —— 它们在掩膜里
        往往只有一两像素宽，按 4-连通会被切成一串碎块，外接框就只框住一小段。
        """
        mask = blank(10, 10)
        mask[0:2, 0:2] = True
        mask[2:4, 2:4] = True
        assert len(connected_boxes(mask)) == 1

    def test_gap_keeps_components_apart(self):
        """真正分开（中间隔了空行）的两块不得合并。"""
        mask = blank(10, 10)
        mask[0:2, 0:2] = True
        mask[3:5, 3:5] = True
        assert len(connected_boxes(mask)) == 2

    def test_edge_touch_does_merge(self):
        mask = blank(10, 10)
        mask[0:2, 0:2] = True
        mask[2:4, 0:2] = True
        assert len(connected_boxes(mask)) == 1

    def test_l_shape_is_single_component(self):
        mask = blank(20, 20)
        mask[5:15, 5:8] = True
        mask[12:15, 5:15] = True
        assert connected_boxes(mask) == [Box(5, 5, 15, 15)]

    def test_min_area_filters_specks(self):
        mask = blank()
        mask[10:12, 10:12] = True   # 只有 4 px
        mask[40:60, 40:60] = True
        assert connected_boxes(mask, min_area=10) == [Box(40, 40, 60, 60)]

    def test_accepts_non_boolean_input(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        assert connected_boxes(mask) == [Box(10, 10, 20, 20)]

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            connected_boxes(np.zeros((2, 3, 4), dtype=bool))

    def test_handles_thin_diagonal_line(self):
        """一像素宽的斜线必须被认成一个连通域。"""
        mask = blank(30, 30)
        for step in range(12):
            mask[step, step] = True
        boxes = connected_boxes(mask)
        assert len(boxes) == 1
        assert boxes[0] == Box(0, 0, 12, 12)


class TestCleanup:
    def test_zero_radius_is_identity(self):
        mask = blank(20, 20)
        mask[5:10, 5:10] = True
        assert np.array_equal(cleanup(mask, open_radius=0, close_radius=0), mask)

    def test_open_removes_isolated_speck(self):
        mask = blank(40, 40)
        mask[5:30, 5:30] = True
        mask[36, 36] = True  # 1 像素噪点
        cleaned = cleanup(mask, open_radius=2)
        assert not cleaned[36, 36]
        assert cleaned[20, 20]

    def test_close_fills_small_hole(self):
        mask = blank(40, 40)
        mask[5:35, 5:35] = True
        mask[18:22, 18:22] = False  # 4x4 的洞
        closed = cleanup(mask, close_radius=3)
        assert closed[20, 20]

    def test_open_keeps_solid_block_size(self):
        """开运算不该把实体块永久缩小（先腐蚀后膨胀会复原外轮廓）。"""
        mask = blank(40, 40)
        mask[10:30, 10:30] = True
        cleaned = cleanup(mask, open_radius=2)
        assert cleaned[10, 10] and cleaned[29, 29]
        assert connected_boxes(cleaned) == [Box(10, 10, 30, 30)]


class TestSimilar:
    def test_identical_boxes_are_similar(self):
        assert similar(Box(0, 0, 100, 100), Box(0, 0, 100, 100))

    def test_disjoint_boxes_are_not_similar(self):
        assert not similar(Box(0, 0, 10, 10), Box(50, 50, 60, 60))

    def test_barely_overlapping_boxes_are_not_similar(self):
        assert not similar(Box(0, 0, 100, 100), Box(90, 90, 190, 190))

    def test_slightly_different_boxes_are_similar(self):
        assert similar(Box(0, 0, 100, 100), Box(2, 2, 102, 102), tolerance=0.8)


class TestCandidates:
    def test_single_subject_yields_one_box(self):
        """三个阈值结果相同 -> 去重后只留一个候选。"""
        grid = np.zeros((200, 200), dtype=np.float32)
        grid[50:150, 60:140] = 0.9
        assert candidates(grid) == [Box(60, 50, 140, 150)]

    def test_soft_edges_produce_multiple_candidates(self):
        """软边缘下不同阈值给出大小不同的框 —— 这正是"候选快切"的基础。

        顺序是**从紧到松**：第一候选来自最紧的那一档（最贴合主体），
        往后一档比一档宽松。反过来的顺序会让默认候选永远是最松的框。
        """
        grid = np.zeros((200, 200), dtype=np.float32)
        grid[50:150, 50:150] = 0.4
        grid[70:130, 70:130] = 0.9
        boxes = candidates(grid)
        assert len(boxes) >= 2
        assert boxes[0] == Box(70, 70, 130, 130)
        assert boxes[0].area < boxes[1].area

    def test_two_subjects_both_offered(self):
        grid = np.zeros((300, 300), dtype=np.float32)
        grid[20:120, 20:120] = 0.9
        grid[180:280, 180:280] = 0.9
        boxes = candidates(grid)
        assert len(boxes) >= 2

    def test_specks_are_filtered_out(self):
        grid = np.zeros((400, 400), dtype=np.float32)
        grid[100:300, 100:300] = 0.9
        grid[5:8, 5:8] = 0.9  # 9 像素的误检
        boxes = candidates(grid)
        assert boxes
        assert all(box.area > 1000 for box in boxes)

    def test_empty_grid_gives_nothing(self):
        assert candidates(np.zeros((100, 100), dtype=np.float32)) == []

    def test_top_n_limits_output(self):
        grid = np.zeros((400, 400), dtype=np.float32)
        for index in range(5):
            top = 20 + index * 70
            grid[top:top + 50, 20:70] = 0.9
        assert len(candidates(grid, top_n=2)) == 2

    def test_accepts_batch_and_channel_dims(self):
        """模型输出常是 (1,1,H,W)，进来前要能自己压平。"""
        grid = np.zeros((1, 1, 100, 100), dtype=np.float32)
        grid[0, 0, 20:60, 30:70] = 0.9
        assert candidates(grid) == [Box(30, 20, 70, 60)]

    def test_rejects_unsqueezable_shape(self):
        with pytest.raises(ValueError):
            candidates(np.zeros((2, 3, 4, 5), dtype=np.float32))
