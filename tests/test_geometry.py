"""裁剪几何内核的单元测试。

这里覆盖的三类问题都是"错了不会崩、只会静默产出错图"的类型，
所以必须用数值和方向断言来锁死：

1. Pillow 的 QUAD 四点顺序（传错会得到一张垂直翻转的图）
2. letterbox 坐标往返（错了掩膜就整体偏移）
3. 主体归一化取景（错了批内主体大小就不一致）
4. 越界补背景后的坐标平移（错了裁剪窗就整体错位）
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imgcrop.geometry import (
    Box,
    Letterbox,
    frame_for_subject,
    mask_angle,
    mask_to_box,
    order_quad,
    pad_to_frame,
    quad_output_size,
    rectify_quad,
    rotate_box,
    rotate_image,
    tilt_is_meaningful,
)

MARK_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
MARK_NAMES = ["红", "绿", "蓝", "黄"]
BACKGROUND = (40, 40, 40)


def marked_image(quad, size=(800, 600), radius=30):
    """在四边形四角画不同颜色的实心圆，用于校验矫正后的方向。"""
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    for (x, y), color in zip(quad, MARK_COLORS):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return image


def corner_mark(image, corner, span=70):
    """识别输出图某个角落附近是哪个颜色的标记。"""
    arr = np.asarray(image).astype(int)
    width, height = image.size
    regions = {
        "tl": (0, 0, span, span),
        "tr": (width - span, 0, width, span),
        "br": (width - span, height - span, width, height),
        "bl": (0, height - span, span, height),
    }
    x1, y1, x2, y2 = regions[corner]
    pixels = arr[y1:y2, x1:x2].reshape(-1, 3)
    # 重采样会留下边缘混合色，所以按"落在某个标记色附近的像素个数"判定，
    # 而不是取均值 —— 均值会被混合色带偏。
    counts = [int(np.sum(np.sum((pixels - color) ** 2, axis=1) < 3000)) for color in MARK_COLORS]
    best = int(np.argmax(counts))
    return MARK_NAMES[best] if counts[best] >= 20 else "无标记"


class TestBox:
    def test_basic_measures(self):
        box = Box(10, 20, 110, 70)
        assert box.width == 100
        assert box.height == 50
        assert box.long_side == 100
        assert box.short_side == 50
        assert box.area == 5000
        assert box.center == (60.0, 45.0)
        assert box.aspect == pytest.approx(2.0)
        assert box.elongations == pytest.approx(2.0)
        assert not box.is_empty

    def test_clipped_and_empty(self):
        assert Box(-10, -10, 50, 50).clipped(100, 100) == Box(0, 0, 50, 50)
        assert Box(90, 90, 200, 200).clipped(100, 100) == Box(90, 90, 100, 100)
        assert Box(500, 500, 600, 600).clipped(100, 100).is_empty

    def test_expanded_is_relative_to_own_size(self):
        """边距按主体自身尺寸算，所以宽扁主体与高瘦主体的绝对边距不同。"""
        wide = Box(0, 0, 100, 50).expanded(0.1)
        assert (wide.width, wide.height) == (120, 60)
        assert wide.center == (50.0, 25.0)

    def test_expanded_zero_is_identity(self):
        box = Box(10, 10, 110, 60)
        assert box.expanded(0) == box
        assert box.expanded(-0.5) == box

    def test_union_and_intersection(self):
        a = Box(0, 0, 100, 100)
        b = Box(50, 50, 150, 150)
        assert a.union(b) == Box(0, 0, 150, 150)
        assert a.intersection(b) == Box(50, 50, 100, 100)


class TestLetterbox:
    def test_fit_wide_source_pads_vertically(self):
        lb = Letterbox.fit(1000, 500, 1024, 1024)
        assert lb.scale == pytest.approx(1.024)
        assert lb.pad_x == pytest.approx(0.0)
        assert lb.pad_y == pytest.approx(256.0)  # (1024 - 512) / 2

    def test_fit_square_source_has_no_padding(self):
        lb = Letterbox.fit(512, 512, 1024, 1024)
        assert lb.scale == pytest.approx(2.0)
        assert lb.pad_x == 0.0
        assert lb.pad_y == 0.0

    def test_fit_tall_source_pads_horizontally(self):
        lb = Letterbox.fit(500, 1000, 1024, 1024)
        assert lb.scale == pytest.approx(1.024)
        assert lb.pad_x == pytest.approx(256.0)
        assert lb.pad_y == pytest.approx(0.0)

    def test_roundtrip_is_exact(self):
        """坐标必须能原样走回来，否则掩膜贴回原图会整体偏移。"""
        lb = Letterbox.fit(1000, 500, 1024, 1024)
        for x, y in [(0.0, 0.0), (1000.0, 500.0), (537.0, 213.0)]:
            mx, my = lb.to_model(x, y)
            rx, ry = lb.to_source(mx, my)
            assert rx == pytest.approx(x)
            assert ry == pytest.approx(y)

    def test_source_point_maps_into_padding_when_outside(self):
        """原图之外的模型坐标会落到填充区（负坐标），这是判断越界的方式。"""
        lb = Letterbox.fit(1000, 500, 1024, 1024)
        sx, sy = lb.to_source(500.0, 100.0)  # 落在上方填充区
        assert sy < 0

    def test_box_mapping_clips_to_source(self):
        lb = Letterbox.fit(1000, 500, 1024, 1024)
        model_box = Box(0, 0, 1024, 1024)  # 整个模型输入
        source_box = lb.to_source_box(model_box)
        assert source_box == Box(0, 0, 1000, 500)  # 填充区被裁掉

    def test_content_bounds_exclude_padding(self):
        lb = Letterbox.fit(1000, 500, 1024, 1024)
        assert lb.content_bounds() == (0, 256, 1024, 768)

    def test_source_mask_keeps_geometry_for_skewed_images(self):
        """掩膜回原图必须"先裁内容区再等比缩放"，否则角度会被拉歪。

        直接把整块模型输入 resize 回原图（非等比）会让 20 度的长条算出
        10 度 —— 倾斜校正于是越修越歪。
        """
        from PIL import ImageDraw

        width, height = 2000, 1000
        source = Image.new("L", (width, height), 0)
        ImageDraw.Draw(source).rectangle(
            [width // 2 - 700, height // 2 - 60, width // 2 + 700, height // 2 + 60], fill=255
        )
        source = source.rotate(20.0, resample=Image.Resampling.BILINEAR,
                               center=(width // 2, height // 2), fillcolor=0)

        lb = Letterbox.fit(width, height, 1024, 1024)
        canvas = Image.new("L", (1024, 1024), 0)
        scaled = (round(width * lb.scale), round(height * lb.scale))
        canvas.paste(source.resize(scaled, Image.Resampling.BILINEAR),
                     (round(lb.pad_x), round(lb.pad_y)))

        restored = lb.to_source_mask(np.asarray(canvas) > 127)
        assert restored.shape == (height, width)
        assert mask_angle(restored) == pytest.approx(-20.0, abs=1.0)

    def test_source_mask_accepts_a_mask_already_in_source_space(self):
        lb = Letterbox.fit(1000, 500, 1024, 1024)
        mask = np.zeros((500, 1000), dtype=bool)
        mask[100:200, 300:400] = True
        assert np.array_equal(lb.to_source_mask(mask), mask)


class TestMaskToBox:
    def test_finds_tight_box(self):
        mask = np.zeros((100, 200), dtype=np.float32)
        mask[30:70, 40:150] = 0.9
        assert mask_to_box(mask) == Box(40, 30, 150, 70)

    def test_empty_mask_returns_none(self):
        assert mask_to_box(np.zeros((50, 50), dtype=np.float32)) is None

    def test_threshold_is_respected(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:20, 10:20] = 0.9
        mask[30:40, 30:40] = 0.3
        assert mask_to_box(mask, threshold=0.8) == Box(10, 10, 20, 20)
        assert mask_to_box(mask, threshold=0.2) == Box(10, 10, 40, 40)

    def test_accepts_pil_image(self):
        image = Image.new("L", (60, 40), 0)
        ImageDraw.Draw(image).rectangle([10, 5, 40, 30], fill=255)
        assert mask_to_box(image) == Box(10, 5, 41, 31)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            mask_to_box(np.zeros((2, 3, 4, 5), dtype=np.float32))


class TestFrameForSubject:
    def test_zero_margin_matches_subject(self):
        subject = Box(100, 100, 300, 200)
        frame = frame_for_subject(subject, margin=0.0, aspect=None)
        assert frame.box == subject
        assert frame.exact_scale

    def test_margin_scales_both_sides(self):
        subject = Box(0, 0, 100, 50)
        frame = frame_for_subject(subject, margin=0.1, aspect=None)
        assert frame.box.width == 120
        assert frame.box.height == 60
        assert frame.box.center == (50.0, 25.0)

    def test_batch_consistency_same_margin_gives_same_share(self):
        """同一 margin + 同一比例下，不同大小的主体在成品里的占比必须相同。

        这是"批内主体大小与留白一致"的核心保证：主体占比只由参数决定，
        与拍摄时主体多大（远近）无关。
        """
        shares = []
        for size in [(200, 200), (800, 800), (400, 400), (120, 120)]:
            subject = Box(0, 0, *size)
            frame = frame_for_subject(subject, margin=0.08, aspect=1.0)
            assert frame.exact_scale
            shares.append(subject.long_side / frame.box.long_side)
        assert all(share == pytest.approx(shares[0], abs=0.005) for share in shares)
        assert shares[0] == pytest.approx(1 / 1.16, abs=0.005)

    def test_batch_consistency_with_rectangular_subjects(self):
        """主体形状不同但同批时，长边占比仍然一致。"""
        shares = []
        for size in [(400, 200), (600, 300), (200, 100)]:
            subject = Box(0, 0, *size)
            frame = frame_for_subject(subject, margin=0.1, aspect=None)
            shares.append(subject.long_side / frame.box.long_side)
        assert all(share == pytest.approx(1 / 1.2, abs=0.01) for share in shares)

    def test_aspect_conflict_keeps_subject_intact(self):
        """方形主体配 3:1 画面：宁可放大画面，也不切掉主体。"""
        subject = Box(500, 500, 600, 600)
        frame = frame_for_subject(subject, margin=0.08, aspect=3.0)
        assert not frame.exact_scale
        assert frame.box.x1 <= subject.x1 and frame.box.x2 >= subject.x2
        assert frame.box.y1 <= subject.y1 and frame.box.y2 >= subject.y2
        assert frame.box.aspect == pytest.approx(3.0, rel=0.02)

    def test_subject_always_inside_frame(self):
        """参数扫一遍，主体永远不得被切到画面外。"""
        subject = Box(300, 200, 900, 500)
        for aspect in [None, 0.5, 0.75, 1.0, 4 / 3, 16 / 9, 3.0]:
            for margin in [0.0, 0.05, 0.2, 0.5]:
                frame = frame_for_subject(subject, margin=margin, aspect=aspect)
                assert frame.box.x1 <= subject.x1
                assert frame.box.y1 <= subject.y1
                assert frame.box.x2 >= subject.x2
                assert frame.box.y2 >= subject.y2

    def test_frame_may_overflow_source_boundary(self):
        """归一化取景允许裁剪窗越界，由裁剪阶段补背景。"""
        subject = Box(0, 0, 200, 200)  # 主体紧贴左上角
        frame = frame_for_subject(subject, margin=0.2, aspect=None)
        assert frame.box.x1 < 0 and frame.box.y1 < 0
        assert frame.needs_padding((1000, 1000))

    def test_empty_subject_rejected(self):
        with pytest.raises(ValueError):
            frame_for_subject(Box(10, 10, 10, 10))

    def test_negative_margin_rejected(self):
        with pytest.raises(ValueError):
            frame_for_subject(Box(0, 0, 10, 10), margin=-0.1)


class TestQuad:
    QUAD = [(300, 120), (700, 200), (640, 560), (240, 480)]  # TL, TR, BR, BL

    def test_order_quad_normalizes_shuffled_input(self):
        shuffled = [self.QUAD[2], self.QUAD[0], self.QUAD[3], self.QUAD[1]]
        assert order_quad(shuffled) == self.QUAD

    def test_order_quad_is_idempotent(self):
        """已经是标准顺序时不该再被改动 —— 重复调用必须稳定。"""
        once = order_quad(self.QUAD)
        assert order_quad(once) == once

    def test_order_quad_rejects_wrong_count(self):
        with pytest.raises(ValueError):
            order_quad([(0, 0), (1, 1), (2, 2)])

    def test_order_quad_rejects_degenerate(self):
        with pytest.raises(ValueError):
            order_quad([(0, 0), (0, 0), (5, 5), (9, 1)])

    def test_output_size_follows_edge_lengths(self):
        """输出尺寸按四边形实际边长算：宽 hypot(400,80)=408，高 hypot(60,360)=365。"""
        width, height = quad_output_size(self.QUAD)
        assert width == pytest.approx(408, abs=1)
        assert height == pytest.approx(365, abs=1)

    def test_rectify_quad_orientation_is_locked(self):
        """对角标记必须各就各位。

        这是防"静默垂直翻转"的哨兵测试：Pillow 的 Image.Transform.QUAD
        要的是 左上/左下/右下/右上 这一非直观顺序，传成直观顺序不会报错，
        只会给出一张上下颠倒的图（纯色测试图根本看不出来）。
        """
        image = marked_image(self.QUAD)
        out = rectify_quad(image, self.QUAD, out_size=(400, 400))
        assert corner_mark(out, "tl") == "红"
        assert corner_mark(out, "tr") == "绿"
        assert corner_mark(out, "br") == "蓝"
        assert corner_mark(out, "bl") == "黄"

    def test_rectify_quad_accepts_shuffled_corners(self):
        """乱序传入的四个角也要得到同样的结果。"""
        image = marked_image(self.QUAD)
        shuffled = [self.QUAD[1], self.QUAD[3], self.QUAD[0], self.QUAD[2]]
        out = rectify_quad(image, shuffled, out_size=(400, 400))
        assert corner_mark(out, "tl") == "红"
        assert corner_mark(out, "br") == "蓝"

    def test_rectify_quad_default_size_from_edges(self):
        image = marked_image(self.QUAD)
        out = rectify_quad(image, self.QUAD)
        assert out.size == quad_output_size(self.QUAD)

    def test_rectified_content_is_upright(self):
        """横向条纹在矫正后仍应是横向的（不是被转 90 度）。"""
        image = Image.new("RGB", (800, 600), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        quad = [(100, 100), (700, 100), (700, 500), (100, 500)]
        for y in range(140, 460, 60):
            draw.rectangle([120, y, 680, y + 20], fill=(0, 0, 0))
        out = rectify_quad(image, quad, out_size=(600, 400))
        arr = np.asarray(out.convert("L"))
        row_darkness = (arr < 128).mean(axis=1)
        col_darkness = (arr < 128).mean(axis=0)
        # 条纹沿行方向成片出现，所以行的方差远大于列的方差
        assert row_darkness.std() > 3 * col_darkness.std()


class TestTilt:
    def test_mask_angle_detects_rotation(self):
        base = Image.new("L", (400, 400), 0)
        ImageDraw.Draw(base).rectangle([80, 180, 320, 220], fill=255)  # 240x40 横条
        rotated = base.rotate(-20, resample=Image.Resampling.BICUBIC, fillcolor=0)
        angle = mask_angle(np.asarray(rotated) > 127)
        assert abs(abs(angle) - 20) < 2.0

    def test_mask_angle_of_level_bar_is_zero(self):
        mask = np.zeros((200, 300), dtype=bool)
        mask[90:110, 50:250] = True
        assert abs(mask_angle(mask)) < 0.5

    def test_mask_angle_of_empty_mask_is_zero(self):
        assert mask_angle(np.zeros((50, 50), dtype=bool)) == 0.0

    def test_tilt_only_for_elongated_subjects(self):
        slender = Box(0, 0, 300, 60)   # 长宽比 5
        blobby = Box(0, 0, 100, 100)   # 长宽比 1
        assert tilt_is_meaningful(slender, 5.0)
        assert not tilt_is_meaningful(blobby, 5.0)

    def test_tilt_rejected_outside_trusted_band(self):
        slender = Box(0, 0, 300, 60)
        assert not tilt_is_meaningful(slender, 0.2)    # 太小，可能是噪声
        assert not tilt_is_meaningful(slender, 30.0)   # 太大，可能是主体本身斜放
        assert not tilt_is_meaningful(slender, -25.0)

    def test_rotate_image_expands_canvas(self):
        image = Image.new("RGB", (100, 50), (255, 255, 255))
        out = rotate_image(image, 90)
        assert out.size == (50, 100)


class TestRotateBox:
    """旋转后的坐标必须和 rotate_image 的实际变换对齐 —— 差一点框就整体偏。"""

    def _marked(self, size=(200, 100), box=Box(60, 30, 140, 70)) -> Image.Image:
        image = Image.new("RGB", size, (0, 0, 0))
        ImageDraw.Draw(image).rectangle(
            [box.x1, box.y1, box.x2 - 1, box.y2 - 1], fill=(255, 255, 255)
        )
        return image

    def test_matches_the_real_rotation(self):
        image = self._marked()
        truth = Box(60, 30, 140, 70)
        for angle in (0.0, 7.5, 10.0, -20.0, 30.0, 45.0):
            rotated = rotate_image(image, angle, (0, 0, 0))
            actual = mask_to_box(np.asarray(rotated)[:, :, 0] > 127)
            predicted = rotate_box(truth, image.size, rotated.size, angle)
            # 允许 1 像素：Pillow 内部与这里各做一次取整
            assert abs(predicted.x1 - actual.x1) <= 1
            assert abs(predicted.y1 - actual.y1) <= 1
            assert abs(predicted.x2 - actual.x2) <= 1
            assert abs(predicted.y2 - actual.y2) <= 1

    def test_zero_angle_is_identity(self):
        box = Box(10, 20, 110, 220)
        assert rotate_box(box, (400, 400), (400, 400), 0.0) == box

    def test_rotated_box_never_shrinks(self):
        """旋转后主体的轴对齐包围盒只会变大 —— 这是几何必然，不是误差。"""
        box = Box(60, 30, 140, 70)
        image = self._marked()
        for angle in (5.0, 15.0, 30.0):
            rotated = rotate_image(image, angle, (0, 0, 0))
            grown = rotate_box(box, image.size, rotated.size, angle)
            assert grown.width >= box.width
            assert grown.height >= box.height


class TestPadToFrame:
    def test_returns_input_when_inside(self):
        image = Image.new("RGB", (500, 500), (10, 20, 30))
        box = Box(100, 100, 300, 300)
        out_image, out_box = pad_to_frame(image, box)
        assert out_image is image
        assert out_box == box

    def test_pads_and_shifts_coordinates(self):
        """越界取景靠补背景解决，补完之后裁剪窗必须平移到正确位置。"""
        image = Image.new("RGB", (500, 500), (10, 20, 30))
        box = Box(-50, -30, 450, 470)
        out_image, out_box = pad_to_frame(image, box)
        assert out_image.size == (550, 530)  # 左补 50、上补 30
        assert (out_box.width, out_box.height) == (500, 500)
        assert out_box.x1 == 0 and out_box.y1 == 0
        cropped = out_image.crop((out_box.x1, out_box.y1, out_box.x2, out_box.y2))
        assert cropped.size == (500, 500)
        # 补齐的区域是背景色，原图内容向右下平移了 (50, 30)
        assert cropped.getpixel((5, 5)) == (255, 255, 255)
        assert cropped.getpixel((60, 40)) == (10, 20, 30)

    def test_background_color_respected(self):
        image = Image.new("RGB", (100, 100), (10, 20, 30))
        out_image, _ = pad_to_frame(image, Box(-10, -10, 50, 50), background=(0, 0, 0))
        assert out_image.getpixel((0, 0)) == (0, 0, 0)

    def test_grayscale_padding_uses_int_fill(self):
        image = Image.new("L", (100, 100), 40)
        out_image, _ = pad_to_frame(image, Box(-10, -10, 50, 50))
        assert out_image.mode == "L"
        assert out_image.getpixel((0, 0)) == 255
