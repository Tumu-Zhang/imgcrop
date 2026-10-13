"""裁剪链路的单元测试。

重点在三条用户明确要求的行为上：

1. 批内一致 —— 不同大小的主体用同一规格裁剪后，主体占比必须相同
2. 原图零改动 —— 裁剪只产出新图，绝不触碰源图
3. 人工兜底与全自动同路 —— 点选/手动/锁定只是换一个主体框的来源
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imgcrop.crop import ASPECT_PRESETS, CropMode, CropSpec, apply_crop
from imgcrop.geometry import Box, mask_to_box
from imgcrop.model import SpecError

BACKGROUND = (60, 70, 80)
FOREGROUND = (240, 240, 235)


def subject_image(subject: Box, size=(1000, 800)) -> Image.Image:
    """造一张"深色底 + 浅色实心矩形主体"的测试图。"""
    image = Image.new("RGB", size, BACKGROUND)
    ImageDraw.Draw(image).rectangle(
        [subject.x1, subject.y1, subject.x2 - 1, subject.y2 - 1], fill=FOREGROUND
    )
    return image


def tilted_subject(size=(800, 600), angle=-12.0, bar=(300, 280, 500, 320)):
    """造一个倾斜的细长主体，返回 (图像, 布尔掩膜)。"""
    layer = Image.new("L", size, 0)
    ImageDraw.Draw(layer).rectangle(list(bar), fill=255)
    layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=0)
    image = Image.new("RGB", size, BACKGROUND)
    image.paste(FOREGROUND, mask=layer)
    return image, np.asarray(layer) > 127


class TestCropSpecValidation:
    def test_disabled_spec_needs_nothing(self):
        CropSpec(mode=CropMode.NONE).validate()

    def test_subject_mode_requires_subject(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.SUBJECT).validate()

    def test_free_mode_requires_subject(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.FREE, subject=Box(10, 10, 10, 10)).validate()

    def test_subject_mode_ok_with_box(self):
        CropSpec(mode=CropMode.SUBJECT, subject=Box(0, 0, 10, 10)).validate()

    def test_quad_mode_requires_four_points(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.QUAD, quad=[(0, 0), (1, 1), (2, 2)]).validate()
        CropSpec(mode=CropMode.QUAD, quad=[(0, 0), (10, 0), (10, 10), (0, 10)]).validate()

    def test_quad_mode_rejects_degenerate(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.QUAD, quad=[(0, 0), (0, 0), (10, 10), (0, 10)]).validate()

    def test_negative_margin_rejected(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.SUBJECT, subject=Box(0, 0, 10, 10), margin=-0.1).validate()

    def test_non_positive_aspect_rejected(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.SUBJECT, subject=Box(0, 0, 10, 10), aspect=0).validate()

    def test_validate_does_not_mutate(self):
        spec = CropSpec(mode=CropMode.SUBJECT, subject=Box(0, 0, 10, 10))
        spec.validate()
        assert spec.warnings == []


class TestDescribe:
    def test_disabled(self):
        assert CropSpec(mode=CropMode.NONE).describe() == "不裁剪"

    def test_subject_share_reflects_margin(self):
        spec = CropSpec(mode=CropMode.SUBJECT, subject=Box(0, 0, 10, 10), margin=0.08)
        assert "86%" in spec.describe()

    def test_quad_mode_mentions_output_size(self):
        spec = CropSpec(
            mode=CropMode.QUAD,
            quad=[(0, 0), (10, 0), (10, 10), (0, 10)],
            quad_size=(400, 300),
        )
        assert "400x300" in spec.describe()


class TestDisabledCrop:
    def test_returns_same_object(self):
        image = subject_image(Box(100, 100, 200, 200))
        out = apply_crop(image, CropSpec(mode=CropMode.NONE))
        assert out.image is image
        assert out.box is None
        assert out.exact_scale


class TestSubjectCrop:
    def test_margin_expands_around_subject(self):
        subject = Box(300, 200, 700, 600)  # 400x400
        out = apply_crop(
            subject_image(subject),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.1),
        )
        assert out.image.size == (480, 480)
        assert out.exact_scale

    def test_zero_margin_crops_exactly_to_subject(self):
        subject = Box(300, 200, 700, 600)
        out = apply_crop(
            subject_image(subject),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.0),
        )
        assert out.image.size == (400, 400)
        assert out.box == subject

    def test_crop_keeps_subject_content(self):
        subject = Box(300, 200, 700, 600)
        out = apply_crop(
            subject_image(subject),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.25),
        )
        # 中间应该是主体本身，四周是背景（含 25% 留白）
        width, height = out.image.size
        assert out.image.getpixel((width // 2, height // 2)) == FOREGROUND
        assert out.image.getpixel((2, 2)) == BACKGROUND

    def test_aspect_is_applied(self):
        subject = Box(500, 400, 900, 800)  # 400x400
        out = apply_crop(
            subject_image(subject, size=(1400, 1200)),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.1, aspect=16 / 9),
        )
        width, height = out.image.size
        assert width / height == pytest.approx(16 / 9, rel=0.02)

    def test_aspect_conflict_reports_warning(self):
        subject = Box(500, 400, 900, 800)  # 方形主体
        out = apply_crop(
            subject_image(subject, size=(2400, 1600)),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.1, aspect=3.0),
        )
        assert not out.exact_scale
        assert any("比例冲突" in warning for warning in out.warnings)

    def test_pads_background_when_subject_touches_edge(self):
        subject = Box(0, 0, 200, 200)
        out = apply_crop(
            subject_image(subject, size=(400, 400)),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.2),
        )
        assert out.image.size == (280, 280)  # 200 x 1.4
        assert out.image.getpixel((5, 5)) == (255, 255, 255)  # 补出来的白边
        assert any("补背景" in warning for warning in out.warnings)

    def test_custom_background_color(self):
        subject = Box(0, 0, 200, 200)
        out = apply_crop(
            subject_image(subject, size=(400, 400)),
            CropSpec(
                mode=CropMode.SUBJECT, subject=subject, margin=0.2,
                background=(0, 0, 0),
            ),
        )
        assert out.image.getpixel((5, 5)) == (0, 0, 0)


class TestFreeCrop:
    def test_uses_box_directly(self):
        image = subject_image(Box(0, 0, 1000, 800), size=(1000, 800))
        out = apply_crop(image, CropSpec(mode=CropMode.FREE, subject=Box(100, 50, 400, 300)))
        assert out.image.size == (300, 250)
        assert out.box == Box(100, 50, 400, 300)

    def test_free_crop_pads_when_outside(self):
        image = subject_image(Box(0, 0, 400, 400), size=(400, 400))
        out = apply_crop(image, CropSpec(mode=CropMode.FREE, subject=Box(-50, -50, 350, 350)))
        assert out.image.size == (400, 400)

    def test_free_crop_ignores_margin(self):
        image = subject_image(Box(0, 0, 500, 500), size=(500, 500))
        out = apply_crop(
            image,
            CropSpec(mode=CropMode.FREE, subject=Box(100, 100, 300, 300), margin=0.5),
        )
        assert out.image.size == (200, 200)


class TestQuadCrop:
    def test_rectifies_to_requested_size(self):
        image = subject_image(Box(0, 0, 800, 600), size=(800, 600))
        quad = [(100, 100), (700, 140), (680, 540), (80, 500)]
        out = apply_crop(
            image,
            CropSpec(mode=CropMode.QUAD, quad=quad, quad_size=(400, 400)),
        )
        assert out.image.size == (400, 400)
        assert out.box is None
        assert any("透视矫正" in warning for warning in out.warnings)

    def test_default_size_follows_edges(self):
        """输出尺寸按四边形边长推：宽 hypot(600,40)=601，高 hypot(20,400)=400。"""
        image = subject_image(Box(0, 0, 800, 600), size=(800, 600))
        quad = [(100, 100), (700, 140), (680, 540), (80, 500)]
        out = apply_crop(image, CropSpec(mode=CropMode.QUAD, quad=quad))
        assert out.image.size[0] == pytest.approx(601, abs=2)
        assert out.image.size[1] == pytest.approx(400, abs=2)


class TestTiltCorrection:
    def test_tilt_is_corrected_with_mask(self):
        image, mask = tilted_subject()
        subject = mask_to_box(mask)
        assert subject is not None
        out = apply_crop(
            image,
            CropSpec(
                mode=CropMode.SUBJECT, subject=subject,
                margin=0.15, rectify_tilt=True,
            ),
            mask=mask,
        )
        assert any("自动转正" in warning for warning in out.warnings)
        assert out.image.size[0] > 0 and out.image.size[1] > 0

    def test_tilt_correction_needs_mask(self):
        subject = Box(300, 280, 500, 320)
        out = apply_crop(
            subject_image(subject),
            CropSpec(
                mode=CropMode.SUBJECT, subject=subject,
                margin=0.1, rectify_tilt=True,
            ),
        )
        assert not any("倾斜" in warning for warning in out.warnings)

    def test_blobby_subject_is_not_rotated(self):
        """方形主体没有明确朝向，强行旋转只会把它转歪。"""
        mask = np.zeros((400, 400), dtype=bool)
        mask[150:250, 150:250] = True
        subject = Box(150, 150, 250, 250)
        out = apply_crop(
            subject_image(subject, size=(400, 400)),
            CropSpec(
                mode=CropMode.SUBJECT, subject=subject,
                margin=0.1, rectify_tilt=True,
            ),
            mask=mask,
        )
        assert not any("自动转正" in warning for warning in out.warnings)

    def test_tilt_can_be_disabled(self):
        image, mask = tilted_subject()
        subject = mask_to_box(mask)
        assert subject is not None
        out = apply_crop(
            image,
            CropSpec(
                mode=CropMode.SUBJECT, subject=subject,
                margin=0.15, rectify_tilt=False,
            ),
            mask=mask,
        )
        assert out.warnings == []

    def test_tilt_never_overrides_a_free_box(self):
        """自由裁剪框是用户亲手定的：倾斜校正不能拿掩膜推出来的框顶掉它。

        否则"拖到哪就裁到哪"是假的 —— 勾上倾斜校正，用户的框会整体挪走。
        """
        image, mask = tilted_subject()
        free = Box(120, 100, 420, 300)
        out = apply_crop(
            image,
            CropSpec(mode=CropMode.FREE, subject=free, rectify_tilt=True),
            mask=mask,
        )
        assert out.box is not None
        assert (out.box.x1, out.box.y1) == (free.x1, free.y1)
        assert not any("倾斜" in warning for warning in out.warnings)


class TestManualRotation:
    """手动角度：画面转、裁剪框保持水平，成品永远是正的。"""

    def test_manual_angle_rotates_the_frame(self):
        subject = Box(300, 280, 500, 320)
        out = apply_crop(
            subject_image(subject, size=(600, 600)),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.0, rotate_deg=12.0),
        )
        assert any("手动角度旋转 +12.0°" in warning for warning in out.warnings)
        assert out.box is not None
        # 200x40 转 12° 之后，包围盒大约 204x81
        assert out.box.width > 200
        assert out.box.height > 60

    def test_free_mode_box_is_rotated_too(self):
        free = Box(200, 200, 400, 300)
        out = apply_crop(
            subject_image(free, size=(600, 600)),
            CropSpec(mode=CropMode.FREE, subject=free, rotate_deg=10.0),
        )
        assert out.box is not None
        assert out.box.height > 100  # 100 高转 10° 之后装不下原来的内容

    def test_auto_and_manual_angles_are_added(self):
        image, mask = tilted_subject()
        subject = mask_to_box(mask)
        assert subject is not None
        out = apply_crop(
            image,
            CropSpec(
                mode=CropMode.SUBJECT, subject=subject, margin=0.1,
                rectify_tilt=True, rotate_deg=5.0,
            ),
            mask=mask,
        )
        note = next(w for w in out.warnings if "旋转" in w and "倾斜" not in w)
        assert "自动转正" in note and "手动" in note

    def test_rejects_impossible_angles(self):
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.FREE, subject=Box(0, 0, 10, 10), rotate_deg=float("nan")).validate()
        with pytest.raises(SpecError):
            CropSpec(mode=CropMode.FREE, subject=Box(0, 0, 10, 10), rotate_deg=720.0).validate()

    def test_no_rotation_leaves_everything_alone(self):
        subject = Box(300, 280, 500, 320)
        out = apply_crop(
            subject_image(subject, size=(600, 600)),
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.0),
        )
        assert out.warnings == []
        assert out.box is not None
        assert (out.box.x1, out.box.y1) == (subject.x1, subject.y1)


class TestSourceIsNeverTouched:
    def test_pixels_and_size_unchanged(self):
        """裁剪只产出新图 —— "原图零改动"是这个工具的硬约定。"""
        subject = Box(100, 100, 300, 300)
        image = subject_image(subject, size=(500, 500))
        before_pixels = image.tobytes()
        before_size = image.size

        apply_crop(
            image,
            CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.3),
        )
        apply_crop(image, CropSpec(mode=CropMode.FREE, subject=Box(0, 0, 100, 100)))

        assert image.size == before_size
        assert image.tobytes() == before_pixels


class TestBatchConsistency:
    def test_same_margin_gives_same_subject_share(self):
        """用户要求"同类型批量照片主体与边距保持一致" —— 这条测试看住它。

        主体在原图里大小差 3 倍，只要用同一份规格，成品里的占比就相同。
        这是"主体归一化取景"相对"固定像素边距"的关键优势。
        """
        shares = []
        for side in (200, 400, 600):
            subject = Box(0, 0, side, side)
            image = subject_image(subject, size=(1000, 1000))
            out = apply_crop(
                image,
                CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.08),
            )
            shares.append(side / out.image.size[0])
        assert all(share == pytest.approx(shares[0], abs=0.005) for share in shares)
        assert shares[0] == pytest.approx(1 / 1.16, abs=0.005)

    def test_rectangular_subjects_keep_long_side_share(self):
        shares = []
        for width, height in ((400, 200), (600, 300), (200, 100)):
            subject = Box(0, 0, width, height)
            image = subject_image(subject, size=(1000, 1000))
            out = apply_crop(
                image,
                CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.1),
            )
            shares.append(max(width, height) / out.image.size[0])
        assert all(share == pytest.approx(shares[0], abs=0.01) for share in shares)

    def test_aspect_locked_batch_produces_uniform_ratio(self):
        """固定比例 + 固定边距时，整批输出的宽高比完全一致。"""
        ratios = []
        for width, height in ((300, 200), (500, 400), (800, 600)):
            subject = Box(0, 0, width, height)
            image = subject_image(subject, size=(1600, 1200))
            out = apply_crop(
                image,
                CropSpec(
                    mode=CropMode.SUBJECT, subject=subject,
                    margin=0.1, aspect=4 / 3,
                ),
            )
            out_width, out_height = out.image.size
            ratios.append(out_width / out_height)
        assert all(ratio == pytest.approx(4 / 3, rel=0.02) for ratio in ratios)


class TestAspectPresets:
    def test_contains_common_ratios(self):
        labels = [label for label, _ in ASPECT_PRESETS]
        assert "原始比例" in labels
        assert "1:1" in labels
        assert "4:3" in labels
        assert "16:9" in labels

    def test_first_preset_is_native_ratio(self):
        label, value = ASPECT_PRESETS[0]
        assert label == "原始比例"
        assert value is None
