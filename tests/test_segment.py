"""分割推理层的单元测试。

不需要模型的部分（预处理、坐标映射、缺模型时的提示）直接测；真正跑推理的
用例只在模型已安装时执行，否则跳过。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pathlib import Path

from imgcrop import models
from imgcrop.geometry import Box, Letterbox
from imgcrop.segment import (
    INPUT_SIZE,
    EfficientSamSegmenter,
    IsnetSegmenter,
    Proposal,
    SegmentError,
    build_tensor,
)


def make_image(size=(800, 600), subject=None) -> Image.Image:
    """深色台面 + 可选一块浅色主体。"""
    image = Image.new("RGB", size, (40, 45, 55))
    if subject is not None:
        ImageDraw.Draw(image).rectangle(
            [subject[0], subject[1], subject[2] - 1, subject[3] - 1],
            fill=(240, 238, 230),
        )
    return image


class TestBuildTensor:
    def test_shape_dtype_and_contiguity(self):
        tensor, _ = build_tensor(make_image((800, 600)))
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
        assert tensor.dtype == np.float32
        assert tensor.flags["C_CONTIGUOUS"]

    def test_normalisation_stays_within_half_range(self):
        tensor, _ = build_tensor(make_image())
        assert float(tensor.min()) >= -0.5 - 1e-6
        assert float(tensor.max()) <= 0.5 + 1e-6

    def test_wide_image_is_padded_vertically(self):
        _, box = build_tensor(make_image((1000, 500)))
        assert box.scale == pytest.approx(1.024)
        assert box.pad_y == pytest.approx(256.0)
        assert box.pad_x == pytest.approx(0.0)

    def test_tall_image_is_padded_horizontally(self):
        _, box = build_tensor(make_image((500, 1000)))
        assert box.pad_x == pytest.approx(256.0)

    def test_padding_is_normalised_to_zero(self):
        """填充区归一化后必须是 0（中性灰）—— 黑边会被模型读成"很暗的前景"。"""
        tensor, box = build_tensor(make_image((1000, 500)))
        row = int(box.pad_y) // 2
        assert np.allclose(tensor[0, :, row, :], 0.0, atol=0.01)

    def test_source_pixel_maps_into_content_area(self):
        _, box = build_tensor(make_image((1000, 500)))
        x, y = box.to_model(500.0, 250.0)
        assert abs(x - 512) < 1.0
        assert abs(y - 512) < 1.0  # 250 * 1.024 + 256

    def test_square_image_needs_no_padding(self):
        _, box = build_tensor(make_image((512, 512)))
        assert box.pad_x == 0.0 and box.pad_y == 0.0


class TestProposal:
    def _proposal(self, boxes):
        return Proposal(
            boxes=boxes,
            mask=np.zeros((10, 10), dtype=np.float32),
            letterbox=Letterbox.fit(10, 10, 10, 10),
        )

    def test_best_is_the_largest(self):
        proposal = self._proposal([Box(0, 0, 100, 100), Box(0, 0, 10, 10)])
        assert proposal.best == Box(0, 0, 100, 100)

    def test_best_is_none_without_candidates(self):
        assert self._proposal([]).best is None

    def test_source_mask_lands_in_source_space(self):
        """倾斜校正用的掩膜必须在原图坐标里 —— 模型输入空间那份会算歪角度。"""
        proposal = Proposal(
            boxes=[Box(0, 0, 10, 10)],
            mask=np.zeros((1024, 1024), dtype=np.float32),
            letterbox=Letterbox.fit(800, 600, 1024, 1024),
        )
        mask = proposal.source_mask()
        assert mask.shape == (600, 800)
        assert mask.dtype == np.bool_


class TestMissingModel:
    def test_resolve_path_rejects_missing_file(self, tmp_path):
        with pytest.raises(SegmentError):
            IsnetSegmenter(path=tmp_path / "nope.onnx").resolve_path()

    def test_tells_user_how_to_get_the_model(self, tmp_path):
        segmenter = IsnetSegmenter(path=tmp_path / "missing.onnx")
        with pytest.raises(SegmentError, match="fetch_models"):
            _ = segmenter.session


@pytest.mark.skipif(
    not models.is_installed(models.ISNET),
    reason="ISNet 模型未安装（先运行 python tools/fetch_models.py）",
)
class TestIsnetInference:
    SUBJECT = (250, 200, 550, 420)

    def test_finds_a_bright_subject_on_a_dark_plate(self):
        proposal = IsnetSegmenter().saliency(make_image((800, 600), subject=self.SUBJECT))

        assert proposal.mask.shape == (INPUT_SIZE, INPUT_SIZE)
        assert proposal.best is not None

        center_x, center_y = proposal.best.center
        assert 200 < center_x < 600
        assert 150 < center_y < 470

    def test_box_covers_most_of_the_subject(self):
        """外接框应当把主体基本框住 —— 盖不住就说明坐标映射或分割有问题。"""
        proposal = IsnetSegmenter().saliency(make_image((800, 600), subject=self.SUBJECT))
        best = proposal.best
        assert best is not None

        truth = Box(*self.SUBJECT)
        overlap = best.intersection(truth)
        assert overlap.area / truth.area > 0.5

    def test_flat_image_does_not_crash(self):
        proposal = IsnetSegmenter().saliency(make_image((400, 400)))
        assert isinstance(proposal.boxes, list)

    def test_padding_is_never_treated_as_subject(self):
        """填充区不是前景：宽高比悬殊的图曾经会被"框住整幅画面"。

        1200x300 的横条 letterbox 之后，灰边占了画布的 2/3，而 ISNet 会把灰边
        也判成前景 —— 候选框换算回原图便贴满整张图。
        """
        image = make_image((1200, 300), subject=(400, 80, 800, 220))
        proposal = IsnetSegmenter().saliency(image)

        left, top, right, bottom = proposal.letterbox.content_bounds(proposal.mask.shape)
        assert float(proposal.mask[:top, :].max(initial=0.0)) == 0.0
        assert float(proposal.mask[bottom:, :].max(initial=0.0)) == 0.0
        assert float(proposal.mask[:, :left].max(initial=0.0)) == 0.0
        assert float(proposal.mask[:, right:].max(initial=0.0)) == 0.0

        assert proposal.boxes
        for box in proposal.boxes:
            assert 0 <= box.x1 < box.x2 <= 1200
            assert 0 <= box.y1 < box.y2 <= 300


class TestMissingEfficientSam:
    def test_resolve_path_rejects_missing_file(self, tmp_path):
        with pytest.raises(SegmentError):
            EfficientSamSegmenter(path=tmp_path / "nope.onnx").resolve_path()

    def test_tells_user_how_to_get_the_model(self, tmp_path):
        segmenter = EfficientSamSegmenter(path=tmp_path / "missing.onnx")
        with pytest.raises(SegmentError, match="fetch_models"):
            _ = segmenter.session

    def test_rejects_empty_points_before_loading_the_model(self):
        """没给点就直接拒绝 —— 不必先把 46 MB 模型读进内存。"""
        segmenter = EfficientSamSegmenter(path=Path("不存在的模型.onnx"))
        with pytest.raises(SegmentError, match="至少需要一个提示点"):
            segmenter.segment(make_image(), [])


@pytest.mark.skipif(
    not models.is_installed(models.EFFICIENT_SAM),
    reason="EfficientSAM 模型未安装（先运行 python tools/fetch_models.py）",
)
class TestEfficientSamInference:
    SUBJECT = (250, 200, 550, 420)

    def test_point_inside_subject_returns_that_object(self):
        """这是点选的全部意义：点哪就框哪，坐标还要换回原图。"""
        image = make_image((800, 600), subject=self.SUBJECT)
        hit = EfficientSamSegmenter().segment(image, [(400.0, 310.0)])

        assert hit is not None
        assert hit.mask.shape == (INPUT_SIZE, INPUT_SIZE)
        assert hit.mask.dtype == np.bool_
        assert 0.0 <= hit.score <= 1.0

        truth = Box(*self.SUBJECT)
        overlap = hit.box.intersection(truth)
        assert overlap.area / truth.area > 0.5, f"点中主体却没框住它：{hit.box}"

    def test_background_point_returns_a_valid_hit_or_nothing(self):
        """点在背景上时结果不可靠 —— 实测这条路会把整张图当成一块前景。

        所以界面上的用法是"自动结果不对时点目标物体"，而不是靠点背景来排除
        （要排除得再补一个 label=0 的点，见 segment() 的说明）。
        这里只保证不抛异常、给出的框落在原图范围内。
        """
        image = make_image((800, 600), subject=self.SUBJECT)
        hit = EfficientSamSegmenter().segment(image, [(60.0, 60.0)])

        if hit is None:
            return
        assert not hit.box.is_empty
        assert 0 <= hit.box.x1 and 0 <= hit.box.y1
        assert hit.box.x2 <= 800 and hit.box.y2 <= 600

    def test_negative_label_is_accepted(self):
        """label=0 表示"这里不对"，接口要收得下 —— 界面用它来纠正误框。"""
        image = make_image((800, 600), subject=self.SUBJECT)
        hit = EfficientSamSegmenter().segment(
            image, [(400.0, 310.0), (60.0, 60.0)], [1, 0]
        )
        assert hit is None or not hit.box.is_empty
        if hit is not None:
            assert hit.mask.shape == (INPUT_SIZE, INPUT_SIZE)
