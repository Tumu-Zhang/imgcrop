"""裁剪 × 输出落盘的集成测试。

1.1.0 起输出语义改为「所见即所得」：成品像素 = 裁剪窗原像素，不缩放、不放大，
格式跟随源图（详见 imgcrop/model.py 的说明）。本文件顺着这条语义，验证裁剪与
落盘连起来之后对用户的最终承诺：一批照片进去，一批主体占比一致的成品出来，
而原图一个字节都没动。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from conftest import make_photo, read_size
from imgcrop.crop import CropMode, CropSpec
from imgcrop.geometry import Box
from imgcrop.model import OutputSpec
from imgcrop.pipeline import CropPlan, crop_batch


def spec_for(workdir: Path, **kwargs) -> OutputSpec:
    return OutputSpec(output_dir=workdir, **kwargs)


def plan_for(subject: Box, margin: float = 0.08) -> CropPlan:
    return CropPlan(spec=CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=margin))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCroppedOutput:
    def test_output_is_crop_window_one_to_one(self, tmp_path, workdir):
        """成品像素 = 裁剪窗原像素：取景框多大，出来的图就多大。"""
        subject = Box(100, 50, 500, 450)  # 400 × 400
        src = make_photo(tmp_path / "photo.png", (100, 50, 500, 450))
        report = crop_batch([src], spec_for(workdir), crop_provider=lambda _p: plan_for(subject, margin=0.0))

        assert report.success, [r.error for r in report.results]
        result = report.results[0]
        assert read_size(result.out_path) == (400, 400)
        assert result.out_size == (400, 400)

    def test_subject_share_is_uniform_across_batch(self, tmp_path, workdir):
        """批内一致：主体在原图里差 2 倍，成品里主体的像素占比必须相同。"""
        subjects = [Box(50, 50, 250, 250), Box(50, 50, 450, 450)]
        paths = []
        plans = {}
        for index, subject in enumerate(subjects):
            path = make_photo(tmp_path / f"photo{index}.png", (subject.x1, subject.y1, subject.x2, subject.y2))
            paths.append(path)
            plans[path] = plan_for(subject, margin=0.08)

        report = crop_batch(paths, spec_for(workdir), crop_provider=plans.get)
        assert report.success, [r.error for r in report.results]

        shares = []
        for result in report.results:
            with Image.open(result.out_path) as image:
                gray = image.convert("L")
                width, height = gray.size
                middle = [gray.getpixel((x, height // 2)) for x in range(width)]
            hits = [x for x, value in enumerate(middle) if value > 200]
            shares.append((hits[-1] - hits[0] + 1) / width)

        # 留白 8% 时主体约占画面 1/1.16；两张源图主体差 2 倍，占比必须一致
        assert abs(shares[0] - shares[1]) < 0.005
        assert abs(shares[0] - 1 / 1.16) < 0.02

    def test_sources_are_never_modified(self, tmp_path, workdir):
        """全程只读源文件 —— 原图零改动是这个工具的硬约定。"""
        subject = Box(100, 100, 500, 400)
        src = make_photo(tmp_path / "photo.png", (100, 100, 500, 400))
        before = file_hash(src)

        report = crop_batch([src], spec_for(workdir), crop_provider=lambda _p: plan_for(subject, margin=0.2))
        assert report.success

        assert file_hash(src) == before
        assert read_size(src) == (1000, 800)  # 源图尺寸照旧

    def test_output_name_template_and_directory(self, tmp_path, workdir):
        subject = Box(100, 100, 600, 500)
        src = make_photo(tmp_path / "sample.png", (100, 100, 600, 500))
        report = crop_batch(
            [src],
            spec_for(workdir, name_template="{stem}_crop"),
            crop_provider=lambda _p: plan_for(subject, margin=0.1),
        )

        assert report.success
        assert (workdir / "sample_crop.png").exists()
        assert (tmp_path / "sample.png").exists()  # 原图仍在原处


class TestCropIsOptional:
    def test_without_provider_behaviour_is_unchanged(self, tmp_path, workdir):
        """没有裁剪规格时按原图 1:1 落盘（不裁剪时的基线行为）。"""
        src = make_photo(tmp_path / "photo.png", (0, 0, 1000, 800))
        report = crop_batch([src], spec_for(workdir))

        assert report.success
        assert read_size(report.results[0].out_path) == (1000, 800)

    def test_disabled_spec_is_a_no_op(self, tmp_path, workdir):
        """即使提供了 provider，返回「不裁剪」规格也应当走原路径。"""
        src = make_photo(tmp_path / "photo.png", (0, 0, 1000, 800))
        report = crop_batch(
            [src],
            spec_for(workdir),
            crop_provider=lambda _p: CropPlan(spec=CropSpec(mode=CropMode.NONE)),
        )

        assert report.success
        assert read_size(report.results[0].out_path) == (1000, 800)

    def test_missing_entry_falls_back_to_no_crop(self, tmp_path, workdir):
        """provider 里没有这张图（例如用户的裁剪还没做完）时按不裁剪处理。"""
        src = make_photo(tmp_path / "photo.png", (0, 0, 1000, 800))
        report = crop_batch([src], spec_for(workdir), crop_provider=lambda _p: None)

        assert report.success
        assert read_size(report.results[0].out_path) == (1000, 800)


class TestCropWarnings:
    def test_padding_warning_reaches_the_report(self, tmp_path, workdir):
        """主体贴边补背景时，用户必须能在结果里看到，而不是静默发生。"""
        subject = Box(0, 0, 200, 200)  # 紧贴左上角
        src = make_photo(tmp_path / "photo.png", (0, 0, 200, 200), size=(800, 800))
        report = crop_batch([src], spec_for(workdir), crop_provider=lambda _p: plan_for(subject, margin=0.2))

        result = report.results[0]
        assert result.ok
        assert any("补背景" in warning for warning in result.warnings)
