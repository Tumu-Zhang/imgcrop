"""裁剪接入输出管线的集成测试。

这里验证的是"裁剪 + 期刊规格输出"合起来的行为，也就是这个功能对用户的
最终承诺：一批照片进去，一批主体大小一致、分辨率一致、命名可控的图出来，
而原图一个字节都没动。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from imgcrop.crop import CropMode, CropSpec
from imgcrop.geometry import Box
from imgcrop.model import OutputFormat, OutputSpec, SizeMode, Unit
from imgcrop.pipeline import Cropper

PLATE = (60, 70, 80)
SPECIMEN = (235, 235, 230)


def make_photo(path: Path, subject: Box, size=(1000, 800)) -> Path:
    """造一张模拟实验照片：深色台面 + 浅色主体。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, PLATE)
    ImageDraw.Draw(image).rectangle(
        [subject.x1, subject.y1, subject.x2 - 1, subject.y2 - 1], fill=SPECIMEN
    )
    image.save(path, dpi=(72, 72))
    return path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


class TestCroppedOutput:
    def test_pixel_mode_gives_uniform_size_across_batch(self, tmp_path):
        """批内一致：主体在原图里差 3 倍，成品必须是同一个像素尺寸。

        这是用户要求的"同类型批量照片大小分辨率均保持一致"。
        """
        plans: dict[Path, CropSpec] = {}
        paths: list[Path] = []
        for index, side in enumerate((200, 400, 600)):
            subject = Box(50, 50, 50 + side, 50 + side)
            path = make_photo(tmp_path / f"photo{index}.png", subject)
            paths.append(path)
            plans[path] = CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.08)

        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=1200, px_height=1200,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        report = Cropper(spec, crop_provider=plans.get).run(paths)

        assert all(result.ok for result in report.results)
        sizes = {read_size(out / f"{path.stem}.png") for path in paths}
        assert sizes == {(1200, 1200)}

    def test_subject_share_is_uniform_in_output_pixels(self, tmp_path):
        """成品里主体的实际像素大小也必须一致，而不只是画布一致。"""
        plans: dict[Path, CropSpec] = {}
        paths: list[Path] = []
        for index, side in enumerate((200, 400)):
            subject = Box(50, 50, 50 + side, 50 + side)
            path = make_photo(tmp_path / f"photo{index}.png", subject)
            paths.append(path)
            plans[path] = CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.08)

        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=1000, px_height=1000,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        Cropper(spec, crop_provider=plans.get).run(paths)

        shares = []
        for path in paths:
            with Image.open(out / f"{path.stem}.png") as image:
                gray = image.convert("L")
                width, height = gray.size
                # 取中间一行，量浅色主体横向占了多少像素
                middle = [gray.getpixel((x, height // 2)) for x in range(width)]
            hits = [x for x, value in enumerate(middle) if value > 200]
            shares.append((hits[-1] - hits[0] + 1) / width)

        # 两张源图里主体差 2 倍，成品里主体的像素占比必须相同
        assert abs(shares[0] - shares[1]) < 0.005
        assert abs(shares[0] - 1 / 1.16) < 0.02

    def test_aspect_resolution_uses_cropped_shape(self, tmp_path):
        """只给宽度时，高度按"裁剪后"的形状推 —— 证明裁剪早于尺寸解析生效。"""
        subject = Box(0, 0, 400, 200)  # 2:1 的样品
        path = make_photo(tmp_path / "photo.png", subject, size=(1000, 1000))
        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.CM, phys_width=8.5,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        plans = {path: CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.0)}
        report = Cropper(spec, crop_provider=plans.get).run([path])

        assert report.results[0].ok
        # 8.5cm @300dpi = 1004 px 宽；裁剪后是 2:1，所以高 = 502
        assert read_size(out / "photo.png") == (1004, 502)

    def test_sources_are_never_modified(self, tmp_path):
        """全程只读源文件 —— 原图零改动是这个工具的硬约定。"""
        subject = Box(100, 100, 500, 400)
        path = make_photo(tmp_path / "photo.png", subject)
        before = file_hash(path)

        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=800, px_height=800,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        plans = {path: CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.2)}
        Cropper(spec, crop_provider=plans.get).run([path])

        assert file_hash(path) == before
        assert read_size(path) == (1000, 800)  # 源图尺寸照旧

    def test_output_name_template_and_directory(self, tmp_path):
        subject = Box(100, 100, 600, 500)
        path = make_photo(tmp_path / "sample.png", subject)
        out = tmp_path / "cropped"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=600, px_height=600,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
            name_template="{stem}_crop",
        )
        plans = {path: CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.1)}
        report = Cropper(spec, crop_provider=plans.get).run([path])

        assert report.results[0].ok
        assert (out / "sample_crop.png").exists()
        assert (tmp_path / "sample.png").exists()  # 原图仍在原处


class TestCropIsOptional:
    def test_without_provider_behaviour_is_unchanged(self, tmp_path):
        """没有裁剪规格时，管线行为与加裁剪之前完全一致（回归保护）。"""
        path = make_photo(tmp_path / "photo.png", Box(0, 0, 1000, 800))
        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=500, px_height=400,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        report = Cropper(spec).run([path])

        assert report.results[0].ok
        assert read_size(out / "photo.png") == (500, 400)

    def test_disabled_spec_is_a_no_op(self, tmp_path):
        """即使提供了 provider，返回"不裁剪"规格也应当走原路径。"""
        path = make_photo(tmp_path / "photo.png", Box(0, 0, 1000, 800))
        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=500, px_height=400,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        plans = {path: CropSpec(mode=CropMode.NONE)}
        report = Cropper(spec, crop_provider=plans.get).run([path])

        assert report.results[0].ok
        assert read_size(out / "photo.png") == (500, 400)

    def test_missing_entry_falls_back_to_no_crop(self, tmp_path):
        """provider 里没有这张图（例如用户的裁剪还没做完）时按不裁剪处理。"""
        path = make_photo(tmp_path / "photo.png", Box(0, 0, 1000, 800))
        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=400, px_height=320,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        report = Cropper(spec, crop_provider=lambda _p: None).run([path])

        assert report.results[0].ok
        assert read_size(out / "photo.png") == (400, 320)


class TestCropWarnings:
    def test_padding_warning_reaches_the_report(self, tmp_path):
        """主体贴边补背景时，用户必须能在结果里看到，而不是静默发生。"""
        subject = Box(0, 0, 200, 200)  # 紧贴左上角
        path = make_photo(tmp_path / "photo.png", subject, size=(800, 800))
        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=500, px_height=500,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        plans = {path: CropSpec(mode=CropMode.SUBJECT, subject=subject, margin=0.2)}
        report = Cropper(spec, crop_provider=plans.get).run([path])

        result = report.results[0]
        assert result.ok
        assert any("补背景" in warning for warning in result.warnings)
