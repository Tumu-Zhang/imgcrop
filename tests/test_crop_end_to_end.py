"""裁剪功能的端到端验证：真实模型 + 合成照片，走完整链路。

这条链路把前面所有模块串起来：ISNet 找主体 -> CropSpec 取景 -> OutputSpec 定规格
-> 编码输出。它要跑一遍模型，比别的测试慢，但能一次性发现"模块各自都对、拼起来错"
这类问题 —— 尤其是坐标映射。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from imgcrop import ingest, models
from imgcrop.crop import CropMode, CropSpec
from imgcrop.model import OutputFormat, OutputSpec, SizeMode
from imgcrop.pipeline import Cropper
from imgcrop.segment import IsnetSegmenter

pytestmark = pytest.mark.skipif(
    not models.is_installed(models.ISNET),
    reason="ISNet 模型未安装（先运行 python tools/fetch_models.py）",
)


def make_photo(path: Path, subject: tuple[int, int, int, int], size=(1200, 900)) -> Path:
    """造一张"深色台面 + 浅色试样"的模拟实验照片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (48, 52, 58))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = subject
    draw.rectangle([left, top, right - 1, bottom - 1], fill=(236, 232, 224))
    # 加一点内部结构，避免主体是一块纯色而过于理想化
    draw.ellipse([left + 40, top + 40, left + 140, top + 140], fill=(96, 116, 168))
    image.save(path, dpi=(72, 72))
    return path


def load_crop_cli():
    """命令行入口本身就是一个模块，直接拿来用（不再需要按路径加载脚本）。"""
    from imgcrop import cli

    return cli


class TestAutoCropPipeline:
    def test_batch_with_different_subject_sizes_comes_out_uniform(self, tmp_path):
        """这是这个功能的中心承诺：主体大小差 2.3 倍的照片，成品完全一致。"""
        photos: list[Path] = []
        for index, scale in enumerate((0.6, 1.0, 1.4)):
            width, height = int(300 * scale), int(240 * scale)
            left, top = 200 + index * 60, 150 + index * 40
            photos.append(
                make_photo(
                    tmp_path / "src" / f"p{index}.png",
                    (left, top, left + width, top + height),
                )
            )

        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=1000, px_height=1000,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )

        segmenter = IsnetSegmenter()
        plans: dict[Path, CropSpec] = {}
        for photo in photos:
            with Image.open(photo) as image:
                best = segmenter.saliency(image).best
            assert best is not None, f"{photo.name} 没有找到主体"
            # 固定比例是"整批像素尺寸完全一致"的必要条件：只做主体归一化时，
            # 各图检测框的宽高比会有微差，等比内接后高度就会差几个像素。
            plans[photo] = CropSpec(
                mode=CropMode.SUBJECT, subject=best, margin=0.08, aspect=4 / 3
            )

        report = Cropper(spec, crop_provider=plans.get).run(photos)

        assert report.fail_count == 0, [r.error for r in report.results if not r.ok]
        sizes = set()
        for photo in photos:
            with Image.open(out / f"{photo.stem}.png") as produced:
                sizes.add(produced.size)

        widths = {width for width, _ in sizes}
        heights = {height for _, height in sizes}
        assert widths == {1000}, sizes
        # 高度允许 1 像素的取整差：裁剪窗是整数像素矩形，其宽高比无法精确等于
        # 4:3，等比内接后就会差 1 像素 —— 与 model.py 里记录的"整数像素的必然
        # 结果"是同一回事，Photoshop 也一样。要像素级严格一致就用 keep_aspect=False。
        assert max(heights) - min(heights) <= 1, f"整批高度不应相差超过 1 像素：{sizes}"
        assert min(heights) >= 748, f"比例明显不对：{sizes}"

    def test_source_files_are_untouched(self, tmp_path):
        photo = make_photo(tmp_path / "src" / "one.png", (300, 220, 800, 620))
        before = photo.read_bytes()

        out = tmp_path / "out"
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=800, px_height=800,
            dpi=300, fmt=OutputFormat.PNG, output_dir=out,
        )
        with Image.open(photo) as image:
            best = IsnetSegmenter().saliency(image).best
        assert best is not None

        plans = {photo: CropSpec(mode=CropMode.SUBJECT, subject=best, margin=0.1)}
        Cropper(spec, crop_provider=plans.get).run([photo])

        assert photo.read_bytes() == before, "源照片被改动了 —— 这违反了硬约定"


class TestCommandLine:
    def test_dry_run_finds_subject_without_writing_anything(self, tmp_path):
        photo = make_photo(tmp_path / "src" / "one.png", (300, 220, 800, 620))
        cli = load_crop_cli()

        code = cli.main([str(photo), "--dry-run"])

        assert code == 0
        assert list((tmp_path / "src").glob("*")) == [photo]

    def test_aspect_flag_is_parsed(self):
        cli = load_crop_cli()
        assert cli.parse_aspect("1:1") == pytest.approx(1.0)
        assert cli.parse_aspect("4:3") == pytest.approx(4 / 3)
        assert cli.parse_aspect("none") is None
        assert cli.parse_aspect("原始") is None

    def test_directory_input_is_expanded(self, tmp_path):
        make_photo(tmp_path / "src" / "a.png", (100, 100, 400, 400))
        make_photo(tmp_path / "src" / "b.jpg", (100, 100, 400, 400))
        (tmp_path / "src" / "notes.txt").write_text("ignore me", encoding="utf-8")

        found = ingest.gather([str(tmp_path / "src")])

        assert {p.name for p in found} == {"a.png", "b.jpg"}
