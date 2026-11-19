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
from imgcrop.model import OutputSpec
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
    def test_batch_with_different_subject_sizes_keeps_uniform_share(self, tmp_path):
        """中心承诺（1.1.0 语义）：主体大小差 2.3 倍的照片，主体在成品里的占比一致。

        1.1.0 起输出 = 裁剪窗原像素（1:1，无重采样），"批内一致"因此表现为
        主体占比一致 + 成品尺寸与主体尺寸成正比，而不是像素完全相同。
        """
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
        spec = OutputSpec(output_dir=out)

        segmenter = IsnetSegmenter()
        plans: dict[Path, CropSpec] = {}
        for photo in photos:
            with Image.open(photo) as image:
                best = segmenter.saliency(image).best
            assert best is not None, f"{photo.name} 没有找到主体"
            plans[photo] = CropSpec(
                mode=CropMode.SUBJECT, subject=best, margin=0.08, aspect=4 / 3
            )

        report = Cropper(spec, crop_provider=plans.get).run(photos)

        assert report.fail_count == 0, [r.error for r in report.results if not r.ok]
        shares = []
        for photo in photos:
            with Image.open(out / f"{photo.stem}.png") as produced:
                assert produced.size[1] == pytest.approx(
                    produced.size[0] * 3 / 4, rel=0.02
                ), f"比例应固定为 4:3：{photo.name} {produced.size}"
                shares.append(
                    plans[photo].subject.width * (1 + 2 * 0.08) / produced.size[0]
                )
        assert max(shares) - min(shares) <= 0.03, f"主体占比应一致：{shares}"

    def test_source_files_are_untouched(self, tmp_path):
        photo = make_photo(tmp_path / "src" / "one.png", (300, 220, 800, 620))
        before = photo.read_bytes()

        out = tmp_path / "out"
        spec = OutputSpec(output_dir=out)
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
