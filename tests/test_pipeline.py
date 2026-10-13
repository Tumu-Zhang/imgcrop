"""端到端管线测试：从图片文件到最终落盘图片的完整链路。"""

from __future__ import annotations

import pytest
from PIL import Image

from conftest import (
    PNG_DPI_TOLERANCE,
    TIFF_LZW,
    make_raster,
    read_dpi,
    read_size,
    read_tiff_compression,
)
from imgcrop.model import (
    ConflictPolicy,
    OutputFormat,
    OutputSpec,
    SizeMode,
    TiffCompression,
    Unit,
)
from imgcrop.pipeline import crop_batch, default_output_dir


def base_spec(out_dir, **kwargs) -> OutputSpec:
    params = dict(
        fmt=OutputFormat.TIFF,
        size_mode=SizeMode.PHYSICAL,
        phys_unit=Unit.CM,
        phys_width=8.5,
        dpi=300,
        output_dir=out_dir,
    )
    params.update(kwargs)
    return OutputSpec(**params)


class TestImageOutput:
    def test_png_to_tiff_physical_size(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig1.png", (1000, 800))
        report = crop_batch([src], base_spec(workdir))
        assert report.success, [r.error for r in report.results]

        out = report.results[0].out_path
        assert out.suffix == ".tif"
        assert read_size(out) == (1004, 803)
        assert read_dpi(out) == (300.0, 300.0)

    def test_tiff_is_lzw_compressed(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (600, 400))
        report = crop_batch([src], base_spec(workdir, tiff_compression=TiffCompression.LZW))
        assert read_tiff_compression(report.results[0].out_path) == TIFF_LZW

    def test_pixel_mode_produces_exact_size(self, tmp_path, workdir):
        """源比例与目标框一致时，像素模式给出精确尺寸。"""
        src = make_raster(tmp_path / "fig.png", (800, 600))
        spec = base_spec(
            workdir, size_mode=SizeMode.PIXELS, px_width=1200, px_height=900, dpi=600
        )
        report = crop_batch([src], spec)
        out = report.results[0].out_path
        assert read_size(out) == (1200, 900)
        assert read_dpi(out) == (600.0, 600.0)

    def test_pixel_box_fits_within_when_aspect_differs(self, tmp_path, workdir):
        """源比例与目标框不一致时应等比内接，而不是拉伸变形。"""
        src = make_raster(tmp_path / "fig.png", (1000, 800))  # 比例 1.25
        spec = base_spec(
            workdir, size_mode=SizeMode.PIXELS, px_width=1200, px_height=900, dpi=600
        )
        report = crop_batch([src], spec)
        # 高度顶满 900，宽度 = 900 * 1.25 = 1125，小于框宽 1200
        assert read_size(report.results[0].out_path) == (1125, 900)

    def test_jpeg_output(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (800, 600))
        report = crop_batch([src], base_spec(workdir, fmt=OutputFormat.JPEG, jpeg_quality=90))
        out = report.results[0].out_path
        assert out.suffix == ".jpg"
        with Image.open(out) as img:
            assert img.format == "JPEG"
        assert read_dpi(out) == (300.0, 300.0)

    def test_png_output(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (800, 600))
        report = crop_batch([src], base_spec(workdir, fmt=OutputFormat.PNG))
        out = report.results[0].out_path
        with Image.open(out) as img:
            assert img.format == "PNG"
        # PNG 的 pHYs 以「像素/米」存储，300dpi 只能近似表示，这是格式固有精度
        dx, dy = read_dpi(out)
        assert dx == pytest.approx(300.0, abs=PNG_DPI_TOLERANCE)
        assert dy == pytest.approx(300.0, abs=PNG_DPI_TOLERANCE)

    def test_pdf_output(self, tmp_path, workdir):
        """输出成 PDF 是投稿常见要求，编码层仍然支持。"""
        src = make_raster(tmp_path / "fig.png", (800, 600))
        report = crop_batch([src], base_spec(workdir, fmt=OutputFormat.PDF))
        out = report.results[0].out_path
        assert out.suffix == ".pdf"
        assert out.stat().st_size > 0

    def test_sixteen_bit_tiff_preserved(self, tmp_path, workdir):
        """16 位科学图像在 TIFF 输出下不能被静默截断到 8 位。"""
        src = make_raster(tmp_path / "deep.tif", (400, 300), mode="I;16")
        report = crop_batch([src], base_spec(workdir, fmt=OutputFormat.TIFF))
        with Image.open(report.results[0].out_path) as img:
            assert img.mode in ("I;16", "I;16B", "I;16L", "I;16N")
            assert img.size == (1004, 753)

    def test_sixteen_bit_to_jpeg_warns_about_bit_depth(self, tmp_path, workdir):
        src = make_raster(tmp_path / "deep.tif", (400, 300), mode="I;16")
        report = crop_batch([src], base_spec(workdir, fmt=OutputFormat.JPEG))
        result = report.results[0]
        assert result.ok
        assert any("16 位" in w for w in result.warnings)
        with Image.open(result.out_path) as img:
            assert img.mode == "L"

    def test_no_upscale_keeps_source_pixels(self, tmp_path, workdir):
        src = make_raster(tmp_path / "small.png", (200, 160))
        report = crop_batch([src], base_spec(workdir, allow_upscale=False))
        result = report.results[0]
        assert read_size(result.out_path) == (200, 160)
        assert any("未放大" in w for w in result.warnings)


class TestOutputPlumbing:
    def test_default_output_dir_is_cropped_subfolder(self, tmp_path):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        report = crop_batch([src], OutputSpec(phys_width=5, dpi=300))
        expected_dir = default_output_dir(tmp_path)
        assert report.results[0].out_path.parent == expected_dir
        assert expected_dir.name == "cropped"

    def test_name_template(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (800, 600))
        report = crop_batch([src], base_spec(workdir, name_template="{stem}_300dpi"))
        assert report.results[0].out_path.name == "fig_300dpi.tif"

    def test_conflict_rename_keeps_both(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        spec = base_spec(workdir, on_conflict=ConflictPolicy.RENAME)
        first = crop_batch([src], spec).results[0].out_path
        second = crop_batch([src], spec).results[0].out_path
        assert first != second
        assert second.name == "fig_1.tif"
        assert first.exists() and second.exists()

    def test_conflict_skip(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        spec = base_spec(workdir, on_conflict=ConflictPolicy.SKIP)
        assert crop_batch([src], spec).results[0].ok
        second = crop_batch([src], spec).results[0]
        assert not second.ok
        assert "跳过" in second.error

    def test_conflict_overwrite(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        spec = base_spec(workdir, on_conflict=ConflictPolicy.OVERWRITE)
        first = crop_batch([src], spec).results[0]
        second = crop_batch([src], spec).results[0]
        assert first.out_path == second.out_path


class TestBatchRobustness:
    def test_unsupported_file_reports_error_and_batch_continues(self, tmp_path, workdir):
        good = make_raster(tmp_path / "ok.png", (400, 300))
        bad = tmp_path / "drawing.cdr"
        bad.write_bytes(b"RIFFxxxxCDR")
        missing = tmp_path / "nope.png"

        report = crop_batch([good, bad, missing], base_spec(workdir))
        assert report.ok_count == 1
        assert report.fail_count == 2
        errors = " ".join(r.error or "" for r in report.results)
        assert "CorelDRAW" in errors or "不支持" in errors
        assert "不存在" in errors

    def test_invalid_spec_fails_every_item(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        spec = base_spec(workdir, phys_width=None)
        report = crop_batch([src], spec)
        assert report.fail_count == 1
        assert "输出规格有误" in report.results[0].error

    def test_cancel_after_first_file(self, tmp_path, workdir):
        """取消在文件边界生效：已开始的这张做完，后面的不再开始。"""
        srcs = [make_raster(tmp_path / f"f{i}.png", (400, 300)) for i in range(3)]
        state = {"cancel": False}

        def progress(event):
            if event.completed >= 1:
                state["cancel"] = True

        report = crop_batch(
            srcs,
            base_spec(workdir),
            progress=progress,
            should_cancel=lambda: state["cancel"],
        )
        assert report.ok_count == 2
        # 已经通知开始的那张会做完，之后不再开始新的 —— 取消不会把半张图留在半路
        assert {r.source.name for r in report.results} == {"f0.png", "f1.png"}

    def test_progress_callback_reports_files(self, tmp_path, workdir):
        srcs = [make_raster(tmp_path / f"f{i}.png", (200, 200)) for i in range(3)]
        events = []
        crop_batch(srcs, base_spec(workdir), progress=events.append)
        assert [e.completed for e in events] == [0, 1, 2]
        assert all(e.total == 3 for e in events)
        assert all(e.current for e in events)  # 每次都要带上正在处理的那张

    def test_pixel_sizes_reports_inconsistency(self, tmp_path, workdir):
        """整批尺寸不一致时必须能被报告出来，界面据此提示用户指定比例。"""
        srcs = [make_raster(tmp_path / f"f{i}.png", (500 + i * 40, 400)) for i in range(2)]
        spec = base_spec(workdir, size_mode=SizeMode.PIXELS, px_width=800, dpi=300)
        report = crop_batch(srcs, spec)
        assert len(report.pixel_sizes()) == 2


class TestIdempotence:
    def test_rerunning_on_own_output_keeps_geometry(self, tmp_path, workdir):
        """把产物再跑一遍，尺寸与 DPI 必须稳定（可复现）。"""
        src = make_raster(tmp_path / "fig.png", (1000, 800))
        spec = base_spec(workdir, size_mode=SizeMode.PIXELS, px_width=1000, dpi=300)
        first = crop_batch([src], spec).results[0].out_path

        dest = workdir / "round2"
        spec2 = base_spec(dest, size_mode=SizeMode.PIXELS, px_width=1000, dpi=300)
        second = crop_batch([first], spec2).results[0].out_path

        assert read_size(second) == read_size(first)
        assert read_dpi(second) == read_dpi(first)
