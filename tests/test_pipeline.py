"""端到端管线测试：从图片文件到最终落盘图片的完整链路。

1.1.0 瘦身后（D5/D6/D11）：输出 = 裁剪窗原像素（1:1，无重采样），
格式跟随源、DPI 跟随源 —— 本文件顺着这条新语义验证落盘与容错。
"""

from __future__ import annotations

import pytest
from PIL import Image

from conftest import PNG_DPI_TOLERANCE, make_raster, read_dpi, read_size
from imgcrop.crop import CropMode, CropSpec
from imgcrop.geometry import Box
from imgcrop.model import ConflictPolicy, OutputSpec
from imgcrop.pipeline import CropPlan, crop_batch, default_output_dir


def base_spec(out_dir, **kwargs) -> OutputSpec:
    params = dict(output_dir=out_dir)
    params.update(kwargs)
    return OutputSpec(**params)


def plan_for(box=None, mode=CropMode.NONE, **kwargs) -> CropPlan:
    spec_kwargs = dict(mode=mode, **kwargs)
    if box is not None:
        spec_kwargs["subject"] = box if isinstance(box, Box) else Box(*box)
    return CropPlan(spec=CropSpec(**spec_kwargs))


class TestImageOutput:
    def test_output_equals_crop_window_one_to_one(self, tmp_path, workdir):
        """核心语义：成品像素 = 裁剪窗原像素，不缩放不放大。"""
        src = make_raster(tmp_path / "fig1.png", (1000, 800))
        report = crop_batch(
            [src],
            base_spec(workdir),
            crop_provider=lambda _p: plan_for(mode=CropMode.FREE, box=(100, 80, 600, 480)),
        )
        assert report.success, [r.error for r in report.results]

        result = report.results[0]
        out = result.out_path
        assert out.suffix == ".png"  # 格式跟随源
        assert read_size(out) == (500, 400)
        assert result.out_size == (500, 400)

    def test_no_crop_passes_source_through(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (800, 600))
        report = crop_batch([src], base_spec(workdir))
        assert read_size(report.results[0].out_path) == (800, 600)

    def test_dpi_defaults_to_300_when_source_has_none(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300))  # 未写 DPI
        report = crop_batch([src], base_spec(workdir))
        out = report.results[0].out_path
        dx, dy = read_dpi(out)
        assert dx == pytest.approx(300.0, abs=PNG_DPI_TOLERANCE + 1)
        assert dy == pytest.approx(300.0, abs=PNG_DPI_TOLERANCE + 1)

    def test_dpi_follows_source_metadata(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300), dpi=(96, 96))
        report = crop_batch([src], base_spec(workdir))
        out = report.results[0].out_path
        dx, _dy = read_dpi(out)
        assert dx == pytest.approx(96.0, abs=1)

    def test_jpg_source_produces_jpg(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.jpg", (800, 600))
        report = crop_batch([src], base_spec(workdir))
        out = report.results[0].out_path
        assert out.suffix == ".jpg"
        with Image.open(out) as img:
            assert img.format == "JPEG"

    def test_tiff_source_produces_png(self, tmp_path, workdir):
        """D6：TIFF→PNG（投稿场景统一到通用无损容器）。"""
        src = make_raster(tmp_path / "fig.tif", (800, 600))
        report = crop_batch([src], base_spec(workdir))
        out = report.results[0].out_path
        assert out.suffix == ".png"
        with Image.open(out) as img:
            assert img.format == "PNG"

    def test_sixteen_bit_preserved_for_png_output(self, tmp_path, workdir):
        """16 位科学图经 TIFF→PNG 链路不能被静默截断到 8 位。"""
        src = make_raster(tmp_path / "deep.tif", (400, 300), mode="I;16")
        report = crop_batch([src], base_spec(workdir))
        result = report.results[0]
        assert result.ok
        with Image.open(result.out_path) as img:
            assert img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I")
            assert img.size == (400, 300)


class TestOutputPlumbing:
    def test_default_output_dir_is_cropped_subfolder(self, tmp_path):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        report = crop_batch([src], OutputSpec())
        expected_dir = default_output_dir(tmp_path)
        assert report.results[0].out_path.parent == expected_dir
        assert expected_dir.name == "cropped"

    def test_name_template(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (800, 600))
        report = crop_batch([src], base_spec(workdir, name_template="{stem}_cropped"))
        assert report.results[0].out_path.name == "fig_cropped.png"

    def test_conflict_rename_keeps_both(self, tmp_path, workdir):
        src = make_raster(tmp_path / "fig.png", (400, 300))
        spec = base_spec(workdir, on_conflict=ConflictPolicy.RENAME)
        first = crop_batch([src], spec).results[0].out_path
        second = crop_batch([src], spec).results[0].out_path
        assert first != second
        assert second.name == "fig_1.png"
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
        """输出尺寸随取景框走 —— 框不同尺寸就不同，这正是要提示用户复核的信号。"""
        srcs = [make_raster(tmp_path / f"f{i}.png", (500 + i * 40, 400)) for i in range(2)]
        specs = [CropSpec(mode=CropMode.FREE, subject=Box(0, 0, 300 + i * 50, 200)) for i in range(2)]
        report = crop_batch(
            srcs,
            base_spec(workdir),
            crop_provider=lambda p: next(
                (CropPlan(spec=s) for s, src in zip(specs, srcs) if src == p), None
            ),
        )
        assert len(report.pixel_sizes()) == 2


class TestIdempotence:
    def test_rerunning_on_own_output_keeps_geometry(self, tmp_path, workdir):
        """把产物再跑一遍，尺寸与 DPI 必须稳定（可复现）。"""
        src = make_raster(tmp_path / "fig.png", (1000, 800))
        plan = plan_for(mode=CropMode.FREE, box=Box(100, 100, 700, 600))
        spec = base_spec(workdir)
        first = crop_batch([src], spec, crop_provider=lambda _p: plan).results[0].out_path

        dest = workdir / "round2"
        spec2 = base_spec(dest)
        second = crop_batch([first], spec2).results[0].out_path

        assert read_size(second) == read_size(first)
        assert read_dpi(second) == read_dpi(first)
