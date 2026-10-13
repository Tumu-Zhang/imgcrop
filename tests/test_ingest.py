"""输入探测测试：识别图片、源分辨率，并给出可读的错误提示。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from conftest import make_raster
from imgcrop.ingest import SourceKind, classify, gather, is_image, probe


class TestImage:
    def test_basic_probe(self, tmp_path):
        path = make_raster(tmp_path / "fig.png", (1200, 900))
        info = probe(path)
        assert info.kind is SourceKind.IMAGE
        assert info.pixel_size == (1200, 900)
        assert info.aspect == 1200 / 900
        assert info.frames == 1
        assert info.error is None

    def test_reads_embedded_dpi(self, tmp_path):
        path = make_raster(tmp_path / "fig.png", (600, 600), dpi=(300, 300))
        info = probe(path)
        # PNG 用 pHYs 存 DPI（单位像素/米），300dpi 只能近似表示，允许千分位误差
        assert info.dpi is not None
        assert abs(info.dpi[0] - 300.0) < 0.01
        assert "300" in info.source_dpi_text

    def test_missing_dpi_is_flagged(self, tmp_path):
        path = make_raster(tmp_path / "fig.png", (600, 600))
        info = probe(path)
        assert info.dpi is None
        assert any("无 DPI 元数据" in note for note in info.notes)

    def test_alpha_channel_note(self, tmp_path):
        path = tmp_path / "alpha.png"
        Image.new("RGBA", (100, 100), (0, 0, 0, 128)).save(path)
        info = probe(path)
        assert any("透明" in note for note in info.notes)

    def test_zero_dpi_treated_as_missing(self, tmp_path):
        """有些工具会写 dpi=(0, 0)，那不是可用信息，不能被显示成「0 dpi」。"""
        path = tmp_path / "zero.png"
        Image.new("RGB", (100, 100), "white").save(path, dpi=(0, 0))
        info = probe(path)
        assert info.dpi is None
        assert any("无 DPI 元数据" in note for note in info.notes)

    def test_multipage_gif_collapses_to_first_frame(self, tmp_path):
        path = tmp_path / "anim.gif"
        frames = [Image.new("RGB", (40, 40), (i * 40, 0, 0)) for i in range(3)]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        info = probe(path)
        assert info.frames == 1
        assert any("多帧" in note for note in info.notes)

    def test_tiff_extension(self, tmp_path):
        path = make_raster(tmp_path / "scan.tiff", (300, 300))
        assert probe(path).kind is SourceKind.IMAGE


class TestGather:
    def test_directory_is_expanded_recursively(self, tmp_path):
        make_raster(tmp_path / "a.png", (100, 100))
        make_raster(tmp_path / "nested" / "b.jpg", (100, 100))
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

        found = gather([str(tmp_path)])

        assert {p.name for p in found} == {"a.png", "b.jpg"}

    def test_duplicates_are_collapsed(self, tmp_path):
        path = make_raster(tmp_path / "a.png", (100, 100))
        found = gather([str(path), str(tmp_path), str(path)])
        assert found == [path]

    def test_explicit_files_pass_through(self, tmp_path):
        path = make_raster(tmp_path / "a.png", (100, 100))
        other = tmp_path / "missing.png"
        assert gather([str(path), str(other)]) == [path, other]

    def test_is_image(self):
        assert is_image("x.PNG")
        assert is_image("x.jpeg")
        assert not is_image("x.pdf")


class TestFailures:
    def test_missing_file(self, tmp_path):
        info = probe(tmp_path / "nope.png")
        assert info.error == "文件不存在"

    def test_unsupported_with_hint(self, tmp_path):
        path = tmp_path / "drawing.cdr"
        path.write_bytes(b"RIFFxxxxCDR")
        info = probe(path)
        assert info.kind is SourceKind.UNSUPPORTED
        assert "CorelDRAW" in info.error

    def test_pdf_points_at_the_converter(self, tmp_path):
        """PDF 不是裁剪器的输入，但错误信息必须告诉用户下一步怎么办。"""
        path = tmp_path / "paper.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        info = probe(path)
        assert info.kind is SourceKind.UNSUPPORTED
        assert "图片转换器" in info.error

    def test_eps_hint_mentions_ghostscript(self, tmp_path):
        path = tmp_path / "old.eps"
        path.write_bytes(b"%!PS")
        info = probe(path)
        assert "Ghostscript" in info.error

    def test_unknown_extension(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("hello", encoding="utf-8")
        info = probe(path)
        assert info.kind is SourceKind.UNSUPPORTED

    def test_corrupt_image_reports_error(self, tmp_path):
        path = tmp_path / "fake.png"
        path.write_bytes(b"this is not a png")
        info = probe(path)
        assert info.error is not None


class TestClassify:
    def test_categories(self):
        assert classify(Path("a.jpeg")) is SourceKind.IMAGE
        assert classify(Path("a.TIF")) is SourceKind.IMAGE
        assert classify(Path("a.PDF")) is SourceKind.UNSUPPORTED
        assert classify(Path("a.pptx")) is SourceKind.UNSUPPORTED
        assert classify(Path("a.docx")) is SourceKind.UNSUPPORTED
