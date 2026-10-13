"""编码与体积上限控制测试。

体积上限是期刊投稿的硬约束（常见"单图 ≤ 10MB"），也是最容易悄悄降质的地方，
所以这里既要验"能压到"，也要验"降质一定有警告"。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import TIFF_ADOBE_DEFLATE, TIFF_JPEG, TIFF_LZW
from imgcrop import encode
from imgcrop.encode import VolumeLimitUnreachable
from imgcrop.model import OutputFormat, OutputSpec, TiffCompression


def noise_image(size: tuple[int, int] = (700, 700)) -> Image.Image:
    """随机噪声图：无损压缩几乎压不动，用来逼出有损兜底路径。"""
    rng = np.random.default_rng(20240516)
    data = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


def spec_for(fmt: OutputFormat, *, limit: int | None = None, **kwargs) -> OutputSpec:
    params = dict(
        fmt=fmt,
        phys_width=2.0,
        dpi=300,
        max_bytes=limit,
        output_dir=None,
    )
    params.update(kwargs)
    return OutputSpec(**params)


class TestNoLimit:
    def test_jpeg_untouched_when_no_limit(self):
        image = noise_image((300, 300))
        _, result = encode.encode_image(image, spec_for(OutputFormat.JPEG, limit=None))
        assert result.quality_used == 95
        assert not any("上限" in w for w in result.warnings)

    def test_generous_limit_does_not_downgrade(self):
        image = Image.new("RGB", (300, 300), "white")
        _, result = encode.encode_image(
            image, spec_for(OutputFormat.JPEG, limit=50 * 1024 * 1024)
        )
        assert result.quality_used == 95


class TestJpegVolume:
    def test_binary_search_stays_under_limit(self):
        image = noise_image()
        limit = 120 * 1024
        data, result = encode.encode_image(
            image, spec_for(OutputFormat.JPEG, limit=limit)
        )
        assert len(data) <= limit
        assert result.quality_used < 95
        assert any("质量" in w for w in result.warnings)

    def test_lower_limit_yields_smaller_file(self):
        image = noise_image()
        big, _ = encode.encode_image(
            image, spec_for(OutputFormat.JPEG, limit=400 * 1024)
        )
        small, _ = encode.encode_image(
            image, spec_for(OutputFormat.JPEG, limit=60 * 1024)
        )
        assert len(small) < len(big)

    def test_impossible_limit_raises(self):
        image = noise_image((1200, 1200))
        with pytest.raises(VolumeLimitUnreachable, match="请降低 DPI"):
            encode.encode_image(image, spec_for(OutputFormat.JPEG, limit=200))


class TestTiffVolume:
    def test_lossless_kept_when_it_fits(self):
        image = Image.new("RGB", (600, 600), (12, 34, 56))
        data, result = encode.encode_image(
            image, spec_for(OutputFormat.TIFF, limit=5 * 1024 * 1024)
        )
        assert result.compression_used == "lzw"
        assert not any("有损" in w for w in result.warnings)

    def test_falls_back_to_deflate_before_lossy(self):
        """无损档位之间先互相尝试，不应直接跳到有损。"""
        image = noise_image((700, 700))
        limit = 4 * 1024 * 1024
        _, result = encode.encode_image(
            image, spec_for(OutputFormat.TIFF, limit=limit, tiff_compression=TiffCompression.LZW)
        )
        assert result.compression_used in ("lzw", "deflate")
        assert not any("有损" in w for w in result.warnings)

    def test_lossy_fallback_warns_explicitly(self):
        image = noise_image((900, 900))
        # 无损档位必然压不进（噪声 TIFF/LZW 约 2.4MB），有损档位则能达标
        limit = 300 * 1024
        data, result = encode.encode_image(
            image, spec_for(OutputFormat.TIFF, limit=limit)
        )
        assert result.compression_used == "jpeg"
        assert len(data) <= limit
        assert any("有损" in w for w in result.warnings)

    def test_lossy_fallback_can_be_forbidden(self):
        image = noise_image((900, 900))
        with pytest.raises(VolumeLimitUnreachable, match="无损"):
            encode.encode_image(
                image,
                spec_for(OutputFormat.TIFF, limit=150 * 1024, lossy_fallback=False),
            )


class TestPngVolume:
    def test_lossless_reports_instead_of_silently_degrading(self):
        image = noise_image((800, 800))
        limit = 100 * 1024
        data, result = encode.encode_image(
            image, spec_for(OutputFormat.PNG, limit=limit)
        )
        # PNG 无法有损压缩：仍输出无损文件，但必须明确告知未达标
        assert len(data) > limit
        assert any("无损格式" in w for w in result.warnings)


class TestBitDepthHandling:
    def test_sixteen_bit_kept_for_tiff(self):
        arr = np.linspace(0, 65535, 256 * 256, dtype=np.uint16).reshape(256, 256)
        image = Image.fromarray(arr)
        data, _ = encode.encode_image(image, spec_for(OutputFormat.TIFF))
        import io

        with Image.open(io.BytesIO(data)) as back:
            assert back.mode in ("I;16", "I;16B", "I;16L", "I;16N")

    def test_sixteen_bit_downshifted_for_jpeg(self):
        arr = np.full((64, 64), 65535, dtype=np.uint16)
        image = Image.fromarray(arr)
        data, result = encode.encode_image(image, spec_for(OutputFormat.JPEG))
        import io

        with Image.open(io.BytesIO(data)) as back:
            assert back.mode == "L"
        assert any("16 位" in w for w in result.warnings)


class TestCompressionMapping:
    """确保压缩参数真的写进了 TIFF tag，而不是被 Pillow 静默忽略。"""

    @pytest.mark.parametrize(
        "compression,expected_tag",
        [
            (TiffCompression.LZW, TIFF_LZW),
            (TiffCompression.DEFLATE, TIFF_ADOBE_DEFLATE),
        ],
    )
    def test_tags_written(self, compression, expected_tag, tmp_path):
        image = Image.new("RGB", (200, 200), (200, 10, 10))
        data, _ = encode.encode_image(
            image, spec_for(OutputFormat.TIFF, tiff_compression=compression)
        )
        path = tmp_path / "x.tif"
        path.write_bytes(data)
        with Image.open(path) as back:
            assert int(back.tag_v2.get(259, 1)) == expected_tag

    def test_jpeg_compression_tag(self, tmp_path):
        image = noise_image((300, 300))
        data, _ = encode.encode_image(
            image,
            spec_for(OutputFormat.TIFF, tiff_compression=TiffCompression.JPEG),
        )
        path = tmp_path / "x.tif"
        path.write_bytes(data)
        with Image.open(path) as back:
            assert int(back.tag_v2.get(259, 1)) == TIFF_JPEG


class TestDpiWritten:
    @pytest.mark.parametrize("fmt", [OutputFormat.TIFF, OutputFormat.JPEG])
    def test_exact_dpi(self, fmt, tmp_path):
        image = Image.new("RGB", (300, 300), "white")
        spec = spec_for(fmt)
        spec.dpi = 600
        data, _ = encode.encode_image(image, spec)
        path = tmp_path / f"x{fmt.extension}"
        path.write_bytes(data)
        with Image.open(path) as back:
            assert back.info["dpi"] == (600.0, 600.0)
