"""编码层测试：格式跟随源（D6）、DPI 跟随源（D11）、透明通道处理。

体积上限与 TIFF 压缩链路已随 1.1.0 瘦身删除（D5/D7 定案），
编码只保留两条路径：PNG（无损、保透明）与 JPEG（quality 95、透明合成白底）。
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from imgcrop import encode
from imgcrop.encode import DEFAULT_DPI, JPEG_QUALITY, encode_image, resolve_output_format


class TestResolveOutputFormat:
    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".jpg", "JPEG"),
            (".jpeg", "JPEG"),
            (".jfif", "JPEG"),
            (".bmp", "JPEG"),
            (".webp", "JPEG"),
            ("", "JPEG"),
            (".png", "PNG"),
            (".tif", "PNG"),
            (".tiff", "PNG"),
        ],
    )
    def test_mapping(self, suffix, expected):
        assert resolve_output_format(suffix) == expected

    def test_case_insensitive(self):
        assert resolve_output_format(".PNG") == "PNG"
        assert resolve_output_format(".JPG") == "JPEG"


class TestJpegEncoding:
    def test_quality_is_fixed_constant(self):
        image = Image.new("RGB", (200, 200), (200, 30, 60))
        data, _result = encode_image(image, "JPEG")
        assert data[:2] == b"\xff\xd8"  # JPEG 魔数
        assert JPEG_QUALITY == 95  # D6：质量是内部常量，不设 UI

    def test_dpi_written(self, tmp_path):
        image = Image.new("RGB", (300, 300), "white")
        path = tmp_path / "x.jpg"
        encode.encode_to_file(image, path, "JPEG", dpi=350)
        with Image.open(path) as back:
            assert back.info["dpi"] == (350.0, 350.0)

    def test_default_dpi_is_300(self, tmp_path):
        image = Image.new("RGB", (300, 300), "white")
        path = tmp_path / "x.jpg"
        encode.encode_to_file(image, path, "JPEG")
        with Image.open(path) as back:
            assert back.info["dpi"][0] == pytest.approx(DEFAULT_DPI, abs=1)

    def test_rgba_flattens_to_white_background(self):
        """RGBA 编码成 JPEG 时透明区必须合成白底（直接 convert 会变黑）。"""
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))  # 全透明红
        data, result = encode_image(image, "JPEG")
        assert any("白" in w for w in result.warnings)
        back = Image.open(io.BytesIO(data)).convert("RGB")
        r, g, b = back.getpixel((50, 50))
        assert (r, g, b) == (255, 255, 255)

    def test_sixteen_bit_downshifted(self):
        arr = np.full((64, 64), 65535, dtype=np.uint16)
        image = Image.fromarray(arr)
        data, result = encode_image(image, "JPEG")
        with Image.open(io.BytesIO(data)) as back:
            assert back.mode == "L"
        assert any("16 位" in w for w in result.warnings)


class TestPngEncoding:
    def test_rgba_preserved(self):
        """D6：PNG→PNG 保透明 —— RGBA 进 RGBA 出。"""
        image = Image.new("RGBA", (100, 100), (200, 30, 60, 120))
        data, result = encode_image(image, "PNG")
        assert not any("透明" in w for w in result.warnings)
        back = Image.open(io.BytesIO(data))
        assert back.mode == "RGBA"
        assert back.getpixel((50, 50))[3] == pytest.approx(120, abs=2)

    def test_cmyk_converted(self):
        image = Image.new("CMYK", (80, 80), (10, 20, 30, 5))
        data, result = encode_image(image, "PNG")
        with Image.open(io.BytesIO(data)) as back:
            assert back.mode == "RGB"
        assert any("CMYK" in w for w in result.warnings)


class TestResolveSourceDpi:
    def test_follows_source(self):
        assert encode.resolve_source_dpi((600.0, 600.0)) == 600

    def test_default_when_missing(self):
        assert encode.resolve_source_dpi(None) == 300
        assert encode.resolve_source_dpi(()) == 300

    def test_invalid_values_fall_back(self):
        assert encode.resolve_source_dpi((0.0, 0.0)) == 300
        assert encode.resolve_source_dpi(("bad", "bad")) == 300
