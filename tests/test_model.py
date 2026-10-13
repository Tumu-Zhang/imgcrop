"""几何解析的单元测试 —— 这是整个工具最容易出错的地方。"""

from __future__ import annotations

import pytest

from imgcrop.model import (
    OutputFormat,
    OutputSpec,
    SizeMode,
    SpecError,
    TiffCompression,
    Unit,
    fit_within,
    resolve_geometry,
)


class TestPhysicalMode:
    def test_single_width_keeps_aspect(self):
        """只给宽度：宽度精确，高度按源比例推出。"""
        spec = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.CM,
            phys_width=8.5, dpi=300,
        )
        geo = spec.resolve(src_aspect=1000 / 800)
        assert geo.pixel_size == (1004, 803)  # 8.5cm = 3.34646in * 300 = 1003.9 -> 1004
        assert geo.dpi == 300

    def test_mm_and_inch_units(self):
        spec_mm = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.MM, phys_width=170, dpi=300
        )
        geo_mm = spec_mm.resolve(src_aspect=1.0)
        assert geo_mm.pixel_size[0] == 2008  # 170mm = 6.6929in * 300 = 2007.87 -> 2008

        spec_in = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.INCH, phys_width=2, dpi=300
        )
        geo_in = spec_in.resolve(src_aspect=1.0)
        assert geo_in.pixel_size[0] == 600

    def test_single_height_keeps_aspect(self):
        spec = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.CM, phys_height=6.0, dpi=600
        )
        geo = spec.resolve(src_aspect=2.0)
        # 6cm = 2.362205in，@600dpi 高 = 1417.3 -> 1417；宽 = 4.724409in * 600 = 2834.6 -> 2835
        assert geo.pixel_size == (2835, 1417)

    def test_box_both_sides_fits_within(self):
        """宽高都给时是"内接"，不得超出框，也不得变形。"""
        spec = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.INCH,
            phys_width=3.0, phys_height=1.0, dpi=300, keep_aspect=True,
        )
        # 源比例 2.0 < 框比例 3.0 -> 高度受限
        wide = spec.resolve(src_aspect=2.0)
        assert wide.pixel_size == (600, 300)

        # 源比例 4.0 > 框比例 3.0 -> 宽度受限
        flat = spec.resolve(src_aspect=4.0)
        assert flat.pixel_size == (900, 225)
        for geo, box in ((wide, (900, 300)), (flat, (900, 300))):
            assert geo.pixel_size[0] <= box[0]
            assert geo.pixel_size[1] <= box[1]

    def test_stretch_when_aspect_disabled(self):
        spec = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.INCH,
            phys_width=3.0, phys_height=1.0, dpi=300, keep_aspect=False,
        )
        geo = spec.resolve(src_aspect=2.0)
        assert geo.pixel_size == (900, 300)


class TestPixelMode:
    def test_exact_pixels(self):
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=2550, px_height=3300, dpi=300
        )
        geo = spec.resolve(src_aspect=2550 / 3300)
        assert geo.pixel_size == (2550, 3300)
        assert geo.dpi == 300

    def test_pixels_derive_physical_size(self):
        spec = OutputSpec(size_mode=SizeMode.PIXELS, px_width=1200, dpi=600)
        geo = spec.resolve(src_aspect=1.0)
        assert geo.pixel_size == (1200, 1200)
        assert geo.width_in == pytest.approx(2.0)
        assert geo.height_in == pytest.approx(2.0)

    def test_pixel_width_only_follows_source_aspect(self):
        spec = OutputSpec(size_mode=SizeMode.PIXELS, px_width=1000, dpi=300)
        geo = spec.resolve(src_aspect=1.5)
        assert geo.pixel_size == (1000, 667)


class TestValidation:
    def test_missing_both_sides(self):
        with pytest.raises(SpecError, match="至少填写"):
            OutputSpec(phys_width=None, phys_height=None).validate()

    def test_stretch_requires_both_sides(self):
        with pytest.raises(SpecError, match="取消等比"):
            OutputSpec(phys_width=8, keep_aspect=False).validate()

    def test_bad_dpi(self):
        with pytest.raises(SpecError, match="DPI"):
            OutputSpec(phys_width=8, dpi=0).validate()

    def test_bad_quality(self):
        with pytest.raises(SpecError, match="JPEG 质量"):
            OutputSpec(phys_width=8, jpeg_quality=0).validate()

    def test_zero_size_rejected(self):
        with pytest.raises(SpecError, match="必须大于 0"):
            OutputSpec(phys_width=-1).validate()

    def test_unknown_aspect_with_both_sides_ok(self):
        spec = OutputSpec(phys_unit=Unit.INCH, phys_width=2, phys_height=2)
        geo = spec.resolve(src_aspect=None)
        assert geo.pixel_size == (600, 600)


class TestNoUpscale:
    def test_shrinks_when_source_too_small(self):
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=1000, dpi=300, allow_upscale=False
        )
        target = spec.resolve(src_aspect=1.0)
        limited, warnings = fit_within(target, (400, 400), allow_upscale=False)
        assert limited.pixel_size == (400, 400)
        assert warnings and "未放大" in warnings[0]

    def test_untouched_when_scaling_down(self):
        spec = OutputSpec(
            size_mode=SizeMode.PIXELS, px_width=500, dpi=300, allow_upscale=False
        )
        target = spec.resolve(src_aspect=1.0)
        limited, warnings = fit_within(target, (4000, 4000), allow_upscale=False)
        assert limited.pixel_size == target.pixel_size
        assert not warnings

    def test_untouched_when_upscale_allowed(self):
        spec = OutputSpec(size_mode=SizeMode.PIXELS, px_width=5000, dpi=300)
        target = spec.resolve(src_aspect=1.0)
        limited, warnings = fit_within(target, (400, 400), allow_upscale=True)
        assert limited.pixel_size == (5000, 5000)
        assert not warnings


class TestFormatMetadata:
    def test_extensions(self):
        assert OutputFormat.TIFF.extension == ".tif"
        assert OutputFormat.JPEG.extension == ".jpg"
        assert OutputFormat.PNG.extension == ".png"
        assert OutputFormat.PDF.extension == ".pdf"

    def test_lossless_flags(self):
        assert OutputFormat.TIFF.is_lossless
        assert OutputFormat.PNG.is_lossless
        assert not OutputFormat.JPEG.is_lossless

    def test_default_compression(self):
        assert OutputSpec().tiff_compression is TiffCompression.LZW


class TestResolveGeometryFunction:
    def test_matches_method(self):
        spec = OutputSpec(
            size_mode=SizeMode.PHYSICAL, phys_unit=Unit.CM, phys_width=8.5, dpi=300
        )
        assert resolve_geometry(spec, 1.25).pixel_size == spec.resolve(1.25).pixel_size
