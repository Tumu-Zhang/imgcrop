"""落盘规格（OutputSpec）的单元测试。

1.1.0 瘦身后规格只剩"落盘三件套 + 背景色"：像素尺寸由裁剪窗决定（1:1），
格式与 DPI 由编码层按源决定 —— 那两部分的规则在 test_encode / test_pipeline 里。
"""

from __future__ import annotations

import pytest

from imgcrop.model import ConflictPolicy, OutputSpec, SpecError, format_bytes


class TestOutputSpecValidation:
    def test_default_spec_is_valid(self):
        OutputSpec().validate()

    def test_blank_template_rejected(self):
        with pytest.raises(SpecError, match="命名模板不能为空"):
            OutputSpec(name_template="   ").validate()

    def test_unknown_placeholder_rejected(self):
        with pytest.raises(SpecError, match="命名模板不合法"):
            OutputSpec(name_template="{stem}_{index}").validate()

    def test_stem_and_ext_placeholders_ok(self):
        OutputSpec(name_template="{stem}_crop").validate()


class TestConflictPolicy:
    def test_values(self):
        assert ConflictPolicy.RENAME.value == "rename"
        assert ConflictPolicy.OVERWRITE.value == "overwrite"
        assert ConflictPolicy.SKIP.value == "skip"

    def test_default_is_rename(self):
        assert OutputSpec().on_conflict is ConflictPolicy.RENAME


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(512) == "512 B"

    def test_kilobytes_and_megabytes(self):
        assert format_bytes(2048) == "2.00 KB"
        assert format_bytes(3 * 1024 * 1024) == "3.00 MB"
