"""输出规格的数据模型。

1.1.0 瘦身定案（D5-D7、D11，见《功能瘦身与补足讨论稿.md》）：
界面与核心库一起删掉"输出工程"链路 —— 尺寸（像素/物理）、DPI 输入、
格式选择、体积上限全部移除。导出语义从「规格拟合」变成**「所见即所得」**：

    成品像素 = 裁剪窗在当前旋转角下的原始像素，1:1 输出，不缩放、不放大。

OutputSpec 于是只剩"落盘三件套 + 背景色"：
    - 输出目录 / 命名模板 / 同名策略；
    - 取景窗越出原图边界时补的背景色。
输出格式跟随源文件（encode.resolve_output_format），DPI 元数据跟随源图
（缺省 300），两者都是编码层的内部决策，不再进入规格。

SpecError 仍在这里定义：裁剪规格（CropSpec）与命名模板的校验共用它。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ConflictPolicy(str, Enum):
    """输出文件重名时的处理方式。"""

    OVERWRITE = "overwrite"
    RENAME = "rename"
    SKIP = "skip"


class SpecError(ValueError):
    """输出规格本身不合法（例如命名模板坏了、同名策略缺失）。"""


@dataclass
class OutputSpec:
    """一次导出的落盘要求。所有输入源的设置共用同一份规格。

    像素尺寸不由这里决定 —— 成品就是裁剪窗本身（1:1）。格式与 DPI 由
    编码层按源文件决定（见 encode.resolve_output_format）。
    """

    # --- 输出落盘 ---
    output_dir: Path | None = None
    name_template: str = "{stem}"  # 可用 {stem} {ext}
    on_conflict: ConflictPolicy = ConflictPolicy.RENAME

    # --- 画面 ---
    background: tuple[int, int, int] = (255, 255, 255)
    """取景窗越出原图边界时补的背景色。"""

    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name_template.strip():
            raise SpecError("命名模板不能为空")
        try:
            self.name_template.format(stem="x", ext="png")
        except (KeyError, IndexError, ValueError) as exc:
            raise SpecError(f"命名模板不合法：{self.name_template} —— {exc}") from exc


def format_bytes(num: int) -> str:
    """人类可读的体积。"""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"  # pragma: no cover


__all__ = [
    "ConflictPolicy",
    "OutputSpec",
    "SpecError",
    "format_bytes",
]
