"""界面用的小控件 —— 从 qfluentwidgets 派生，只做"库不做的那一点"。

放这里的东西有两个共同点：
1. 颜色一律从 `gui.theme` 取，本文件不写死任何色值；
2. 对外暴露的是"业务能直接用的量"（例如留白的百分数），换算只在这里做一次。

Card
    qfluentwidgets 的 CardWidget 底色是**半透明白**（叠加在窗口背景上），
    浅色下发灰、深色下发亮，都不是 Google 卡片那种干净的纯色底。这里派生一层
    覆盖配色，圆角也放大到与窗口、画布同一档。

NumberField（为什么不用 SpinBox）
    qfluentwidgets 的 `SpinBox` 在输入框右侧挂了两个上下按钮，宽度需求约 184px。
    两个并排放进"宽 × 高"，或者塞进滑块旁边，布局就会把**输入区**压到只剩几十
    像素 —— 表现就是"数字显示不全"（1600 只剩 16）。这里的数值输入框不带按钮，
    宽度按"最长的数字"算，宽度需求降到 ~90px；要加减用滑块、预设或键盘上下键。
"""

from __future__ import annotations

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QFontMetrics
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CardWidget, LineEdit, Slider

from gui import theme

SPIN_BUTTON_MARGIN = 26
"""输入框两侧留出的内边距（控件像素）：数字不至于顶着边框。"""


def text_width(text: str, *, point: int = 10) -> int:
    """按界面字体量一段文字的宽度 —— 所有"最小宽度"都从这里来，不写魔数。"""
    return QFontMetrics(theme.ui_font(point)).horizontalAdvance(text)


class Card(CardWidget):
    """Google 风格的卡片：纯色底 + 大圆角 + 一圈浅描边。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setBorderRadius(theme.CARD_RADIUS)

    def _normalBackgroundColor(self) -> QColor:  # noqa: N802 - 库的命名
        return QColor(theme.palette().card_bg)

    def _hoverBackgroundColor(self) -> QColor:  # noqa: N802
        # 卡片不可点击，悬停不该变色（否则整块面板在鼠标经过时闪一下）
        return self._normalBackgroundColor()

    def _pressedBackgroundColor(self) -> QColor:  # noqa: N802
        return self._normalBackgroundColor()


class NumberField(LineEdit):
    """紧凑的数值输入框：没有加减按钮，**留空表示"自动"**。

    - `value()` / `setValue()`：数值。空文本与 0 都读作 0（调用方据此判断"自动"）。
    - `setDecimals()`：物理单位（cm／in）要小数，像素不需要 —— 换单位时调这个。
    - 失焦时会把文本规整一遍（`1600.00` → `1600`），免得越打越长。
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        *,
        minimum: float = 0.0,
        maximum: float = 20000.0,
        decimals: int = 0,
        placeholder: str = "自动",
        sample: str = "000000",
        show_zero: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._show_zero = show_zero
        self._syncing = False

        self.setPlaceholderText(placeholder)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setClearButtonEnabled(False)
        # 宽度按"最长的数字"定：够用即可，绝不因为库的默认 sizeHint（很大）把同行别的控件挤掉
        comfortable = text_width(sample) + SPIN_BUTTON_MARGIN
        self.setMinimumWidth(comfortable)
        self.setMaximumWidth(comfortable + 48)

        self._validator = QDoubleValidator(minimum, maximum, decimals, self)
        self._validator.setNotation(QDoubleValidator.StandardNotation)
        # 固定用 C 区域：否则中文/欧洲区域下小数点会跟着变，解析回来就出错
        self._validator.setLocale(QLocale.c())
        self.setValidator(self._validator)

        self.textChanged.connect(self._on_text_changed)
        self.editingFinished.connect(self._normalize)

    # -- 取值 -----------------------------------------------------------
    def value(self) -> float:
        text = self.text().strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:  # 输入中途（"1e"、"1."）不该炸
            return 0.0

    def setValue(self, value: float) -> None:  # noqa: N802 - 跟 Qt 的命名对齐
        text = self._format(value)
        if text == self.text():
            return
        self._syncing = True
        self.setText(text)
        self._syncing = False
        self.valueChanged.emit(self.value())

    def setDecimals(self, decimals: int) -> None:  # noqa: N802
        if decimals == self._decimals:
            return
        self._decimals = decimals
        self._validator.setDecimals(decimals)
        # 改了校验器的小数位要重新装一次才生效
        self.setValidator(self._validator)
        self._normalize()

    # -- 内部 -----------------------------------------------------------
    def _format(self, value: float) -> str:
        if not value and not self._show_zero:
            return ""
        if self._decimals <= 0:
            return str(int(round(value)))
        return f"{value:.{self._decimals}f}"

    def _normalize(self) -> None:
        if not self.text().strip():
            return
        text = self._format(self.value())
        if text == self.text():
            return
        self._syncing = True
        self.setText(text)
        self._syncing = False

    def _on_text_changed(self, _text: str) -> None:
        if not self._syncing:
            self.valueChanged.emit(self.value())


class PercentSlider(QWidget):
    """「滑块 + 能直接打字的输入框」的百分数控件。

    存在的理由：留白这类参数，光有滑块调不准、光有输入框又看不出量级。
    两者绑定后，拖得动也打得准。

    - `value()` / `setValue()`：百分数本身（0–100）
    - `fraction()`：0–1 的小数（业务侧要的比例）
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        value: float = 4.0,
        *,
        minimum: float = 0.0,
        maximum: float = 100.0,
        decimals: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._syncing = False

        self.slider = Slider(Qt.Horizontal)
        self.slider.setRange(int(minimum), int(maximum))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)

        self.spin = NumberField(
            minimum=minimum,
            maximum=maximum,
            decimals=decimals,
            placeholder="0",
            sample="000.0",
        )
        self.unit = CaptionLabel("%")
        self.unit.setFont(theme.ui_font(10))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        row.addWidget(self.unit)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self._syncing = True
        self.slider.setValue(int(round(value)))
        self.spin.setValue(value)
        self._syncing = False

    # -- 取值 -----------------------------------------------------------
    def value(self) -> float:
        """百分数，例如 4.0 表示 4%。"""
        return float(self.spin.value())

    def fraction(self) -> float:
        """0–1 的小数，例如 0.04。"""
        return self.value() / 100.0

    def setValue(self, value: float) -> None:  # noqa: N802 - 跟 Qt 的命名对齐
        self.spin.setValue(float(value))

    # -- 联动 -----------------------------------------------------------
    def _on_slider(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(float(value))
        self._syncing = False
        self.valueChanged.emit(self.value())

    def _on_spin(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(int(round(value)))
        self._syncing = False
        self.valueChanged.emit(float(value))


class AngleSlider(QWidget):
    """「滑块 + 能直接打字的输入框」的角度控件。

    滑块给整数度（拖起来稳、能一路拖到 0），输入框可以键入 0.1° 的精确角度
    （转正要靠的就是这半度）。范围默认 ±45°：再大的偏斜该用"旋转图片"而不是
    "把照片摆正"。
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        value: float = 0.0,
        *,
        minimum: float = -45.0,
        maximum: float = 45.0,
        decimals: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._syncing = False

        self.slider = Slider(Qt.Horizontal)
        self.slider.setRange(int(minimum), int(maximum))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)

        self.spin = NumberField(
            minimum=minimum,
            maximum=maximum,
            decimals=decimals,
            placeholder="0",
            sample="-00.0",
            show_zero=True,
        )
        self.unit = CaptionLabel("°")
        self.unit.setFont(theme.ui_font(10))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        row.addWidget(self.unit)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self._syncing = True
        self.slider.setValue(int(round(value)))
        self.spin.setValue(value)
        self._syncing = False

    # -- 取值 -----------------------------------------------------------
    def value(self) -> float:
        """角度（度），正 = 逆时针。"""
        return float(self.spin.value())

    def setValue(self, value: float) -> None:  # noqa: N802 - 跟 Qt 的命名对齐
        self.spin.setValue(float(value))

    # -- 联动 -----------------------------------------------------------
    def _on_slider(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(float(value))
        self._syncing = False
        self.valueChanged.emit(self.value())

    def _on_spin(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(int(round(value)))
        self._syncing = False
        self.valueChanged.emit(float(value))


__all__ = ["AngleSlider", "Card", "NumberField", "PercentSlider", "text_width"]
