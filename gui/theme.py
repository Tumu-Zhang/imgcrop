"""界面配色、字体、图标与样式表 —— 按 Google（Material）风格校准。

颜色只在这里定义一次，窗口与画布都从这里取 —— 同一个语义在两处用了不同的
灰，是桌面程序最容易散架的地方。

设计目标
--------
- Google 风：清爽的浅灰底 + 纯白卡片 + Google 蓝强调色，圆角与留白都放大一号。
- 内容优先：画布是视觉中心，两侧退后成为工具面板。
- 深浅色一致：两套 palette 的语义一一对应，切换时只换色值、不换结构。

令牌结构
--------
按 "surface / text / control / accent / status / canvas" 分组：
- surface_*: 背景层级（窗口、卡片、浮层）
- text_*: 文字层级（主/次/占位）
- control_*: 控件边框、悬停、激活、禁用
- accent_*: 强调色及其状态
- status_*: 成功/警告/错误
- canvas_*: 画布自绘专用

样式表挂在哪一级（踩过的坑）
----------------------------
**不要**用 `QApplication.setStyleSheet()`：qfluentwidgets 把自己那套控件样式
（含深色下的卡片底色、描边）也挂在 QApplication 上，整体替换会把它冲掉 ——
现象就是"深色模式下卡片还是浅底、文字却已变白，几乎看不见"。所以这里的
样式表只发给**窗口自己**（`MainWindow.setStyleSheet(build_stylesheet())`），
库的样式原样保留。

主题跟随系统
------------
`apply()` 用 qfluentwidgets 的 Theme.AUTO（内部 darkdetect 监听系统主题），
切换时 palette() 会给出另一套色，窗口与画布据此重绘，不需要重启。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QIcon, QPalette

UI_FAMILIES = (
    "Google Sans",
    "Roboto",
    "Microsoft YaHei UI",
    "Segoe UI",
    "Noto Sans CJK SC",
)
"""界面字体：Google 系字体优先，中文回落到微软雅黑 —— 缺字时 Qt 自己往下找。"""


@dataclass(frozen=True)
class Palette:
    """一整套界面配色。字符串都是 QSS/十六进制色，QColor 给自绘用。"""

    name: str
    dark: bool

    # 背景层级
    window_bg: str
    card_bg: str
    layer_bg: str
    elevated_bg: str

    # 文字层级
    text_primary: str
    text_secondary: str
    text_tertiary: str
    text_disabled: str

    # 控件边框与分隔
    border: str
    control_border: str
    control_border_hover: str
    divider: str

    # 强调色
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_light: str
    selection_bg: str

    # 状态色
    danger: str
    warning: str
    success: str
    info: str

    # 画布自绘
    canvas_bg: QColor
    canvas_frame: QColor
    frame_line: QColor
    frame_shade: QColor
    subject_line: QColor
    subject_fill: QColor
    handle_fill: QColor
    handle_edge: QColor
    pick_mark: QColor
    pick_mark_ring: QColor
    pick_negative: QColor
    """背景点（负点）的填充色 —— 与前景点区分开，一眼看出"这个点表示不要"。"""
    pick_negative_ring: QColor
    grid_line: QColor


# 浅色主题：Google 的干净灰底 + 纯白卡片 + Google 蓝
LIGHT = Palette(
    name="light",
    dark=False,
    window_bg="#f8f9fa",
    card_bg="#ffffff",
    layer_bg="#f1f3f4",
    elevated_bg="#ffffff",
    text_primary="#202124",
    text_secondary="#5f6368",
    text_tertiary="#80868b",
    text_disabled="#bdc1c6",
    border="#dadce0",
    control_border="#dadce0",
    control_border_hover="#bdc1c6",
    divider="#e8eaed",
    accent="#1a73e8",
    accent_hover="#1765cc",
    accent_pressed="#185abc",
    accent_light="#e8f0fe",
    selection_bg="#e8f0fe",
    danger="#d93025",
    warning="#e37400",
    success="#188038",
    info="#1a73e8",
    canvas_bg=QColor("#202124"),
    canvas_frame=QColor("#5f6368"),
    frame_line=QColor(255, 255, 255, 245),
    frame_shade=QColor(0, 0, 0, 140),
    subject_line=QColor(251, 188, 4, 235),
    subject_fill=QColor(251, 188, 4, 36),
    handle_fill=QColor(255, 255, 255),
    handle_edge=QColor("#1a73e8"),
    pick_mark=QColor("#d93025"),
    pick_mark_ring=QColor(255, 255, 255, 220),
    pick_negative=QColor("#1a73e8"),
    pick_negative_ring=QColor(255, 255, 255, 220),
    grid_line=QColor(255, 255, 255, 55),
)

# 深色主题：Google 深灰（#202124 系）+ 浅蓝强调
DARK = Palette(
    name="dark",
    dark=True,
    window_bg="#202124",
    card_bg="#303134",
    layer_bg="#28292c",
    elevated_bg="#3c4043",
    text_primary="#e8eaed",
    text_secondary="#9aa0a6",
    text_tertiary="#80868b",
    text_disabled="#5f6368",
    border="#3c4043",
    control_border="#5f6368",
    control_border_hover="#80868b",
    divider="#3c4043",
    accent="#8ab4f8",
    accent_hover="#aecbfa",
    accent_pressed="#d2e3fc",
    accent_light="#28292c",
    selection_bg="#394457",
    danger="#f28b82",
    warning="#fdd663",
    success="#81c995",
    info="#8ab4f8",
    canvas_bg=QColor("#131314"),
    canvas_frame=QColor("#5f6368"),
    frame_line=QColor(255, 255, 255, 245),
    frame_shade=QColor(0, 0, 0, 175),
    subject_line=QColor(253, 214, 99, 235),
    subject_fill=QColor(253, 214, 99, 42),
    handle_fill=QColor(255, 255, 255),
    handle_edge=QColor("#8ab4f8"),
    pick_mark=QColor("#f28b82"),
    pick_mark_ring=QColor(255, 255, 255, 220),
    pick_negative=QColor("#8ab4f8"),
    pick_negative_ring=QColor(255, 255, 255, 220),
    grid_line=QColor(255, 255, 255, 50),
)

CARD_RADIUS = 14
"""卡片圆角（控件像素）—— 窗口、画布、卡片都用同一档，视觉上才是一套。"""

CANVAS_RADIUS = 12
"""画布面板圆角。"""


def resource_path(name: str) -> Path:
    """定位随包资源：打包后从 _MEIPASS 或 exe 同级目录找。"""
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / name)
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / name)
    candidates.append(Path(__file__).resolve().parents[1] / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def is_dark() -> bool:
    """当前是不是深色主题。

    以 qfluentwidgets 的 qconfig 为准 —— 它在 Theme.AUTO 下已经解析成实际的
    浅色/深色（内部用 darkdetect 跟随系统）。取不到时按浅色处理。
    """
    try:
        from qfluentwidgets import Theme, qconfig

        return qconfig.theme == Theme.DARK
    except Exception:  # noqa: BLE001 - 库不在时不影响界面能用
        return False


def palette() -> Palette:
    """当前主题的配色。"""
    return DARK if is_dark() else LIGHT


def ui_font(point_size: int = 10, *, bold: bool = False, italic: bool = False, weight: int | None = None) -> QFont:
    """界面字体：Google 系优先，可按需加粗/倾斜/指定字重（层次感都从这里来）。"""
    font = QFont()
    font.setFamilies(list(UI_FAMILIES))
    font.setPointSize(point_size)
    if weight is not None:
        font.setWeight(weight)
    elif bold:
        font.setWeight(QFont.DemiBold)
    font.setItalic(italic)
    return font


def app_icon() -> QIcon:
    icon = resource_path("icon.ico")
    return QIcon(str(icon)) if icon.is_file() else QIcon()


def qt_palette() -> QPalette:
    """原生控件的调色板 —— 深色主题下必须换掉。

    Fusion 自带的调色板是浅色的：不换的话，容器类控件（central widget、
    滚动区的 viewport）还是白底，qfluentwidgets 的半透明卡片叠上去就变成
    一片白，而文字已经按深色主题变白 —— 结果就是"白卡片配白字，看不见"。
    """
    p = palette()
    text = QColor(p.text_primary)
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(p.window_bg))
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, QColor(p.layer_bg if p.dark else p.card_bg))
    pal.setColor(QPalette.AlternateBase, QColor(p.card_bg))
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, QColor(p.card_bg))
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.Highlight, QColor(p.accent))
    pal.setColor(
        QPalette.HighlightedText, QColor("#ffffff") if not p.dark else QColor("#202124")
    )
    pal.setColor(QPalette.ToolTipBase, QColor(p.card_bg))
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.PlaceholderText, QColor(p.text_tertiary))
    pal.setColor(QPalette.Mid, QColor(p.border))
    pal.setColor(QPalette.Dark, QColor(p.border))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(p.text_disabled))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(p.text_disabled))
    return pal


def build_stylesheet() -> str:
    """只给原生控件补样式 —— 始终发给窗口，不要再往 QApplication 上挂。

    刻意**不写** `QPushButton` / `QComboBox` 这类裸选择器：qfluentwidgets 的控件
    都从这些原生类派生，一条全局规则会把它们的 Fluent 外观整个盖掉。这里只处理
    窗口底色、分隔条、菜单、滚动条、工具提示这些"库不管"的部分。
    """
    p = palette()
    return f"""
    QMainWindow, QDialog {{ background: {p.window_bg}; }}

    QSplitter::handle {{ background: {p.window_bg}; }}
    QSplitter::handle:horizontal {{ width: 8px; }}
    QSplitter::handle:vertical {{ height: 8px; }}
    QSplitter::handle:hover {{ background: {p.border}; }}
    QSplitter::handle:pressed {{ background: {p.accent}; }}

    QMenuBar {{
        background: transparent;
        color: {p.text_primary};
        padding: 2px 6px;
        border: none;
    }}
    QMenuBar::item {{ padding: 6px 12px; border-radius: 6px; background: transparent; }}
    QMenuBar::item:selected {{ background: {p.layer_bg}; }}
    QMenuBar::item:pressed {{ background: {p.layer_bg}; }}

    QMenu {{
        background: {p.card_bg};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: 10px;
        padding: 6px;
    }}
    QMenu::item {{ padding: 7px 30px 7px 14px; border-radius: 8px; }}
    QMenu::item:selected {{ background: {p.selection_bg}; }}
    QMenu::item:disabled {{ color: {p.text_disabled}; }}
    QMenu::separator {{ height: 1px; background: {p.divider}; margin: 5px 8px; }}

    QStatusBar {{ background: {p.window_bg}; color: {p.text_secondary}; }}
    QStatusBar::item {{ border: none; }}

    QToolTip {{
        background: {p.card_bg};
        color: {p.text_primary};
        border: 1px solid {p.border};
        border-radius: 8px;
        padding: 5px 9px;
    }}

    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px 1px 2px 1px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.control_border_hover}; border-radius: 4px; min-height: 32px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {p.text_tertiary}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 1px 2px 1px 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.control_border_hover}; border-radius: 4px; min-width: 32px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {p.text_tertiary}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

    QLabel#hint {{ color: {p.text_secondary}; font-style: italic; }}
    QLabel#warning {{ color: {p.warning}; }}
    QLabel#danger {{ color: {p.danger}; }}
    QLabel#ok {{ color: {p.success}; }}
    QLabel#info {{ color: {p.info}; }}

    /* 清单：Google 的列表是"选中一块浅蓝圆角"，没有整块边框 */
    QListWidget {{
        outline: none;
        border: none;
        border-radius: {CARD_RADIUS}px;
        background: {p.card_bg};
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 8px 10px;
        border-radius: 8px;
        margin: 1px 2px;
    }}
    QListWidget::item:selected {{
        background: {p.selection_bg};
        color: {p.text_primary};
    }}
    QListWidget::item:hover:!selected {{
        background: {p.layer_bg};
    }}
    """


def _repolish(app) -> None:
    """主题变了：把新样式表发给每个窗口，并让它们重画。

    注意这里**不碰** `app.setStyleSheet()` —— qfluentwidgets 的控件样式也挂在
    QApplication 上，整体替换会把 Fluent 外观（尤其深色下的卡片底色）一并冲掉。
    """
    app.setPalette(qt_palette())
    for widget in app.topLevelWidgets():
        widget.setStyleSheet(build_stylesheet())
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)
        widget.update()


def apply(app) -> None:
    """把主题装到 QApplication 上 —— 界面启动的第一步。

    `Theme.AUTO` 让 qfluentwidgets 跟随系统浅/深色（内部监听注册表变化）。
    这里只设样式/字体/图标与库的主题，**样式表留给窗口自己**（见模块开头）。
    """
    try:
        from qfluentwidgets import Theme, setFontFamilies, setTheme, setThemeColor

        setTheme(Theme.AUTO)
        setThemeColor(palette().accent)
        setFontFamilies(list(UI_FAMILIES))
        try:
            from qfluentwidgets import qconfig

            qconfig.themeChangedFinished.connect(lambda: _repolish(app))
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001 - 库不可用时退化成纯 Qt 界面
        pass

    app.setStyle("Fusion")
    app.setFont(ui_font(10))
    # 关键：Fusion 默认调色板是浅色的，深色主题下容器会白底
    app.setPalette(qt_palette())
    app.setWindowIcon(app_icon())


__all__ = [
    "CARD_RADIUS",
    "CANVAS_RADIUS",
    "DARK",
    "LIGHT",
    "Palette",
    "UI_FAMILIES",
    "app_icon",
    "apply",
    "build_stylesheet",
    "is_dark",
    "palette",
    "qt_palette",
    "resource_path",
    "ui_font",
]
