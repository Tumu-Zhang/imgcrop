"""主窗口：左边清单、中间预览、右边参数，底部进度。

界面只做三件事：把参数讲清楚、把当前这张图的裁剪窗画准、把批量任务丢给
后台线程。所有裁剪几何都来自 imgcrop —— 界面自己不算坐标，这样"预览里看到的"
和"导出的"必然一致。

线程规矩：模型推理与批量导出都在 worker 的独立线程里跑，主线程只负责绘制与
状态刷新；界面回调一律不碰磁盘、不建会话，否则拖一下滑杆就能感到卡顿。

取景的三种模式（见 VIEW_MODES）
--------------------------------
- **智能取景**：自动/手动主体框 + 留白与比例算出裁剪窗 —— 批量一致靠它。
- **自由裁剪**：用户拖出的框就是成品范围，所见即所得，不套留白。
- **点选**：EfficientSAM 交互式分割；点选常常只命中主体的一部分，
  所以这个模式下也允许直接拖框补全（"处处可拖"）。

同一个动作在三个模式下含义不同（拖动裁剪框＝改主体范围还是改成品范围），
所以落库一律等"松手"（canvas.boxCommitted），拖动中只做提示。

布局规矩（按"功能区聚合"来分，而不是按位置摊开）：
- **清单功能区**（左栏）：清单一屏到底 + 最常用的两个动作（添加图片、识别主体）；
  其余清单操作（添加文件夹、可疑优先、包含排除、移除、清空）收进「⋮」与右键菜单。
- **预览功能区**（中栏）：模式分段控件 + 三个动作 + 画布 + 随模式变化的提示。
- **参数功能区**（右栏）：取景 / 尺寸与格式 / 导出 三张卡片。

外观规矩（Google 风格）：
- 控件一律用 qfluentwidgets 的版本（圆角、悬停、涟漪、跟随深浅色都由它负责）；
- 配色与字体只从 theme 里取，本文件不写死颜色；卡片用 gui.widgets.Card；
- 文字层次：卡片标题加粗、字段标签常规、说明文字小号倾斜。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    ListWidget,
    MessageBox,
    PrimaryPushButton,
    PrimaryToolButton,
    ProgressBar,
    PushButton,
    SegmentedWidget,
    SmoothScrollArea,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
    TransparentToolButton,
)

from imgcrop import ingest
from imgcrop.crop import (
    ASPECT_PRESETS,
    CropMode,
    CropSpec,
    normalize_angle,
    planned_box,
    planned_rotation,
)
from imgcrop.geometry import Box, rotate_box, rotate_image
from imgcrop.model import ConflictPolicy, OutputFormat, OutputSpec, SizeMode, Unit
from imgcrop.pipeline import CropPlan
from imgcrop.segment import (
    EfficientSamSegmenter,
    IsnetSegmenter,
    Proposal,
    missing_models,
)

from gui import theme
from gui.canvas import CropCanvas
from gui.widgets import AngleSlider, Card, NumberField, PercentSlider, text_width
from gui.worker import ExportWorker, PickWorker, ScanWorker

IMAGE_FILTER = (
    "图片 (" + " ".join(f"*{suffix}" for suffix in sorted(ingest.IMAGE_SUFFIXES)) + ");;所有文件 (*)"
)

FORMATS = (
    ("PNG　无损", OutputFormat.PNG),
    ("JPEG　小体积", OutputFormat.JPEG),
    ("TIFF　无损投稿", OutputFormat.TIFF),
)

CONFLICTS = (
    ("重命名（自动加序号）", ConflictPolicy.RENAME),
    ("跳过已存在的", ConflictPolicy.SKIP),
    ("直接覆盖", ConflictPolicy.OVERWRITE),
)

# 命名方式：data 是模板；None = 交给用户自己写模板（这时候输入框才可编辑）
NAMING_MODES = (
    ("原文件名 + _crop", "{stem}_crop"),
    ("原文件名", "{stem}"),
    ("自定义模板", None),
)

ROTATION_BACKGROUND = (255, 255, 255)
"""旋转/取景越界时补的背景色 —— 与 CropSpec 的默认值保持一致。"""

# 尺寸单位：像素走 px 模式；其余走物理尺寸，配合 DPI 换算成像素。
UNITS = (
    ("像素　px", (SizeMode.PIXELS, None)),
    ("毫米　mm", (SizeMode.PHYSICAL, Unit.MM)),
    ("厘米　cm", (SizeMode.PHYSICAL, Unit.CM)),
    ("英寸　in", (SizeMode.PHYSICAL, Unit.INCH)),
)

# 常用出图尺寸预设（宽, 高），单位跟随上面的选择
SIZE_PRESETS = (
    ("1600 × 1200", 1600, 1200),
    ("2048 × 1536", 2048, 1536),
    ("2560 × 1920", 2560, 1920),
    ("1080 × 1080", 1080, 1080),
    ("1920 × 1080", 1920, 1080),
)

# 焦点在这些控件里时，单字母/数字快捷键必须让位给打字
TEXT_INPUTS = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)

# 取景模式（互斥三选一）。拖动裁剪框在不同模式下含义不同：
#   智能取景 —— 框是"主体范围"，裁剪窗由留白与比例算出来；
#   自由裁剪 —— 框就是最终裁剪窗，所见即所得；
#   点选     —— 点一下分割出主体，拖动仍然可以自己框（点选常只选到一部分）。
MODE_SMART = "smart"
MODE_FREE = "free"
MODE_PICK = "pick"

VIEW_MODES: tuple[tuple[str, str, str], ...] = (
    (MODE_SMART, "智能取景", "按识别到的主体 + 留白/比例自动取景；拖框可修正主体范围"),
    (MODE_FREE, "自由裁剪", "自己拖框，拖到哪就裁到哪（不再套用留白与比例）"),
    (MODE_PICK, "点选", "在图上点一下要保留的物体；右键点多余处做负点，也可直接拖框补全"),
)

DEFAULT_MARGIN_PERCENT = 4.0
"""界面默认留白（百分数，每边）—— 主体约占画面的 92%，投稿时不白占版面。"""

MODE_HINTS = {
    MODE_SMART: "拖框＝修正主体范围，留白与比例继续生效；想精确框选就切到「自由裁剪」。",
    MODE_FREE: "拖框＝直接定下裁剪窗，拖到哪就裁到哪（不套留白）；没有框时在图上拖一下即可。",
    MODE_PICK: (
        "点一下要保留的物体＝点选分割（框内框外都算，拖动才是改框）；"
        "多选了就在多余处右键点一下做负点，Esc 清除全部提示点。"
    ),
}
"""画布下方那句随模式变化的说明。"""

BACKGROUND_PICK_SHARE = 0.85
"""点选命中范围超过画面的这个比例时，视为"点到背景"而不是"选中了物体"。

SAM 类模型点在背景上会把整幅画面当成一块前景。真要点整幅画面的人极少，
而误点背景很常见 —— 所以这里当成"没点中"，不覆盖用户原本可用的框。
"""


# 状态行颜色：语义 -> theme.Palette 上的字段名
STATUS_COLORS = {
    "hint": "text_secondary",
    "warning": "warning",
    "danger": "danger",
    "ok": "success",
    "info": "info",
}


def _icon(name: str):
    """按名字取 FluentIcon；名字不存在就返回 None（宁可少个图标，也别让窗口起不来）。"""
    return getattr(FIF, name, None)


def _fill_combo(combo, entries) -> None:
    """往 Fluent 的 ComboBox 里塞一串 (文字, 数据)。

    坑 1：qfluentwidgets 的 `addItem` 第二个位置参数是**图标**，不是 userData ——
    照 Qt 的写法 `addItem(text, data)` 会把数据丢给 setIcon，`currentData()`
    永远返回 None（单位、比例、格式、候选都会读空）。所以这里两段式地填。

    坑 2：下拉框默认按"最短的那一项"给宽度，选中长选项（"2048 × 1536"、
    "重命名（自动加序号）"）时文字会被裁掉。这里按**最长的一项**定最小宽度。
    """
    for label, data in entries:
        combo.addItem(label)
        combo.setItemData(combo.count() - 1, data)
    longest = max((label for label, _data in entries), key=len, default="")
    combo.setMinimumWidth(max(110, text_width(longest) + 56))


def _button(icon_name: str, text: str, *, primary: bool = False):
    """带图标的 Fluent 按钮。"""
    button = PrimaryPushButton() if primary else PushButton()
    icon = _icon(icon_name)
    if icon is not None:
        button.setIcon(icon)
    button.setText(text)
    return button


def _tool_button(icon_name: str, tooltip: str, *, primary: bool = False):
    """纯图标的小方按钮（工具栏用），说明文字放 tooltip。"""
    button = PrimaryToolButton() if primary else TransparentToolButton()
    icon = _icon(icon_name)
    if icon is not None:
        button.setIcon(icon)
    button.setToolTip(tooltip)
    button.setFixedSize(32, 32)
    return button


@dataclass(eq=False)
class Photo:
    """清单里的一张图及其当前状态。"""

    path: Path
    image: Image.Image | None = None
    proposal: Proposal | None = None
    subject: Box | None = None
    mask: np.ndarray | None = None
    origin: str = "待识别"
    status: str = ""
    error: str = ""
    included: bool = True
    manual: bool = False
    """用户直接干预过（拖框 / 点选 / 锁定），重新扫描时不再覆盖。"""

    free: Box | None = None
    """自由裁剪窗。设了就"所见即所得"，不再套用留白与比例（见 _display_box）。"""

    pick_box: Box | None = None
    """最近一次点选命中的物体范围（**原图坐标**）。只用来告诉用户"点到了什么"。"""

    pick_score: float | None = None
    """最近一次点选的模型自评 IoU（0..1），随 pick_box 一起清空。"""

    rotate_deg: float = 0.0
    """手动旋转角（度，正 = 逆时针）。每张图各自一份，"套用到全部"会把它带过去。"""

    subject_src: Box | None = None
    """主体框的**原图坐标**（识别/点选/拖框的原始结果，与旋转角无关）。

    这是唯一的真相：`subject` 是由它按当前角度算出来的显示值。换角度时从
    这里重新算 —— 拿转过的框反推会一次比一次大（轴对齐包围盒不可逆）。
    """

    free_src: Box | None = None
    """自由裁剪框的原图坐标。与 subject_src 同理。"""

    rotated_image: Image.Image | None = None
    """按当前角度旋转后的图（显示与取景都用它，缓存免得每帧重转）。"""

    rotated_angle: float = 0.0
    """rotated_image 是按多少角度转出来的 —— 角度变了就作废。"""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def candidates(self) -> list[Box]:
        return list(self.proposal.boxes) if self.proposal else []

    @property
    def suspicious(self) -> bool:
        """是否需要人工复核。"""
        subject = self.subject_src or self.subject
        free = self.free_src or self.free
        if self.error or subject is None or self.image is None:
            return True
        if free is not None:
            return False
        if len(self.candidates) > 2:
            return True
        width, height = self.image.size
        share = (subject.width * subject.height) / max(1, width * height)
        return share < 0.03 or share > 0.7

    @property
    def short_status(self) -> str:
        """列表中显示的精简状态。"""
        if self.error:
            return "读取失败"
        if self.free is not None:
            return "自由裁剪"
        if self.subject is None:
            return "待识别"
        if self.manual:
            return "已手动调整"
        return "自动识别"


class MainWindow(QMainWindow):
    def __init__(self, paths: list[Path] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("图片裁剪器")
        self.setWindowIcon(theme.app_icon())
        self.resize(1320, 860)

        self._photos: list[Photo] = []
        self._current: int = -1

        self._canvas_photo: Photo | None = None
        """画布当前显示的是哪张图 —— 换了它才清提示点（见 _update_canvas）。"""

        self._mode: str = MODE_SMART
        """当前取景模式（智能取景 / 自由裁剪 / 点选），见 VIEW_MODES。"""

        # 尺寸区：宽/高联动时用来防止信号互相触发，以及记住上一个单位做换算
        self._syncing_size = False
        self._last_unit: tuple[SizeMode, Unit | None] = (SizeMode.PIXELS, None)

        self._isnet = IsnetSegmenter()
        self._sam = EfficientSamSegmenter()
        self._scan_thread: QThread | None = None
        self._scan_worker: ScanWorker | None = None
        self._pick_thread: QThread | None = None
        self._pick_worker: PickWorker | None = None
        self._export_thread: QThread | None = None
        self._export_worker: ExportWorker | None = None
        self._busy = False

        # 样式表挂在**窗口**上，不要挂 QApplication —— 那会把 qfluentwidgets
        # 挂在 app 上的控件样式整个替换掉（深色下会变成白卡片配白字）。
        self.setStyleSheet(theme.build_stylesheet())
        self._build_ui()
        self._wire_shortcuts()
        self._refresh_size_hint()

        missing = missing_models()
        if missing:
            self._set_status(
                "缺少分割模型：" + "、".join(missing)
                + " —— 运行 python tools/fetch_models.py 下载（首次约 90 MB）",
                kind="warning",
            )
        else:
            self._set_status("就绪。把图片或文件夹拖进来，或点「添加图片」。")

        self.setAcceptDrops(True)
        if paths:
            self.add_paths(paths)

    # ==================================================================
    # 界面搭建
    # ==================================================================
    def _build_ui(self) -> None:
        self._build_menu()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(8)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self._build_list_panel())
        self.splitter.addWidget(self._build_canvas_panel())
        self.splitter.addWidget(self._build_settings_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        # 六像素的调节带：够抓住、又不占地方；不许拖到消失（拖没了很难找回来）
        self.splitter.setHandleWidth(6)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([260, 620, 400])
        outer.addWidget(self.splitter, 1)

        # 底部状态行
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        bottom.addWidget(self.progress, 1)

        self.status = CaptionLabel()
        self.status.setWordWrap(True)
        bottom.addWidget(self.status)
        outer.addLayout(bottom)

    def _add_action(self, menu, text: str, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menu(self) -> None:
        """菜单栏：同一个功能的命令聚在同一个菜单里。"""
        menu = self.menuBar()

        file_menu = menu.addMenu("文件(&F)")
        self._add_action(file_menu, "添加图片…", self._choose_files, QKeySequence.Open)
        self._add_action(file_menu, "添加文件夹…", self._choose_dir, "Ctrl+Shift+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "导出当前选中", self._export_current, "Ctrl+Shift+E")
        self._add_action(file_menu, "导出全部…", self._export_all, "Ctrl+E")
        file_menu.addSeparator()
        self._add_action(file_menu, "退出", self.close, QKeySequence.Quit)

        edit_menu = menu.addMenu("编辑(&E)")
        self._add_action(edit_menu, "从清单移除", self._remove_selected, QKeySequence.Delete)
        self._add_action(edit_menu, "包含 / 排除选中", self._toggle_included, "Space")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "可疑优先", self._sort_suspicious_first)
        self._add_action(edit_menu, "清空清单", self._clear_all)

        view_menu = menu.addMenu("取景(&V)")
        for key, label, tip in VIEW_MODES:
            action = self._add_action(
                view_menu, label, lambda _=False, k=key: self._set_mode(k)
            )
            action.setToolTip(tip)
        view_menu.addSeparator()
        self._add_action(view_menu, "识别主体", self._scan_listed, "F5")
        self._add_action(view_menu, "回到自动结果", self._reset_current, "R")
        self._add_action(view_menu, "应用当前框到全部", self._apply_current_box_to_all, "L")

        help_menu = menu.addMenu("帮助(&H)")
        self._add_action(help_menu, "查看运行日志", self._open_run_log)
        self._add_action(help_menu, "关于 图片裁剪器", self._about)

    def _build_list_panel(self) -> QWidget:
        """左栏＝清单功能区：清单一屏到底，动作只留最常用的两个 + 一个「更多」。

        "输入"和"清单操作"本来就是一件事（往清单里加、或对清单动手），
        分成两张卡片、摆七个按钮，只会让人找不到重点。
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 标题行：标题 + 计数 + 更多
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(SubtitleLabel("图片清单"))
        head.addStretch(1)
        self.label_count = CaptionLabel("0 张")
        self.label_count.setFont(theme.ui_font(10, bold=True))
        head.addWidget(self.label_count)
        self.btn_more = _tool_button("MORE", "更多清单操作")
        self.btn_more.clicked.connect(self._show_more_menu)
        head.addWidget(self.btn_more)
        layout.addLayout(head)

        # 清单
        self.list = ListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_list_context_menu)
        self.list.currentRowChanged.connect(self._on_row_changed)
        self.list.itemChanged.connect(self._on_item_changed)
        # 长文件名不适合用横向滚动条看 —— 关掉它，名字自然截断
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.list, 1)

        # 底部只用两个按钮：进来（添加）与动手（识别）
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.btn_add_files = _button("ADD", "添加图片", primary=True)
        self.btn_add_files.setToolTip("选择图片或文件夹（Ctrl+O）；也可以直接把图片拖进窗口")
        self.btn_add_files.clicked.connect(self._choose_files)
        self.btn_scan = _button("SEARCH", "识别主体")
        self.btn_scan.setToolTip("为清单中所有图片识别主体（F5）")
        self.btn_scan.clicked.connect(self._scan_listed)
        for button in (self.btn_add_files, self.btn_scan):
            button.setMinimumHeight(36)
        actions.addWidget(self.btn_add_files, 1)
        actions.addWidget(self.btn_scan, 1)
        layout.addLayout(actions)
        return panel

    def _show_more_menu(self) -> None:
        """清单的次要动作收在这一个菜单里（右键菜单里也有一份）。"""
        menu = QMenu(self)
        self._add_action(menu, "添加文件夹…", self._choose_dir)
        self._add_action(menu, "添加图片…", self._choose_files)
        menu.addSeparator()
        self._add_action(menu, "可疑优先", self._sort_suspicious_first)
        self._add_action(menu, "包含 / 排除选中", self._toggle_included)
        self._add_action(menu, "从清单移除", self._remove_selected)
        self._add_action(menu, "清空清单", self._clear_all)
        menu.exec(self.btn_more.mapToGlobal(self.btn_more.rect().bottomLeft()))

    def _build_canvas_panel(self) -> QWidget:
        """中栏＝预览功能区：取景工具条 + 画布 + 随模式变化的操作提示。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 画布上方工具条
        layout.addLayout(self._build_crop_toolbar())

        self.canvas = CropCanvas()
        self.canvas.boxChanged.connect(self._on_canvas_box_changed)
        self.canvas.boxCommitted.connect(self._on_canvas_box_committed)
        self.canvas.pickRequested.connect(self._on_pick_clicked)
        self.canvas.resetRequested.connect(self._reset_current)
        self.canvas.cycleRequested.connect(self._on_canvas_cycle)
        layout.addWidget(self.canvas, 1)

        self.hint = CaptionLabel()
        self.hint.setWordWrap(True)
        self.hint.setFont(theme.ui_font(9, italic=True))
        layout.addWidget(self.hint)
        self._update_canvas_hint()
        return panel

    def _build_crop_toolbar(self) -> QHBoxLayout:
        """画布上方的取景工具条：模式 + 候选切换 + 核心动作。

        把"锁定此框"和"套用到全部"合并成一步：用户只要调好当前这张图，
        点一下"应用当前框到全部"即可。候选切换用左右箭头，复核时手不用离开图。
        """
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.mode_bar = SegmentedWidget()
        for key, label, _tip in VIEW_MODES:
            self.mode_bar.addItem(key, label)
        # 先设好默认档再连接信号：SegmentedWidget 在 setCurrentItem 时会发信号，
        # 那时画布还没建出来
        self.mode_bar.setCurrentItem(MODE_SMART)
        self.mode_bar.setFixedHeight(34)
        self.mode_bar.currentItemChanged.connect(self._on_mode_changed)
        bar.addWidget(self.mode_bar, 1)

        self.btn_prev_candidate = _tool_button("LEFT", "上一个候选 (←)")
        self.btn_next_candidate = _tool_button("RIGHT", "下一个候选 (→)")
        self.btn_prev_candidate.clicked.connect(self._prev_candidate)
        self.btn_next_candidate.clicked.connect(self._next_candidate)
        bar.addWidget(self.btn_prev_candidate)
        bar.addWidget(self.btn_next_candidate)

        self.btn_reset = _button("SYNC", "回到自动")
        self.btn_reset.setToolTip("恢复自动识别结果（R；画布双击也行，「点选」模式下请用 R）")
        self.btn_reset.clicked.connect(self._reset_current)
        self.btn_apply_box = _button("COPY", "应用当前框到全部")
        self.btn_apply_box.setToolTip("把当前可见裁剪框按比例套到所有图片")
        self.btn_apply_box.clicked.connect(self._apply_current_box_to_all)
        for button in (self.btn_reset, self.btn_apply_box):
            button.setMinimumHeight(34)
            bar.addWidget(button)
        return bar

    def _update_canvas_hint(self) -> None:
        """画布下方的说明：每个模式只讲这个模式下最该知道的那句话。"""
        if not hasattr(self, "hint"):
            return
        self.hint.setText(
            f"{MODE_HINTS.get(self._mode, '')}"
            "　快捷键：← → 切候选，↑ ↓ 切换图片，R 重置，L 应用当前框到全部，"
            "T 智能取景 / P 点选 / F 自由裁剪，Esc 清除提示点。"
        )

    def _build_settings_panel(self) -> QWidget:
        """右栏＝参数功能区：取景 / 尺寸与格式 / 导出 三张卡片，可整体滚动。

        最小宽度是**量出来的**：标签列（"留白（每边）"）+ 字段列（最宽的控件，
        例如"重命名（自动加序号）"下拉框）+ 卡片内边距与间距。窄于这个数，
        库就会把某个输入框压扁，数字显示不全。
        """
        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(372)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_crop_card())
        layout.addWidget(self._build_size_card())
        layout.addWidget(self._build_export_card())

        self.label_params = CaptionLabel("—")
        self.label_params.setWordWrap(True)
        self.label_params.setFont(theme.ui_font(9))
        layout.addWidget(self.label_params)
        layout.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    @staticmethod
    def _card() -> tuple[Card, QVBoxLayout]:
        """几张卡片共用同一套内边距与节奏，左右栏看起来才是一套东西。"""
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)
        return card, layout

    @staticmethod
    def _form() -> QFormLayout:
        form = QFormLayout()
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(12)
        return form

    def _build_crop_card(self) -> Card:
        card, layout = self._card()
        layout.addWidget(StrongBodyLabel("取景"))

        form = self._form()

        self.combo_aspect = ComboBox()
        _fill_combo(self.combo_aspect, ASPECT_PRESETS)
        self.combo_aspect.currentIndexChanged.connect(self._refresh_view)

        # 留白：滑块拖得快、输入框打得准。值就是百分数（每边各留多少）
        self.slider_margin = PercentSlider(DEFAULT_MARGIN_PERCENT, maximum=60.0)
        self.slider_margin.setToolTip(
            "主体四周各留出的比例。4% 表示主体占画面约 92%；"
            "留白越大，主体在成品里越小。"
        )
        self.slider_margin.valueChanged.connect(self._refresh_view)

        self.check_tilt = CheckBox("自动按主体转正（与手动角度叠加）")
        self.check_tilt.toggled.connect(self._refresh_view)

        self.slider_rotate = AngleSlider(0.0)
        self.slider_rotate.setToolTip(
            "手动旋转画面（-45°~+45°，正=逆时针）。转完仍按水平矩形取景，"
            "所以成品永远是正的；勾了上面的自动转正就是「自动打底 + 手动微调」。"
        )
        self.slider_rotate.valueChanged.connect(self._on_rotation_changed)

        self.combo_candidate = ComboBox()
        self.combo_candidate.currentIndexChanged.connect(self._on_candidate_changed)

        form.addRow(BodyLabel("比例"), self.combo_aspect)
        form.addRow(BodyLabel("留白（每边）"), self.slider_margin)
        form.addRow(BodyLabel("旋转"), self.slider_rotate)
        form.addRow(BodyLabel("候选"), self.combo_candidate)
        form.addRow(BodyLabel(""), self.check_tilt)
        layout.addLayout(form)

        margin_note = CaptionLabel(
            "留白与比例只作用于「智能取景」；旋转对三种模式都生效，转完后框仍是水平的。"
            "「自由裁剪」里框到哪就裁到哪。"
        )
        margin_note.setWordWrap(True)
        margin_note.setFont(theme.ui_font(9, italic=True))
        layout.addWidget(margin_note)
        return card

    def _build_size_card(self) -> Card:
        """尺寸与格式 —— 单位、宽、高、「锁定原比例」（同 PowerPoint 改图片大小的逻辑）。"""
        card, layout = self._card()
        layout.addWidget(StrongBodyLabel("尺寸与格式"))

        form = self._form()

        self.combo_unit = ComboBox()
        _fill_combo(self.combo_unit, UNITS)
        self.combo_unit.currentIndexChanged.connect(self._on_unit_changed)

        self.combo_size_preset = ComboBox()
        _fill_combo(
            self.combo_size_preset,
            [("自定义", None)] + [(label, (w, h)) for label, w, h in SIZE_PRESETS],
        )
        self.combo_size_preset.currentIndexChanged.connect(self._on_size_preset_changed)

        self.spin_width = NumberField(sample="000000")
        self.spin_height = NumberField(sample="000000")
        for field in (self.spin_width, self.spin_height):
            field.setToolTip("留空＝按主体自动；填了就是这一边的输出尺寸")
        self.spin_width.setValue(1600)
        self.spin_height.setValue(1200)
        self.spin_width.valueChanged.connect(self._on_width_changed)
        self.spin_height.valueChanged.connect(self._on_height_changed)

        # 单位单独标在行尾：输入框里只留数字，宽度需求就小得多（见 NumberField）
        self.label_size_unit = CaptionLabel("px")
        self.label_size_unit.setFont(theme.ui_font(10))

        size_row = QWidget()
        size_box = QHBoxLayout(size_row)
        size_box.setContentsMargins(0, 0, 0, 0)
        size_box.setSpacing(8)
        size_box.addWidget(self.spin_width)
        size_box.addWidget(BodyLabel("×"))
        size_box.addWidget(self.spin_height)
        size_box.addWidget(self.label_size_unit)
        size_box.addStretch(1)

        self.switch_lock = SwitchButton()
        self.switch_lock.setOnText("锁定原比例")
        self.switch_lock.setOffText("锁定原比例")
        self.switch_lock.setChecked(True)
        self.switch_lock.checkedChanged.connect(self._on_lock_toggled)

        self.spin_dpi = SpinBox()
        self.spin_dpi.setRange(36, 1200)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSuffix(" dpi")
        self.spin_dpi.valueChanged.connect(self._refresh_size_hint)

        self.combo_format = ComboBox()
        _fill_combo(self.combo_format, FORMATS)

        form.addRow(BodyLabel("单位"), self.combo_unit)
        form.addRow(BodyLabel("宽 × 高"), size_row)
        form.addRow(BodyLabel("预设"), self.combo_size_preset)
        form.addRow(BodyLabel(""), self.switch_lock)
        form.addRow(BodyLabel("分辨率"), self.spin_dpi)
        form.addRow(BodyLabel("格式"), self.combo_format)
        layout.addLayout(form)

        self.label_size = CaptionLabel("—")
        self.label_size.setWordWrap(True)
        self.label_size.setFont(theme.ui_font(9))
        layout.addWidget(self.label_size)
        return card

    def _build_export_card(self) -> Card:
        """导出（含落盘）：目录、命名、同名策略、体积上限与"真的写文件"按钮同一区。"""
        card, layout = self._card()
        layout.addWidget(StrongBodyLabel("导出"))

        form = self._form()

        dir_row = QWidget()
        dir_box = QHBoxLayout(dir_row)
        dir_box.setContentsMargins(0, 0, 0, 0)
        dir_box.setSpacing(6)
        self.edit_out = LineEdit()
        self.edit_out.setPlaceholderText("留空＝同级 cropped 文件夹")
        browse = _tool_button("FOLDER", "选择输出文件夹…")
        browse.clicked.connect(self._choose_output_dir)
        dir_box.addWidget(self.edit_out, 1)
        dir_box.addWidget(browse)

        self.combo_name = ComboBox()
        _fill_combo(self.combo_name, NAMING_MODES)
        self.combo_name.currentIndexChanged.connect(self._on_name_mode_changed)

        self.edit_name = LineEdit()
        self.edit_name.setText("{stem}_crop")
        self.edit_name.setToolTip("模板变量：{stem} = 源文件名（不含扩展名）")
        self.combo_conflict = ComboBox()
        _fill_combo(self.combo_conflict, CONFLICTS)

        self.spin_limit = DoubleSpinBox()
        self.spin_limit.setRange(0.0, 200.0)
        self.spin_limit.setDecimals(1)
        self.spin_limit.setValue(0.0)
        self.spin_limit.setSuffix(" MB")
        self.spin_limit.setSpecialValueText("不限")

        form.addRow(BodyLabel("目录"), dir_row)
        form.addRow(BodyLabel("命名方式"), self.combo_name)
        form.addRow(BodyLabel("命名"), self.edit_name)
        form.addRow(BodyLabel("同名时"), self.combo_conflict)
        form.addRow(BodyLabel("体积上限"), self.spin_limit)
        layout.addLayout(form)

        # 导出按钮区：主按钮突出，再加一个次要的「导出当前」
        export_buttons = QHBoxLayout()
        export_buttons.setSpacing(8)
        self.btn_export = _button("SAVE", "导出全部…", primary=True)
        self.btn_export.clicked.connect(self._export_all)
        self.btn_export.setMinimumHeight(38)
        self.btn_export_current = PushButton("导出当前")
        self.btn_export_current.setToolTip("只导出当前选中的这一张（Ctrl+Shift+E）")
        self.btn_export_current.clicked.connect(self._export_current)
        export_buttons.addWidget(self.btn_export, 2)
        export_buttons.addWidget(self.btn_export_current, 1)
        layout.addLayout(export_buttons)

        note = CaptionLabel("到这一步才真正写文件，之前都没动过原图。")
        note.setWordWrap(True)
        note.setFont(theme.ui_font(9, italic=True))
        layout.addWidget(note)
        return card

    def _wire_shortcuts(self) -> None:
        """键盘：数字切候选，字母在三个取景模式间跳 —— 批量复核时手不用离开图。"""
        for index in range(4):
            shortcut = QShortcut(QKeySequence(str(index + 1)), self)
            shortcut.activated.connect(lambda order=index: self._shortcut_candidate(order))
        QShortcut(QKeySequence("R"), self).activated.connect(self._shortcut(self._reset_current))
        QShortcut(QKeySequence("L"), self).activated.connect(
            self._shortcut(self._apply_current_box_to_all)
        )
        QShortcut(QKeySequence("Left"), self).activated.connect(self._shortcut(self._prev_candidate))
        QShortcut(QKeySequence("Right"), self).activated.connect(self._shortcut(self._next_candidate))
        QShortcut(QKeySequence("Up"), self).activated.connect(self._prev_photo)
        QShortcut(QKeySequence("Down"), self).activated.connect(self._next_photo)
        QShortcut(QKeySequence("T"), self).activated.connect(
            self._shortcut(lambda: self._set_mode(MODE_SMART))
        )
        QShortcut(QKeySequence("P"), self).activated.connect(
            self._shortcut(lambda: self._toggle_mode(MODE_PICK))
        )
        QShortcut(QKeySequence("F"), self).activated.connect(
            self._shortcut(lambda: self._toggle_mode(MODE_FREE))
        )
        QShortcut(QKeySequence("Esc"), self).activated.connect(self._shortcut(self._clear_picks))
        QShortcut(QKeySequence("Ctrl+Shift+E"), self).activated.connect(self._export_current)

    def _toggle_mode(self, mode: str) -> None:
        """同一个键按两次回到智能取景 —— 点选／自由裁剪都是"临时用一下"的。"""
        self._set_mode(MODE_SMART if self._mode == mode else mode)

    @staticmethod
    def _typing() -> bool:
        """焦点在输入框里时，快捷键要让位给打字，否则没法输入数字。"""
        return isinstance(QApplication.focusWidget(), TEXT_INPUTS)

    def _shortcut(self, action):
        """把一个动作包成"不在打字时才执行"。"""

        def run() -> None:
            if not self._typing():
                action()

        return run

    def _shortcut_candidate(self, order: int) -> None:
        if not self._typing():
            self._pick_candidate(order)

    def _show_list_context_menu(self, position) -> None:
        """清单右键菜单：单张常用操作。"""
        item = self.list.itemAt(position)
        if item is None:
            return
        row = self.list.row(item)
        if not (0 <= row < len(self._photos)):
            return

        menu = QMenu(self)
        self._add_action(menu, "导出当前", self._export_current)
        self._add_action(menu, "包含 / 排除", self._toggle_included)
        menu.addSeparator()
        self._add_action(menu, "从清单移除", self._remove_selected)
        menu.exec(self.list.mapToGlobal(position))

    def _open_run_log(self) -> None:
        """打开运行日志 —— 真出问题时，这是最省事的排查入口。"""
        from gui import crashlog

        path = crashlog.log_path()
        if not path.exists():
            self._set_status("还没有运行日志（只在界面模式与 --gui-check 下才会写）。", kind="warning")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _about(self) -> None:
        MessageBox(
            "关于 图片裁剪器",
            "自动找主体、按统一规格批量裁剪。\n\n"
            "· 识别：ISNet 通用显著性分割（可多候选切换）\n"
            "· 点选：EfficientSAM 交互式分割\n"
            "· 取景与输出：本程序自己算，界面预览与导出结果一致\n\n"
            "界面基于 PySide6 与 qfluentwidgets，配色按专业影像工具校准。",
            self,
        ).exec()

    # ==================================================================
    # 尺寸设置（单位 / 宽高联动 / 换算提示）
    # ==================================================================
    @staticmethod
    def _unit_suffix(mode: SizeMode, unit: Unit | None) -> str:
        if mode is SizeMode.PIXELS:
            return " px"
        return {"mm": " mm", "cm": " cm", "inch": " in"}.get(
            unit.value if unit else "cm", " cm"
        )

    def _size_decimals(self) -> int:
        mode, _unit = self.combo_unit.currentData()
        return 0 if mode is SizeMode.PIXELS else 2

    def _round_size(self, value: float) -> float:
        return round(value, self._size_decimals())

    def _crop_ratio(self) -> float | None:
        """锁定比例的基准：优先用当前裁剪框的比例，其次用「比例」预设。"""
        photo = self._photo()
        if photo is not None:
            box = self._display_box(photo)
            if box is not None and box.height > 0:
                return box.width / box.height
        return self._aspect()

    def _on_unit_changed(self, _index: int) -> None:
        """换单位：数值跟着换算（像素 <-> 长度），并更新后缀与小数位。"""
        mode, unit = self.combo_unit.currentData()
        old_mode, old_unit = self._last_unit
        width = self.spin_width.value()
        height = self.spin_height.value()
        dpi = float(self.spin_dpi.value())

        self._syncing_size = True
        suffix = self._unit_suffix(mode, unit)
        decimals = self._size_decimals()
        for field in (self.spin_width, self.spin_height):
            field.setDecimals(decimals)
        self.label_size_unit.setText(suffix.strip())

        if (old_mode, old_unit) != (mode, unit):
            self.spin_width.setValue(
                self._convert_size(width, old_mode, old_unit, mode, unit, dpi)
            )
            self.spin_height.setValue(
                self._convert_size(height, old_mode, old_unit, mode, unit, dpi)
            )
            self._last_unit = (mode, unit)
        self._syncing_size = False

        self._refresh_size_hint()

    @staticmethod
    def _to_inches(value: float, mode: SizeMode, unit: Unit | None, dpi: float) -> float:
        if not value:
            return 0.0
        if mode is SizeMode.PIXELS:
            return value / max(1.0, dpi)
        return value / (unit or Unit.CM).per_inch

    def _convert_size(
        self,
        value: float,
        old_mode: SizeMode,
        old_unit: Unit | None,
        mode: SizeMode,
        unit: Unit | None,
        dpi: float,
    ) -> float:
        if not value:
            return 0.0
        inches = self._to_inches(value, old_mode, old_unit, dpi)
        if mode is SizeMode.PIXELS:
            return round(inches * dpi)
        return round(inches * (unit or Unit.CM).per_inch, 2)

    def _on_size_preset_changed(self, _index: int) -> None:
        data = self.combo_size_preset.currentData()
        if not data:
            return
        width, height = data
        self._syncing_size = True
        self.spin_width.setValue(self._round_size(float(width)))
        self.spin_height.setValue(self._round_size(float(height)))
        self._syncing_size = False
        self._refresh_size_hint()

    def _on_width_changed(self, value: float) -> None:
        """改宽度：锁着比例时高度跟着变（PPT 里拖角点的感觉）。"""
        if self._syncing_size:
            return
        if self.switch_lock.isChecked() and value > 0:
            ratio = self._crop_ratio()
            if ratio:
                self._syncing_size = True
                self.spin_height.setValue(self._round_size(value / ratio))
                self._syncing_size = False
        self._refresh_size_hint()

    def _on_height_changed(self, value: float) -> None:
        if self._syncing_size:
            return
        if self.switch_lock.isChecked() and value > 0:
            ratio = self._crop_ratio()
            if ratio:
                self._syncing_size = True
                self.spin_width.setValue(self._round_size(value * ratio))
                self._syncing_size = False
        self._refresh_size_hint()

    def _on_lock_toggled(self, checked: bool) -> None:
        if checked:
            value = self.spin_width.value()
            if value > 0:
                ratio = self._crop_ratio()
                if ratio:
                    self._syncing_size = True
                    self.spin_height.setValue(self._round_size(value / ratio))
                    self._syncing_size = False
        self._refresh_size_hint()

    def _refresh_size_hint(self) -> None:
        """把"这些设置到底会出多大的图"说清楚 —— 单位/比例两条线都在这里对上。"""
        try:
            spec = self._output_spec()
            spec.validate()
            geometry = spec.resolve(self._crop_ratio())
        except Exception as exc:  # noqa: BLE001 - 提示行不该因为参数没填全就报错
            self.label_size.setText(str(exc))
            return

        width, height = geometry.pixel_size
        dpi = max(1, spec.dpi)
        cm_w = width / dpi * 2.54
        cm_h = height / dpi * 2.54
        lines = [f"输出 {width} × {height} px ｜ {cm_w:.2f} × {cm_h:.2f} cm ｜ {dpi} dpi"]

        if not spec.keep_aspect:
            lines.append("比例已解锁：严格按所填宽高输出（不符时会拉伸）。")
        else:
            ratio = self._crop_ratio()
            if ratio:
                target = width / max(1, height)
                if abs(ratio - target) > 0.01:
                    lines.append("已按原比例等比内接（解锁可精确拉伸）。")
        self.label_size.setText("\n".join(lines))

    # ==================================================================
    # 拖放与添加
    # ==================================================================
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        texts = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile().strip()
        ]
        if texts:
            self.add_paths([Path(text) for text in texts])
            event.acceptProposedAction()

    def add_paths(self, paths: list[Path]) -> None:
        found = ingest.gather([str(path) for path in paths])
        known = {photo.path.resolve() for photo in self._photos if photo.path.exists()}
        fresh: list[int] = []
        for path in found:
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in known:
                continue
            known.add(key)
            self._photos.append(Photo(path=path))
            fresh.append(len(self._photos) - 1)

        if not fresh:
            self._set_status("没有新增可处理的图片。", kind="warning")
            return

        # 如果用户还没指定输出目录，默认用第一张新增图片所在目录的 cropped 子文件夹
        if not self.edit_out.text().strip() and fresh:
            default_dir = self._photos[fresh[0]].path.parent / "cropped"
            self.edit_out.setText(str(default_dir))

        self._reload_list(select=fresh[0])
        self._start_scan(fresh)

    # ==================================================================
    # 清单
    # ==================================================================
    def _reload_list(self, select: int | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for photo in self._photos:
            item = QListWidgetItem(self._label_for(photo))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if photo.included else Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.label_count.setText(f"{len(self._photos)} 张")
        if select is not None and 0 <= select < len(self._photos):
            self.list.setCurrentRow(select)
        elif self.list.count():
            self.list.setCurrentRow(max(0, min(self._current, self.list.count() - 1)))
        else:
            self._current = -1
            self.canvas.set_image(None)
            self.canvas.clear_picks()
            self._canvas_photo = None
            self._update_param_labels()

    @staticmethod
    def _label_for(photo: Photo) -> str:
        marks = []
        if photo.error:
            marks.append("✗")
        elif photo.subject is None and photo.free is None:
            marks.append("…")
        elif photo.suspicious:
            marks.append("?")
        else:
            marks.append("✓")
        if photo.manual:
            marks.append("手")
        if not photo.included:
            marks.append("跳过")
        head = f"{' '.join(marks)}  {photo.name}"
        detail = f"{photo.short_status} · {photo.status or photo.error}".rstrip(" ·")
        return f"{head}\n{detail}".rstrip()

    def _refresh_row(self, index: int) -> None:
        if not (0 <= index < self.list.count()):
            return
        self.list.blockSignals(True)
        self.list.item(index).setText(self._label_for(self._photos[index]))
        self.list.blockSignals(False)

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._photos):
            return
        self._current = row
        self._sync_rotation_slider(self._photo())
        self._update_canvas()
        self._update_candidate_combo()
        self._update_param_labels()
        self._refresh_size_hint()

    def _prev_photo(self) -> None:
        """上一张图。焦点在清单里时让清单自己处理，避免和全局快捷键重复。"""
        if self._typing() or QApplication.focusWidget() is self.list:
            return
        row = self._current - 1
        if row < 0:
            row = len(self._photos) - 1
        if 0 <= row < len(self._photos):
            self.list.setCurrentRow(row)

    def _next_photo(self) -> None:
        """下一张图。焦点在清单里时让清单自己处理，避免和全局快捷键重复。"""
        if self._typing() or QApplication.focusWidget() is self.list:
            return
        row = self._current + 1
        if row >= len(self._photos):
            row = 0
        if 0 <= row < len(self._photos):
            self.list.setCurrentRow(row)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        row = self.list.row(item)
        if 0 <= row < len(self._photos):
            self._photos[row].included = item.checkState() == Qt.Checked
            self._refresh_row(row)

    def _update_canvas(self) -> None:
        photo = self._photo()
        if photo is None:
            self.canvas.set_image(None)
            self._canvas_photo = None
            return
        # 换了图才清提示点：刷新同一张图（改留白、改角度）要留住用户刚点的点
        if photo is not self._canvas_photo:
            self.canvas.clear_picks()
            self._canvas_photo = photo
        # 喂给画布的是"旋转后的图"，框也随之在同一个坐标系里 —— 画布本身
        # 不需要知道有旋转这回事
        self.canvas.set_image(self._rotated_view(photo))
        # 智能取景下主体轮廓说明"识别到了什么"；点选出了结果之后同样要画出来 ——
        # 用户在点选模式下最需要的反馈就是"刚才那一下点中了什么"。
        # 自由裁剪里框本身就是结果，再叠一层"识别到的主体"只会看花眼。
        show_subject = photo.free is None and (
            self._mode == MODE_SMART or photo.pick_box is not None
        )
        self.canvas.set_subject(photo.subject if show_subject else None)
        self.canvas.set_box(self._display_box(photo))

    def _margin(self) -> float:
        """当前留白（0–1 的小数，每边）。"""
        return self.slider_margin.fraction()

    # -- 旋转与坐标换算 -------------------------------------------------
    def _rotation_angle(self, photo: Photo) -> float:
        """这张图当前会转多少度：自动转正（勾选时，来自掩膜）+ 手动角度。

        与导出走的是同一个 planned_rotation —— 预览和成品不会各算一套角度。
        """
        spec = self._spec_for(photo)
        if spec is None:
            # 还没取景：手动角度照样生效（用户可能就是想先把画面转正）
            return normalize_angle(photo.rotate_deg)
        auto, manual, _warnings = planned_rotation(spec, photo.mask)
        return normalize_angle(auto + manual)

    def _rotated_view(self, photo: Photo):
        """按当前角度旋转后的图 —— 画布显示与取景都用它。

        角度不变时直接用缓存：拖动留白滑杆之类的高频刷新不该反复重转整图。
        """
        angle = self._rotation_angle(photo)
        if photo.image is None or abs(angle) < 1e-6:
            return photo.image
        if photo.rotated_image is None or abs(photo.rotated_angle - angle) > 1e-6:
            photo.rotated_image = rotate_image(photo.image, angle, ROTATION_BACKGROUND)
            photo.rotated_angle = angle
        return photo.rotated_image

    def _to_view_box(self, photo: Photo, box: Box | None) -> Box | None:
        """原图坐标 -> 当前旋转角坐标（给画布显示/摆框用）。"""
        if box is None or photo.image is None:
            return box
        angle = self._rotation_angle(photo)
        if abs(angle) < 1e-6:
            return box
        rotated = self._rotated_view(photo)
        if rotated is None:
            return box
        return rotate_box(box, photo.image.size, rotated.size, angle)

    def _to_source_box(self, photo: Photo, box: Box | None) -> Box | None:
        """当前旋转角坐标 -> 原图坐标（用户拖完框之后存这个）。

        逆变换只是"反着转同样的角度"，不是严格的数学逆 —— 轴对齐包围盒
        转过去再转回来会略大。所以只在"用户刚拖完"这一刻用它存 src，
        显示仍然沿用用户拖的那个框（见 _set_subject 的 view 参数）。
        """
        if box is None or photo.image is None:
            return box
        angle = self._rotation_angle(photo)
        if abs(angle) < 1e-6:
            return box
        rotated = self._rotated_view(photo)
        if rotated is None:
            return box
        return rotate_box(box, rotated.size, photo.image.size, -angle)

    def _set_subject(self, photo: Photo, box_src: Box | None, *, view: Box | None = None) -> None:
        """更新主体框：src 是原图坐标（真相），subject 是当前角度的显示值。"""
        photo.subject_src = box_src
        if box_src is None:
            photo.subject = None
        elif view is not None:
            photo.subject = view  # 用户刚拖的框：原样保留，别让他看到框"自己长大"
        else:
            photo.subject = self._to_view_box(photo, box_src)

    def _set_free(self, photo: Photo, box_src: Box | None, *, view: Box | None = None) -> None:
        """更新自由裁剪框（同样：src 是原图坐标，free 是当前角度的显示值）。"""
        photo.free_src = box_src
        if box_src is None:
            photo.free = None
        elif view is not None:
            photo.free = view
        else:
            photo.free = self._to_view_box(photo, box_src)

    def _sync_rotation_slider(self, photo: Photo | None) -> None:
        """滑块显示当前这张图的角度（切图/重置之后要跟上）。"""
        value = photo.rotate_deg if photo is not None else 0.0
        if abs(self.slider_rotate.value() - value) > 1e-6:
            self.slider_rotate.setValue(value)

    def _on_rotation_changed(self, angle: float) -> None:
        """手动角度变了：重算这张图的显示框，画面立刻跟着转。"""
        photo = self._photo()
        if photo is None:
            return
        if abs(photo.rotate_deg - angle) < 1e-6:
            return
        photo.rotate_deg = float(angle)
        photo.rotated_image = None  # 缓存作废，下次取用时按新角度重转
        # 提示点是画在"当时那张画面"上的坐标，画面一转就过期了 —— 清掉，
        # 免得留下几个飘着不动的点骗人（点选的取景结果本身还在）
        if self.canvas.picks:
            self.canvas.clear_picks()
        # 注意从 src 重算，而不是拿上一帧的框反变换 —— 否则拖滑块会让框越滚越大
        self._set_subject(photo, photo.subject_src)
        self._set_free(photo, photo.free_src)
        self._update_canvas()
        self._update_param_labels()
        self._refresh_size_hint()
        if photo.image is not None:
            total = self._rotation_angle(photo)
            self._set_status(
                f"手动旋转 {angle:+.1f}°（当前画面合计 {total:+.1f}°）。"
                "裁剪框仍是水平的，成品会是正的。"
            )

    def _display_box(self, photo: Photo) -> Box | None:
        """当前这张图"最终会裁出的窗"（**当前旋转角坐标系**）。

        自由裁剪窗优先（它就是成品范围）；否则以主体框为准，按留白与比例
        扩张 —— 与 imgcrop 里那条裁剪链路用的是同一个 planned_box。
        """
        if photo.free is not None:
            return photo.free
        if photo.subject is None:
            return None
        return planned_box(
            photo.subject,
            margin=self._margin(),
            aspect=self._aspect(),
        )

    # ==================================================================
    # 识别
    # ==================================================================
    def _start_scan(self, indices: list[int]) -> None:
        if self._scan_thread is not None:
            self._set_status("上一批识别还没跑完，请稍候。", kind="warning")
            return
        targets = [index for index in indices if 0 <= index < len(self._photos)]
        if not targets:
            return
        if missing_models():
            self._set_status(
                "缺少分割模型，无法识别主体。运行 python tools/fetch_models.py 下载。",
                kind="danger",
            )
            return

        self._busy = True
        self.btn_scan.setEnabled(False)
        self._set_status(f"正在识别主体（{len(targets)} 张）…（只识别，不写任何文件）")

        worker = ScanWorker(
            [self._photos[i].path for i in targets], self._isnet, indices=targets
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_scan_progress)
        worker.loaded.connect(self._on_scan_loaded)
        worker.detected.connect(self._on_scan_detected)
        worker.failed.connect(self._on_scan_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(self._on_scan_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_scan_refs)

        self._scan_worker = worker
        self._scan_thread = thread
        thread.start()

    def _clear_scan_refs(self) -> None:
        self._scan_thread = None
        self._scan_worker = None

    def _on_scan_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if name:
            self._set_status(f"正在识别（{done + 1}/{total}）：{name}")

    def _on_scan_loaded(self, index: int, image) -> None:
        photo = self._photos[index]
        photo.image = image
        if index == self._current:
            self._update_canvas()

    def _on_scan_detected(self, index: int, proposal) -> None:
        photo = self._photos[index]
        photo.proposal = proposal
        # 掩膜立刻换回原图空间：下游（倾斜校正）只认原图坐标
        photo.mask = proposal.source_mask() if proposal is not None else None
        best = proposal.best if proposal is not None else None
        if best is None:
            photo.status = "没找到主体"
            self._set_subject(photo, None)
        elif photo.manual:
            photo.status = f"候选 {len(proposal.boxes)} 个（保留手动框）"
        else:
            self._set_subject(photo, best)
            photo.origin = "自动"
            photo.status = f"候选 {len(proposal.boxes)} 个"
        self._refresh_row(index)
        if index == self._current:
            self._update_canvas()
            self._update_candidate_combo()
            self._update_param_labels()
            self._refresh_size_hint()

    def _on_scan_failed(self, index: int, message: str) -> None:
        photo = self._photos[index]
        photo.error = message
        photo.status = ""
        self._set_subject(photo, None)
        self._refresh_row(index)
        if index == self._current:
            self._update_param_labels()

    def _on_scan_finished(self) -> None:
        self._busy = False
        self.btn_scan.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        done = sum(1 for photo in self._photos if self._display_box(photo) is not None)
        bad = sum(1 for photo in self._photos if photo.error)
        suspicious = sum(1 for p in self._photos if p.suspicious)
        self._set_status(
            f"识别完成：{done} 张有可取景范围，{bad} 张读取失败，{suspicious} 张建议复核。"
            "核对好裁剪框后点右栏「导出全部」或「导出当前」。",
            kind="ok" if not bad else "warning",
        )
        self._refresh_size_hint()

    def _scan_listed(self) -> None:
        self._start_scan(list(range(len(self._photos))))

    # ==================================================================
    # 参数与取景
    # ==================================================================
    def _aspect(self) -> float | None:
        data = self.combo_aspect.currentData()
        return None if data is None else float(data)

    def _refresh_view(self, *_args) -> None:
        """留白／比例／倾斜变了：只刷新"看"的部分。

        这里**不遍历清单** —— 留白与比例只影响"以主体为准算出的窗"，
        不动任何一张图的数据。拖动滑块时每动一下都刷一遍整张清单，
        几十张图就会卡顿。
        """
        self._update_canvas()
        self._update_param_labels()
        self._refresh_size_hint()

    def _update_param_labels(self) -> None:
        photo = self._photo()
        if photo is None:
            self.label_params.setText("没有选中图片。")
            return
        if photo.error:
            self.label_params.setText(f"读取失败：{photo.error}")
            return
        spec = self._spec_for(photo)
        if spec is None:
            self.label_params.setText("还没有确定裁剪范围：先「识别主体」，或在图上直接拖一个框。")
            return
        box = self._display_box(photo)
        where = f"裁剪窗 {box.width} × {box.height} @ ({box.x1}, {box.y1})" if box else ""
        parts = [spec.describe(), where, f"来源：{photo.origin or '自动'}"]
        if photo.pick_box is not None:
            # 把"命中的物体"和"套了留白的成品窗"分两段写清楚 —— 否则点选后
            # 看到的框比物体大一圈，会以为点选不准
            hit = self._to_view_box(photo, photo.pick_box) or photo.pick_box
            score = f"，置信度 {photo.pick_score:.2f}" if photo.pick_score is not None else ""
            parts.insert(1, f"点选命中 {hit.width} × {hit.height}{score}")
        self.label_params.setText(" ｜ ".join(part for part in parts if part))

    def _spec_for(self, photo: Photo) -> CropSpec | None:
        """导出规格：框用**当前（转正后）坐标系**里的值，并声明这一点。

        界面里的框是在"已经转正的画面"上摆的 —— 用户拖的、点选命中的、
        套用过来的都是。所以直接把这个框交给 apply_crop，别让它再转一遍：
        轴对齐框在旋转下不可精确往返，转两遍框会莫名变大，预览和导出就对
        不上了。命令行那边给的是原图坐标，走默认的 box_in_rotated_frame=False。
        """
        if photo.free is not None:
            return CropSpec(
                mode=CropMode.FREE,
                subject=photo.free,
                rotate_deg=photo.rotate_deg,
                box_in_rotated_frame=True,
                origin=photo.origin or "自由裁剪",
            )
        if photo.subject is None:
            return None
        return CropSpec(
            mode=CropMode.SUBJECT,
            subject=photo.subject,
            margin=self._margin(),
            aspect=self._aspect(),
            rectify_tilt=self.check_tilt.isChecked(),
            rotate_deg=photo.rotate_deg,
            box_in_rotated_frame=True,
            origin=photo.origin or "自动",
        )

    def _plan_for(self, photo: Photo) -> CropPlan | None:
        spec = self._spec_for(photo)
        if spec is None:
            return None
        return CropPlan(spec=spec, mask=photo.mask if spec.rectify_tilt else None)

    # ==================================================================
    # 候选
    # ==================================================================
    def _update_candidate_combo(self) -> None:
        photo = self._photo()
        self.combo_candidate.blockSignals(True)
        self.combo_candidate.clear()
        if photo is None:
            # 没选图片时也别留一个空框 —— 空框看起来像坏了
            _fill_combo(self.combo_candidate, [("（未选图片）", None)])
            self.combo_candidate.blockSignals(False)
            return
        candidates = photo.candidates
        if not candidates:
            _fill_combo(self.combo_candidate, [("（没有候选）", None)])
        else:
            _fill_combo(
                self.combo_candidate,
                [
                    (f"{order}. {box.width} × {box.height}", order - 1)
                    for order, box in enumerate(candidates, start=1)
                ],
            )
            if photo.subject is not None and not photo.manual:
                try:
                    index = candidates.index(photo.subject)
                except ValueError:
                    index = 0
                self.combo_candidate.setCurrentIndex(min(index, len(candidates) - 1))
        self.combo_candidate.blockSignals(False)

    def _on_candidate_changed(self, position: int) -> None:
        if position < 0:
            return
        data = self.combo_candidate.itemData(position)
        if data is None:
            return
        self._pick_candidate(int(data))

    def _pick_candidate(self, order: int) -> None:
        photo = self._photo()
        if photo is None:
            return
        candidates = photo.candidates
        if not (0 <= order < len(candidates)):
            return
        self._set_subject(photo, candidates[order])
        self._set_free(photo, None)
        photo.manual = False
        photo.origin = "候选"
        photo.status = f"已选候选 {order + 1}"
        photo.pick_box = None
        photo.pick_score = None
        if photo.proposal is not None:
            # 候选来自 ISNet，掩膜也要跟着回到自动那一份，否则倾斜校正会用错掩膜
            photo.mask = photo.proposal.source_mask()
        self._refresh_row(self._current)
        self._update_canvas()
        self._update_param_labels()

    def _prev_candidate(self) -> None:
        """切到上一个候选。"""
        photo = self._photo()
        if photo is None:
            return
        candidates = photo.candidates
        if not candidates:
            return
        current = 0
        if photo.subject is not None and not photo.manual:
            try:
                current = candidates.index(photo.subject)
            except ValueError:
                current = 0
        self._pick_candidate((current - 1) % len(candidates))

    def _next_candidate(self) -> None:
        """切到下一个候选。"""
        photo = self._photo()
        if photo is None:
            return
        candidates = photo.candidates
        if not candidates:
            return
        current = -1
        if photo.subject is not None and not photo.manual:
            try:
                current = candidates.index(photo.subject)
            except ValueError:
                current = -1
        self._pick_candidate((current + 1) % len(candidates))

    def _reset_current(self) -> None:
        photo = self._photo()
        if photo is None or photo.proposal is None:
            return
        # 先归零角度、再重算框：反过来会用旧角度算一遍显示值
        photo.rotate_deg = 0.0
        photo.rotated_image = None
        self._sync_rotation_slider(photo)
        self._set_subject(photo, photo.proposal.best)
        self._set_free(photo, None)
        photo.manual = False
        photo.origin = "自动"
        photo.status = "已回到自动结果"
        photo.pick_box = None
        photo.pick_score = None
        photo.mask = photo.proposal.source_mask()
        self.canvas.clear_picks()
        self._refresh_row(self._current)
        self._update_canvas()
        self._update_candidate_combo()
        self._update_param_labels()

    # ==================================================================
    # 取景模式
    # ==================================================================
    def _set_mode(self, mode: str) -> None:
        """切换取景模式：工具条、画布光标、提示与预览一起跟上。"""
        if not hasattr(self, "canvas") or mode == self._mode:
            return
        self._mode = mode
        self.mode_bar.blockSignals(True)
        self.mode_bar.setCurrentItem(mode)
        self.mode_bar.blockSignals(False)
        self.canvas.set_pick_mode(mode == MODE_PICK)
        if mode != MODE_PICK:
            self.canvas.clear_picks()
        self._update_canvas_hint()
        self._update_canvas()
        self._update_param_labels()
        if mode == MODE_FREE:
            self._set_status("自由裁剪：在图上拖出一个框，拖到哪就裁到哪（不再套留白与比例）。")
        elif mode == MODE_PICK:
            self._set_status(
                "点选：在图上点一下要保留的物体（框内框外都算），"
                "多选了就在多余处右键点一下做负点，Esc 清除提示点。"
            )
        else:
            self._set_status("智能取景：拖动框是修正主体范围，留白与比例继续生效。")

    def _on_mode_changed(self, route_key: str) -> None:
        """分段控件被点了 —— 它自己的状态已经更新，这里只做切换动作。"""
        if route_key and route_key != self._mode:
            self._set_mode(route_key)

    # ==================================================================
    # 手动框 / 点选 / 锁定
    # ==================================================================
    def _on_canvas_box_changed(self, box: Box) -> None:
        """拖动过程中的实时反馈：只报尺寸，不写数据。

        真正落库要等松手（boxCommitted）—— 拖动中改状态会让画布上的框被
        "按留白重算"的结果顶掉，手感就断了。
        """
        self._set_status(
            f"正在框选：{box.width} × {box.height} px"
            + ("（松手后按留白与比例取景）" if self._mode == MODE_SMART else "（松手即生效）")
        )

    def _on_canvas_cycle(self) -> None:
        """画布上"点了一下空白"：智能取景下切到下一个候选，其他模式忽略。"""
        if self._mode != MODE_SMART:
            return
        photo = self._photo()
        if photo is None:
            return
        candidates = photo.candidates
        if not candidates:
            return
        current = 0
        if photo.subject is not None and not photo.manual:
            try:
                current = candidates.index(photo.subject)
            except ValueError:
                current = -1
        self._pick_candidate((current + 1) % len(candidates))

    def _on_canvas_box_committed(self, box: Box) -> None:
        """拖出一个框，松手了 —— 按当前模式决定这个框是什么。

        画布给的是"当前旋转角坐标系"里的框：显示就用它（原样，别让用户看到
        框自己变大），同时换算一份原图坐标存起来 —— 将来改角度时要从这一份
        重新算。
        """
        photo = self._photo()
        if photo is None:
            return

        # 手动拖框之后，"上次点选命中了什么"这层提示已经过期
        photo.pick_box = None
        photo.pick_score = None
        box_src = self._to_source_box(photo, box)

        if self._mode == MODE_SMART:
            # 智能取景：这个框是"主体范围"，裁剪窗仍由留白与比例算出来
            self._set_subject(photo, box_src, view=box)
            self._set_free(photo, None)
            photo.manual = True
            photo.origin = "手动"
            photo.status = f"手动主体框 {box.width} × {box.height}"
            hint = "已修正主体范围；留白与比例仍在生效。想要精确框选请切到「自由裁剪」。"
        else:
            # 自由裁剪 / 点选：框就是成品范围，所见即所得
            self._set_free(photo, box_src, view=box)
            photo.manual = True
            photo.origin = "自由裁剪"
            photo.status = f"自由框选 {box.width} × {box.height}"
            hint = (
                "已设为自由裁剪框（所见即所得）；按 R 可回到自动结果。"
                if self._mode == MODE_FREE
                else "已用拖框定下裁剪范围（所见即所得）；按 R 回到自动结果，或继续点选。"
            )

        self._refresh_row(self._current)
        self._update_canvas()
        self._update_param_labels()
        self._refresh_size_hint()
        self._set_status(hint)

    def _on_pick_clicked(self, x: float, y: float) -> None:
        """画布上点了一下（左键=前景点，右键=背景点）—— 带全部提示点重跑一次分割。"""
        photo = self._photo()
        if photo is None:
            return
        if photo.image is None:
            self._set_status(
                "这张图还没读进来，暂时点不了：先点「识别主体」把它读进来。", kind="warning"
            )
            return
        if self._pick_thread is not None:
            self._set_status("上一次点选还在计算，请稍候。", kind="info")
            return
        if "efficientsam-ti" in missing_models():
            self._set_status("缺少 EfficientSAM 模型，无法点选。", kind="danger")
            return

        # 提示点是累积的：前景点说"要这个"，背景点（右键）说"不要这块"，
        # 模型据此把选多了的区域推开 —— 比清空重来更好用。
        points = list(self.canvas.picks)
        labels = list(self.canvas.pick_labels)
        negatives = sum(1 for label in labels if label < 0.5)

        self._busy = True
        self._set_status(
            f"正在按你的点击分割（{len(points) - negatives} 个前景点、{negatives} 个背景点）…"
        )
        worker = PickWorker(self._current, photo.image, points, labels, self._sam)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_pick_finished)
        worker.failed.connect(self._on_pick_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_pick_refs)
        self._pick_thread = thread
        self._pick_worker = worker
        thread.start()

    def _clear_pick_refs(self) -> None:
        self._pick_thread = None
        self._pick_worker = None

    def _clear_picks(self) -> None:
        """Esc：清掉画布上的提示点（点选模式专用）。"""
        if self._mode != MODE_PICK or not self.canvas.picks:
            return
        self.canvas.clear_picks()
        self._set_status("已清除全部提示点。")

    @staticmethod
    def _hit_is_background(box: Box, photo: Photo) -> bool:
        """这次点选是不是"把整幅画面都选上了"（= 点在了背景上）。"""
        if photo.image is None:
            return False
        width, height = photo.image.size
        share = (box.width * box.height) / max(1, width * height)
        return share >= BACKGROUND_PICK_SHARE

    def _on_pick_finished(self, index: int, hit) -> None:
        self._busy = False
        photo = self._photos[index]
        if hit is None:
            self._set_status("这一点上没找到物体，换个位置再点。", kind="warning")
            return
        if self._hit_is_background(hit.box, photo):
            # 点在背景上时 SAM 会把整张图当成一块前景。与其接受这个"结果"，
            # 不如当成没点中 —— 用户原本可用的框不该被一张整图顶掉。
            self._set_status(
                "这一下点在背景上了（选中的范围几乎覆盖整幅画面），"
                "请点你要保留的物体；要去掉多选的部分，就在多余处右键点一下做负点。",
                kind="warning",
            )
            return
        photo.pick_box = hit.box
        photo.pick_score = float(hit.score)
        # 掩膜换回原图空间再存：倾斜校正要用它估朝向，模型空间那份会算歪
        photo.mask = hit.source_mask()
        photo.manual = True
        photo.origin = "点选"
        photo.status = f"点选（置信度 {hit.score:.2f}）"
        # hit.box 是原图坐标：src 存它，显示值按当前旋转角算
        self._set_subject(photo, hit.box)
        self._set_free(photo, None)
        self._refresh_row(index)
        if index == self._current:
            self._update_canvas()
            self._update_param_labels()
        self._set_status(
            f"已按点选结果取景：命中 {hit.box.width} × {hit.box.height} px"
            f"（置信度 {hit.score:.2f}）；成品窗由留白与比例算出。"
            "选得不完整就直接在图上拖个框补上。"
        )

    def _on_pick_failed(self, index: int, message: str) -> None:
        self._busy = False
        self._set_status(f"点选失败：{message}", kind="danger")

    def _apply_current_box_to_all(self) -> None:
        """把当前这张图可见的裁剪框（连同旋转角）按比例套用到全部图片。

        旧的"先锁定、再套用"两步合并成一步：用户调好当前图，直接应用即可。
        角度一起套过去 —— 整批的朝向才会一致，这正是"手动固定角度"那条路。
        """
        photo = self._photo()
        if photo is None or photo.image is None:
            self._set_status("先选中一张已取景的图片。", kind="warning")
            return
        box = self._display_box(photo)
        if box is None:
            self._set_status("当前图片还没有可应用的裁剪框。", kind="warning")
            return

        # 比例要拿**原图坐标**去算：显示框已经在旋转后坐标系里，而旋转画布
        # 比原图大，用它除原图尺寸会把框算小。
        box_src = self._to_source_box(photo, box)
        if box_src is None:
            self._set_status("当前图片还没有可应用的裁剪框。", kind="warning")
            return
        angle = photo.rotate_deg
        width, height = photo.image.size
        rect = (box_src.x1 / width, box_src.y1 / height, box_src.x2 / width, box_src.y2 / height)

        applied = 0
        for index, target in enumerate(self._photos):
            if target.image is None:
                continue
            tw, th = target.image.size
            target_box = Box(
                round(rect[0] * tw), round(rect[1] * th),
                round(rect[2] * tw), round(rect[3] * th),
            )
            if target_box.is_empty:
                continue
            # 写进"自由裁剪窗"：套用到不同尺寸时直接采用，不再被留白/比例加工
            target.rotate_deg = angle
            target.rotated_image = None
            self._set_free(target, target_box)
            target.manual = True
            target.origin = "套用"
            target.status = "已套用当前框"
            applied += 1
            self._refresh_row(index)
        self._update_canvas()
        self._update_param_labels()
        self._set_status(
            f"已把当前框（含旋转 {angle:+.1f}°）套用到 {applied} 张图。"
        )

    # ==================================================================
    # 排序与增删
    # ==================================================================
    def _sort_suspicious_first(self) -> None:
        current = self._photo()
        self._photos.sort(key=lambda photo: (not photo.suspicious, photo.name))
        self._reload_list(select=self._photos.index(current) if current in self._photos else 0)
        suspicious = sum(1 for p in self._photos if p.suspicious)
        self._set_status(f"已把 {suspicious} 张需要复核的图排到前面。")

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", IMAGE_FILTER)
        if paths:
            self.add_paths([Path(path) for path in paths])

    def _choose_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if directory:
            self.add_paths([Path(directory)])

    def _choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if directory:
            self.edit_out.setText(directory)

    def _remove_selected(self) -> None:
        rows = sorted((self.list.row(item) for item in self.list.selectedItems()), reverse=True)
        if not rows:
            self._set_status("先选中要移除的图片。", kind="info")
            return
        count = len(rows)
        for row in rows:
            del self._photos[row]
        self._reload_list(select=0)
        self._set_status(f"已从清单移除 {count} 张图。")

    def _clear_all(self) -> None:
        self._photos.clear()
        self._current = -1
        self._reload_list()
        self._set_status("清单已清空。")

    def _toggle_included(self) -> None:
        rows = [self.list.row(item) for item in self.list.selectedItems()]
        if not rows:
            self._set_status("先选中要包含或排除的图片。", kind="info")
            return
        flip = not all(self._photos[row].included for row in rows)
        action = "包含" if flip else "排除"
        for row in rows:
            self._photos[row].included = flip
            self._refresh_row(row)
        self.list.blockSignals(True)
        for row in rows:
            self.list.item(row).setCheckState(Qt.Checked if flip else Qt.Unchecked)
        self.list.blockSignals(False)
        self._set_status(f"已{action} {len(rows)} 张图（{'' if flip else '不'}参与导出）。")

    # ==================================================================
    # 导出
    # ==================================================================
    def _on_name_mode_changed(self, index: int) -> None:
        """命名方式：挑预设就把模板填进输入框并锁住，选「自定义模板」才让改。"""
        template = self.combo_name.itemData(index)
        custom = template is None
        self.edit_name.setEnabled(custom)
        if not custom:
            self.edit_name.setText(str(template))

    def _output_spec(self) -> OutputSpec:
        mode, unit = self.combo_unit.currentData()
        limit = self.spin_limit.value()
        width = self.spin_width.value() or None
        height = self.spin_height.value() or None

        if mode is SizeMode.PIXELS:
            px_width = int(round(width)) if width else None
            px_height = int(round(height)) if height else None
            phys_width = phys_height = None
        else:
            px_width = px_height = None
            phys_width, phys_height = width, height

        return OutputSpec(
            size_mode=mode,
            phys_unit=unit or Unit.CM,
            phys_width=phys_width,
            phys_height=phys_height,
            px_width=px_width,
            px_height=px_height,
            dpi=self.spin_dpi.value(),
            keep_aspect=self.switch_lock.isChecked(),
            fmt=OutputFormat(self.combo_format.currentData()),
            max_bytes=int(limit * 1024 * 1024) if limit > 0 else None,
            output_dir=Path(self.edit_out.text().strip()) if self.edit_out.text().strip() else None,
            name_template=self.edit_name.text().strip() or "{stem}_crop",
            on_conflict=ConflictPolicy(self.combo_conflict.currentData()),
        )

    def _export_all(self) -> None:
        # 判据是"有没有可取景的范围"而不是"有没有识别出主体"：只拖了自由框的图
        # 一样要导出（识别失败时全靠它兜底）
        selected = [
            photo
            for photo in self._photos
            if photo.included and self._display_box(photo) is not None
        ]
        self._run_export(selected, "导出全部")

    def _export_current(self) -> None:
        photo = self._photo()
        if photo is None:
            self._set_status("先选中一张要导出的图片。", kind="info")
            return
        if self._display_box(photo) is None:
            self._set_status(
                "当前图片还没有可导出的裁剪范围：先「识别主体」，或者在图上拖一个框。",
                kind="warning",
            )
            return
        if not photo.included:
            self._set_status("当前图片已被排除，导出前先把它包含回来。", kind="warning")
            return
        self._run_export([photo], "导出当前")

    def _run_export(self, photos: list[Photo], task_name: str) -> None:
        if self._busy or self._export_thread is not None:
            self._set_status("还有任务在跑，请稍候。", kind="warning")
            return
        if not photos:
            self._set_status(
                "没有可导出的图片：这些图要么还没取景（识别主体或手动拖框），要么被排除了。",
                kind="warning",
            )
            return

        spec = self._output_spec()
        try:
            spec.validate()
        except Exception as exc:  # noqa: BLE001
            InfoBar.warning(
                title="输出设置有问题",
                content=str(exc),
                duration=5000,
                position=InfoBarPosition.TOP,
                parent=self,
            )
            self._set_status(f"输出设置有问题：{exc}", kind="danger")
            return

        plans = {photo.path: self._plan_for(photo) for photo in photos}
        plans = {path: plan for path, plan in plans.items() if plan is not None}

        self._busy = True
        self.btn_export.setEnabled(False)
        self.btn_export_current.setEnabled(False)
        self._set_status(f"正在导出 {len(plans)} 张…")

        worker = ExportWorker(list(plans), plans, spec)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_export_refs)
        self._export_thread = thread
        self._export_worker = worker
        thread.start()

    def _clear_export_refs(self) -> None:
        self._export_thread = None
        self._export_worker = None

    def _on_export_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done + 1)
        self._set_status(f"正在导出（{done + 1}/{total}）：{name}")

    def _on_export_finished(self, report) -> None:
        self._busy = False
        self.btn_export.setEnabled(True)
        self.btn_export_current.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        sizes = report.pixel_sizes()
        lines = [f"成功 {report.ok_count} 张，失败 {report.fail_count} 张。"]
        if len(sizes) > 1:
            picked = "、".join(f"{w}×{h}" for w, h in sorted(sizes))
            lines.append(
                f"输出出现了 {len(sizes)} 种尺寸（{picked}）。"
                "要整批完全一致：解锁「锁定原比例」并填好宽和高，"
                "或在「比例」里选一个固定比例（例如 4:3）。"
            )
        failed = [result for result in report.results if not result.ok]
        if failed:
            lines.append("失败的文件：")
            lines.extend(f"  · {result.source.name}：{result.error}" for result in failed[:10])
        warned = [result for result in report.results if result.warnings]
        if warned:
            lines.append(f"（{len(warned)} 张带提示，例如补背景、非等比放大）")

        self._set_status(" ".join(lines), kind="danger" if failed else "ok")

        first = next((result.out_path for result in report.results if result.out_path), None)
        content = "\n".join(lines)
        if first is not None:
            content += f"\n\n输出目录：{first.parent}"
        box = MessageBox("导出完成" if not failed else "导出有失败项", content, self)
        box.exec()

    def _on_export_failed(self, message: str) -> None:
        self._busy = False
        self.btn_export.setEnabled(True)
        self.btn_export_current.setEnabled(True)
        self._set_status(f"导出失败：{message}", kind="danger")
        InfoBar.error(
            title="导出失败",
            content=message,
            duration=8000,
            position=InfoBarPosition.TOP,
            parent=self,
        )

    # ==================================================================
    # 小工具
    # ==================================================================
    def _photo(self) -> Photo | None:
        if 0 <= self._current < len(self._photos):
            return self._photos[self._current]
        return None

    def _set_status(self, text: str, *, kind: str = "hint") -> None:
        """底部状态行：文案常驻，颜色按语义给（浅/深两套色都交给控件自己挑）。"""
        self.status.setText(text)
        field = STATUS_COLORS.get(kind, "text_secondary")
        light = QColor(getattr(theme.LIGHT, field))
        dark = QColor(getattr(theme.DARK, field))
        try:
            self.status.setTextColor(light, dark)
        except AttributeError:  # pragma: no cover - 控件 API 变动时的退路
            self.status.setStyleSheet(f"color: {light.name()};")


__all__ = ["MainWindow", "Photo"]
