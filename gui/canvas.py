"""预览画布：显示图片与裁剪框，并支持直接拖拽调整。

拖拽规则（三种模式共用同一套动作）
----------------------------------
按下左键：命中手柄就缩放、框内就移动、其余位置拖出一个新框。**没有框时也能拖**
（识别失败、或点选只选到主体一部分时，用户总有办法自己框），框一律夹在图像
范围内 —— 越界的框导出时要补背景，纯属意外。

"点一下"与"拖一下"由位移区分（见 DRAG_THRESHOLD_PX），点选模式下这一点尤其
重要：主体永远落在裁剪框里（框 = 主体 + 留白），所以**点框内也必须算点选**，
否则点选永远点不中主体。于是按下时先不定性，松手没动就是一次点选，拖动了才按
上面那条规则解释成缩放／移动／重画。两种意图不打架。

右键在点选模式下是"这里不是主体"—— 加一个背景点（负点），用来把多选进来的
区域推开。

拖动过程中只发 `boxChanged`（界面据此报尺寸），松手才发 `boxCommitted`；怎么
解释这个框（主体范围还是成品范围）由主窗口按当前取景模式决定。"点一下"不发
`boxCommitted` —— 点选不是改框，不该顺手把框钉成自由裁剪框。

坐标只在一个地方换算
--------------------
画布内部一律用"原图坐标"保存框（QRectF，浮点），绘制时才映射到控件像素。
反过来（在控件里存框）会在窗口缩放、切图、缩放图片时反复累积舍入误差，
最后表现为"框自己慢慢飘"。对外只给出 imgcrop.geometry.Box（整数、原图坐标），
与裁剪链路用的是同一套坐标。点选提示点同样存原图坐标。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from imgcrop.geometry import Box
from imgcrop.segment import MAX_POINTS

from gui import theme

HANDLE_PX = 9.0
"""手柄的显示边长（控件像素）。"""

MIN_DISPLAY_PX = 24.0
"""框在屏幕上至少要这么大，否则很难再抓住。"""

HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")

DRAG_THRESHOLD_PX = 4.0
"""点选模式下，按下后移动超过这么多像素才算"拖框"，否则算一次点选。"""


def _to_pixmap(image) -> QPixmap:
    """PIL 图 -> QPixmap。优先用 Pillow 自带的 ImageQt（无额外编码开销）。"""
    try:
        from PIL.ImageQt import ImageQt

        return QPixmap.fromImage(ImageQt(image))
    except Exception:  # noqa: BLE001 - 退回最稳的路径
        import io

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        return pixmap


class CropCanvas(QWidget):
    """显示当前图片与裁剪框。"""

    boxChanged = Signal(object)          # Box，原图坐标（拖动过程中持续发）
    boxCommitted = Signal(object)        # Box，原图坐标（拖动结束时发一次）
    pickRequested = Signal(float, float)  # 点选模式下的点击（原图坐标）
    resetRequested = Signal()             # 双击画布，希望回到自动结果
    cycleRequested = Signal()             # 在智能取景下点一下空白处，切换下一个候选

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._image_size: tuple[int, int] = (1, 1)
        self._box: QRectF | None = None
        self._subject: QRectF | None = None
        self._picks: list[tuple[float, float]] = []
        self._pick_labels: list[float] = []
        self._pick_mode = False

        self._drag: str | None = None
        self._drag_handle: str | None = None
        self._drag_start: QPointF | None = None
        self._drag_box: QRectF | None = None
        self._hover_handle: str | None = None
        self._hover_cursor = None

    # -- 数据 -----------------------------------------------------------
    def set_image(self, image) -> None:
        """换一张图；传 None 表示清空。

        提示点**不在这里清**：刷新预览（改留白、改角度、重算取景）也会走
        这条路，顺手清掉用户刚点的点会让人以为"点了没反应"。要清的地方
        （换图、切模式、重置）显式调用 `clear_picks()`。
        """
        if image is None:
            self._pixmap = None
            self._image_size = (1, 1)
        else:
            self._pixmap = _to_pixmap(image)
            self._image_size = (image.size[0], image.size[1])
        self._box = None
        self._subject = None
        self._hover_handle = None
        self.update()

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def set_box(self, box: Box | None) -> None:
        self._box = None if box is None else QRectF(box.x1, box.y1, box.width, box.height)
        self.update()

    def set_subject(self, box: Box | None) -> None:
        self._subject = None if box is None else QRectF(box.x1, box.y1, box.width, box.height)
        self.update()

    def box(self) -> Box | None:
        if self._box is None:
            return None
        return self._as_box(self._box)

    def set_pick_mode(self, enabled: bool) -> None:
        self._pick_mode = enabled
        self._hover_handle = None
        self._hover_cursor = Qt.CrossCursor if enabled else Qt.ArrowCursor
        self.setCursor(self._hover_cursor)
        self.update()


    @property
    def pick_mode(self) -> bool:
        return self._pick_mode

    @property
    def picks(self) -> list[tuple[float, float]]:
        """已经点过的提示点（原图坐标）—— 主窗口据此发起点选分割。"""
        return list(self._picks)

    @property
    def pick_labels(self) -> list[float]:
        """与 `picks` 一一对应的 label：1.0 = 前景点（要保留的），0.0 = 背景点（负点）。"""
        return list(self._pick_labels)

    def clear_picks(self) -> None:
        self._picks = []
        self._pick_labels = []
        self.update()

    def add_pick(self, x: float, y: float, label: float = 1.0) -> None:
        """加一个提示点（label=1 前景、0 背景）。

        超过模型上限时丢掉最早的那个：EfficientSAM 只有 6 个提示点槽位，
        多出来的点模型根本看不到，留在画布上只会骗人。
        """
        self._picks.append((float(x), float(y)))
        self._pick_labels.append(float(label))
        if len(self._picks) > MAX_POINTS:
            del self._picks[0]
            del self._pick_labels[0]
        self.update()

    # -- 坐标换算 -------------------------------------------------------
    def _fit(self) -> tuple[float, float, float]:
        """返回 (缩放, 左上角偏移x, 左上角偏移y)。"""
        width, height = max(1, self.width()), max(1, self.height())
        image_w, image_h = self._image_size
        margin = 12.0
        scale = min((width - margin * 2) / image_w, (height - margin * 2) / image_h)
        scale = max(scale, 1e-6)
        return scale, (width - image_w * scale) / 2, (height - image_h * scale) / 2

    def _to_widget(self, rect: QRectF) -> QRectF:
        scale, ox, oy = self._fit()
        return QRectF(
            ox + rect.x() * scale, oy + rect.y() * scale,
            rect.width() * scale, rect.height() * scale,
        )

    def _point_to_widget(self, x: float, y: float) -> QPointF:
        scale, ox, oy = self._fit()
        return QPointF(ox + x * scale, oy + y * scale)

    def _to_source(self, x: float, y: float) -> QPointF:
        scale, ox, oy = self._fit()
        return QPointF((x - ox) / scale, (y - oy) / scale)

    @staticmethod
    def _as_box(rect: QRectF) -> Box:
        left = int(round(rect.left()))
        top = int(round(rect.top()))
        width = max(1, int(round(rect.width())))
        height = max(1, int(round(rect.height())))
        return Box(left, top, left + width, top + height)

    # -- 边界约束 -------------------------------------------------------
    def _clamp_point(self, point: QPointF) -> QPointF:
        """把原图坐标的点夹进图像范围 —— 框只会落在图上。"""
        width, height = self._image_size
        return QPointF(
            min(max(point.x(), 0.0), float(width)),
            min(max(point.y(), 0.0), float(height)),
        )

    def _clamp_rect(self, rect: QRectF) -> QRectF:
        """整体平移进图像，尺寸够不下的就压到图中。"""
        width, height = self._image_size
        box_w = min(rect.width(), float(width))
        box_h = min(rect.height(), float(height))
        x = min(max(rect.x(), 0.0), width - box_w)
        y = min(max(rect.y(), 0.0), height - box_h)
        return QRectF(x, y, box_w, box_h)

    # -- 绘制 -----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        palette = theme.palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 画布是一块圆角深色面板：整块铺满会显得很"平"，也让三栏缺少层次
        backdrop = QPainterPath()
        backdrop.addRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
            theme.CANVAS_RADIUS,
            theme.CANVAS_RADIUS,
        )
        painter.fillPath(backdrop, palette.canvas_bg)

        if self._pixmap is None:
            self._draw_empty_state(painter, palette)
            return

        scale, ox, oy = self._fit()
        target = QRectF(ox, oy, self._image_size[0] * scale, self._image_size[1] * scale)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        # 主体轮廓：虚线框 + 轻微填充，让用户看到"识别到了什么"
        if self._subject is not None:
            subject_rect = self._to_widget(self._subject)
            painter.setBrush(palette.subject_fill)
            painter.setPen(QPen(palette.subject_line, 1.2, Qt.DashLine))
            painter.drawRect(subject_rect)

        # 裁剪框：压暗外部 + 亮边 + 三分线 + 手柄
        if self._box is not None:
            rect = self._to_widget(self._box)
            self._draw_crop_frame(painter, palette, rect)
        elif not self._picks:
            self._draw_framing_hint(painter, palette, target)

        # 点选标记：前景点是实心圆，背景点（负点）是另一个颜色 + 一道斜杠
        for (x, y), label in zip(self._picks, self._pick_labels):
            point = self._point_to_widget(x, y)
            positive = label >= 0.5
            fill = palette.pick_mark if positive else palette.pick_negative
            ring = palette.pick_mark_ring if positive else palette.pick_negative_ring

            painter.setBrush(QColor(0, 0, 0, 0))
            painter.setPen(QPen(ring, 2.0))
            painter.drawEllipse(point, 7.0, 7.0)
            painter.setBrush(fill)
            painter.setPen(QPen(ring, 1.2))
            painter.drawEllipse(point, 5.0, 5.0)
            if not positive:
                painter.setPen(QPen(ring, 1.6))
                painter.drawLine(
                    QPointF(point.x() - 3.0, point.y() + 3.0),
                    QPointF(point.x() + 3.0, point.y() - 3.0),
                )

    def _draw_empty_state(self, painter: QPainter, palette) -> None:
        """没有图片时的空状态：明确告诉用户可以拖放或点击添加。"""
        empty = QRectF(self.rect()).adjusted(28, 28, -28, -28)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(palette.canvas_frame, 1.8, Qt.DashLine))
        painter.drawRoundedRect(empty, 12.0, 12.0)

        painter.setPen(QColor(palette.text_tertiary))
        font = theme.ui_font(11, weight=theme.QFont.Medium if hasattr(theme.QFont, "Medium") else 500)
        painter.setFont(font)
        painter.drawText(
            empty,
            Qt.AlignCenter,
            "把图片或文件夹拖到这里\n\n"
            "或点左侧「添加图片」",
        )

    def _draw_framing_hint(self, painter: QPainter, palette, target: QRectF) -> None:
        """还没有框时浮一句提示 —— 让"可以自己拖"这件事不用猜。"""
        text = (
            "点一下要保留的物体，也可直接拖框"
            if self._pick_mode
            else "在图上拖动，直接框出要保留的范围"
        )

        painter.setFont(theme.ui_font(11))
        metrics = painter.fontMetrics()
        chrome = QRectF(
            0, 0,
            metrics.horizontalAdvance(text) + 28,
            metrics.height() + 14,
        )
        chrome.moveCenter(target.center())

        painter.setBrush(QColor(0, 0, 0, 145))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(chrome, 9.0, 9.0)

        color = QColor(palette.frame_line)
        color.setAlpha(205)
        painter.setPen(color)
        painter.drawText(chrome, Qt.AlignCenter, text)

    def _draw_crop_frame(self, painter: QPainter, palette, rect: QRectF) -> None:
        """绘制裁剪框及其辅助元素：压暗、边框、三分线、手柄。"""
        # 框外压暗：一眼看出"成品会是什么"
        outer = QPainterPath()
        outer.addRect(QRectF(self.rect()))
        inner = QPainterPath()
        inner.addRect(rect)
        painter.fillPath(outer.subtracted(inner), palette.frame_shade)

        # 主边框：白色亮线 + 强调色内描边，增强对比
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(palette.frame_line, 2.0))
        painter.drawRect(rect)
        painter.setPen(QPen(palette.handle_edge, 1.0))
        painter.drawRect(rect.adjusted(1, 1, -1, -1))

        # 三分线：构图时最实用的一层参考
        painter.setPen(QPen(palette.grid_line, 1.0))
        for step in (1, 2):
            x = rect.left() + rect.width() * step / 3
            y = rect.top() + rect.height() * step / 3
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        # 手柄：白底 + 强调色描边的圆角方块，hover 时放大
        for name, point in self._handle_points(rect).items():
            is_hover = name == self._hover_handle
            size = HANDLE_PX * (1.25 if is_hover else 1.0)
            painter.setBrush(palette.handle_fill)
            pen = QPen(palette.handle_edge, 1.8 if is_hover else 1.4)
            painter.setPen(pen)
            painter.drawRoundedRect(
                QRectF(
                    point.x() - size / 2, point.y() - size / 2,
                    size, size,
                ),
                3.0,
                3.0,
            )

    @staticmethod
    def _handle_points(rect: QRectF) -> dict[str, QPointF]:
        cx, cy = rect.center().x(), rect.center().y()
        return {
            "nw": QPointF(rect.left(), rect.top()),
            "n": QPointF(cx, rect.top()),
            "ne": QPointF(rect.right(), rect.top()),
            "e": QPointF(rect.right(), cy),
            "se": QPointF(rect.right(), rect.bottom()),
            "s": QPointF(cx, rect.bottom()),
            "sw": QPointF(rect.left(), rect.bottom()),
            "w": QPointF(rect.left(), cy),
        }

    # -- 鼠标 -----------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        """按下左键：命中手柄就缩放、框内就移动、其余位置拖出一个新框。

        这条规则在所有模式下都一样（"处处可拖"）—— 没有框时也能拖，识别失败
        或点选只选到一部分时，用户总有办法自己框。

        点选模式多一层：按下时**先不定性**，松手若几乎没动就是一次点选 ——
        落在框内也算，因为主体本来就在框里（框 = 主体 + 留白），否则点选永远
        点不中主体。拖动了才按上面的规则解释。

        右键在点选模式下加一个背景点（负点）。
        """
        if self._pixmap is None:
            return

        if event.button() == Qt.RightButton:
            if self._pick_mode:
                source = self._to_source(event.position().x(), event.position().y())
                if self._inside_image(source):
                    self.add_pick(source.x(), source.y(), label=0.0)
                    self.pickRequested.emit(source.x(), source.y())
            return

        if event.button() != Qt.LeftButton:
            return
        position = event.position()
        source = self._to_source(position.x(), position.y())
        if not self._inside_image(source):
            return

        self._drag_start = position
        self._drag_box = QRectF(self._box) if self._box is not None else None

        if self._box is None:
            if self._pick_mode:
                # 有图没框：先不定性，松手没动就是"点一下分割"
                self._drag, self._drag_handle = "pick_or_new", None
                return
            self._drag, self._drag_handle = "new", None
            start = self._clamp_point(source)
            self._box = QRectF(start, start)
            return

        rect = self._to_widget(self._box)
        handle = self._hit_handle(rect, position)
        if handle:
            # 手柄上的点击留给缩放：那里本来就是个明确的目标
            self._drag, self._drag_handle = "handle", handle
        elif rect.contains(position):
            # 点选模式下框内按下：松手没动 = 点选，拖动了 = 移动框
            self._drag = "pick_or_move" if self._pick_mode else "move"
            self._drag_handle = None
        elif self._pick_mode:
            # 点选模式下框外按下：先不定性 —— 松手没动就是"点一下分割"，
            # 动了就当成拖框（见 mouseMoveEvent）。
            self._drag, self._drag_handle = "pick_or_new", None
        else:
            self._drag, self._drag_handle = "new", None
            start = self._clamp_point(source)
            self._box = QRectF(start, start)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        position = event.position()

        if self._drag is None:
            self._update_hover(position)
            return

        if self._drag in ("pick_or_new", "pick_or_move"):
            if self._drag_start is None or (
                position - self._drag_start
            ).manhattanLength() < DRAG_THRESHOLD_PX:
                return  # 还没动够：意图未定，什么都不做
            if self._drag == "pick_or_move":
                self._drag = "move"
            else:
                self._drag = "new"
                start = self._clamp_point(
                    self._to_source(self._drag_start.x(), self._drag_start.y())
                )
                self._box = QRectF(start, start)

        source = self._clamp_point(self._to_source(position.x(), position.y()))
        if self._drag == "move" and self._drag_box is not None and self._drag_start is not None:
            anchor = self._clamp_point(
                self._to_source(self._drag_start.x(), self._drag_start.y())
            )
            delta = source - anchor
            self._box = self._clamp_rect(
                QRectF(
                    self._drag_box.x() + delta.x(), self._drag_box.y() + delta.y(),
                    self._drag_box.width(), self._drag_box.height(),
                )
            )
        elif self._drag == "new" and self._drag_start is not None:
            start = self._clamp_point(
                self._to_source(self._drag_start.x(), self._drag_start.y())
            )
            self._box = QRectF(start, source).normalized()
        elif self._drag == "handle" and self._drag_box is not None:
            self._box = self._resize(self._drag_box, self._drag_handle, source)

        self.update()
        current = self.box()
        if current is not None:
            self.boxChanged.emit(current)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        drag = self._drag
        start = self._drag_start
        previous_box = self._drag_box
        self._drag = None
        self._drag_handle = None
        self._drag_start = None
        self._drag_box = None

        if drag is None:
            return

        if drag in ("pick_or_new", "pick_or_move"):
            # 几乎没动 → 这是一次点选，不是框选。框保持原样、不发 boxCommitted：
            # 点选不是"改框"，顺手把框钉成自由裁剪框只会掩盖点选本身。
            source = self._to_source(event.position().x(), event.position().y())
            if self._inside_image(source):
                self.add_pick(source.x(), source.y())
                self.pickRequested.emit(source.x(), source.y())
            return

        # 点选模式下点在框沿／手柄上没拖动：什么都不该改（拖动才是缩放）。
        # 否则"点一下框沿"会把取景钉成自由裁剪框，看着像框自己缩了一圈。
        if (
            drag == "handle"
            and self._pick_mode
            and start is not None
            and (event.position() - start).manhattanLength() < DRAG_THRESHOLD_PX
        ):
            return

        tiny = self._box is not None and self._box.width() < MIN_DISPLAY_PX / max(
            self._fit()[0], 1e-6
        )
        if tiny:
            # 框小到看不见了就丢掉，避免留下一个抓不住的碎片。
            # 如果是"点一下"而不是"拖一下"，智能取景下可以视为"切换候选"的意图。
            if drag == "new":
                self.cycleRequested.emit()
                # 保留原来的框（如果有），这次点击只是"切候选"，不是重新框选
                self._box = QRectF(previous_box) if previous_box is not None else None
            else:
                self._box = None
        self.update()
        current = self.box()
        if current is not None:
            self.boxChanged.emit(current)
            self.boxCommitted.emit(current)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """双击画布：只要点在图上就视为"回到自动结果"的请求。

        点选模式下直接忽略：Qt 的双击序列是 press / release / dblclick，第一次
        release 已经触发过一次点选了，再重置只会让"点选结果"和"自动结果"打架。
        点选模式下回到自动结果请用 R 或工具栏的「回到自动」。
        """
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        if self._pick_mode:
            return
        source = self._to_source(event.position().x(), event.position().y())
        if self._inside_image(source):
            self.resetRequested.emit()

    def _resize(self, base: QRectF, handle: str | None, source: QPointF) -> QRectF:
        left, top, right, bottom = base.left(), base.top(), base.right(), base.bottom()
        x, y = source.x(), source.y()
        if handle and "w" in handle:
            left = min(x, right - 1)
        if handle and "e" in handle:
            right = max(x, left + 1)
        if handle and "n" in handle:
            top = min(y, bottom - 1)
        if handle and "s" in handle:
            bottom = max(y, top + 1)
        return QRectF(QPointF(left, top), QPointF(right, bottom))

    def _hit_handle(self, rect: QRectF, position: QPointF) -> str | None:
        for name, point in self._handle_points(rect).items():
            hover_size = HANDLE_PX * 1.25
            if (point - position).manhattanLength() <= hover_size:
                return name
        return None

    def _update_hover(self, position: QPointF) -> None:
        rect = self._to_widget(self._box) if self._box is not None else None
        handle = self._hit_handle(rect, position) if rect is not None else None
        inside = rect is not None and rect.contains(position)
        on_image = self._inside_image(self._to_source(position.x(), position.y()))

        if handle is not None:
            # 手柄优先：那里点了就是缩放，光标如实说明
            cursor = self._cursor_for(handle, inside)
        elif self._pick_mode:
            # 点选模式下框内框外都是"点一下分割" —— 光标不该在框沿上突变，
            # 否则用户会以为框内点不动
            cursor = Qt.CrossCursor if on_image else Qt.ArrowCursor
        elif inside:
            cursor = Qt.SizeAllCursor
        elif on_image:
            cursor = Qt.CrossCursor  # 框外／还没有框：拖一下就能框出一块
        else:
            cursor = Qt.ArrowCursor

        if handle != self._hover_handle:
            self._hover_handle = handle
            self.update()
        if cursor != self._hover_cursor:
            self._hover_cursor = cursor
            self.setCursor(cursor)


    @staticmethod
    def _cursor_for(handle: str | None, inside: bool):
        if handle in ("nw", "se"):
            return Qt.SizeFDiagCursor
        if handle in ("ne", "sw"):
            return Qt.SizeBDiagCursor
        if handle in ("n", "s"):
            return Qt.SizeVerCursor
        if handle in ("e", "w"):
            return Qt.SizeHorCursor
        return Qt.SizeAllCursor if inside else Qt.ArrowCursor

    def _inside_image(self, point: QPointF) -> bool:
        width, height = self._image_size
        return 0 <= point.x() < width and 0 <= point.y() < height


__all__ = ["CropCanvas"]
