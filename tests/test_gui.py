"""图形界面的冒烟测试：不开窗口，但把关键交互真跑一遍。

用 offscreen 平台跑，所以能在没有显示器的环境（CI、打包机）里执行。
不碰模型：识别那一步用假数据注入，这样测试快且不依赖 90 MB 的 onnx 文件；
真正需要模型的部分由 test_segment.py 负责。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw  # noqa: E402

from imgcrop.crop import CropMode  # noqa: E402
from imgcrop.geometry import Box, Letterbox  # noqa: E402
from imgcrop.model import SizeMode, Unit  # noqa: E402
from imgcrop.render import load_raster  # noqa: E402
from imgcrop.segment import PointHit, Proposal  # noqa: E402

pytest.importorskip("PySide6")

from gui import main_window as mw  # noqa: E402


class FakeMessageBox:
    """替掉真弹框：测试不能被模态对话框卡住。

    qfluentwidgets 的 MessageBox 构造是 (title, content, parent)，用法比原来短。
    """

    def __init__(self, title: str = "", content: str = "", parent=None) -> None:
        self.title = title
        self.text = content

    def exec(self) -> int:
        return 0


class FakeInfoBar:
    """替掉 InfoBar：它会在窗口上飘一条提示，测试里不需要真飘。"""

    @staticmethod
    def warning(*args, **kwargs) -> None:
        return None

    @staticmethod
    def error(*args, **kwargs) -> None:
        return None


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(mw, "MessageBox", FakeMessageBox)
    monkeypatch.setattr(mw, "InfoBar", FakeInfoBar)
    # 不让界面真的去跑模型：识别那步在这里没有意义
    monkeypatch.setattr(mw.MainWindow, "_start_scan", lambda self, indices: None)
    win = mw.MainWindow()
    yield win
    win.close()


def make_photo(path: Path, size=(1000, 800)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (60, 70, 80))
    ImageDraw.Draw(image).rectangle([200, 150, 599, 449], fill=(235, 235, 230))
    image.save(path)
    return path


def make_item(path: Path, subject: Box | None = None, candidates: int = 1) -> mw.Photo:
    """造一个"已经识别过"的清单项，省掉模型推理。"""
    loaded = load_raster(path)
    boxes = [subject or Box(200, 150, 600, 450)]
    for step in range(1, candidates):
        boxes.append(Box(100 * step, 80 * step, 300 * step, 240 * step))
    proposal = Proposal(
        boxes=boxes,
        mask=np.zeros((16, 16), dtype=np.float32),
        letterbox=Letterbox.fit(loaded.image.size[0], loaded.image.size[1], 1024, 1024),
    )
    return mw.Photo(
        path=path,
        image=loaded.image,
        proposal=proposal,
        subject=proposal.best,
        subject_src=proposal.best,
        mask=proposal.mask,
        origin="自动",
        status=f"候选 {len(boxes)} 个",
    )


def wait_for_export(gui, qapp, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while gui._export_thread is not None and time.time() < deadline:  # noqa: SLF001
        qapp.processEvents()
        time.sleep(0.01)
    assert gui._export_thread is None, "导出线程没有结束"


class TestStartup:
    def test_window_starts_empty(self, window):
        assert window.list.count() == 0
        assert window._photo() is None  # noqa: SLF001
        assert window.btn_export.isEnabled()

    def test_adding_photos_fills_the_list(self, window, tmp_path):
        make_photo(tmp_path / "a.png")
        make_photo(tmp_path / "b.png", size=(800, 600))

        window.add_paths([tmp_path])

        assert window.list.count() == 2
        assert {photo.name for photo in window._photos} == {"a.png", "b.png"}  # noqa: SLF001

    def test_duplicates_are_ignored(self, window, tmp_path):
        path = make_photo(tmp_path / "a.png")
        window.add_paths([path])
        window.add_paths([path, tmp_path])
        assert window.list.count() == 1

    def test_unsupported_files_are_left_out(self, window, tmp_path):
        make_photo(tmp_path / "a.png")
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        window.add_paths([tmp_path])
        assert window.list.count() == 1


class TestFraming:
    def test_preview_box_follows_margin(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        small = window._display_box(item)  # noqa: SLF001
        window.slider_margin.setValue(30.0)  # 留白按百分数给：30%
        large = window._display_box(item)  # noqa: SLF001

        assert small is not None and large is not None
        assert large.width > small.width

    def test_margin_accepts_typed_values(self, window, tmp_path):
        """留白既能拖也能直接打字 —— 输入框里的值必须真的生效。"""
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.slider_margin.spin.setValue(37.5)
        assert abs(window.slider_margin.value() - 37.5) < 1e-6
        assert abs(window._margin() - 0.375) < 1e-6  # noqa: SLF001

    def test_aspect_preset_changes_shape(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.combo_aspect.setCurrentIndex(1)  # 1:1
        box = window._display_box(item)  # noqa: SLF001
        assert box is not None
        assert abs(box.width / box.height - 1.0) < 0.02

    def test_dragging_in_smart_mode_keeps_the_margin_rule(self, window, tmp_path):
        """智能取景里拖框＝修正主体范围，留白与比例继续生效。"""
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        dragged = Box(300, 200, 700, 500)
        window._on_canvas_box_committed(dragged)  # noqa: SLF001

        assert item.subject == dragged
        assert item.free is None
        spec = window._spec_for(item)  # noqa: SLF001
        assert spec is not None and spec.mode is CropMode.SUBJECT
        framed = window._display_box(item)  # noqa: SLF001
        assert framed is not None and framed.width > dragged.width

    def test_dragging_in_free_mode_is_the_final_box(self, window, tmp_path):
        """自由裁剪里拖框＝成品范围本身，不再套留白。"""
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        dragged = Box(300, 200, 700, 500)
        window._on_canvas_box_committed(dragged)  # noqa: SLF001

        spec = window._spec_for(item)  # noqa: SLF001
        assert spec is not None
        assert spec.mode is CropMode.FREE
        assert spec.subject == dragged
        assert window._display_box(item) == dragged  # noqa: SLF001
        assert item.short_status == "自由裁剪"

    def test_margin_no_longer_moves_a_free_box(self, window, tmp_path):
        """自由裁剪过的图不该被留白滑块再动一次。"""
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        dragged = Box(300, 200, 700, 500)
        window._on_canvas_box_committed(dragged)  # noqa: SLF001
        window.slider_margin.setValue(40.0)
        assert window._display_box(item) == dragged  # noqa: SLF001

    def test_reset_clears_the_free_box(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        window._on_canvas_box_committed(Box(300, 200, 700, 500))  # noqa: SLF001
        window._reset_current()  # noqa: SLF001

        assert item.free is None
        assert item.subject == item.proposal.boxes[0]
        assert window._spec_for(item).mode is CropMode.SUBJECT  # noqa: SLF001

    def test_candidate_switch_replaces_subject(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"), candidates=3)
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window._pick_candidate(2)  # noqa: SLF001

        assert item.subject == item.proposal.boxes[2]
        assert item.origin == "候选"

    def test_reset_returns_to_automatic(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"), candidates=2)
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window._pick_candidate(1)  # noqa: SLF001
        window._reset_current()  # noqa: SLF001

        assert item.subject == item.proposal.boxes[0]
        assert item.manual is False


class TestModes:
    """三个取景模式互斥，切换时画布的光标与提示跟着走。"""

    def test_mode_switch_drives_canvas_pick_mode(self, window):
        assert window._mode == mw.MODE_SMART  # noqa: SLF001
        assert window.canvas.pick_mode is False

        window._set_mode(mw.MODE_PICK)  # noqa: SLF001
        assert window.canvas.pick_mode is True

        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        assert window.canvas.pick_mode is False
        assert window.mode_bar.currentRouteKey() == mw.MODE_FREE  # noqa: SLF001

    def test_same_key_twice_returns_to_smart(self, window):
        window._toggle_mode(mw.MODE_FREE)  # noqa: SLF001
        assert window._mode == mw.MODE_FREE  # noqa: SLF001
        window._toggle_mode(mw.MODE_FREE)  # noqa: SLF001
        assert window._mode == mw.MODE_SMART  # noqa: SLF001

    def test_hint_follows_the_mode(self, window):
        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        assert "拖到哪就裁到哪" in window.hint.text()


class TestCanvasDrag:
    """画布上的拖框：没有框也能拖、框只能落在图上、点选模式下"点"≠"拖"。"""

    @staticmethod
    def _canvas(size=(1000, 800), widget=(600, 400)):
        from gui.canvas import CropCanvas

        canvas = CropCanvas()
        canvas.resize(*widget)
        canvas.set_image(Image.new("RGB", size, (40, 40, 40)))
        return canvas

    @staticmethod
    def _event(kind, x, y, button=None):
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        if button is None:
            button = Qt.LeftButton
        position = QPointF(x, y)
        return QMouseEvent(
            QEvent.MouseButtonPress if kind == "press" else
            QEvent.MouseMove if kind == "move" else QEvent.MouseButtonRelease,
            position, position, button, button, Qt.NoModifier,
        )

    def _drag(self, canvas, start, end):
        canvas.mousePressEvent(self._event("press", *start))
        canvas.mouseMoveEvent(self._event("move", *end))
        canvas.mouseReleaseEvent(self._event("release", *end))

    def test_can_drag_a_box_when_there_is_none(self):
        """识别失败／还没识别时也拖得动 —— 这是"自由裁剪"能兜底的前提。"""
        canvas = self._canvas()
        assert canvas.box() is None

        committed = []
        canvas.boxCommitted.connect(committed.append)
        self._drag(canvas, (80, 60), (300, 240))

        box = canvas.box()
        assert box is not None and not box.is_empty
        assert len(committed) == 1

    def test_box_is_clamped_into_the_image(self):
        """拖到窗口边上也不会跑出原图 —— 越界的框导出时要补背景，纯属意外。"""
        canvas = self._canvas()
        self._drag(canvas, (80, 60), (9000, 9000))

        box = canvas.box()
        assert box is not None
        assert box.x2 <= 1000 and box.y2 <= 800

    def test_pick_mode_click_does_not_draw_a_box(self):
        """点选模式下"按一下"是一次点选，不是拖出一个 1 像素的框。"""
        canvas = self._canvas()
        canvas.set_pick_mode(True)

        picked = []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))
        canvas.mousePressEvent(self._event("press", 160, 130))
        canvas.mouseReleaseEvent(self._event("release", 160, 130))

        assert len(picked) == 1
        assert canvas.box() is None

    def test_pick_mode_drag_still_draws_a_box(self):
        """点选选不全时，同一个模式下直接拖框补全（不用来回切模式）。"""
        canvas = self._canvas()
        canvas.set_pick_mode(True)

        picked = []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))
        self._drag(canvas, (160, 130), (340, 260))

        assert picked == []
        box = canvas.box()
        assert box is not None and not box.is_empty

    def test_pick_mode_click_inside_the_box_is_still_a_pick(self):
        """框内点击必须是点选 —— 这是"点选无效"的根因。

        主体永远落在裁剪框里（框 = 主体 + 留白），所以一旦"框内按下"被判成
        "移动框"，点主体就永远点不出点选，用户只剩背景可以点。
        """
        canvas = self._canvas()
        canvas.set_pick_mode(True)
        canvas.set_box(Box(350, 250, 650, 550))

        picked, committed = [], []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))
        canvas.boxCommitted.connect(committed.append)

        center = canvas._to_widget(canvas._box).center()  # noqa: SLF001
        canvas.mousePressEvent(self._event("press", center.x(), center.y()))
        canvas.mouseReleaseEvent(self._event("release", center.x(), center.y()))

        assert len(picked) == 1
        assert committed == []  # 点选不是改框：不该顺手把框钉成自由裁剪框
        assert canvas.box() == Box(350, 250, 650, 550)
        assert canvas.pick_labels == [1.0]

    def test_pick_mode_drag_inside_the_box_moves_it(self):
        """点选模式下拖动仍然是"改框"：点一下和拖一下要能分开。"""
        canvas = self._canvas()
        canvas.set_pick_mode(True)
        canvas.set_box(Box(350, 250, 650, 550))

        picked = []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))

        center = canvas._to_widget(canvas._box).center()  # noqa: SLF001
        self._drag(canvas, (center.x(), center.y()), (center.x() - 60, center.y() - 40))

        assert picked == []
        box = canvas.box()
        assert box is not None
        assert box.width == 300 and box.height == 300
        assert (box.x1, box.y1) != (350, 250)

    def test_pick_mode_click_on_a_handle_changes_nothing(self):
        """手柄上的点击留给缩放：点一下不该顺手把取景钉成自由裁剪框。"""
        canvas = self._canvas()
        canvas.set_pick_mode(True)
        canvas.set_box(Box(350, 250, 650, 550))

        picked, committed = [], []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))
        canvas.boxCommitted.connect(committed.append)

        corner = canvas._to_widget(canvas._box).topLeft()  # noqa: SLF001
        canvas.mousePressEvent(self._event("press", corner.x(), corner.y()))
        canvas.mouseReleaseEvent(self._event("release", corner.x(), corner.y()))

        assert picked == []
        assert committed == []
        assert canvas.box() == Box(350, 250, 650, 550)

    def test_pick_mode_right_click_adds_a_negative_point(self):
        """右键＝"这里不是主体"：给模型一个背景点，把多选进来的区域推开。"""
        canvas = self._canvas()
        canvas.set_pick_mode(True)

        picked = []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))
        from PySide6.QtCore import Qt

        canvas.mousePressEvent(self._event("press", 300, 200, button=Qt.RightButton))

        assert len(picked) == 1
        assert canvas.pick_labels == [0.0]

    def test_right_click_is_ignored_outside_pick_mode(self):
        canvas = self._canvas()
        picked = []
        canvas.pickRequested.connect(lambda x, y: picked.append((x, y)))
        from PySide6.QtCore import Qt

        canvas.mousePressEvent(self._event("press", 300, 200, button=Qt.RightButton))
        assert picked == []
        assert canvas.picks == []

    def test_pick_points_stop_at_the_model_slot_count(self):
        """提示点最多 6 个（模型的槽位数）——多的丢掉最早的，别让它骗人。"""
        from imgcrop.segment import MAX_POINTS

        canvas = self._canvas()
        for step in range(MAX_POINTS + 4):
            canvas.add_pick(float(step), float(step))

        assert len(canvas.picks) == MAX_POINTS
        assert canvas.picks[0] == (4.0, 4.0)  # 最早的点被挤掉
        assert canvas.pick_labels == [1.0] * MAX_POINTS

    def test_escape_clamps_a_moved_box(self):
        """把框往图像外拖时，整块被拉住而不是跑掉一半。"""
        canvas = self._canvas()
        canvas.set_box(Box(100, 100, 400, 300))

        rect = canvas._to_widget(canvas._box)  # noqa: SLF001
        center = rect.center()
        canvas.mousePressEvent(self._event("press", center.x(), center.y()))
        canvas.mouseMoveEvent(self._event("move", center.x() + 500, center.y() + 500))
        canvas.mouseReleaseEvent(self._event("release", center.x() + 500, center.y() + 500))

        box = canvas.box()
        assert box is not None
        assert box.x1 >= 0 and box.y1 >= 0
        assert box.x2 <= 1000 and box.y2 <= 800
        assert box.width == 300 and box.height == 200

    def test_double_click_emits_reset(self):
        """双击画布触发 resetRequested，方便快速回到自动结果。"""
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent

        canvas = self._canvas()
        resets = []
        canvas.resetRequested.connect(lambda: resets.append(True))

        position = QPointF(160, 130)
        event = QMouseEvent(
            QEvent.MouseButtonDblClick, position, position,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier,
        )
        canvas.mouseDoubleClickEvent(event)
        assert len(resets) == 1

    def test_tiny_click_outside_box_emits_cycle(self):
        """智能取景下点一下空白处，画布通知外界"切候选"。"""
        canvas = self._canvas()
        canvas.set_box(Box(100, 100, 400, 300))

        cycles = []
        canvas.cycleRequested.connect(lambda: cycles.append(True))
        # 点在框外、图内，几乎没移动
        canvas.mousePressEvent(self._event("press", 500, 300))
        canvas.mouseReleaseEvent(self._event("release", 500, 300))

        assert len(cycles) == 1
        assert canvas.box() is not None  # 原来的框还在


class TestCandidateNavigation:
    def test_next_and_prev_buttons_cycle_candidates(self, window, tmp_path):
        first = make_item(make_photo(tmp_path / "a.png"), candidates=3)
        window._photos = [first]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        initial = first.subject
        window._next_candidate()  # noqa: SLF001
        assert first.subject is not initial
        window._prev_candidate()  # noqa: SLF001
        assert first.subject == initial


class TestPickMode:
    """点选：结果落到哪里、点到背景怎么兜、Esc 清点。"""

    def _open(self, window, tmp_path):
        photo = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [photo]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001
        window._set_mode(mw.MODE_PICK)  # noqa: SLF001
        return photo

    @staticmethod
    def _hit(photo, box, score=0.93):
        return PointHit(
            box=box,
            mask=np.zeros((800, 1000), dtype=bool),
            score=score,
            letterbox=photo.proposal.letterbox,
        )

    def test_result_becomes_the_subject_and_is_reported(self, window, tmp_path):
        photo = self._open(window, tmp_path)
        hit = self._hit(photo, Box(220, 170, 580, 430))

        window._on_pick_finished(0, hit)  # noqa: SLF001

        assert photo.subject == hit.box
        assert photo.pick_box == hit.box
        assert photo.pick_score == pytest.approx(0.93)
        assert photo.origin == "点选"
        assert photo.free is None  # 点选＝主体语义：留白与比例继续生效
        assert photo.manual is True
        # 命中的物体与成品窗要分别写出来，用户才不会以为"点选不准"
        assert "点选命中 360 × 260" in window.label_params.text()

    def test_pick_gives_a_subject_not_a_final_box(self, window, tmp_path):
        """点选给的是主体，成品窗仍由留白与比例算 —— 批内一致靠这条。"""
        photo = self._open(window, tmp_path)
        window._on_pick_finished(0, self._hit(photo, Box(220, 170, 580, 430)))  # noqa: SLF001

        display = window._display_box(photo)  # noqa: SLF001
        assert display is not None
        assert display.width > 360 and display.height > 260  # 已按留白外扩

    def test_pick_that_swallows_the_image_is_rejected(self, window, tmp_path):
        """点在背景上时 SAM 会把整幅画面当成一块前景 —— 不能让它顶掉可用的框。"""
        photo = self._open(window, tmp_path)
        before = photo.subject

        window._on_pick_finished(0, self._hit(photo, Box(0, 0, 1000, 800)))  # noqa: SLF001

        assert photo.subject == before
        assert photo.pick_box is None
        assert "背景" in window.status.text()

    def test_escape_clears_the_pick_points(self, window, tmp_path):
        self._open(window, tmp_path)
        window.canvas.add_pick(120.0, 90.0)
        assert window.canvas.picks

        window._clear_picks()  # noqa: SLF001
        assert window.canvas.picks == []

    def test_escape_does_nothing_outside_pick_mode(self, window, tmp_path):
        self._open(window, tmp_path)
        window._set_mode(mw.MODE_SMART)  # noqa: SLF001
        window.canvas.add_pick(120.0, 90.0)

        window._clear_picks()  # noqa: SLF001
        assert window.canvas.picks == [(120.0, 90.0)]

    def test_dragging_a_box_clears_the_stale_pick_highlight(self, window, tmp_path):
        photo = self._open(window, tmp_path)
        window._on_pick_finished(0, self._hit(photo, Box(220, 170, 580, 430)))  # noqa: SLF001
        assert photo.pick_box is not None

        window._on_canvas_box_committed(Box(100, 100, 500, 400))  # noqa: SLF001

        assert photo.pick_box is None
        assert photo.pick_score is None


class TestLock:
    def test_lock_scales_to_each_image(self, window, tmp_path):
        first = make_item(make_photo(tmp_path / "a.png", size=(1000, 800)))
        second = make_item(make_photo(tmp_path / "b.png", size=(500, 400)))
        window._photos = [first, second]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        locked = window._display_box(first)  # noqa: SLF001
        window._apply_current_box_to_all()  # noqa: SLF001

        assert locked is not None
        assert second.manual is True
        assert second.origin == "套用"
        assert second.free is not None
        # 半尺寸的图，框也应当正好是一半大
        assert abs(second.free.width - locked.width / 2) <= 2
        assert abs(second.free.height - locked.height / 2) <= 2

    def test_lock_without_any_box_warns(self, window):
        window._apply_current_box_to_all()  # noqa: SLF001
        assert "先选中一张" in window.status.text()


class TestList:
    def test_exclude_toggle(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.list.item(0).setSelected(True)
        window._toggle_included()  # noqa: SLF001

        assert item.included is False
        assert "跳过" in window.list.item(0).text()

    def test_suspicious_first_puts_failures_on_top(self, window, tmp_path):
        good = make_item(make_photo(tmp_path / "good.png"))
        broken = make_item(make_photo(tmp_path / "broken.png"))
        broken.error = "读不出来"

        window._photos = [good, broken]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001
        window._sort_suspicious_first()  # noqa: SLF001

        assert window._photos[0] is broken  # noqa: SLF001


class TestSize:
    """尺寸设置：锁定联动、单位换算、拉伸与精确尺寸。"""

    def test_lock_syncs_the_other_edge(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.switch_lock.setChecked(True)
        window.spin_width.setValue(1600)
        # 裁剪框是 4:3，所以高度应当自己跟到 1200
        assert abs(window.spin_height.value() - 1200) <= 2

        window.spin_height.setValue(600)
        assert abs(window.spin_width.value() - 800) <= 2

    def test_unlocking_lets_both_edges_be_set(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.switch_lock.setChecked(False)
        window.spin_width.setValue(1600)
        window.spin_height.setValue(500)

        assert window.spin_width.value() == 1600
        assert window.spin_height.value() == 500

        spec = window._output_spec()  # noqa: SLF001
        assert spec.keep_aspect is False
        assert (spec.px_width, spec.px_height) == (1600, 500)
        # 解锁 = 精确拉伸：结果像素必须等于所填值，哪怕比例和源图不符
        assert spec.resolve(4 / 3).pixel_size == (1600, 500)

    def test_locked_two_edges_fit_inside_the_box(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.switch_lock.setChecked(True)
        window.spin_width.setValue(1600)
        window.spin_height.setValue(1200)
        spec = window._output_spec()  # noqa: SLF001
        assert spec.keep_aspect is True
        # 方图放进 4:3 的框里 -> 短边顶满，不拉伸
        assert spec.resolve(1.0).pixel_size == (1200, 1200)

    def test_unit_switch_converts_the_value(self, window):
        window.combo_unit.setCurrentIndex(0)  # 像素
        window.spin_dpi.setValue(300)
        window.spin_width.setValue(1600)

        window.combo_unit.setCurrentIndex(2)  # 厘米
        assert abs(window.spin_width.value() - 13.55) < 0.02

        window.combo_unit.setCurrentIndex(0)  # 回到像素
        assert abs(window.spin_width.value() - 1600) <= 1

    def test_physical_unit_makes_a_physical_spec(self, window):
        window.combo_unit.setCurrentIndex(3)  # 英寸
        window.spin_dpi.setValue(300)
        window.spin_width.setValue(6)
        window.spin_height.setValue(4)

        spec = window._output_spec()  # noqa: SLF001
        assert spec.size_mode is SizeMode.PHYSICAL
        assert spec.phys_unit is Unit.INCH
        assert spec.resolve(1.5).pixel_size == (1800, 1200)

    def test_size_hint_states_pixels_and_cm(self, window, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window.combo_unit.setCurrentIndex(0)
        window.spin_width.setValue(1600)
        window.spin_height.setValue(1200)

        text = window.label_size.text()  # noqa: SLF001
        assert "px" in text and "cm" in text

    def test_size_preset_fills_both_edges(self, window):
        window.combo_unit.setCurrentIndex(0)
        window.combo_size_preset.setCurrentIndex(1)  # 1600 x 1200

        assert window.spin_width.value() == 1600
        assert window.spin_height.value() == 1200


class TestNaming:
    """导出命名：预设一键切，只有「自定义模板」才让改。"""

    @staticmethod
    def _mode(window, template):
        return window.combo_name.findData(template)

    def test_preset_fills_and_locks_the_template(self, window):
        window._on_name_mode_changed(self._mode(window, "{stem}"))  # noqa: SLF001
        assert window.edit_name.text() == "{stem}"
        assert window.edit_name.isEnabled() is False

        window._on_name_mode_changed(self._mode(window, None))  # noqa: SLF001
        assert window.edit_name.isEnabled() is True

    def test_output_spec_follows_the_mode(self, window):
        window._on_name_mode_changed(self._mode(window, "{stem}"))  # noqa: SLF001
        assert window._output_spec().name_template == "{stem}"  # noqa: SLF001

        window._on_name_mode_changed(self._mode(window, "{stem}_crop"))  # noqa: SLF001
        assert window._output_spec().name_template == "{stem}_crop"  # noqa: SLF001

    def test_default_mode_keeps_the_suffix(self, window):
        assert window._output_spec().name_template == "{stem}_crop"  # noqa: SLF001


class TestRotation:
    """旋转：画面转、裁剪框保持水平；角度可以套用到整批。"""

    def _open(self, window, tmp_path, size=(1000, 800)):
        photo = make_item(make_photo(tmp_path / "a.png", size=size))
        window._photos = [photo]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001
        return photo

    def test_the_picture_and_the_box_turn_together(self, window, tmp_path):
        photo = self._open(window, tmp_path)
        before = window._display_box(photo)  # noqa: SLF001

        window.slider_rotate.setValue(10.0)

        assert photo.rotate_deg == pytest.approx(10.0)
        assert photo.rotated_image is not None
        assert window.canvas.image_size == photo.rotated_image.size
        after = window._display_box(photo)  # noqa: SLF001
        assert before is not None and after is not None
        assert (after.x1, after.y1) != (before.x1, before.y1)

    def test_zero_angle_uses_the_original_pixels(self, window, tmp_path):
        photo = self._open(window, tmp_path)
        before = window._display_box(photo)  # noqa: SLF001

        window.slider_rotate.setValue(0.0)

        assert photo.rotated_image is None
        assert window._display_box(photo) == before  # noqa: SLF001
        assert window.canvas.image_size == photo.image.size

    def test_auto_tilt_shows_up_in_the_preview(self, window, tmp_path):
        """勾了自动转正，预览也要跟着转 —— 预览与导出不能各算一套角度。"""
        photo = self._open(window, tmp_path)
        layer = Image.new("L", photo.image.size, 0)
        ImageDraw.Draw(layer).rectangle([200, 380, 800, 420], fill=255)
        layer = layer.rotate(8.0, resample=Image.Resampling.BILINEAR, fillcolor=0)
        photo.mask = np.asarray(layer) > 127

        window.check_tilt.setChecked(True)

        assert window._rotation_angle(photo) == pytest.approx(-8.0, abs=1.5)  # noqa: SLF001

    def test_dragging_while_rotated_keeps_both_coordinates(self, window, tmp_path):
        photo = self._open(window, tmp_path)
        window.slider_rotate.setValue(12.0)
        window._set_mode(mw.MODE_FREE)  # noqa: SLF001

        view_box = Box(300, 300, 700, 600)
        window._on_canvas_box_committed(view_box)  # noqa: SLF001

        # 显示的就是用户拖的那个框（不该让他看到框自己长大）
        assert photo.free == view_box
        # 存下来的原图坐标换算回当前角度时，必须盖住用户拖的那块
        assert photo.free_src is not None
        back = window._to_view_box(photo, photo.free_src)  # noqa: SLF001
        assert back is not None
        assert back.x1 <= view_box.x1 + 1 and back.y1 <= view_box.y1 + 1
        assert back.x2 >= view_box.x2 - 1 and back.y2 >= view_box.y2 - 1

    def test_spec_declares_the_box_is_already_rotated(self, window, tmp_path):
        """界面给的框在转正后的画面上，规格必须声明这一点 —— 否则会转两遍。"""
        photo = self._open(window, tmp_path)
        window.slider_rotate.setValue(9.0)

        spec = window._spec_for(photo)  # noqa: SLF001
        assert spec is not None
        assert spec.rotate_deg == pytest.approx(9.0)
        assert spec.box_in_rotated_frame is True
        assert spec.subject == photo.subject  # 预览用的就是同一个框

    def test_preview_matches_the_exported_frame(self, window, tmp_path):
        """预览里看到的框必须就是导出时会裁的窗 —— 这是这个工具的硬承诺。"""
        from imgcrop.crop import apply_crop

        photo = self._open(window, tmp_path)
        for angle in (0.0, 7.5, -12.0):
            window.slider_rotate.setValue(angle)
            preview = window._display_box(photo)  # noqa: SLF001
            spec = window._spec_for(photo)  # noqa: SLF001
            out = apply_crop(photo.image, spec, None)
            assert out.box is not None, f"{angle}° 没能算出裁剪窗"
            assert (out.box.x1, out.box.y1, out.box.x2, out.box.y2) == (
                preview.x1, preview.y1, preview.x2, preview.y2
            ), f"{angle}° 时预览与导出不一致"

    def test_preview_matches_export_for_a_free_box(self, window, tmp_path):
        from imgcrop.crop import apply_crop

        photo = self._open(window, tmp_path)
        window.slider_rotate.setValue(10.0)
        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        window._on_canvas_box_committed(Box(300, 300, 700, 600))  # noqa: SLF001

        preview = window._display_box(photo)  # noqa: SLF001
        spec = window._spec_for(photo)  # noqa: SLF001
        out = apply_crop(photo.image, spec, None)
        assert out.box is not None
        assert (out.box.x1, out.box.y1, out.box.x2, out.box.y2) == (
            preview.x1, preview.y1, preview.x2, preview.y2
        )

    def test_apply_to_all_carries_the_angle(self, window, tmp_path):
        first = make_item(make_photo(tmp_path / "a.png"))
        second = make_item(make_photo(tmp_path / "b.png", size=(500, 400)))
        window._photos = [first, second]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001
        window.slider_rotate.setValue(6.0)

        window._apply_current_box_to_all()  # noqa: SLF001

        assert second.rotate_deg == pytest.approx(6.0)
        assert second.free_src is not None
        assert second.free is not None
        assert second.origin == "套用"

    def test_reset_clears_the_manual_angle(self, window, tmp_path):
        photo = self._open(window, tmp_path)
        window.slider_rotate.setValue(7.0)

        window._reset_current()  # noqa: SLF001

        assert photo.rotate_deg == 0.0
        assert window.slider_rotate.value() == pytest.approx(0.0)


class TestRotatedExport:
    def test_export_after_rotating_writes_a_file(self, window, qapp, tmp_path):
        photo = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [photo]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001
        window.slider_rotate.setValue(8.0)

        out = tmp_path / "out"
        window.edit_out.setText(str(out))
        window._export_all()  # noqa: SLF001
        wait_for_export(window, qapp)

        assert (out / "a_crop.png").is_file()

    def test_export_honours_the_naming_mode(self, window, qapp, tmp_path):
        photo = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [photo]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001
        window._on_name_mode_changed(window.combo_name.findData("{stem}"))  # noqa: SLF001

        out = tmp_path / "out"
        window.edit_out.setText(str(out))
        window._export_all()  # noqa: SLF001
        wait_for_export(window, qapp)

        assert (out / "a.png").is_file()


class TestExport:
    def test_export_uses_a_free_box_without_any_subject(self, window, qapp, tmp_path):
        """自动/点选都没结果时，靠自由框也能导出 —— 这是兜底路径。"""
        item = make_item(make_photo(tmp_path / "a.png"))
        item.subject = None
        item.proposal = None
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        window._set_mode(mw.MODE_FREE)  # noqa: SLF001
        window._on_canvas_box_committed(Box(100, 80, 500, 400))  # noqa: SLF001

        out = tmp_path / "out"
        window.edit_out.setText(str(out))
        window.switch_lock.setChecked(False)
        window.spin_width.setValue(400)
        window.spin_height.setValue(320)

        window._export_all()  # noqa: SLF001
        wait_for_export(window, qapp)

        with Image.open(out / "a_crop.png") as image:
            assert image.size == (400, 320)

    def test_export_writes_cropped_files(self, window, qapp, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        out = tmp_path / "out"
        window.edit_out.setText(str(out))
        window.spin_width.setValue(600)

        window._export_all()  # noqa: SLF001
        wait_for_export(window, qapp)

        produced = out / "a_crop.png"
        assert produced.is_file()
        with Image.open(produced) as image:
            assert image.size[0] == 600

    def test_export_honours_exact_size_when_unlocked(self, window, qapp, tmp_path):
        item = make_item(make_photo(tmp_path / "a.png"))
        window._photos = [item]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        out = tmp_path / "out"
        window.edit_out.setText(str(out))
        window.switch_lock.setChecked(False)
        window.spin_width.setValue(600)
        window.spin_height.setValue(600)

        window._export_all()  # noqa: SLF001
        wait_for_export(window, qapp)

        with Image.open(out / "a_crop.png") as image:
            assert image.size == (600, 600)

    def test_export_skips_excluded_images(self, window, qapp, tmp_path):
        first = make_item(make_photo(tmp_path / "a.png"))
        second = make_item(make_photo(tmp_path / "b.png"))
        second.included = False
        window._photos = [first, second]  # noqa: SLF001
        window._reload_list(select=0)  # noqa: SLF001

        out = tmp_path / "out"
        window.edit_out.setText(str(out))
        window._export_all()  # noqa: SLF001
        wait_for_export(window, qapp)

        assert (out / "a_crop.png").is_file()
        assert not (out / "b_crop.png").exists()

    def test_export_without_subject_warns(self, window):
        window._export_all()  # noqa: SLF001
        assert "没有可导出" in window.status.text()
