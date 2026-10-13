"""后台任务：识别主体、点选分割、批量导出。

界面线程只做绘制，模型推理与编码都在这三个 worker 里跑。它们都用
QObject + moveToThread 的方式驱动（而不是继承 QThread），这样槽函数
在哪儿执行一目了然，也不会出现"线程对象本身住在主线程"的经典坑。

模型会话由主窗口持有并传进来 —— 两个模型加起来 90 多 MB，整批只该加载一次。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from imgcrop.model import OutputSpec
from imgcrop.pipeline import CropPlan, CropReport, Cropper
from imgcrop.render import load_raster
from imgcrop.segment import IsnetSegmenter


class ScanWorker(QObject):
    """逐张跑 ISNet，给出主体候选。

    只做"识别"，不做取景 —— 边距/比例这类参数随时会变，取景便宜（纯几何），
    识别贵（一次推理）。分开之后，用户拖一下滑杆不需要重新跑模型。
    """

    progress = Signal(int, int, str)   # 已完成张数, 总数, 当前文件名
    detected = Signal(int, object)     # 清单行号, Proposal（没有主体时为 None）
    loaded = Signal(int, object)       # 清单行号, PIL 图（界面缓存用）
    failed = Signal(int, str)
    finished = Signal()

    def __init__(
        self,
        paths: list[Path],
        segmenter: IsnetSegmenter,
        indices: list[int] | None = None,
    ) -> None:
        """`indices` 是这批图在界面清单里的行号。

        只勾了清单里的其中几张时，worker 要报告"清单第几行"，而不是"我这一批
        的第几张" —— 后者会让结果写到别的图上。不传就按 0..n-1 处理。

        另外：这里的信号只能连**有线程亲和性的 QObject 方法**（例如
        MainWindow._on_scan_detected），别用 lambda。lambda 没有 receiver，
        PySide6 无从判断该在哪个线程执行（连显式 QueuedConnection 也会排到
        发射信号的 worker 线程），结果界面回调在后台线程里碰 QListWidget/
        QProgressBar —— 访问违例，进程直接消失。
        """
        super().__init__()
        self._paths = list(paths)
        self._segmenter = segmenter
        self._indices = list(indices) if indices is not None else list(range(len(self._paths)))
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        total = len(self._paths)
        for local, path in enumerate(self._paths):
            if self._cancelled:
                break
            index = self._indices[local]
            self.progress.emit(local, total, path.name)
            try:
                loaded = load_raster(path)
            except Exception as exc:  # noqa: BLE001 - 单张坏了不该拖垮整批
                self.failed.emit(index, f"{type(exc).__name__}: {exc}")
                continue
            self.loaded.emit(index, loaded.image)
            try:
                proposal = self._segmenter.saliency(loaded.image)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(index, f"{type(exc).__name__}: {exc}")
                continue
            self.detected.emit(index, proposal)
        self.progress.emit(total, total, "")
        self.finished.emit()


class PickWorker(QObject):
    """对一张图跑点选分割。"""

    finished = Signal(int, object)   # 下标, PointHit | None
    failed = Signal(int, str)

    def __init__(
        self,
        index: int,
        image,
        points: list[tuple[float, float]],
        labels: list[float],
        segmenter,
    ) -> None:
        super().__init__()
        self._index = index
        self._image = image
        self._points = points
        self._labels = labels
        self._segmenter = segmenter

    @Slot()
    def run(self) -> None:
        try:
            hit = self._segmenter.segment(self._image, self._points, self._labels)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._index, f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(self._index, hit)


class ExportWorker(QObject):
    """按当前方案批量裁剪导出。"""

    progress = Signal(int, int, str)
    finished = Signal(object)   # CropReport
    failed = Signal(str)

    def __init__(
        self,
        paths: list[Path],
        plans: dict[Path, CropPlan],
        spec: OutputSpec,
    ) -> None:
        super().__init__()
        self._paths = list(paths)
        self._plans = plans
        self._spec = spec

    @Slot()
    def run(self) -> None:
        try:
            report: CropReport = Cropper(
                self._spec,
                crop_provider=self._plans.get,
                progress=lambda event: self.progress.emit(
                    event.completed, event.total, event.current
                ),
            ).run(self._paths)
        except Exception as exc:  # noqa: BLE001 - 兜住一切，避免线程静默死亡
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(report)


# 三个 worker 都在独立线程里跑：主线程只负责画界面与刷新状态
__all__ = ["ExportWorker", "PickWorker", "ScanWorker"]
