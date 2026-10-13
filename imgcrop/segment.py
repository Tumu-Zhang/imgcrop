"""分割推理：把图片喂给本地小模型，拿回主体掩膜。

两个模型，两种用法
------------------
ISNet
    整图进去，显著性掩膜出来。不需要任何提示，用于"拖进来就给出候选框"。
    它回答的是"这张图里什么最显眼"，于是也用于整批自动裁剪。

EfficientSAM
    整图 + 一个提示点进去，该点对应物体的掩膜出来。它回答的是"你点的那个
    东西在哪" —— 自动结果不对时点一下即可，比手动拖框精确得多。

预处理与坐标映射
----------------
两个模型都吃 1024x1024 的正方形输入，所以原图要先等比缩放再居中填充，输出
的掩膜坐标要按同样的映射换回原图 —— 这一步由 geometry.Letterbox 保证可逆，
错一次就会让所有裁剪框整体偏移。

填充色用中性灰（归一化后正好是 0）而不是黑色：黑边在某些图上会被模型当成
"很暗的前景"，灰边最不容易被误读。

会话是懒加载并缓存的：两个模型加起来 90 多 MB，不该在程序启动时就占住内存。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from imgcrop import models, regions
from imgcrop.geometry import Box, Letterbox, mask_to_box

__all__ = [
    "INPUT_SIZE",
    "EfficientSamSegmenter",
    "IsnetSegmenter",
    "MAX_POINTS",
    "PointHit",
    "Proposal",
    "SegmentError",
    "build_tensor",
    "missing_models",
]

INPUT_SIZE = 1024
"""两个模型都是 1024x1024 的正方形输入。"""

MAX_POINTS = 6
"""EfficientSAM 的提示点输入固定 6 个槽位，用不到的槽位用 label=-1 填充。"""

_PIXEL_MEAN = 128.0
_PIXEL_SCALE = 256.0
"""输入归一化：x -> (x - 128) / 256，等价于 x/255 - 0.5，输出落在 -0.5..0.5。"""

_PAD_COLOR = (128, 128, 128)
"""填充色取中性灰：归一化后正好是 0，最不容易被模型读成"很暗的前景"。"""


class SegmentError(RuntimeError):
    """模型缺失、运行时缺失或推理失败。"""


def missing_models() -> list[str]:
    """返回缺失模型的 key（界面据此提示先下载）。"""
    return [entry.key for entry in models.missing()]


@dataclass
class Proposal:
    """一次自动分割给出的结果。"""

    boxes: list[Box]
    """候选主体框，**已换算回原图坐标**，按面积从大到小。"""

    mask: np.ndarray
    """模型输入分辨率上的显著性掩膜（0..1）。做倾斜校正时要用它。"""

    letterbox: Letterbox

    @property
    def best(self) -> Box | None:
        """最可能的那个候选（没有候选时为 None）。

        candidates 按"从紧到松"给出，所以这里拿到的是最贴合主体的那一档，
        而不是面积最大（=最松）的那个。
        """
        return self.boxes[0] if self.boxes else None

    def source_mask(self) -> np.ndarray:
        """掩膜换算回原图尺寸（原图空间的布尔数组）。

        倾斜校正要用掩膜估计主体朝向，必须用这一个：`mask` 还在模型输入
        空间（含 letterbox 填充），直接缩放会得到失真的角度。
        """
        return self.letterbox.to_source_mask(self.mask)


@dataclass
class PointHit:
    """一次点选分割的结果。"""

    box: Box
    """点中物体的包围盒，**已换算回原图坐标**。"""

    mask: np.ndarray
    """模型输入分辨率上的布尔掩膜。apply_crop 会自动缩放到原图，可直接传。"""

    score: float
    """模型自评的 IoU（0..1），可用于判断这次点击是否可信。"""

    letterbox: Letterbox

    def source_mask(self) -> np.ndarray:
        """掩膜换算回原图尺寸（原图空间的布尔数组）—— 倾斜校正用这个。"""
        return self.letterbox.to_source_mask(self.mask)


def build_tensor(
    image: Image.Image,
    size: int = INPUT_SIZE,
) -> tuple[np.ndarray, Letterbox]:
    """等比缩放 + 居中填充 + 归一化，返回 (1,3,size,size) 张量与坐标映射。"""
    width, height = image.size
    box = Letterbox.fit(width, height, size, size)

    scaled_size = (
        min(size, max(1, round(width * box.scale))),
        min(size, max(1, round(height * box.scale))),
    )
    scaled = image.convert("RGB").resize(scaled_size, Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), _PAD_COLOR)
    canvas.paste(scaled, (round(box.pad_x), round(box.pad_y)))

    array = np.asarray(canvas, dtype=np.float32)
    array = (array - _PIXEL_MEAN) / _PIXEL_SCALE
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
    return tensor, box


def _blank_padding(mask: np.ndarray, letterbox: Letterbox) -> None:
    """把 letterbox 的填充区直接抹成 0（原地修改）。

    填充色是中性灰，但它不代表"没有前景"：实测在宽高比悬殊的图上，
    ISNet 会把灰边也判成前景（一张 772x108 的图有 23.7% 的前景落在灰边里），
    候选框于是被拉到贴边、甚至变成整幅画面。掩膜只在内容区有意义。
    """
    left, top, right, bottom = letterbox.content_bounds(mask.shape)
    mask[:top, :] = 0.0
    mask[bottom:, :] = 0.0
    mask[:, :left] = 0.0
    mask[:, right:] = 0.0


def _import_onnxruntime():
    try:
        import onnxruntime  # noqa: PLC0415 - 有意懒加载，避免启动即加载几十 MB
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise SegmentError(
            "分割功能需要 onnxruntime，请先安装：pip install onnxruntime"
        ) from exc
    return onnxruntime


def _build_session(path: Path, threads: int | None):
    onnxruntime = _import_onnxruntime()
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = threads
    return onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _match_inputs(session) -> dict[str, str]:
    """按名字特征找出三个输入。

    模型的输入顺序在两个版本之间变过，硬编码下标迟早会错位；按名字里的
    image / coords / labels 关键词匹配，顺序怎么变都不会拿错张量。
    """
    names = {item.name for item in session.get_inputs()}

    def pick(*keywords: str) -> str | None:
        for name in sorted(names):
            lowered = name.lower()
            if all(keyword in lowered for keyword in keywords):
                return name
        return None

    found = {
        "images": pick("image"),
        "coords": pick("coords") or pick("point", "coord"),
        "labels": pick("labels") or pick("point", "label"),
    }
    missing = [key for key, value in found.items() if value is None]
    if missing:
        raise SegmentError(f"模型输入签名不认识，缺少：{'、'.join(missing)}")
    return found  # type: ignore[return-value]


class IsnetSegmenter:
    """ISNet 显著性分割：不需要提示，自动给出画面主体。

    可以复用同一个实例处理整批图片 —— 会话只建一次。
    """

    def __init__(self, path: Path | None = None, *, threads: int | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._threads = threads
        self._session = None
        self._input_name: str | None = None

    # -- 会话 -----------------------------------------------------------
    def resolve_path(self) -> Path:
        """定位模型文件；找不到时给出可操作的下一步，而不是一句"文件不存在"。"""
        candidate = self._path if self._path is not None else models.find(models.ISNET)
        if candidate is None or not Path(candidate).is_file():
            raise SegmentError(
                "缺少 ISNet 模型（自动找主体用）。"
                "运行 python tools/fetch_models.py 下载，或用离线包通过「导入模型」安装。"
            )
        return Path(candidate)

    @property
    def session(self):
        if self._session is None:
            self._session = _build_session(self.resolve_path(), self._threads)
            self._input_name = self._session.get_inputs()[0].name
        return self._session

    # -- 推理 -----------------------------------------------------------
    def saliency(self, image: Image.Image) -> Proposal:
        """算出一张图的主体候选（坐标已换回原图）。"""
        tensor, letterbox = build_tensor(image)
        session = self.session
        if self._input_name is None:  # pragma: no cover - session 建立后必然已赋值
            raise SegmentError("模型输入名缺失，会话未正确初始化")

        raw = session.run(None, {self._input_name: tensor})[0]
        mask = np.asarray(raw, dtype=np.float32)
        if mask.ndim > 2:
            mask = mask.squeeze()
        mask = np.clip(mask, 0.0, 1.0)
        # 填充区不是前景：不抹掉的话，宽高比悬殊的图会被"框住整幅画面"
        _blank_padding(mask, letterbox)

        boxes: list[Box] = []
        for model_box in regions.candidates(mask):
            source_box = letterbox.to_source_box(model_box)
            if source_box.is_empty:
                continue  # 完全落在填充区里的候选没有意义
            boxes.append(source_box)

        return Proposal(boxes=boxes, mask=mask, letterbox=letterbox)


class EfficientSamSegmenter:
    """EfficientSAM 点选分割：给提示点，返回"点的那个物体"的掩膜与包围盒。

    points 是原图坐标 (x, y)；labels 取 1 表示前景点、0 表示背景点
    （"这里不对，换一个"）。两个提示点通常就够了，模型自身会在 3 个候选
    掩膜里挑 IoU 最高的那个。
    """

    def __init__(self, path: Path | None = None, *, threads: int | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._threads = threads
        self._session = None
        self._inputs: dict[str, str] | None = None

    # -- 会话 -----------------------------------------------------------
    def resolve_path(self) -> Path:
        candidate = (
            self._path if self._path is not None else models.find(models.EFFICIENT_SAM)
        )
        if candidate is None or not Path(candidate).is_file():
            raise SegmentError(
                "缺少 EfficientSAM 模型（点选分割用）。"
                "运行 python tools/fetch_models.py 下载，或用离线包通过「导入模型」安装。"
            )
        return Path(candidate)

    @property
    def session(self):
        if self._session is None:
            self._session = _build_session(self.resolve_path(), self._threads)
            self._inputs = _match_inputs(self._session)
        return self._session

    # -- 推理 -----------------------------------------------------------
    def segment(
        self,
        image: Image.Image,
        points: Sequence[tuple[float, float]],
        labels: Sequence[float] | None = None,
        *,
        threshold: float = 0.0,
    ) -> PointHit | None:
        """按提示点分割；该点处没有物体时返回 None。

        threshold 作用在模型输出的 logit 上：logit > 0 等价于概率 > 0.5。
        """
        chosen = list(points)[:MAX_POINTS]
        if not chosen:
            raise SegmentError("点选分割至少需要一个提示点")
        point_labels = (
            [1.0] * len(chosen) if labels is None else [float(v) for v in labels][: len(chosen)]
        )

        tensor, letterbox = build_tensor(image)
        coords = np.zeros((1, 1, MAX_POINTS, 2), dtype=np.float32)
        flags = np.full((1, 1, MAX_POINTS), -1.0, dtype=np.float32)
        for index, (point, flag) in enumerate(zip(chosen, point_labels)):
            model_x, model_y = letterbox.to_model(float(point[0]), float(point[1]))
            coords[0, 0, index] = (model_x, model_y)
            flags[0, 0, index] = flag

        session = self.session
        assert self._inputs is not None  # session 建立时已解析
        outputs = session.run(
            None,
            {
                self._inputs["images"]: tensor,
                self._inputs["coords"]: coords,
                self._inputs["labels"]: flags,
            },
        )
        if len(outputs) < 2:  # pragma: no cover - 签名探查过，这里只是兜底
            raise SegmentError(f"模型输出数量异常：{len(outputs)}")

        masks = np.asarray(outputs[0], dtype=np.float32).reshape(-1, INPUT_SIZE, INPUT_SIZE)
        scores = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
        best = int(np.argmax(scores))
        hit = masks[best] > threshold

        model_box = mask_to_box(hit)
        if model_box is None:
            return None
        source_box = letterbox.to_source_box(model_box)
        if source_box.is_empty:
            return None
        return PointHit(
            box=source_box,
            mask=hit,
            score=float(scores[best]),
            letterbox=letterbox,
        )
