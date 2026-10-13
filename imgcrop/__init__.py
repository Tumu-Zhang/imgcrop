"""图片裁剪器 —— 自动找出主体，按统一规格批量裁剪。

给实验照片去边留白：拖进一批图，程序用本地小模型（ISNet）找出主体，
按「主体归一化取景」统一构图，再按你指定的像素/DPI/格式输出。
自动结果不对时，点一下物体（EfficientSAM）或直接拖框即可修正。

设计原则
--------
原图零改动
    所有处理都在内存里做，输出写到单独的目录，源文件一个字节也不动。
批内一致
    画面尺寸 = 主体尺寸 x (1 + 2 x 边距)，输出像素固定 —— 于是同批照片里
    主体的大小与留白自然一致，与拍摄远近无关。
全自动与人工兜底同一条路
    自动分割、点选、手拖框都只产出一个 Box，后面的裁剪链路完全一致。

包内模块
--------
geometry   纯几何：坐标映射、取景、透视、旋转
crop       裁剪规格与执行
regions    掩膜区域分析（连通域 + 多阈值候选）
segment    模型推理（ISNet 自动分割、EfficientSAM 点选）
models     模型清单与下载/校验/缓存/离线导入
model      输出规格（像素/物理尺寸、DPI、格式、体积上限）
encode     编码与体积上限搜索
render     位图载入与重采样
ingest     输入探测与清单整理
pipeline   裁剪编排（批处理、进度、报告）
"""

from imgcrop.crop import ASPECT_PRESETS, DEFAULT_MARGIN, CropMode, CropSpec, apply_crop
from imgcrop.geometry import Box
from imgcrop.model import (
    OutputFormat,
    OutputSpec,
    SizeMode,
    TargetGeometry,
    Unit,
    resolve_geometry,
)
from imgcrop.pipeline import CropPlan, CropReport, CropResult, Cropper, crop_batch

__all__ = [
    "ASPECT_PRESETS",
    "Box",
    "CropMode",
    "CropPlan",
    "CropReport",
    "CropResult",
    "CropSpec",
    "Cropper",
    "DEFAULT_MARGIN",
    "OutputFormat",
    "OutputSpec",
    "SizeMode",
    "TargetGeometry",
    "Unit",
    "apply_crop",
    "crop_batch",
    "resolve_geometry",
]

__version__ = "1.0.0"
