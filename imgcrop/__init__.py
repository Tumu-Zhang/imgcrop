"""图片裁剪器 —— 自动找出主体，专注把照片裁正、裁好。

给实验照片、证件、扫描件去边留白：拖进一批图，程序用本地小模型（ISNet）
找出主体，按「主体归一化取景」统一构图；自动结果不对时，点一下物体
（EfficientSAM）或直接拖框即可修正。矫正能力（1.1.0）：

- 透视拉平：拖四个角点把斜拍的证件/画框拉成正视矩形；
- 自动转正：最小面积矩形拟合估角，证件/文档类更稳；
- 手动旋转 ±180° 微调。

输出所见即所得（1.1.0 瘦身定案）：成品像素 = 裁剪窗原像素（1:1，不缩放
不放大），格式跟随源（PNG 保透明），DPI 元数据跟随源图（缺省 300）。

设计原则
--------
原图零改动
    所有处理都在内存里做，输出写到单独的目录，源文件一个字节也不动。
批内一致
    画面尺寸 = 主体尺寸 x (1 + 2 x 边距) —— 同批照片里主体的大小与
    留白自然一致，与拍摄远近无关。
全自动与人工兜底同一条路
    自动分割、点选、手拖框都只产出一个 Box，后面的裁剪链路完全一致。

包内模块
--------
geometry   纯几何：坐标映射、取景、透视、旋转、最小面积矩形
crop       裁剪规格与执行（透视拉平 -> 旋转 -> 取景 的融合管线）
regions    掩膜区域分析（连通域 + 多阈值候选）
segment    模型推理（ISNet 自动分割、EfficientSAM 点选）
models     模型清单与下载/校验/缓存/离线导入
model      落盘规格（目录、命名、同名策略、背景色）
encode     编码（格式跟随源、DPI 跟随源）
render     位图载入与重采样
ingest     输入探测与清单整理
pipeline   裁剪编排（批处理、进度、报告）
"""

from imgcrop.crop import ASPECT_PRESETS, DEFAULT_MARGIN, CropMode, CropSpec, apply_crop
from imgcrop.geometry import Box
from imgcrop.model import ConflictPolicy, OutputSpec
from imgcrop.pipeline import CropPlan, CropReport, CropResult, Cropper, crop_batch

__all__ = [
    "ASPECT_PRESETS",
    "Box",
    "ConflictPolicy",
    "CropMode",
    "CropPlan",
    "CropReport",
    "CropResult",
    "CropSpec",
    "Cropper",
    "DEFAULT_MARGIN",
    "OutputSpec",
    "apply_crop",
    "crop_batch",
]

__version__ = "1.1.0"
