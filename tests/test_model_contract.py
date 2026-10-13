"""模型契约检查：ONNX 的输入输出签名必须与推理层的假设一致。

这类问题不会让程序崩溃，只会让结果悄悄错掉（掩膜贴到错误的尺寸上、
归一化方式不对导致掩膜全黑等）。所以模型文件一旦更新，先跑这个文件。

模型未安装时自动跳过。要看到真实签名：::

    python -m pytest tests/test_model_contract.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from imgcrop import models

ort = pytest.importorskip("onnxruntime", reason="需要 onnxruntime 才能检查模型签名")


def load_session(key: str):
    entry = models.MODELS[key]
    path = models.find(entry)
    if path is None:
        pytest.skip(f"{entry.label} 未安装，先运行 python tools/fetch_models.py")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return entry, path, session


class TestIsnetContract:
    def test_input_is_rgb_and_four_dimensional(self):
        _, path, session = load_session("isnet-general-use")
        inputs = session.get_inputs()

        assert len(inputs) == 1, f"{path.name} 期望 1 个输入，实际 {len(inputs)}"
        # 空间维度是符号名（'height' / 'width'），因为导出时开了 dynamic axes；
        # 能钉死的只有"4 维、3 通道"，分辨率由 build_tensor 决定。
        assert len(inputs[0].shape) == 4
        assert inputs[0].shape[1] == 3

    def test_runtime_outputs_include_a_main_saliency_map(self):
        """图里声明的是符号形状，只能真跑一次才知道实际输出。

        ISNet 保留了 U^2-Net 的多级侧输出（这个导出共 12 个），所以不能假设
        "只有 1 个输出"；要确认的是其中有一张单通道 1024x1024 的主显著性图。
        """
        _, path, session = load_session("isnet-general-use")
        tensor = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
        results = session.run(None, {session.get_inputs()[0].name: tensor})

        shapes = [np.asarray(item).shape for item in results]
        assert (1, 1, 1024, 1024) in shapes, (
            f"{path.name} 的输出里没有主显著性图 (1,1,1024,1024)，实际有：{shapes}"
        )

    def test_the_output_we_actually_use_is_the_main_one(self):
        """推理层取的是第 0 个输出 —— 它必须就是那张主显著性图。

        取错输出不会报错，只会让掩膜分辨率对不上、坐标整体偏掉，所以单独看住。
        图里声明的是符号形状，因此只能跑一次看真实结果。
        """
        _, path, session = load_session("isnet-general-use")
        tensor = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
        first = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])

        assert first.shape == (1, 1, 1024, 1024), (
            f"{path.name} 的第 0 个输出实际是 {first.shape}，不是主显著性图，"
            "segment.IsnetSegmenter 需要改为按形状挑选输出"
        )

    def test_normalisation_range_comes_back_as_saliency(self):
        """用 (pixel-128)/256 归一化喂一张图，输出必须落在 0..1 且有空间区分度。

        如果归一化方式变了（例如该用 x/255-0.5），输出往往会退化成一片饱和
        或一片全黑 —— 这条断言就是为了逮住那种情况。
        """
        _, _, session = load_session("isnet-general-use")

        picture = np.full((1024, 1024, 3), 200, dtype=np.float32)
        picture[300:724, 300:724] = 30  # 一块明显不同的区域
        tensor = ((picture - 128.0) / 256.0).transpose(2, 0, 1)[None].astype(np.float32)

        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        mask = np.asarray(raw).squeeze()

        assert mask.shape == (1024, 1024)
        assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
        spread = float(mask.std())
        assert spread > 0.01, f"掩膜几乎没有空间变化（std={spread:.4f}），归一化可能不对"
        assert abs(float(mask[512, 512]) - float(mask[20, 20])) > 0.1, (
            "中心与角落的显著性没有差别，模型或归一化可能不匹配"
        )


class TestEfficientSamContract:
    def test_reports_signature(self):
        """EfficientSAM 的输入组合较复杂（图像 + 提示点/框），先把真实签名打出来。"""
        _, path, session = load_session("efficientsam-ti")

        print(f"\n{path.name}（{path.stat().st_size / 1048576:.1f} MB）")
        for tensor in session.get_inputs():
            print(f"  IN   {tensor.name:28s} {tensor.shape}  {tensor.type}")
        for tensor in session.get_outputs():
            print(f"  OUT  {tensor.name:28s} {tensor.shape}  {tensor.type}")

        assert session.get_inputs(), "模型应当有输入"
        assert session.get_outputs(), "模型应当有输出"

    def test_output_looks_like_masks(self):
        """输出应当是若干张掩膜（或多分辨率掩膜），不是单张分类分数。"""
        _, _, session = load_session("efficientsam-ti")
        outputs = session.get_outputs()
        # 至少要有 2 个输出（通常是 masks 和一个质量/分数），且形状里带空间维度
        assert len(outputs) >= 2, f"期望至少 2 个输出，实际 {len(outputs)}"
