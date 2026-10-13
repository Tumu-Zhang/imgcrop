"""联网验证：模型来源是否真的可用、哈希是否真的对得上。

默认跳过 —— 网络可达性不该成为常规测试结果的一部分。显式运行::

    IMGCROP_NETWORK_TESTS=1 python -m pytest tests/test_models_online.py -v

test_prepare_project_models 会把模型实际下载到项目 models/ 目录（打包要用），
它同时就是一次端到端验证：来源能下、下完的哈希与清单一致。
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pytest

from imgcrop import models

PROJECT_MODELS = Path(__file__).resolve().parents[1] / "models"

pytestmark = pytest.mark.skipif(
    not os.environ.get("IMGCROP_NETWORK_TESTS"),
    reason="需要联网；设置 IMGCROP_NETWORK_TESTS=1 后重跑",
)


@pytest.mark.parametrize("key", sorted(models.MODELS))
def test_source_reachable(key):
    """HEAD 探测：至少一个来源可用，且长度与清单里的近似值不相矛盾。"""
    entry = models.MODELS[key]
    available: list[tuple[str, int]] = []
    errors: list[str] = []

    for url in entry.urls():
        try:
            request = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "imgcrop/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                available.append((url, int(response.headers.get("Content-Length") or 0)))
        except Exception as exc:  # noqa: BLE001 - 逐个源汇报
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    assert available, f"所有下载源都不可用：{errors}"

    url, length = available[0]
    if entry.size and length:
        assert abs(length - entry.size) / entry.size < 0.05, (
            f"{url} 返回 {length} 字节，与清单里的 {entry.size} 相差过大，"
            "可能源上的文件已经更新，清单需要同步"
        )


@pytest.mark.parametrize("key", sorted(models.MODELS))
def test_prepare_project_models(key):
    """把模型下到项目 models/ 目录并校验哈希 —— 打包时就地取材。"""
    entry = models.MODELS[key]
    path = models.download(entry, target_dir=PROJECT_MODELS)

    assert path.is_file()
    assert models.verify(entry, path), f"{entry.filename} 哈希与清单不符"
