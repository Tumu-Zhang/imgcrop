"""模型管理：下载、校验、缓存、离线导入。

这个模块只管"模型文件在不在、从哪来、对不对"，不碰推理逻辑（推理在
segment.py），也不碰界面。

为什么需要它
------------
两个分割模型各有几十 MB，不可能塞进 pip 包，也不该让用户自己去翻
HuggingFace 找文件。这里把它们登记成一份清单：每次取用前先按"程序自带
-> 用户缓存"的顺序找，找不到才下载，并且下载后一定用 SHA-256 校验。

镜像
----
huggingface.co 在部分网络环境下连不上（实测本机直连超时），所以默认先走
hf-mirror.com，失败再回落官方源。全部来源都失败时会给出逐个来源的失败
原因，而不是一句"下载失败"。

离线导入
--------
实验室机器常常不能上外网，所以除了下载还提供 zip 导入：把两个 onnx 打进
一个压缩包，拷过去导入即可，校验规则与下载完全一致。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_MIRRORS",
    "ENV_MODELS_DIR",
    "MODELS",
    "ModelEntry",
    "ModelError",
    "describe_status",
    "download",
    "find",
    "import_archive",
    "is_installed",
    "missing",
    "search_dirs",
    "status",
    "user_models_dir",
    "verify",
]

# 进度回调：(文件名, 已下载字节, 总字节)。总字节为 0 表示服务端没给长度。
ProgressCallback = Callable[[str, int, int], None]
CancelCheck = Callable[[], bool]

ENV_MODELS_DIR = "IMGCROP_MODELS_DIR"
"""覆盖模型目录（主要给测试与便携部署用）。"""

DEFAULT_MIRRORS: tuple[str, ...] = (
    "https://hf-mirror.com",
    "https://huggingface.co",
)

_CHUNK = 1 << 20  # 1 MiB


class ModelError(RuntimeError):
    """模型缺失、下载失败或校验不通过。"""


@dataclass(frozen=True)
class ModelEntry:
    """一个可下载的模型。"""

    key: str
    label: str
    filename: str
    repo: str
    sha256: str
    purpose: str
    size: int | None = None
    """字节数的近似值，仅用于进度预估；实际以服务端 Content-Length 为准。"""

    def urls(self, mirrors: Sequence[str] = DEFAULT_MIRRORS) -> list[str]:
        return [
            f"{mirror.rstrip('/')}/{self.repo}/resolve/main/{self.filename}"
            for mirror in mirrors
        ]


ISNET = ModelEntry(
    key="isnet-general-use",
    label="ISNet 主体分割",
    filename="isnet-general-use-q8.onnx",
    repo="SacredNoir/isnet-general-use-onnx",
    sha256="feed6f32a5e707ca7e939576b2d891b23fb9eb4114749657a5efc64e8651e43a",
    purpose="自动找出画面主体，给出裁剪候选框",
    size=44_436_071,
)

EFFICIENT_SAM = ModelEntry(
    key="efficientsam-ti",
    label="EfficientSAM-Ti 可提示分割",
    filename="image_segmentation_efficientsam_ti_2025april.onnx",
    repo="opencv/image_segmentation_efficientsam",
    sha256="4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77",
    purpose="点一下物体就精确框住它，用于自动结果不对时的人工兜底",
    size=48_300_000,
)

MODELS: dict[str, ModelEntry] = {entry.key: entry for entry in (ISNET, EFFICIENT_SAM)}


# ----------------------------------------------------------------------
# 目录
# ----------------------------------------------------------------------
def app_root() -> Path:
    """程序所在目录（开发时是仓库根，打包后是 exe 所在的文件夹）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def user_models_dir() -> Path:
    """用户级模型目录：下载与离线导入的落点。"""
    override = os.environ.get(ENV_MODELS_DIR)
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "图片裁剪器" / "models"


def search_dirs() -> list[Path]:
    """按优先级列出模型可能存在的位置。

    顺序有讲究：程序自带的排在用户缓存之前，这样"随包分发"的模型不会被
    用户目录里的旧文件顶掉。
    """
    dirs: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass) / "models")
    dirs.append(app_root() / "models")
    dirs.append(user_models_dir())
    return dirs


# ----------------------------------------------------------------------
# 查找与校验
# ----------------------------------------------------------------------
def find(entry: ModelEntry) -> Path | None:
    """找到模型文件；只看存在性，不做哈希校验（校验 40+MB 太慢，只在下载后做）。"""
    for directory in search_dirs():
        candidate = directory / entry.filename
        if candidate.is_file():
            return candidate
    return None


def is_installed(entry: ModelEntry) -> bool:
    return find(entry) is not None


def missing() -> list[ModelEntry]:
    return [entry for entry in MODELS.values() if not is_installed(entry)]


def status() -> list[tuple[ModelEntry, Path | None]]:
    return [(entry, find(entry)) for entry in MODELS.values()]


def describe_status() -> str:
    lines = []
    for entry, path in status():
        state = str(path) if path else "未安装"
        lines.append(f"{entry.label}（{entry.key}）：{state}")
    return "\n".join(lines)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(entry: ModelEntry, path: Path) -> bool:
    path = Path(path)
    return path.is_file() and sha256_of(path) == entry.sha256


# ----------------------------------------------------------------------
# 下载
# ----------------------------------------------------------------------
def _fetch(
    url: str,
    partial: Path,
    entry: ModelEntry,
    progress: ProgressCallback | None,
    should_cancel: CancelCheck | None,
) -> None:
    """下载到 .part 文件，支持断点续传。取消时保留 .part 以便下次接着下。"""
    resume_at = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "imgcrop/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status_code = getattr(response, "status", 200)
        if resume_at and status_code != 206:
            # 服务端不支持续传，只能重头来
            resume_at = 0
            partial.unlink(missing_ok=True)

        length = int(response.headers.get("Content-Length") or 0)
        total = length + resume_at if length else (entry.size or 0)
        done = resume_at
        with partial.open("ab" if resume_at else "wb") as handle:
            while True:
                if should_cancel and should_cancel():
                    raise ModelError("下载已取消")
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress:
                    progress(entry.filename, done, total)

        # 连接被中途掐断时 read() 返回空串而不是抛异常，所以必须自己核对收到的
        # 字节数：否则会留下一个"看起来下完了"的残缺文件，白等一次校验。
        # 这里抛异常才能让上层按"续传重试"处理 —— .part 不会删。
        expected = resume_at + length
        if length and done < expected:
            raise ModelError(f"连接提前结束：收到 {done} 字节，预期 {expected} 字节")


def download(
    entry: ModelEntry,
    *,
    target_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    mirrors: Sequence[str] = DEFAULT_MIRRORS,
    attempts_per_mirror: int = 4,
) -> Path:
    """下载并校验模型。已经装好（哈希一致）就直接返回，不重复下载。

    同一个来源会重试若干次：镜像对几十 MB 的连接经常中途断开，而 .part 是保留的，
    所以重试等于**续传**而不是从头再来。实测 44 MB 的模型会在 30 MB 附近断一次，
    靠续传两三次就能下完。
    """
    destination = Path(target_dir) if target_dir else user_models_dir()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / entry.filename

    if verify(entry, target):
        return target

    partial = target.with_name(target.name + ".part")
    failures: list[str] = []

    for url in entry.urls(mirrors):
        for attempt in range(1, attempts_per_mirror + 1):
            if should_cancel and should_cancel():
                raise ModelError("下载已取消")

            try:
                _fetch(url, partial, entry, progress, should_cancel)
            except Exception as exc:  # noqa: BLE001 - 逐个来源逐个次数重试，原因要汇总
                if should_cancel and should_cancel():
                    raise ModelError("下载已取消") from exc
                failures.append(f"{url}（第 {attempt} 次）—— {type(exc).__name__}: {exc}")
                continue  # .part 还在，下一轮从中断处续传

            if verify(entry, partial):
                partial.replace(target)
                return target

            failures.append(f"{url} —— 下载完成但 SHA-256 与清单不符，文件可能损坏")
            partial.unlink(missing_ok=True)
            break

    detail = "\n  ".join(failures) if failures else "没有可用的下载源"
    raise ModelError(
        f"{entry.label} 下载失败，已尝试全部来源：\n  {detail}\n"
        f"可以手动下载 {entry.filename} 放进 {destination}，"
        "或用离线包通过「导入模型」安装。"
    )


# ----------------------------------------------------------------------
# 离线导入
# ----------------------------------------------------------------------
def import_archive(archive: Path, *, target_dir: Path | None = None) -> list[str]:
    """从 zip 导入模型文件，返回成功导入的模型 key。

    压缩包里的文件按文件名匹配，目录层级无关；每个文件都要通过 SHA-256 校验，
    校验不过的直接删掉并报错，避免留下一个"看起来装好了"的坏文件。
    """
    archive = Path(archive)
    if not archive.is_file():
        raise ModelError(f"找不到压缩包：{archive}")

    destination = Path(target_dir) if target_dir else user_models_dir()
    destination.mkdir(parents=True, exist_ok=True)

    imported: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        by_name = {Path(name).name: name for name in bundle.namelist()}
        for entry in MODELS.values():
            member = by_name.get(entry.filename)
            if member is None:
                continue
            target = destination / entry.filename
            with bundle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, _CHUNK)
            if not verify(entry, target):
                target.unlink(missing_ok=True)
                raise ModelError(f"{entry.filename} 校验失败，压缩包可能不完整或已被改动")
            imported.append(entry.key)

    if not imported:
        known = "、".join(entry.filename for entry in MODELS.values())
        raise ModelError(f"压缩包里没有找到已知的模型文件（需要 {known}）")
    return imported
