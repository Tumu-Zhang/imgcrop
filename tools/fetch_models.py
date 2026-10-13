"""模型获取工具：状态查询、下载、源可达性探测、离线导入。

裁剪功能需要两个本地小模型（ISNet 自动分割 + EfficientSAM-Ti 点选），
它们不随 pip 包分发。这个脚本负责把它们取到位。

用法::

    python tools/fetch_models.py --status              # 看模型装在哪、缺什么
    python tools/fetch_models.py --check               # 只探测下载源可达性，不下载
    python tools/fetch_models.py                       # 下载所有缺失的模型
    python tools/fetch_models.py --only isnet-general-use
    python tools/fetch_models.py --target models       # 指定落点（打包时用）
    python tools/fetch_models.py --import models.zip   # 内网机器离线导入
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imgcrop import models  # noqa: E402  （需要先补 sys.path）


def _human(num_bytes: int | None) -> str:
    if not num_bytes:
        return "长度未知"
    return f"{num_bytes / 1048576:.1f} MB"


def cmd_status() -> int:
    print("模型目录（按优先级）:")
    for directory in models.search_dirs():
        state = "存在" if directory.is_dir() else "——"
        print(f"  [{state}] {directory}")
    print()
    print(models.describe_status())

    pending = models.missing()
    if pending:
        total = sum(entry.size or 0 for entry in pending)
        print(f"\n还缺 {len(pending)} 个模型，约需下载 {total / 1048576:.1f} MB")
    else:
        print("\n模型齐备。")
    return 0


def cmd_check() -> int:
    """只发 HEAD 请求，确认下载源可达 —— 避免白下几十 MB 才发现 404。"""
    failures = 0
    for entry in models.MODELS.values():
        print(f"{entry.label}（{entry.filename}）")
        for url in entry.urls():
            try:
                request = urllib.request.Request(
                    url, method="HEAD", headers={"User-Agent": "imgcrop/1.0"}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    size = _human(int(response.headers.get("Content-Length") or 0))
                    print(f"    可用  {response.status}  {size}  {url}")
            except Exception as exc:  # noqa: BLE001 - 逐个源汇报，不给用户一个笼统的失败
                failures += 1
                print(f"    不可用  {type(exc).__name__}: {exc}")
        print()
    return 1 if failures else 0


def _progress_printer(entry_label: str):
    def show(filename: str, done: int, total: int) -> None:
        if total:
            percent = done * 100 // total
            text = f"  {filename}: {done / 1048576:6.1f} / {total / 1048576:6.1f} MB ({percent}%)"
        else:
            text = f"  {filename}: {done / 1048576:6.1f} MB"
        print(f"\r{text:<72}", end="", flush=True)

    _ = entry_label
    return show


def cmd_download(only: str | None, target: Path | None) -> int:
    entries = [models.MODELS[only]] if only else list(models.MODELS.values())
    failures = 0
    for entry in entries:
        existing = models.find(entry)
        if existing:
            print(f"{entry.label}：已就绪 {existing}")
            continue
        print(f"{entry.label}：开始下载（约 {_human(entry.size)}）")
        try:
            path = models.download(
                entry, target_dir=target, progress=_progress_printer(entry.label)
            )
            print(f"\r  {entry.filename} -> {path}")
        except models.ModelError as exc:
            failures += 1
            print(f"\r  {entry.filename} 失败：{exc}")
    return 1 if failures else 0


def cmd_import(archive: Path, target: Path | None) -> int:
    try:
        imported = models.import_archive(archive, target_dir=target)
    except models.ModelError as exc:
        print(f"导入失败：{exc}")
        return 1
    print("已导入：" + "、".join(imported))
    print(models.describe_status())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="获取图片转换器裁剪功能所需的本地模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--status", action="store_true", help="只显示模型状态")
    parser.add_argument("--check", action="store_true", help="只探测下载源可达性")
    parser.add_argument("--only", choices=sorted(models.MODELS), help="只处理指定模型")
    parser.add_argument("--target", type=Path, help="模型落点目录，默认用户缓存目录")
    parser.add_argument("--import", dest="archive", type=Path, metavar="ZIP", help="从压缩包离线导入")
    args = parser.parse_args(argv)

    if args.status:
        return cmd_status()
    if args.check:
        return cmd_check()
    if args.archive:
        return cmd_import(args.archive, args.target)
    return cmd_download(args.only, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
