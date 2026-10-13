"""把绿色版产物打成 Release 资产（zip）。

前置：先跑过 tools/build.py（产物：图片裁剪器.exe + _internal）。

用法：
    python tools/make_release_zip.py

产物：
    dist\\release\\imgcrop-v<版本>-win64.zip
    内含 imgcrop-v<版本>-win64\\{图片裁剪器.exe, _internal\\, README.md}

版本号取自 imgcrop.__version__（唯一事实源）—— 发布时只需确认这个值
与即将打的 tag 一致即可，不必在两处手改。

注意：当前 exe 采用「模型内置」策略（build.py 把两个 onnx 打进 _internal/models），
所以 zip 会比较大（约 -30% 压缩后仍在百 MB 级）。「模型外置 + 首启指引
fetch_models.py」是待定案项（见 组织部/PIPELINE.md 阶段规划），定案后本脚本
需同步排除 _internal/models 并更新 README 分发说明。
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imgcrop import __version__  # noqa: E402

APP_NAME = "图片裁剪器"


def main() -> int:
    exe = ROOT / f"{APP_NAME}.exe"
    internal = ROOT / "_internal"
    readme = ROOT / "README.md"

    missing = [str(p) for p in (exe, internal, readme) if not p.exists()]
    if missing:
        print("缺少打包产物：" + "、".join(missing), file=sys.stderr)
        print("请先执行： python tools/build.py", file=sys.stderr)
        return 1

    folder = f"imgcrop-v{__version__}-win64"
    out_dir = ROOT / "dist" / "release"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{folder}.zip"

    print(f"打包 {folder}.zip …", flush=True)
    started = time.perf_counter()
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(exe, f"{folder}/{exe.name}")
        archive.write(readme, f"{folder}/{readme.name}")
        count += 2
        for path in sorted(internal.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    f"{folder}/_internal/{path.relative_to(internal).as_posix()}",
                )
                count += 1

    elapsed = time.perf_counter() - started
    size_mb = target.stat().st_size / 1048576
    print(f"\n完成：{target}")
    print(f"  {count} 个文件 / {size_mb:.0f} MB / 用时 {elapsed:.0f}s")
    print("  解压后进入该文件夹，双击 图片裁剪器.exe 即可使用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
