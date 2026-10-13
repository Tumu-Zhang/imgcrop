"""一键打包成双击即用的「图片裁剪器.exe」。

做的事：
1. 用 PyInstaller 打一个 onedir 发行包，**把两个 onnx 模型（约 90 MB）内置进去**，
   所以用户机器上不需要联网、不需要再装模型；
2. 把 exe、运行库与 README 从 dist/ 提到项目根目录，打开文件夹就能看到；
3. 立刻跑一次 exe 的 `--selftest`：真跑一遍裁剪链路 + 模型推理，
   确保"打出来的东西不是个只能启动、一用就崩的空壳"。

用法：
    python tools/build.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "图片裁剪器"
MODELS_DIR = ROOT / "models"

# 这些包本身不是本工具的依赖，但环境里装了它们，PyInstaller 会顺着某些可选导入
# 把它们整个拖进来（光是 torch 就有 364 MB）。全部排除。
#
# 注意：onnxruntime **不能**排除 —— 它正是跑分割模型要用的。
EXCLUDES = (
    "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "IPython",
    "pytest", "torch", "torchvision", "torchaudio", "pyarrow", "scipy",
    "pandas", "cryptography", "sklearn", "cv2", "numba", "sympy", "h5py",
    "sqlalchemy", "lxml", "wx", "notebook", "jupyter", "setuptools",
    # 转换器时代留下的、裁剪器完全不用的东西
    "fitz", "pymupdf", "pptx",
    # 注意：win32api / win32con / pythoncom / win32com **不能**排除 ——
    # qfluentwidgets 的按钮会 import qframelesswindow，而后者在 Windows 上靠
    # win32api 拖动无边框窗口。这四个一排除，打出来的 exe 一双击就
    # ModuleNotFoundError: No module named 'win32api'。
)

HIDDEN_IMPORTS = ("onnxruntime", "PIL.ImageQt", "qframelesswindow", "darkdetect")

# 需要连资源一起收的包：qfluentwidgets 的图标/主题是编译进 qrc 的二进制资源，
# 只写 --hidden-import 会打出一个「控件在、图标全空」的包，必须 --collect-all。
COLLECT_ALL = ("qfluentwidgets",)

# ----------------------------------------------------------------------
# 打包后精简（PRUNE）：删掉对纯 Widgets 桌面应用完全无用的运行库
# ----------------------------------------------------------------------
# 支持通配符（相对 _internal/）。清单继承图片转换器的逐项 PE 导入表核对结论
# （206.6MB → 136.2MB 同款钩子），并按裁剪器差异调整：
#   · **保留** PySide6 的 Qt6Svg / qsvg / qsvgicon —— qfluentwidgets 的
#     FluentIcon 走 svg 渲染，删了图标全空；
#   · **保留** PIL._avif —— 裁剪器是图片工具，AVIF 输入可能用得上（待定）；
#   · **保留** libcrypto-1_1 / libssl-1_1 —— CPython 的 _ssl/_hashlib 直接
#     依赖，删了运行必崩（转换器反例留档）；
#   · 删的是：Mesa 软件 OpenGL、QML/Quick/虚拟键盘、QtPdf、QtNetwork 及其
#     TLS 后端（libcrypto-3/libssl-3 只被它引用）、Qt 自带翻译（没用
#     QTranslator，.qm 全不会被加载）、Pythonwin（win32ui 的 GUI 壳）。
# 删完的关卡是下面的 --selftest（真跑裁剪 + 模型推理）。
PRUNE = (
    # Mesa 软件 OpenGL 兜底：本程序全程光栅绘制，从不建 GL 上下文
    "PySide6/opengl32sw.dll",
    # QML / Quick / 虚拟键盘：纯 Widgets 应用，全仓无一处 QML
    "PySide6/Qt6Quick.dll",
    "PySide6/Qt6Qml.dll",
    "PySide6/Qt6QmlModels.dll",
    "PySide6/Qt6QmlMeta.dll",
    "PySide6/Qt6QmlWorkerScript.dll",
    "PySide6/Qt6VirtualKeyboard.dll",
    # Qt6OpenGL 只被 Qt6Quick 引用（Quick 已删）
    "PySide6/Qt6OpenGL.dll",
    "PySide6/plugins/platforminputcontexts",
    # Qt PDF：裁剪器不处理 PDF
    "PySide6/Qt6Pdf.dll",
    "PySide6/plugins/imageformats/qpdf.dll",
    # Qt Network：本程序不联网（模型获取是打包前的独立脚本 fetch_models.py）
    "PySide6/Qt6Network.dll",
    "PySide6/QtNetwork.pyd",
    "PySide6/plugins/tls",
    "PySide6/plugins/networkinformation",
    "PySide6/plugins/generic/qtuiotouchplugin.dll",
    # Qt 自带翻译：没装 QTranslator，一个 .qm 都不会被加载
    "PySide6/translations",
    # Pythonwin：win32ui 的 GUI 壳（win32api/win32con 必须保留，勿动）
    "Pythonwin",
    # OpenSSL 3：只被上面已删的 QtNetwork TLS 后端引用
    #（注意 libcrypto-1_1 / libssl-1_1 是 _ssl/_hashlib 的依赖，必须保留）
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll",
    # --add-data 的副产物：README 在项目根已有一份
    "README.md",
)

# 随包带上的资源：模型 + 使用说明 + 图标（窗口图标是运行时从包里取的）
DATA_FILES = (
    (MODELS_DIR, "models"),
    (ROOT / "README.md", "."),
    (ROOT / "icon.ico", "."),
)


def _missing_models() -> list[str]:
    """列出 models/ 下的 onnx —— 少于两个时调用方会拒绝打包。"""
    if not MODELS_DIR.is_dir():
        return ["（models 目录不存在）"]
    return sorted(p.name for p in MODELS_DIR.glob("*.onnx"))


def _measure(root: Path) -> tuple[int, int]:
    """统计目录里的文件数与总字节。"""
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def _mv_to_trash(target: Path, trash: Path) -> None:
    """把旧产物 mv 进临时区（**不要 rmtree**）。

    批量删除会被安全策略整体拦截，rmtree(ignore_errors=True) 被拦时还会
    **静默失败**（手册 §1.8 实测坑）。trash 区在 build/ 下，已被 .gitignore，
    手动清理或下次打包覆盖即可。
    """
    if not target.exists():
        return
    dst = trash / target.name
    if dst.exists():  # 上一次的垃圾还在：换个带时间戳的名字，绝不覆盖
        dst = trash / f"{target.name}.{int(time.time())}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(dst))


def _prune(internal: Path) -> tuple[int, list[str], list[str]]:
    """按 PRUNE 清掉用不上的运行库，返回（实际省下的字节, 已处理项, 缺失项）。

    **目录不 rmtree，一律 mv 进 build/_prune_trash/**：批量删除会被安全策略
    整体拦截、rmtree(ignore_errors=True) 被拦时还会**静默失败**（手册 §1.8
    实测坑）。mv 是瞬时操作，trash 区随 build/ 被 .gitignore、下次打包覆盖。
    省下的字节按**处理前后实测差值**报告，而不是把命中项的预期体积加起来 ——
    这样万一某个目录被占用而没挪走，报告里立刻看得出来（不会假装成功）。
    缺失项不算错：本来就不存在或已被 exclude 挡在收集阶段。
    """
    trash = ROOT / "build" / "_prune_trash"
    before = _measure(internal)[1]
    handled: list[str] = []
    missing: list[str] = []
    for pattern in PRUNE:
        hits = sorted(internal.glob(pattern))
        if not hits:
            missing.append(pattern)
            continue
        for target in hits:
            rel = target.relative_to(internal)
            if target.is_dir():
                size = _measure(target)[1]
                dst = trash / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(dst))
            else:
                size = target.stat().st_size
                target.unlink(missing_ok=True)
            if target.exists():
                print(f"  ! 清理失败（被占用？）：{rel}")
            handled.append(f"{rel}  {size / 1048576:.1f} MB")
    after = _measure(internal)[1]
    return max(0, before - after), handled, missing


def main() -> int:
    try:
        import PyInstaller.__main__
    except ImportError:
        print("没有安装 PyInstaller。请执行： pip install pyinstaller", file=sys.stderr)
        return 1

    models = _missing_models()
    if len(models) < 2:
        print(
            "models/ 下缺少模型文件（应有 ISNet 与 EfficientSAM 两个 onnx）。"
            "先运行 python tools/fetch_models.py",
            file=sys.stderr,
        )
        return 1
    print("将内置模型：" + "、".join(models))

    dist_dir = ROOT / "dist" / APP_NAME
    trash = ROOT / "build" / "_old_trash"
    # 预清理：PyInstaller --noconfirm 会整目录删掉旧 dist 产物，
    # 批量删除可能被安全策略整体拦截 → 先 mv 进临时区
    _mv_to_trash(dist_dir, trash)

    icon = ROOT / "icon.ico"
    if not icon.exists():
        print("缺少 icon.ico，正在生成…")
        from tools.make_icon import main as make_icon

        if make_icon() != 0:
            return 1

    args = [
        "--noconfirm",
        "--windowed",
        "--name", APP_NAME,
        "--icon", str(icon),
    ]
    for source, target in DATA_FILES:
        if Path(source).exists():
            args += ["--add-data", f"{source}{';'}{target}"]
    for name in COLLECT_ALL:
        args += ["--collect-all", name]
    for name in HIDDEN_IMPORTS:
        args += ["--hidden-import", name]
    for name in EXCLUDES:
        args += ["--exclude-module", name]
    args.append(str(ROOT / "main.py"))

    print("开始打包（内置 90 MB 模型，通常需要 1-3 分钟）…", flush=True)
    started = time.perf_counter()
    PyInstaller.__main__.run(args)
    elapsed = time.perf_counter() - started

    dist_dir = ROOT / "dist" / APP_NAME
    exe = dist_dir / f"{APP_NAME}.exe"
    if not exe.exists():
        print("打包似乎失败了：找不到 exe", file=sys.stderr)
        return 1

    # 打包后精简：清掉 hook 硬塞进来的无用运行库（见 PRUNE 上方核对注释）
    internal_dist = dist_dir / "_internal"
    if internal_dist.is_dir():
        saved, handled, missing = _prune(internal_dist)
        print(f"\n精简完成，实测省下 {saved / 1048576:.1f} MB：")
        for item in handled:
            print(f"  - {item}")
        if missing:
            print(f"  （{len(missing)} 项本来就不存在，跳过）")

    # 把说明复制到产物根目录 —— 用户打开文件夹第一眼就能看到
    readme = ROOT / "README.md"
    if readme.exists():
        shutil.copy2(readme, dist_dir / "README.md")

    # 用户要求「双击就能找到，别藏太深」：把 exe 与运行库从 dist/ 提到项目根
    for name in (f"{APP_NAME}.exe", "_internal", "README.md"):
        source = dist_dir / name
        if not source.exists():
            continue
        target = ROOT / name
        # 旧产物 mv 进临时区（别 rmtree：批量删除会被拦截且可能静默失败）
        _mv_to_trash(target, trash)
        shutil.move(str(source), str(target))
    shutil.rmtree(ROOT / "dist", ignore_errors=True)

    exe = ROOT / f"{APP_NAME}.exe"
    size_mb = sum(
        f.stat().st_size
        for base in (exe, ROOT / "_internal")
        if base.exists()
        for f in ([base] if base.is_file() else base.rglob("*"))
        if f.is_file()
    ) / 1048576

    print(f"\n打包完成，用时 {elapsed:.0f}s")
    print(f"  体积：{size_mb:.0f} MB")
    print(f"  双击：{exe}")

    # 打包不等于能用：这里真跑一次，确认模型进包了、onnxruntime 也能加载
    print("\n正在自检打出来的 exe（真跑一次裁剪 + 模型推理）…", flush=True)
    try:
        result = subprocess.run(
            [str(exe), "--selftest"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"自检无法执行：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(result.stdout or result.stderr)
    if result.returncode != 0:
        print("自检未通过 —— 这个包还不能交付。", file=sys.stderr)
        return 1

    print("\n提示：exe 必须和同级的 _internal 文件夹一起拷走才能用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
