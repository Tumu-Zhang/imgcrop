"""图片裁剪器 —— 打包与直接运行用的入口。

双击运行、由 PyInstaller 打包成的 exe、或直接 `python main.py` 都以这个文件为入口：

    图片裁剪器.exe                    打开界面（也可以把图片/文件夹拖到 exe 上）
    图片裁剪器.exe --selftest         不开窗口，真跑一遍裁剪链路并校验（打包验收用）
    图片裁剪器.exe --smoke            开一次界面就自动退出（验收"双击能打开"）
    图片裁剪器.exe --gui-check 图片   开界面、按"拖进来"的路子处理这图片，跑完汇报
    图片裁剪器.exe --cli 照片目录 ...  当命令行用（等价于 python -m imgcrop）
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

SELFTEST_FLAG = "--selftest"
SMOKE_FLAG = "--smoke"
GUI_CHECK_FLAG = "--gui-check"
CLI_FLAG = "--cli"


def _use_utf8_output() -> None:
    """把标准输出切到 UTF-8。

    Windows 下 Python 的 stdout 默认按本地代码页（GBK）编码，命令行版一输出中文
    就成乱码（把结果重定向到文件时尤其明显）。自检与命令行都是给人看结果的，
    所以先统一成 UTF-8；万一没有 reconfigure 就维持原样，宁可乱码也不要抛异常。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 少见环境下的兜底
            pass


def _build_app():
    """建 QApplication 并把主题装配好 —— 界面模式的公共开头。"""
    from PySide6.QtWidgets import QApplication

    from gui import theme

    app = QApplication(sys.argv[:1])
    # 主题（Win11 浅/深色跟随系统）、微软雅黑、样式表与图标都在这里装好
    theme.apply(app)
    return app


def _smoke() -> int:
    """开一次界面然后自动退出。

    "双击能打开"是最难自动断言的一件事（窗口出来没有，程序自己看不见）。
    这个开关把界面完整装配一遍、显示出来，半秒后自行退出：能返回 0 就说明
    打包后的 exe 真的能在真实桌面上把窗口开起来。
    """
    _use_utf8_output()

    from PySide6.QtCore import QTimer

    from gui.main_window import MainWindow

    app = _build_app()
    window = MainWindow()
    window.show()
    QTimer.singleShot(800, app.quit)
    code = app.exec()
    print(f"[通过] 界面启动成功：{window.windowTitle()}")
    return code


def _gui_check(sources: list[Path]) -> int:
    """走一遍"把图片拖进界面"的完整路径，然后退出。

    这条路径是用户最常用的，也是唯一无法靠单元测试覆盖的：真实渲染 + 后台线程
    + 模型推理三件事同时在场。这里刻意**连界面模式的开机流程一起走**（装崩溃
    取证、重定向标准流），这样用"无控制台"的方式启动它，就等于复现双击。
    """
    _use_utf8_output()

    from gui import crashlog

    crashlog.install()
    crashlog.write("--gui-check 开始")

    from gui.main_window import MainWindow

    app = _build_app()
    window = MainWindow(paths=sources or None)
    window.show()

    deadline = time.time() + 300
    while window._scan_thread is not None and time.time() < deadline:  # noqa: SLF001
        app.processEvents()
        time.sleep(0.01)

    if window._scan_thread is not None:  # noqa: SLF001
        crashlog.write("[失败] 识别没有在 300 秒内结束")
        return 1

    photos = window._photos  # noqa: SLF001
    if not photos:
        crashlog.write("[失败] 没有加载到任何图片")
        return 1

    ok = True
    for photo in photos:
        # photo.status 里已经含「候选 N 个」，别再拼一次（否则日志会重复一遍）
        detail = photo.status or photo.error or "（未处理）"
        crashlog.write(f"  {photo.name}: {detail}；裁剪框 {photo.subject}")
        if photo.error:
            ok = False
    crashlog.write(f"[{'通过' if ok else '失败'}] 拖入路径走完，共 {len(photos)} 张")
    return 0 if ok else 1


def main() -> int:
    # 打包成 exe 后，若子进程被意外拉起，没有这行会反复启动新窗口
    multiprocessing.freeze_support()

    argv = sys.argv[1:]

    if SELFTEST_FLAG in argv:
        _use_utf8_output()
        from imgcrop.selftest import main as run_selftest

        return run_selftest()

    if SMOKE_FLAG in argv:
        return _smoke()

    if GUI_CHECK_FLAG in argv:
        return _gui_check([Path(item) for item in argv if not item.startswith("-")])

    if CLI_FLAG in argv:
        _use_utf8_output()
        from imgcrop.cli import main as run_cli

        return run_cli([item for item in argv if item != CLI_FLAG])

    # ---- 界面模式 ----
    # 先装崩溃取证与标准流重定向：打包后没有控制台，这是"闪退无痕"的唯一解药
    from gui import crashlog

    crashlog.install()

    # 剩下的位置参数当"拖到 exe 上的文件/文件夹"，界面启动后直接列进清单
    paths = [Path(item) for item in argv if not item.startswith("-")]

    from gui.app import main as run_app

    return run_app(paths=paths)


if __name__ == "__main__":
    sys.exit(main())
