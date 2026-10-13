"""崩溃取证与标准流重定向。

打包成 `--windowed` 的 exe 之后没有控制台，进程一旦崩掉（原生访问违例、
未捕获异常、子线程异常），外面什么都看不到 —— 用户只看到"闪退"。所以界面
模式启动时先做三件事：

1. **把标准流接进日志文件**（Python 层的 `sys.stdout/stderr` 与 C 层的 fd 1/2）。
   PyInstaller 的 windowed 模式下 `sys.stderr` 是 `None`、fd 2 也可能无效，
   而 onnxruntime 这类原生库会往 stderr 写警告（建会话时必定写一条形状推断
   警告）—— 往无效句柄写是"进程凭空消失"的常见来源。
2. 装上 `faulthandler`：真正的原生崩溃（段错误/访问违例）会留下 Python 调用栈。
3. 接管 `sys.excepthook` 与 `threading.excepthook`：主线程和后台线程的未捕获
   异常都写进同一个文件，而不是丢给一个不存在的 stderr。

日志：`%TEMP%\\图片裁剪器-运行日志.log`
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import traceback
from pathlib import Path

LOG_NAME = "图片裁剪器-运行日志.log"

_stream = None


def log_path() -> Path:
    """日志落点：优先临时目录，实在不行退回用户主目录。"""
    base = os.environ.get("TEMP") or os.environ.get("TMP") or os.path.expanduser("~")
    return Path(base) / LOG_NAME


def install() -> Path:
    """装好日志、崩溃栈与标准流重定向；返回日志路径。"""
    global _stream
    path = log_path()

    try:
        _stream = open(path, "a", encoding="utf-8", errors="replace")
    except OSError:
        _stream = None

    # 1) C 层的 fd 1/2 —— 原生库（onnxruntime 等）写的就是这两个
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if fd > 2:
            os.close(fd)
    except OSError:  # pragma: no cover - 权限异常时保持原样
        pass

    # 2) Python 层的标准流：windowed 打包下它们是 None，直接 print 就会炸
    if _stream is not None:
        if sys.stdout is None:
            sys.stdout = _stream
        if sys.stderr is None:
            sys.stderr = _stream

    # 3) 原生崩溃也要留栈
    try:
        faulthandler.enable(file=_stream if _stream is not None else sys.stderr)
    except Exception:  # noqa: BLE001
        pass

    # 4) 主线程 / 子线程的未捕获异常
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    write(f"\n===== 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    return path


def write(text: str) -> None:
    """往日志里写一行（日志不可用时静默跳过）。"""
    if _stream is None:
        return
    try:
        _stream.write(text + "\n")
        _stream.flush()
    except Exception:  # noqa: BLE001
        pass


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    write("\n[主线程未捕获异常]")
    if _stream is not None:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=_stream)
        _stream.flush()


def _thread_excepthook(args) -> None:
    name = getattr(args.thread, "name", "?")
    write(f"\n[子线程未捕获异常] {name}")
    if _stream is not None:
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=_stream)
        _stream.flush()


__all__ = ["install", "log_path", "write"]
