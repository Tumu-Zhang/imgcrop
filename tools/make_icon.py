"""生成应用图标（icon.ico）。

图标就是界面里那枚套准标记：圆加贯穿的十字，深色方底配套准青。
和规格条两端用的是同一个图形，任务栏、窗口标题栏、文件夹里保持一致。

用法：
    python tools/make_icon.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui import theme  # noqa: E402

ICO_SIZES = (16, 20, 24, 32, 48, 64, 128, 256)


def _png_bytes(pixmap: QPixmap) -> bytes:
    """把 QPixmap 编码成 PNG 字节。

    QByteArray 必须先用名字绑定再交给 QBuffer：QBuffer 不接管它的所有权，
    传临时对象会得到一个悬空引用（实测直接段错误）。
    """
    storage = QByteArray()
    buffer = QBuffer(storage)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    buffer.close()
    return bytes(storage)


def _build_ico(frames: list[tuple[int, bytes]]) -> bytes:
    """手工拼 ICO 容器，每个尺寸内嵌一份精确渲染的 PNG。

    不用 Pillow 的 ICO 保存，是因为它会把最大帧重采样成小尺寸 ——
    16×16 上的圆加十字会糊成一团，而这个图标恰恰全靠那几条细线。
    """
    count = len(frames)
    header = struct.pack("<HHH", 0, 1, count)  # reserved, type=icon, count

    entries = bytearray()
    payload = bytearray()
    offset = 6 + 16 * count
    for size, png in frames:
        dimension = 0 if size >= 256 else size  # 256 在 ICO 里记作 0
        entries += struct.pack(
            "<BBBBHHII",
            dimension, dimension,
            0,     # 调色板颜色数（真彩为 0）
            0,     # 保留
            1,     # 颜色平面
            32,    # 位深
            len(png),
            offset + len(payload),
        )
        payload += png

    return bytes(header) + bytes(entries) + bytes(payload)


def main() -> int:
    # QPixmap 需要 QGuiApplication 存在；引用要活到函数结束，不能让它被回收
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    frames: list[tuple[int, bytes]] = []
    for size in ICO_SIZES:
        pixmap = theme.icon_pixmap(size)
        frames.append((size, _png_bytes(pixmap)))

    target = ROOT / "icon.ico"
    target.write_bytes(_build_ico(frames))
    print(
        f"已生成 {target}（{target.stat().st_size} bytes，"
        f"{len(ICO_SIZES)} 种尺寸：{', '.join(str(s) for s in ICO_SIZES)}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
