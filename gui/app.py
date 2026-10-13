"""图形界面入口。

`python main.py` 与打包后的 exe 都从这里进 —— 界面之外没有别的东西，
所以"双击能用"这件事只需要保证这个函数跑得起来。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None, paths: list[Path] | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    from gui import theme
    from gui.main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("图片裁剪器")
    app.setOrganizationName("imgcrop")

    # Fusion 打底再叠自定义样式表：Windows 原生样式对 QSS 的支持不完整，
    # 下拉箭头、勾选框这类子控件在原生样式下会跟主题打架。
    app.setStyle("Fusion")

    # 字体、样式表、Fluent 控件主题（含跟随系统深浅色）统一在 theme 里装
    theme.apply(app)

    window = MainWindow(paths=list(paths) if paths else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
