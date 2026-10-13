# -*- coding: utf-8 -*-
"""UI 自检 · 几何断言层。

> **移植状态：骨架已就位，尚未按裁剪器 gui 接口适配**（set_theme /
> build_stylesheet / objectName 约定）。与裁剪器 `gui/theme.py`、
> `gui/main_window.py` 的对接在 GUI 专业化（组织部/PIPELINE.md 阶段二）一并完成。

只做「读控件」的确定性检查 —— 这些是「看图」看不出来的东西：
  - 可点控件最小热区（WCAG 2.5.8 ≥24×24）
  - 同一 objectName 的按钮：字号 / 高度 / 菜单弹出模式是否一致
  - 顶栏同行按钮：高度与垂直中线是否对齐
同时把窗口截图落盘，供多模态 agent 做审美评审。

分工（勿重叠）：
  本脚本   → 「齐不齐」（确定性缺陷，读控件）
  人/agent → 「好不好看」（审美，读截图）

在项目根下执行：
    python tools/ui_audit.py
    python tools/ui_audit.py --out "<dir>"
退出码：0 = 无 FAIL；1 = 有 FAIL 项。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PySide6.QtCore import QPoint, QRect, QSettings  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QToolButton  # noqa: E402

from gui import theme  # noqa: E402

DEFAULT_OUT = REPO / ".workbuddy" / "tmp" / "ui_audit"
MIN_HIT = 24        # WCAG 2.5.8 最小目标
SIZE_EPS = 1        # 高度/中线容差（px）

results: list[tuple[str, str, str]] = []   # (level, title, detail)


def add(level: str, title: str, detail: str) -> None:
    results.append((level, title, detail))


# ---------------------------------------------------------------- 建窗

def build_window(app: QApplication, ini: Path):
    from gui import main_window as mw
    from gui.main_window import MainWindow

    mw.MainWindow._settings = lambda _self: QSettings(str(ini), QSettings.Format.IniFormat)
    theme.set_theme("light")
    app.setStyleSheet(theme.build_stylesheet("light"))

    win = MainWindow()
    win.resize(1190, 788)
    win.hide()
    try:
        win.layout().activate()
    except Exception:
        pass
    app.processEvents()
    return win


# ---------------------------------------------------------------- 检查

def check_hit_area(win) -> None:
    bad = []
    for b in win.findChildren(QAbstractButton):
        if b.isHidden() or b.size().width() <= 0:
            continue
        if b.width() < MIN_HIT or b.height() < MIN_HIT:
            bad.append(f"{b.objectName() or b.__class__.__name__} {b.width()}×{b.height()}")
    if bad:
        add("FAIL", "最小热区（≥24×24，WCAG 2.5.8）", "；".join(bad))
    else:
        add("PASS", "最小热区（≥24×24，WCAG 2.5.8）", "所有按钮达标")


def check_same_role_consistency(win) -> None:
    groups: dict[str, list[QAbstractButton]] = {}
    for b in win.findChildren(QAbstractButton):
        if b.isHidden():
            continue
        groups.setdefault(b.objectName() or f"<{b.__class__.__name__}>", []).append(b)

    issues = []
    for name, items in groups.items():
        if len(items) < 2:
            continue
        sizes = {round(i.font().pointSizeF(), 2) for i in items}
        if len(sizes) > 1:
            issues.append(f"{name} 字号不一致 {sorted(sizes)}")
        hs = {i.height() for i in items}
        if max(hs) - min(hs) > SIZE_EPS:
            issues.append(f"{name} 高度不一致 {sorted(hs)}")
        modes = {i.popupMode() for i in items if isinstance(i, QToolButton)}
        if len(modes) > 1:
            issues.append(f"{name} 菜单弹出模式不一致 {[m.name for m in modes]}"
                          f"（整块弹出 vs 分离箭头按钮，交互手感不同）")
    if issues:
        add("FAIL", "同类控件一致性", "；".join(issues))
    else:
        add("PASS", "同类控件一致性", "同 objectName 的字号/高度/弹出模式一致")


def check_topbar_alignment(win) -> None:
    row = [("导出日志", win.export_log_btn),
           ("反馈", win.feedback_btn),
           ("检查更新", win.update_btn)]
    hs = {n: w.height() for n, w in row}
    centers = {n: w.mapTo(win, QPoint(0, 0)).y() + w.height() / 2 for n, w in row}
    issues = []
    if max(hs.values()) - min(hs.values()) > SIZE_EPS:
        issues.append(f"高度不一致 {hs}")
    if max(centers.values()) - min(centers.values()) > SIZE_EPS:
        issues.append(f"垂直中线不对齐 {[f'{n}:{c:.1f}' for n, c in centers.items()]}")
    if issues:
        add("FAIL", "顶栏同行按钮对齐", "；".join(issues))
    else:
        add("PASS", "顶栏同行按钮对齐", f"高度 {list(hs.values())}，中线一致")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    ini = out / "_audit_settings.ini"
    win = build_window(app, ini)

    # 先 grab 一次：强制完成布局与首帧绘制，之后再读几何/截图才准确
    shot = win.grab()
    shot.save(str(out / "window_full.png"))
    shot.copy(QRect(0, 0, win.width(), min(120, win.height()))).save(str(out / "topbar.png"))

    check_hit_area(win)
    check_same_role_consistency(win)
    check_topbar_alignment(win)

    win.close()
    ini.unlink(missing_ok=True)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# UI 自检报告（几何断言层） · {stamp}", "",
             f"> 截图：`{out}`（`topbar.png` / `window_full.png` → 交给多模态 agent 或 OpenDesign 读图评审）",
             "> 本报告只覆盖「齐不齐」；「好不好看」走 OpenDesign design-review。", ""]
    for level, title, detail in results:
        mark = {"PASS": "✓", "FAIL": "✗"}[level]
        lines.append(f"- {mark} **{title}** — {detail}")
    fails = sum(1 for lv, *_ in results if lv == "FAIL")
    lines += ["", f"**汇总**：FAIL {fails} · PASS {len(results) - fails}"]
    report = "\n".join(lines)
    (out / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\n报告与截图已写入：{out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
