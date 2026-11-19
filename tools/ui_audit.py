# -*- coding: utf-8 -*-
"""UI 自检 · 几何断言层（已按裁剪器 gui 接口适配，P1-4）。

只做「读控件」的确定性检查 —— 这些是「看图」看不出来的东西：
  - 可点控件最小热区（WCAG 2.5.8 ≥24×24）
  - 同一 objectName 的按钮：字号 / 高度是否一致
  - 三栏 splitter：中栏画布与右栏参数卡的最小宽度断言
  - 空状态截图、浅/深两主题整窗截图落盘

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

from PySide6.QtWidgets import QAbstractButton, QApplication, QCheckBox  # noqa: E402

DEFAULT_OUT = REPO / ".workbuddy" / "tmp" / "ui_audit"
MIN_HIT = 24        # WCAG 2.5.8 最小目标
SIZE_EPS = 6        # 高度容差（px）：工具条/卡片的按钮本就分档，只抓明显失衡
CANVAS_MIN_WIDTH = 360   # gui/canvas.py：CropCanvas.setMinimumSize
SETTINGS_MIN_WIDTH = 372  # gui/main_window.py：右栏滚动区最小宽度

results: list[tuple[str, str, str]] = []   # (level, title, detail)


def add(level: str, title: str, detail: str) -> None:
    results.append((level, title, detail))


# ---------------------------------------------------------------- 建窗

def build_window(app: QApplication):
    from gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1320, 860)
    win.show()
    app.processEvents()
    return win


def grab_window(win, out: Path, name: str):
    shot = win.grab()
    shot.save(str(out / name))
    return shot


def check_themes(app: QApplication, win, out: Path) -> None:
    """浅/深两主题各截一张整窗图；断言主题切换链路（含 P0-3 强调色跟随）能走通。"""
    try:
        from qfluentwidgets import Theme, setTheme

        setTheme(Theme.LIGHT)
        app.processEvents()
        grab_window(win, out, "window_light.png")
        setTheme(Theme.DARK)
        app.processEvents()
        grab_window(win, out, "window_dark.png")
        setTheme(Theme.AUTO)
        app.processEvents()
        add("PASS", "双主题截图", "window_light.png / window_dark.png 已落盘")
    except Exception as exc:  # noqa: BLE001
        add("FAIL", "双主题截图", f"主题切换或截图失败：{type(exc).__name__}: {exc}")
        grab_window(win, out, "window_current.png")


# ---------------------------------------------------------------- 检查

def check_hit_area(win) -> None:
    bad = []
    for b in win.findChildren(QAbstractButton):
        if b.isHidden() or b.size().width() <= 0:
            continue
        if isinstance(b, QCheckBox) or "ArrowButton" in type(b).__name__:
            # 勾选框（整行标签都可点）与 qfw 下拉框的内部箭头不算独立功能按钮
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
        name = b.objectName()
        if not name:
            # 裁剪器没有 objectName 约定：匿名按钮的高度本来就按所在区分档
            #（工具条 34 / 清单与导出 36/38），不做跨区比较
            continue
        groups.setdefault(name, []).append(b)

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
    if issues:
        add("FAIL", "同类控件一致性", "；".join(issues))
    else:
        add("PASS", "同类控件一致性", "同 objectName 的字号/高度一致")


def check_three_column_layout(win) -> None:
    """三栏骨架：画布与右栏的最小宽度必须保住（否则输入框会被压扁）。"""
    issues = []
    canvas_min = win.canvas.minimumSize().width()
    if canvas_min < CANVAS_MIN_WIDTH:
        issues.append(f"画布最小宽 {canvas_min} < {CANVAS_MIN_WIDTH}")
    right = win.splitter.widget(2)
    right_min = right.minimumWidth()
    if 0 < right_min < SETTINGS_MIN_WIDTH:
        issues.append(f"右栏最小宽 {right_min} < {SETTINGS_MIN_WIDTH}")
    if win.splitter.childrenCollapsible():
        issues.append("splitter 允许拖到消失（childrenCollapsible 应为 False）")
    if issues:
        add("FAIL", "三栏布局骨架", "；".join(issues))
    else:
        add(
            "PASS",
            "三栏布局骨架",
            f"画布最小宽 {canvas_min}，右栏最小宽 {right_min}，splitter 不可折叠",
        )


def check_required_widgets(win) -> None:
    """1.1.0 结构断言：右栏三卡（矫正/取景/导出）的关键控件必须在场。"""
    required = {
        "check_tilt": "自动转正勾选",
        "slider_rotate": "旋转滑块（±180°）",
        "btn_quad": "透视拉平按钮",
        "btn_quad_undo": "撤销拉平按钮",
        "combo_aspect": "比例预设",
        "slider_margin": "留白滑块",
        "combo_candidate": "候选下拉",
        "edit_out": "输出目录",
        "combo_conflict": "同名策略",
        "btn_export": "导出全部",
        "btn_cancel": "取消任务按钮",
    }
    missing = [label for attr, label in required.items() if not hasattr(win, attr)]
    if missing:
        add("FAIL", "1.1.0 三卡控件", "缺少：" + "、".join(missing))
        return
    rot = win.slider_rotate
    if rot.slider.minimum() > -180 or rot.slider.maximum() < 180:
        add("FAIL", "1.1.0 三卡控件", f"旋转范围应为 ±180°，实际 {rot.slider.minimum()}~{rot.slider.maximum()}")
    else:
        add("PASS", "1.1.0 三卡控件", "矫正/取景/导出关键控件齐全，旋转 ±180°")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    win = build_window(app)

    # 先 grab 一次：强制完成布局与首帧绘制，之后再读几何/截图才准确
    grab_window(win, out, "window_empty.png")

    check_hit_area(win)
    check_same_role_consistency(win)
    check_three_column_layout(win)
    check_required_widgets(win)
    check_themes(app, win, out)

    win.close()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# UI 自检报告（几何断言层） · {stamp}", "",
             f"> 截图：`{out}`（window_empty / window_light / window_dark → 交给多模态 agent 读图评审）",
             "> 本报告只覆盖「齐不齐」；「好不好看」走人工/OpenDesign 评审。", ""]
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
