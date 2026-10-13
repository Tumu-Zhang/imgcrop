"""命令行：自动找出主体并批量裁剪照片。

没有界面时（或想批量脚本化时）用它，也是图形界面背后同一条链路的验证入口。

用法::

    python -m imgcrop 照片目录 --out cropped --aspect 4:3 --px 1600
    python -m imgcrop a.jpg b.jpg --aspect 1:1 --margin 0.1 --px 1600
    python -m imgcrop 照片目录 --dry-run          # 只看识别到什么，不写文件

尺寸/格式/体积上限与图形界面共用同一套 OutputSpec，所以命令行裁出来的图
与界面里导出的完全一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imgcrop import ingest
from imgcrop.crop import ASPECT_PRESETS, CropMode, CropSpec
from imgcrop.model import OutputFormat, OutputSpec, SizeMode
from imgcrop.pipeline import CropPlan, Cropper
from imgcrop.render import load_raster
from imgcrop.segment import IsnetSegmenter, SegmentError


def parse_aspect(text: str) -> float | None:
    """把 "1:1" / "4:3" / "none" 解析成宽高比。"""
    normalized = text.strip().lower()
    if normalized in {"none", "original", "auto", "原始"}:
        return None
    if ":" in normalized:
        left, right = normalized.split(":", 1)
        return float(left) / float(right)
    return float(normalized)


def build_output_spec(args) -> OutputSpec:
    if args.px is not None:
        size_mode, px_width, px_height = SizeMode.PIXELS, args.px, args.px_height
    elif args.px_width is not None or args.px_height is not None:
        size_mode, px_width, px_height = SizeMode.PIXELS, args.px_width, args.px_height
    else:
        raise SystemExit("请给出输出尺寸：--px 1600（长边）或 --px-width/--px-height")

    return OutputSpec(
        size_mode=size_mode,
        px_width=px_width,
        px_height=px_height,
        dpi=args.dpi,
        fmt=OutputFormat(args.format),
        keep_aspect=not args.exact_size,
        output_dir=args.out,
        name_template=args.name,
        max_bytes=int(args.limit * 1024 * 1024) if args.limit else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自动找出主体并裁剪照片")
    parser.add_argument("inputs", nargs="+", help="照片文件或目录（目录会递归展开）")
    parser.add_argument("--out", type=Path, help="输出目录；留空则落在源文件同级 cropped/")
    parser.add_argument("--name", default="{stem}_crop", help="命名模板，默认 {stem}_crop")
    parser.add_argument("--margin", type=float, default=0.08, help="主体四周留白比例，默认 0.08")
    parser.add_argument(
        "--aspect",
        default="none",
        help="目标比例：1:1 / 4:3 / 16:9 / none（跟随主体）",
    )
    parser.add_argument("--px", type=int, help="输出长边像素（配合 --aspect 可让整批尺寸完全一致）")
    parser.add_argument("--px-width", type=int, help="输出宽度（像素），与 --px 等价")
    parser.add_argument("--px-height", type=int, help="输出高度（像素），可选")
    parser.add_argument("--dpi", type=int, default=300, help="输出 DPI，默认 300")
    parser.add_argument("--format", default="png", choices=[f.value for f in OutputFormat])
    parser.add_argument("--limit", type=float, help="单张体积上限（MB）")
    parser.add_argument("--exact-size", action="store_true", help="输出严格等于指定像素尺寸")
    parser.add_argument("--rectify-tilt", action="store_true", help="按主体长轴自动校正倾斜")
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="手动旋转角（度，正 = 逆时针）；与 --rectify-tilt 叠加",
    )
    parser.add_argument("--dry-run", action="store_true", help="只报告检测到什么，不写文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    files = ingest.gather(args.inputs)
    if not files:
        print("没有找到可处理的图片。")
        return 1

    aspect = parse_aspect(args.aspect)
    try:
        segmenter = IsnetSegmenter()
        segmenter.resolve_path()
    except SegmentError as exc:
        print(f"无法开始：{exc}")
        return 1

    plans: dict[Path, CropPlan] = {}
    print(f"正在识别主体（{len(files)} 张）…")
    for path in files:
        try:
            loaded = load_raster(path)
        except Exception as exc:  # noqa: BLE001 - 单张失败不该中断整批
            print(f"  [跳过] {path.name}：{exc}")
            continue

        proposal = segmenter.saliency(loaded.image)
        best = proposal.best
        if best is None:
            print(f"  [跳过] {path.name}：没有检测到主体")
            continue

        plans[path] = CropPlan(
            spec=CropSpec(
                mode=CropMode.SUBJECT,
                subject=best,
                margin=args.margin,
                aspect=aspect,
                rectify_tilt=args.rectify_tilt,
                rotate_deg=args.rotate,
                origin="自动",
            ),
            mask=proposal.source_mask(),
        )
        width, height = loaded.image.size
        print(
            f"  {path.name}: {width}x{height} -> 主体 {best.width}x{best.height}"
            f" @ ({best.x1},{best.y1})  候选 {len(proposal.boxes)} 个"
        )

    if not plans:
        print("没有任何一张图识别出主体。")
        return 1

    if args.dry_run:
        print("\n--dry-run：未写出任何文件。")
        return 0

    spec = build_output_spec(args)
    print(f"\n开始裁剪输出（{len(plans)} 张）…")
    report = Cropper(spec, crop_provider=plans.get).run(list(plans))

    for result in report.results:
        if result.ok:
            print(f"  ✓ {result.source.name} -> {result.out_path}")
        else:
            print(f"  ✗ {result.source.name}：{result.error}")
        for warning in result.warnings:
            print(f"      · {warning}")

    print(f"\n完成：成功 {report.ok_count} 张，失败 {report.fail_count} 张。")

    sizes = report.pixel_sizes()
    if len(sizes) > 1:
        picked = "、".join(f"{w}x{h}" for w, h in sorted(sizes))
        print(
            f"注意：本次输出出现了 {len(sizes)} 种像素尺寸（{picked}）。"
            "这是因为各图主体形状略有差异 —— 指定 --aspect（例如 --aspect 4:3）"
            "可让整批尺寸完全一致。"
        )

    return 0 if report.fail_count == 0 else 1


__all__ = ["ASPECT_PRESETS", "build_output_spec", "build_parser", "main", "parse_aspect"]


if __name__ == "__main__":
    raise SystemExit(main())
