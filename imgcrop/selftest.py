"""打包后自检：不开窗口，真跑一遍裁剪链路，确认程序"还能干活"。

打包最容易坏的两件事是「资源没带上」和「原生库没配好」—— 它们都不会在
源码运行时报错。所以这里刻意跑真实流程：合成图 -> 自动主体框 -> 裁剪 ->
1:1 落盘 -> 回读校验，最后再确认模型与 onnxruntime 真的可用。

用法::

    图片裁剪器.exe --selftest

任何一步不通过都返回非 0，并打印到底哪一步断了。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from imgcrop.crop import CropMode, CropSpec
from imgcrop.geometry import Box, mask_to_box
from imgcrop.model import OutputSpec
from imgcrop.pipeline import CropPlan, Cropper

# 合成图的构图：白底 + 居中的深色主体块，模拟"浅底上放着一个样品"
CANVAS = (1200, 800)
SUBJECT = Box(400, 250, 800, 550)
SUBJECT_COLOR = (40, 40, 40)
BACKGROUND = (255, 255, 255)


class CheckFailed(RuntimeError):
    """自检失败。"""


def _make_sample(path: Path) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    for x in range(SUBJECT.x1, SUBJECT.x2):
        for y in range(SUBJECT.y1, SUBJECT.y2):
            image.putpixel((x, y), SUBJECT_COLOR)
    image.save(path)


def _check_crop(workdir: Path) -> str:
    """裁剪 -> 1:1 落盘 -> 回读校验。

    1.1.0 起输出不再重采样：成品尺寸 = 裁剪窗尺寸 = 主体 x (1+2*margin)。
    """
    source = workdir / "sample.png"
    _make_sample(source)

    spec = OutputSpec(
        output_dir=workdir / "out",
        name_template="{stem}_crop",
    )
    plan = CropPlan(
        spec=CropSpec(mode=CropMode.SUBJECT, subject=SUBJECT, margin=0.08, origin="自检")
    )
    report = Cropper(spec, crop_provider=lambda _p: plan).run([source])
    if report.fail_count or not report.ok_count:
        failure = report.results[0] if report.results else None
        raise CheckFailed(f"裁剪失败：{getattr(failure, 'error', '没有任何结果')}")

    out_path = report.results[0].out_path
    assert out_path is not None
    if not out_path.is_file():
        raise CheckFailed(f"没有生成输出文件：{out_path}")

    expected = (round(400 * 1.16), round(300 * 1.16))  # 464 x 348
    with Image.open(out_path) as produced:
        produced.load()
        width, height = produced.size
        center = produced.convert("RGB").getpixel((width // 2, height // 2))
        dpi = produced.info.get("dpi")
        mode = produced.mode

    if out_path.suffix != ".png":
        raise CheckFailed(f"PNG 源应输出 PNG（格式跟随源），实际 {out_path.suffix}")
    if (width, height) != expected:
        raise CheckFailed(f"输出应为裁剪窗原像素 {expected}，实际 {width}x{height}")
    # 主体应当被裁到画面正中：中心像素必须是主体色而不是背景色
    if sum(abs(a - b) for a, b in zip(center, SUBJECT_COLOR)) > 60:
        raise CheckFailed(f"画面中心不是主体（取到 {center}，应为 {SUBJECT_COLOR} 附近）")
    if not dpi or abs(float(dpi[0]) - 300) > 1:
        raise CheckFailed(f"DPI 元数据应缺省 300，实际 {dpi}")
    if mode != "RGB":
        raise CheckFailed(f"不带透明的源应输出 RGB，实际 {mode}")

    return f"输出 {width}x{height}（1:1）@ {dpi or '无 dpi 元数据'}"


def _check_format_follows_source(workdir: Path) -> str:
    """D6：JPG 源 -> JPG 输出。"""
    source = workdir / "sample.jpg"
    _make_sample(source)

    spec = OutputSpec(output_dir=workdir / "out", name_template="{stem}_fmt")
    plan = CropPlan(
        spec=CropSpec(mode=CropMode.SUBJECT, subject=SUBJECT, margin=0.08, origin="自检")
    )
    report = Cropper(spec, crop_provider=lambda _p: plan).run([source])
    if report.fail_count or not report.ok_count:
        failure = report.results[0] if report.results else None
        raise CheckFailed(f"JPG 裁剪失败：{getattr(failure, 'error', '没有任何结果')}")

    out_path = report.results[0].out_path
    assert out_path is not None
    if out_path.suffix != ".jpg":
        raise CheckFailed(f"JPG 源应输出 JPG，实际 {out_path.suffix}")
    return "JPG 源 -> JPG 输出"


def _check_transparency(workdir: Path) -> str:
    """D6：带透明的 PNG 源 -> PNG 输出保留透明通道。"""
    source = workdir / "alpha.png"
    Image.new("RGBA", (200, 160), (200, 30, 60, 120)).save(source)

    spec = OutputSpec(output_dir=workdir / "out", name_template="{stem}_alpha")
    plan = CropPlan(
        spec=CropSpec(mode=CropMode.FREE, subject=Box(10, 10, 190, 150), origin="自检")
    )
    report = Cropper(spec, crop_provider=lambda _p: plan).run([source])
    if report.fail_count or not report.ok_count:
        failure = report.results[0] if report.results else None
        raise CheckFailed(f"透明图裁剪失败：{getattr(failure, 'error', '没有任何结果')}")

    out_path = report.results[0].out_path
    assert out_path is not None
    with Image.open(out_path) as produced:
        produced.load()
        mode = produced.mode
    if mode != "RGBA":
        raise CheckFailed(f"带透明的 PNG 源应输出 RGBA（保透明），实际 {mode}")
    return "PNG 透明通道保留"


def _check_quad(workdir: Path) -> str:
    """D9 融合管线：quad 正交前置 + mode=NONE 时拉平结果就是成品。"""
    source = workdir / "quad.png"
    Image.new("RGB", (600, 400), (200, 210, 220)).save(source)

    spec = OutputSpec(
        output_dir=workdir / "out",
        name_template="{stem}_quad",
    )
    plan = CropPlan(
        spec=CropSpec(
            mode=CropMode.NONE,
            quad=[(0.0, 0.0), (600.0, 0.0), (600.0, 400.0), (0.0, 400.0)],
            quad_size=(300, 200),
        )
    )
    report = Cropper(spec, crop_provider=lambda _p: plan).run([source])
    if report.fail_count or not report.ok_count:
        failure = report.results[0] if report.results else None
        raise CheckFailed(f"四点矫正失败：{getattr(failure, 'error', '没有任何结果')}")

    out_path = report.results[0].out_path
    assert out_path is not None
    with Image.open(out_path) as produced:
        produced.load()
        size = produced.size
    if size != (300, 200):
        raise CheckFailed(f"四点矫正输出应为 300x200，实际 {size[0]}x{size[1]}")
    return f"四点矫正输出 {size[0]}x{size[1]}"


def _check_rotation(workdir: Path) -> str:
    """旋转后裁剪：画面转了，主体也得跟着转正（不是只把框挪一下）。"""
    source = workdir / "tilted.png"
    layer = Image.new("L", CANVAS, 0)
    ImageDraw.Draw(layer).rectangle([300, 370, 900, 430], fill=255)
    layer = layer.rotate(12.0, resample=Image.Resampling.BICUBIC, fillcolor=0)
    image = Image.new("RGB", CANVAS, BACKGROUND)
    image.paste(SUBJECT_COLOR, mask=layer)
    image.save(source)

    subject = mask_to_box(np.asarray(layer) > 127)
    if subject is None:
        raise CheckFailed("合成倾斜主体失败（没得到掩膜框）")

    spec = OutputSpec(
        output_dir=workdir / "out",
        name_template="{stem}_straight",
    )
    plan = CropPlan(
        spec=CropSpec(
            mode=CropMode.SUBJECT,
            subject=subject,
            margin=0.1,
            rotate_deg=-12.0,
            origin="自检",
        )
    )
    report = Cropper(spec, crop_provider=lambda _p: plan).run([source])
    if report.fail_count or not report.ok_count:
        failure = report.results[0] if report.results else None
        raise CheckFailed(f"旋转裁剪失败：{getattr(failure, 'error', '没有任何结果')}")

    result = report.results[0]
    out_path = result.out_path
    assert out_path is not None
    if not any("旋转" in warning for warning in result.warnings):
        raise CheckFailed(f"结果里没说明旋转过：{result.warnings}")

    with Image.open(out_path) as produced:
        produced.load()
        array = np.asarray(produced.convert("L"))

    dark = array < 128  # 主体是深色块
    if not dark.any():
        raise CheckFailed("输出里找不到主体")
    rows = np.flatnonzero(dark.any(axis=1))
    cols = np.flatnonzero(dark.any(axis=0))
    width = int(cols[-1] - cols[0] + 1)
    height = int(rows[-1] - rows[0] + 1)
    ratio = width / max(1, height)
    # 原来的细长条是 600x60（约 10:1），歪 12° 时包围盒会变成约 3:1。
    # 转正之后必须还是一条细长条 —— 比例小就说明只是挪了框、没转画面。
    if ratio < 6.0:
        raise CheckFailed(f"画面没被转正：输出里主体的长宽比只有 {ratio:.1f}（应接近 10）")

    return f"按 -12° 转正后主体仍是一条 {width}x{height} 的细长条"


def _check_auto_quad(workdir: Path) -> str:
    """UI-007 自动选角：斜矩形掩膜 → 四角估计 → 拉平输出与真值一致。"""
    source = workdir / "tilted_doc.png"
    layer = Image.new("L", (800, 600), 0)
    ImageDraw.Draw(layer).rectangle([200, 150, 600, 450], fill=255)
    layer = layer.rotate(10.0, resample=Image.Resampling.BICUBIC, fillcolor=0)
    image = Image.new("RGB", (800, 600), (48, 52, 58))
    image.paste((236, 232, 224), mask=layer)
    image.save(source)
    mask = np.asarray(layer) > 127

    from imgcrop.geometry import quad_from_mask, quad_output_size

    quad = quad_from_mask(mask)
    if quad is None:
        raise CheckFailed("自动选角没能从斜矩形掩膜估出四边形")
    expected = quad_output_size(list(quad))
    if abs(expected[0] - 400) > 8 or abs(expected[1] - 300) > 8:
        raise CheckFailed(
            f"自动估出的拉平尺寸 {expected} 偏离真值矩形 400x300 太远"
        )

    plan = CropPlan(spec=CropSpec(mode=CropMode.NONE, quad=quad))
    report = Cropper(
        OutputSpec(output_dir=workdir / "out", name_template="{stem}_auto"),
        crop_provider=lambda _p: plan,
    ).run([source])
    if report.fail_count or not report.ok_count:
        failure = report.results[0] if report.results else None
        raise CheckFailed(f"自动选角拉平失败：{getattr(failure, 'error', '没有任何结果')}")
    out_path = report.results[0].out_path
    assert out_path is not None
    with Image.open(out_path) as produced:
        produced.load()
        size = produced.size
    if size != expected:
        raise CheckFailed(f"拉平输出 {size} 与估计尺寸 {expected} 不一致")
    return f"斜矩形自动选角 -> 拉平输出 {size[0]}x{size[1]}"


def _check_models() -> str:
    """确认模型文件与 onnxruntime 在打包后依然可用。"""
    from imgcrop import models
    from imgcrop.segment import IsnetSegmenter, SegmentError

    missing = models.missing()
    if missing:
        return "跳过（缺少模型：" + "、".join(entry.key for entry in missing) + "）"

    segmenter = IsnetSegmenter()
    try:
        segmenter.resolve_path()
        segmenter.session  # 真正建一次会话：onnxruntime 的原生库在这一步才会被加载
    except SegmentError as exc:
        raise CheckFailed(f"模型不可用：{exc}") from exc

    sample = Image.new("RGB", CANVAS, BACKGROUND)
    for x in range(SUBJECT.x1, SUBJECT.x2, 2):
        for y in range(SUBJECT.y1, SUBJECT.y2, 2):
            sample.putpixel((x, y), SUBJECT_COLOR)

    proposal = segmenter.saliency(sample)
    if not proposal.boxes:
        return "onnxruntime 与模型可用，但合成图上没有候选（不算失败）"
    return f"模型可用，合成图上给出 {len(proposal.boxes)} 个候选"


def main(argv: list[str] | None = None) -> int:
    _ = argv
    print("图片裁剪器 自检")
    print("=" * 40)
    steps = (
        ("裁剪 -> 1:1 落盘 -> 回读校验", _check_crop),
        ("格式跟随源（JPG -> JPG）", _check_format_follows_source),
        ("PNG 透明通道保留", _check_transparency),
        ("四点透视矫正", _check_quad),
        ("自动选角（掩膜 -> 四角 -> 拉平）", _check_auto_quad),
        ("旋转后裁剪", _check_rotation),
    )

    with tempfile.TemporaryDirectory(prefix="imgcrop-selftest-") as raw:
        workdir = Path(raw)
        for label, step in steps:
            try:
                detail = step(workdir)
            except CheckFailed as exc:
                print(f"[失败] {label}：{exc}")
                return 1
            except Exception as exc:  # noqa: BLE001 - 自检要报出真实原因，不能吞
                print(f"[失败] {label}：{type(exc).__name__}: {exc}")
                return 1
            print(f"[通过] {label} —— {detail}")

        try:
            detail = _check_models()
        except CheckFailed as exc:
            print(f"[失败] 模型与推理：{exc}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] 模型与推理：{type(exc).__name__}: {exc}")
            return 1
        print(f"[通过] 模型与推理 —— {detail}")

    print("=" * 40)
    print("自检全部通过。")
    return 0


__all__ = ["CheckFailed", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
