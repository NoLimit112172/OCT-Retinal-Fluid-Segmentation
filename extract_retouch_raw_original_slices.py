
"""
extract_retouch_raw_original_slices.py

从 RETOUCH 原始 oct.raw / oct.mhd 中导出多张未处理 OCT B-scan 图像，并生成总览拼图。
不会修改原始数据，只会在 OUTPUT_DIR 下生成新的 PNG 文件。

数据根目录：
D:\Graduation_Design\retouch

运行：
cd /d D:\Graduation_Design\OCT_Segmentation
.venv\Scripts\python.exe extract_retouch_raw_original_slices.py
"""

import os
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_ROOT = Path(r"D:\Graduation_Design\retouch")
OUTPUT_DIR = Path(r"D:\Graduation_Design\原始图像展示\RETOUCH")

SLICES_PER_VOLUME = 6
MAX_TOTAL_SLICES_FOR_SHEET = 30
THUMB_SIZE = (160, 120)
SAVE_INDIVIDUAL_SLICES = True


def parse_mhd(mhd_path: Path):
    info = {}
    with open(mhd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def mhd_dtype(element_type: str):
    mapping = {
        "MET_UCHAR": np.uint8,
        "MET_CHAR": np.int8,
        "MET_USHORT": np.uint16,
        "MET_SHORT": np.int16,
        "MET_UINT": np.uint32,
        "MET_INT": np.int32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    if element_type not in mapping:
        raise ValueError(f"暂不支持 ElementType: {element_type}")
    return mapping[element_type]


def normalize_to_uint8(img):
    """
    仅用于保存 PNG 可视化，不会改动原始 raw 文件。
    用 1%~99% 百分位拉伸，避免直接保存全黑。
    """
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - lo) / (hi - lo)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def read_mhd_raw_volume(mhd_path: Path):
    """
    读取 oct.mhd + oct.raw。
    MetaImage DimSize 通常为 X Y Z，这里 reshape 为 [Z, Y, X]。
    """
    info = parse_mhd(mhd_path)

    if "DimSize" not in info:
        raise ValueError(f"{mhd_path} 缺少 DimSize")
    if "ElementType" not in info:
        raise ValueError(f"{mhd_path} 缺少 ElementType")

    dim = [int(x) for x in info["DimSize"].split()]
    if len(dim) != 3:
        raise ValueError(f"{mhd_path} DimSize 不是三维: {dim}")

    x, y, z = dim
    dtype = mhd_dtype(info["ElementType"])

    raw_name = info.get("ElementDataFile", "oct.raw")
    raw_path = mhd_path.parent / raw_name
    if not raw_path.exists():
        raw_path = mhd_path.parent / "oct.raw"
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到 raw 文件: {raw_path}")

    data = np.fromfile(raw_path, dtype=dtype)
    expected = x * y * z
    if data.size != expected:
        raise ValueError(
            f"raw 大小不匹配: {raw_path}, 实际={data.size}, 期望={expected}, DimSize={dim}"
        )

    volume = data.reshape((z, y, x))
    return volume, info, raw_path


def infer_device(path: Path):
    s = str(path).lower()
    if "cirrus" in s:
        return "Cirrus"
    if "spectralis" in s:
        return "Spectralis"
    if "topcon" in s:
        return "Topcon"
    return "UnknownDevice"


def select_indices(n, k):
    if n <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [n // 2]
    start = int(n * 0.15)
    end = int(n * 0.85)
    if end <= start:
        start, end = 0, n - 1
    return np.linspace(start, end, k).astype(int).tolist()


def make_contact_sheet(records, save_path: Path, title: str):
    if not records:
        print("没有可生成总览图的切片。")
        return

    cols = 5
    rows = math.ceil(len(records) / cols)
    tw, th = THUMB_SIZE
    label_h = 24
    pad = 18
    title_h = 42

    w = cols * tw + (cols + 1) * pad
    h = title_h + rows * (th + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_label = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    draw.text((pad, 10), title, fill=(0, 0, 0), font=font_title)

    for i, rec in enumerate(records):
        r = i // cols
        c = i % cols
        x0 = pad + c * (tw + pad)
        y0 = title_h + pad + r * (th + label_h + pad)

        img = rec["img"].convert("L").resize(THUMB_SIZE, Image.Resampling.BILINEAR)
        sheet.paste(Image.merge("RGB", (img, img, img)), (x0, y0))

        label = rec["label"][:24]
        draw.text((x0 + 4, y0 + th + 4), label, fill=(0, 0, 0), font=font_label)

    sheet.save(save_path)
    print(f"总览图已保存: {save_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mhd_files = sorted(DATA_ROOT.rglob("oct.mhd"))
    if not mhd_files:
        mhd_files = sorted(DATA_ROOT.rglob("*.mhd"))

    if not mhd_files:
        print(f"未找到 .mhd 文件: {DATA_ROOT}")
        print("说明：RETOUCH 的 oct.raw 需要 oct.mhd 头文件提供尺寸信息。")
        return

    print(f"找到 {len(mhd_files)} 个 RETOUCH mhd 文件。")

    records = []
    exported = 0

    for mhd_path in mhd_files:
        try:
            volume, info, raw_path = read_mhd_raw_volume(mhd_path)
        except Exception as e:
            print(f"跳过 {mhd_path}: {e}")
            continue

        device = infer_device(mhd_path)
        case_id = mhd_path.parent.name
        out_dir = OUTPUT_DIR / device / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        indices = select_indices(volume.shape[0], SLICES_PER_VOLUME)
        print(f"[{device}] {case_id}: shape={volume.shape}, slices={indices}")

        for sid in indices:
            img_u8 = normalize_to_uint8(volume[sid])
            pil_img = Image.fromarray(img_u8, mode="L")
            label = f"{device}_{case_id}_z{sid:03d}"

            if SAVE_INDIVIDUAL_SLICES:
                pil_img.save(out_dir / f"{label}.png")

            if len(records) < MAX_TOTAL_SLICES_FOR_SHEET:
                records.append({"img": pil_img, "label": f"{device} z{sid:03d}"})
            exported += 1

    sheet_path = OUTPUT_DIR / "RETOUCH_raw_oct_contact_sheet.png"
    make_contact_sheet(records, sheet_path, "RETOUCH Dataset - Raw OCT B-scans")

    print("=" * 70)
    print("RETOUCH 原始 OCT 切片导出完成")
    print(f"导出目录: {OUTPUT_DIR}")
    print(f"单张切片数量: {exported}")
    print(f"总览图: {sheet_path}")
    print("不会修改原始 retouch 数据。")


if __name__ == "__main__":
    main()
