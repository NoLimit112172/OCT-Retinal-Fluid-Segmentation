
"""
extract_duke_mat_original_slices.py

从 Duke 2015_BOE_Chiu 数据集的 .mat 文件中导出多张未处理 OCT B-scan 图像，并生成总览拼图。
不会修改原始 .mat 数据，只会在 OUTPUT_DIR 下生成新的 PNG 文件。

数据根目录：
D:\Graduation_Design\2015_BOE_Chiu

运行：
cd /d D:\Graduation_Design\OCT_Segmentation
.venv\Scripts\python.exe extract_duke_mat_original_slices.py

依赖：
pip install scipy h5py pillow numpy
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import scipy.io as sio
except Exception:
    sio = None

try:
    import h5py
except Exception:
    h5py = None


DATA_ROOT = Path(r"D:\Graduation_Design\2015_BOE_Chiu")
OUTPUT_DIR = Path(r"D:\Graduation_Design\原始图像展示\DUKE")

SLICES_PER_SUBJECT = 6
MAX_TOTAL_SLICES_FOR_SHEET = 30
THUMB_SIZE = (160, 120)
SAVE_INDIVIDUAL_SLICES = True


def normalize_to_uint8(img):
    """
    仅用于保存 PNG 可视化，不会改动原始 mat 文件。
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


def is_candidate_volume(arr):
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim < 3:
        return False
    if not np.issubdtype(arr.dtype, np.number):
        return False
    if min(arr.shape) < 5:
        return False
    return True


def load_mat_regular(mat_path: Path):
    if sio is None:
        raise ImportError("未安装 scipy，请先 pip install scipy")

    data = sio.loadmat(str(mat_path))
    candidates = {}

    for k, v in data.items():
        if k.startswith("__"):
            continue
        if is_candidate_volume(v):
            candidates[k] = np.asarray(v)

    if not candidates:
        raise ValueError("未找到 3D OCT 体数据变量")

    preferred = ["images", "Images", "OCT", "oct", "volume", "Volume", "data", "Data"]
    for key in preferred:
        if key in candidates:
            return key, candidates[key]

    key = max(candidates.keys(), key=lambda kk: candidates[kk].size)
    return key, candidates[key]


def collect_h5_datasets(group, prefix=""):
    found = {}

    for key in group.keys():
        item = group[key]
        name = f"{prefix}/{key}" if prefix else key

        if h5py is not None and isinstance(item, h5py.Dataset):
            try:
                arr = np.array(item)
            except Exception:
                continue
            if is_candidate_volume(arr):
                found[name] = arr

        elif h5py is not None and isinstance(item, h5py.Group):
            found.update(collect_h5_datasets(item, name))

    return found


def load_mat_hdf5(mat_path: Path):
    if h5py is None:
        raise ImportError("未安装 h5py，请先 pip install h5py")

    with h5py.File(str(mat_path), "r") as f:
        candidates = collect_h5_datasets(f)

    if not candidates:
        raise ValueError("未找到 3D OCT 体数据变量")

    preferred_fragments = ["images", "oct", "volume", "data"]
    for frag in preferred_fragments:
        for key in candidates:
            if frag.lower() in key.lower():
                return key, np.asarray(candidates[key])

    key = max(candidates.keys(), key=lambda kk: candidates[kk].size)
    return key, np.asarray(candidates[key])


def load_duke_volume(mat_path: Path):
    """
    自动兼容普通 .mat 和 MATLAB v7.3 HDF5 .mat。
    """
    try:
        return load_mat_regular(mat_path)
    except NotImplementedError:
        return load_mat_hdf5(mat_path)
    except Exception as first_error:
        try:
            return load_mat_hdf5(mat_path)
        except Exception:
            raise first_error


def volume_to_slices(volume):
    """
    将 volume 转成 [N, H, W]。
    Duke 可能是 H×W×N，也可能是 N×H×W。
    这里自动把最像“切片数”的轴移到最前面。
    """
    arr = np.squeeze(np.asarray(volume))

    if arr.ndim != 3:
        raise ValueError(f"压缩后不是 3D volume: shape={arr.shape}")

    shape = arr.shape

    # 如果第 0 维比较像切片数，直接用第 0 维
    if shape[0] <= 300 and shape[1] >= 100 and shape[2] >= 100:
        slice_axis = 0
    else:
        # 否则通常最小维度是切片数
        slice_axis = int(np.argmin(shape))

    slices = np.moveaxis(arr, slice_axis, 0)
    return slices


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

    mat_files = sorted(DATA_ROOT.glob("*.mat"))
    if not mat_files:
        print(f"没有找到 .mat 文件: {DATA_ROOT}")
        return

    print(f"找到 {len(mat_files)} 个 Duke .mat 文件。")

    records = []
    exported = 0

    for mat_path in mat_files:
        subject = mat_path.stem
        out_dir = OUTPUT_DIR / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            var_name, volume = load_duke_volume(mat_path)
            slices = volume_to_slices(volume)
        except Exception as e:
            print(f"跳过 {mat_path}: {e}")
            continue

        indices = select_indices(slices.shape[0], SLICES_PER_SUBJECT)
        print(
            f"{subject}: variable={var_name}, volume_shape={volume.shape}, "
            f"slices_shape={slices.shape}, slices={indices}"
        )

        for sid in indices:
            img_u8 = normalize_to_uint8(slices[sid])
            pil_img = Image.fromarray(img_u8, mode="L")
            label = f"{subject}_z{sid:03d}"

            if SAVE_INDIVIDUAL_SLICES:
                pil_img.save(out_dir / f"{label}.png")

            if len(records) < MAX_TOTAL_SLICES_FOR_SHEET:
                records.append({"img": pil_img, "label": f"{subject} z{sid:03d}"})
            exported += 1

    sheet_path = OUTPUT_DIR / "DUKE_raw_oct_contact_sheet.png"
    make_contact_sheet(records, sheet_path, "Duke Dataset - Raw OCT B-scans")

    print("=" * 70)
    print("Duke 原始 OCT 切片导出完成")
    print(f"导出目录: {OUTPUT_DIR}")
    print(f"单张切片数量: {exported}")
    print(f"总览图: {sheet_path}")
    print("不会修改原始 Duke 数据。")


if __name__ == "__main__":
    main()
