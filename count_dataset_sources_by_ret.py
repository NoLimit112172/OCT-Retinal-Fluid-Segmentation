"""
count_dataset_sources_by_ret.py

功能：
1. 统计 Dataset_Unified 中 train / val / test 的 images 和 masks 数量；
2. 按文件名来源统计：
   - 文件名中包含 "RET" / "ret" 的图像，记为 RETOUCH；
   - 其他图像全部记为 DUKE；
3. 同时统计 train / val / test 各自的 DUKE / RETOUCH 数量；
4. 输出并保存：
   - dataset_source_statistics_by_ret.csv
   - dataset_source_statistics_by_ret.txt

适用你的当前命名规则：
- RET 开头或文件名含 RET = RETOUCH
- 其他 = DUKE
"""

import os
import csv
from pathlib import Path


# ===================== 路径配置 =====================

DATASET_ROOT = Path(r"D:\Graduation_Design\Dataset_Unified")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_retouch_file(filename: str) -> bool:
    """
    你的命名规则：
    文件名中含 RET，则认为来源是 RETOUCH；
    其他全部认为是 DUKE。
    """
    return "ret" in filename.lower()


def count_images_in_dir(folder: Path):
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and is_image_file(p))


def count_masks_in_dir(folder: Path):
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and is_image_file(p))


def count_split_source(split: str):
    image_dir = DATASET_ROOT / split / "images"
    mask_dir = DATASET_ROOT / split / "masks"

    stats = {
        "split": split,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "images_total": 0,
        "masks_total": 0,
        "RETOUCH": 0,
        "DUKE": 0,
    }

    if not image_dir.exists():
        return stats

    image_files = [
        p for p in image_dir.iterdir()
        if p.is_file() and is_image_file(p)
    ]

    stats["images_total"] = len(image_files)
    stats["masks_total"] = count_masks_in_dir(mask_dir)

    for p in image_files:
        if is_retouch_file(p.name):
            stats["RETOUCH"] += 1
        else:
            stats["DUKE"] += 1

    return stats


def write_csv(split_stats, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Split",
            "Images_Total",
            "Masks_Total",
            "DUKE_Count",
            "RETOUCH_Count",
            "Image_Dir",
            "Mask_Dir",
        ])

        for s in split_stats:
            writer.writerow([
                s["split"],
                s["images_total"],
                s["masks_total"],
                s["DUKE"],
                s["RETOUCH"],
                s["image_dir"],
                s["mask_dir"],
            ])

        writer.writerow([])
        writer.writerow([
            "TOTAL",
            sum(s["images_total"] for s in split_stats),
            sum(s["masks_total"] for s in split_stats),
            sum(s["DUKE"] for s in split_stats),
            sum(s["RETOUCH"] for s in split_stats),
            "",
            "",
        ])


def write_txt(split_stats, output_path: Path):
    total_images = sum(s["images_total"] for s in split_stats)
    total_masks = sum(s["masks_total"] for s in split_stats)
    total_duke = sum(s["DUKE"] for s in split_stats)
    total_retouch = sum(s["RETOUCH"] for s in split_stats)

    lines = []
    lines.append("=" * 72)
    lines.append("数据集来源数量统计结果")
    lines.append("=" * 72)
    lines.append(f"DATASET_ROOT: {DATASET_ROOT}")
    lines.append("")
    lines.append("识别规则：")
    lines.append("  文件名包含 RET / ret  → RETOUCH")
    lines.append("  其他文件              → DUKE")
    lines.append("")

    lines.append("[按划分统计]")
    for s in split_stats:
        lines.append(
            f"{s['split']:>5s}: images={s['images_total']}, masks={s['masks_total']}, "
            f"DUKE={s['DUKE']}, RETOUCH={s['RETOUCH']}"
        )

    lines.append("")
    lines.append("[总计]")
    lines.append(f"Images Total : {total_images}")
    lines.append(f"Masks Total  : {total_masks}")
    lines.append(f"DUKE         : {total_duke}")
    lines.append(f"RETOUCH      : {total_retouch}")

    if total_images != total_duke + total_retouch:
        lines.append("")
        lines.append("警告：DUKE + RETOUCH 数量与 images 总数不一致，请检查统计规则。")
    else:
        lines.append("")
        lines.append("校验：DUKE + RETOUCH = Images Total，统计通过。")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not DATASET_ROOT.exists():
        print(f"错误：找不到数据集根目录：{DATASET_ROOT}")
        return

    split_stats = []
    for split in ["train", "val", "test"]:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            split_stats.append(count_split_source(split))

    if not split_stats:
        print(f"错误：{DATASET_ROOT} 下未找到 train / val / test 文件夹。")
        return

    csv_path = Path("dataset_source_statistics_by_ret.csv")
    txt_path = Path("dataset_source_statistics_by_ret.txt")

    write_csv(split_stats, csv_path)
    write_txt(split_stats, txt_path)

    total_images = sum(s["images_total"] for s in split_stats)
    total_masks = sum(s["masks_total"] for s in split_stats)
    total_duke = sum(s["DUKE"] for s in split_stats)
    total_retouch = sum(s["RETOUCH"] for s in split_stats)

    print("=" * 72)
    print("数据集来源数量统计结果")
    print("=" * 72)
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print()
    print("识别规则：文件名包含 RET/ret 记为 RETOUCH，其他全部记为 DUKE")
    print()
    print("[按划分统计]")
    for s in split_stats:
        print(
            f"{s['split']:>5s}: images={s['images_total']}, masks={s['masks_total']}, "
            f"DUKE={s['DUKE']}, RETOUCH={s['RETOUCH']}"
        )

    print()
    print("[总计]")
    print(f"Images Total : {total_images}")
    print(f"Masks Total  : {total_masks}")
    print(f"DUKE         : {total_duke}")
    print(f"RETOUCH      : {total_retouch}")

    print()
    print(f"已保存: {csv_path.resolve()}")
    print(f"已保存: {txt_path.resolve()}")


if __name__ == "__main__":
    main()
