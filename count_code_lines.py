import os
import csv
from pathlib import Path

# 自动统计当前脚本所在目录
# 你现在代码基本都在 .venv 目录下，所以这里会统计 .venv 目录里的 .py 文件
ROOT = Path(__file__).resolve().parent

# 排除虚拟环境内部依赖包、缓存、模型权重等
EXCLUDE_DIRS = {
    "__pycache__",
    "Lib",
    "Scripts",
    "Include",
    "share",
    ".pytest_cache",

    # checkpoint 权重目录一般不算源代码
    "checkpoints",
    "checkpoints_dicece_unet",
    "checkpoints_dicece_attention",
    "checkpoints_dicece_segnet",
    "checkpoints_nnunet_base",
    "checkpoints_nnunet_sda",
    "checkpoints_nnunet_sasc",
    "checkpoints_nnunet_sda_sasc",
    "checkpoints_nnunet_eca_encoder",
    "checkpoints_nnunet_cbam_decoder",
    "checkpoints_nnunet_eca_cbam",
}

# 这些文件不计入你的毕设主体代码
EXCLUDE_FILES = {
    "count_code_lines.py",
}

file_stats = []
total_lines = 0
total_code_lines = 0
total_blank_lines = 0
total_comment_lines = 0

for root, dirs, files in os.walk(ROOT):
    # 排除指定目录
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

    for file in files:
        if not file.endswith(".py"):
            continue

        if file in EXCLUDE_FILES:
            continue

        path = Path(root) / file

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"跳过无法读取文件: {path}，原因: {e}")
            continue

        line_count = len(lines)
        code_count = 0
        blank_count = 0
        comment_count = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                blank_count += 1
            elif stripped.startswith("#"):
                comment_count += 1
            else:
                code_count += 1

        total_lines += line_count
        total_code_lines += code_count
        total_blank_lines += blank_count
        total_comment_lines += comment_count

        relative_path = path.relative_to(ROOT)
        file_stats.append({
            "file": str(relative_path),
            "total_lines": line_count,
            "code_lines": code_count,
            "blank_lines": blank_count,
            "comment_lines": comment_count,
        })

file_stats.sort(key=lambda x: x["file"])

print("=" * 90)
print("Python 源代码量统计")
print("=" * 90)
print(f"统计根目录: {ROOT}")
print("=" * 90)

for item in file_stats:
    print(
        f"{item['file']:<45} "
        f"总行数: {item['total_lines']:<5} "
        f"有效代码行: {item['code_lines']:<5} "
        f"空行: {item['blank_lines']:<5} "
        f"注释行: {item['comment_lines']:<5}"
    )

print("=" * 90)
print(f"Python 文件数量: {len(file_stats)}")
print(f"总代码行数: {total_lines}")
print(f"有效代码行数: {total_code_lines}")
print(f"空行数: {total_blank_lines}")
print(f"注释行数: {total_comment_lines}")
print("=" * 90)

# 保存 TXT
txt_path = ROOT / "code_line_statistics.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("Python 源代码量统计\n")
    f.write("=" * 90 + "\n")
    f.write(f"统计根目录: {ROOT}\n")
    f.write("=" * 90 + "\n")

    for item in file_stats:
        f.write(
            f"{item['file']:<45} "
            f"总行数: {item['total_lines']:<5} "
            f"有效代码行: {item['code_lines']:<5} "
            f"空行: {item['blank_lines']:<5} "
            f"注释行: {item['comment_lines']:<5}\n"
        )

    f.write("=" * 90 + "\n")
    f.write(f"Python 文件数量: {len(file_stats)}\n")
    f.write(f"总代码行数: {total_lines}\n")
    f.write(f"有效代码行数: {total_code_lines}\n")
    f.write(f"空行数: {total_blank_lines}\n")
    f.write(f"注释行数: {total_comment_lines}\n")
    f.write("=" * 90 + "\n")

# 保存 CSV
csv_path = ROOT / "code_line_statistics.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "total_lines", "code_lines", "blank_lines", "comment_lines"]
    )
    writer.writeheader()
    writer.writerows(file_stats)

print(f"已保存 TXT: {txt_path}")
print(f"已保存 CSV: {csv_path}")