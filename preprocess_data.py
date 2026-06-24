import os
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import scipy.io as sio  # 必须安装: pip install scipy

# ================= 配置区域 =================
# RETOUCH 路径
RETOUCH_ROOT = r'D:\Graduation_Design\retouch'
# Duke 路径
DUKE_ROOT = r'D:\Graduation_Design\2015_BOE_Chiu'
# 输出路径
OUTPUT_ROOT = r'D:\Graduation_Design\Dataset_Unified'


# =====================================

def get_args():
    parser = argparse.ArgumentParser(description='OCT 毕设数据预处理 - RETOUCH + Duke (修复版)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    return parser.parse_args()


def save_image_mask(img_arr, mask_arr, save_dir, prefix, filename):
    img_dir = os.path.join(save_dir, 'images')
    mask_dir = os.path.join(save_dir, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 图像归一化
    if img_arr.max() > 0:
        img_arr = (img_arr / img_arr.max() * 255).astype(np.uint8)
    else:
        img_arr = img_arr.astype(np.uint8)

    # Mask 保持类别值 (0,1,2,3...)
    mask_arr = mask_arr.astype(np.uint8)

    img_pil = Image.fromarray(img_arr)
    mask_pil = Image.fromarray(mask_arr)

    save_name = f"{prefix}_{filename}.png"
    img_pil.save(os.path.join(img_dir, save_name))
    mask_pil.save(os.path.join(mask_dir, save_name))


# --- Duke 数据集处理逻辑 ---
def process_duke(args):
    print("\n>>> [1/2] 正在处理 Duke 数据集 (2015_BOE_Chiu)...")

    if not os.path.exists(DUKE_ROOT):
        print(f"    [ERROR] 路径不存在 {DUKE_ROOT}")
        return

    # 扫描所有的 .mat 文件
    mat_files = glob.glob(os.path.join(DUKE_ROOT, 'Subject_*.mat'))
    print(f"    找到 {len(mat_files)} 个 Duke 受试者数据 (.mat)")

    count_total = 0
    keys_printed = False

    for mat_path in tqdm(mat_files, desc="    Processing Duke"):
        try:
            # 读取 .mat 文件
            mat_data = sio.loadmat(mat_path)
            file_name = os.path.splitext(os.path.basename(mat_path))[0]

            if not keys_printed:
                clean_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                print(f"    [DEBUG] 文件变量检查: {clean_keys}")
                keys_printed = True

            # 1. 提取图像
            if 'images' in mat_data:
                images_vol = mat_data['images']
            else:
                continue

            # 2. 提取 Mask (关键修改：适配你的变量名)
            mask_vol = None

            # 优先级: manualFluid1 > manualFluid2 > manualFluid > automaticFluidDME
            if 'manualFluid1' in mat_data:
                mask_vol = mat_data['manualFluid1']  # 优先用专家1
            elif 'manualFluid2' in mat_data:
                mask_vol = mat_data['manualFluid2']  # 备用专家2
            elif 'manualFluid' in mat_data:
                mask_vol = mat_data['manualFluid']  # 标准名
            elif 'automaticFluidDME' in mat_data:
                mask_vol = mat_data['automaticFluidDME']  # 自动分割结果
            elif 'automaticFluid' in mat_data:
                mask_vol = mat_data['automaticFluid']
            elif 'fluid' in mat_data:
                mask_vol = mat_data['fluid']
            else:
                # 只有分层没有积液标注
                continue

            # 3. 维度调整
            # 检查是否有 nan (MATLAB数据常见问题)
            if np.isnan(images_vol).any(): images_vol = np.nan_to_num(images_vol)
            if np.isnan(mask_vol).any(): mask_vol = np.nan_to_num(mask_vol)

            if images_vol.ndim == 3:
                # 假设最小的维度是 Depth (切片数)
                # Duke通常是 [H, W, D]，需要转为 [D, H, W]
                # 比如 (496, 768, 61) -> (61, 496, 768)
                if images_vol.shape[2] < images_vol.shape[0] and images_vol.shape[2] < images_vol.shape[1]:
                    images_vol = np.transpose(images_vol, (2, 0, 1))
                    mask_vol = np.transpose(mask_vol, (2, 0, 1))

            if images_vol.shape != mask_vol.shape:
                print(f"    [WARNING] {file_name} 尺寸不匹配跳过: Img {images_vol.shape} vs Mask {mask_vol.shape}")
                continue

            for i in range(images_vol.shape[0]):
                slice_img = images_vol[i]
                slice_mask = mask_vol[i]

                # 只有当 Mask 里有东西才保存，或者 5% 概率存背景
                if np.max(slice_mask) > 0 or np.random.rand() < 0.05:
                    split = 'train' if np.random.rand() > args.val_ratio else 'val'
                    save_dir = os.path.join(OUTPUT_ROOT, split)

                    pil_img = Image.fromarray(slice_img).convert('L').resize((args.image_size, args.image_size))
                    pil_mask = Image.fromarray(slice_mask.astype(np.uint8)).resize((args.image_size, args.image_size),
                                                                                   Image.Resampling.NEAREST)

                    save_filename = f"Duke_{file_name}_slice{i:03d}"
                    save_image_mask(np.array(pil_img), np.array(pil_mask), save_dir, "Duke", save_filename)
                    count_total += 1

        except Exception as e:
            print(f"    [ERROR] 处理 {mat_path} 失败: {e}")
            continue

    print(f"    -> Duke 处理完成: 共生成 {count_total} 张切片。")


# --- RETOUCH 处理逻辑 (保持原样) ---
def process_retouch(args):
    print("\n>>> [2/2] 正在处理 RETOUCH 数据集...")

    if not os.path.exists(RETOUCH_ROOT):
        print(f"    [ERROR] 路径不存在 {RETOUCH_ROOT}")
        return

    # 直接寻找所有的 .mhd
    all_mhd = glob.glob(os.path.join(RETOUCH_ROOT, '**', '*.mhd'), recursive=True)

    # 筛选 mask 文件
    mask_files = []
    for f in all_mhd:
        fname = os.path.basename(f).lower()
        if ('reference' in fname or 'segmentation' in fname or 'seg' in fname) and 'test' not in f.lower():
            mask_files.append(f)

    print(f"    找到 {len(mask_files)} 个 RETOUCH 标签文件。")

    count_total = 0

    for mask_path in tqdm(mask_files, desc="    Processing RETOUCH"):
        try:
            dir_name = os.path.dirname(mask_path)
            # 寻找 Volume
            possible_vols = [
                os.path.join(dir_name, "oct.mhd"),
                os.path.join(dir_name,
                             os.path.basename(mask_path).replace('_segmentation', '').replace('_Reference', '').replace(
                                 '_reference', ''))
            ]
            mask_name = os.path.basename(mask_path)
            if '_segmentation' in mask_name.lower():
                possible_vols.append(os.path.join(dir_name, mask_name.lower().replace('_segmentation', '')))

            img_path = None
            for p in possible_vols:
                if os.path.exists(p) and p.endswith('.mhd'):
                    img_path = p;
                    break

            if not img_path:
                siblings = glob.glob(os.path.join(dir_name, "*.mhd"))
                for s in siblings:
                    if s != mask_path and 'ref' not in s.lower() and 'seg' not in s.lower():
                        img_path = s;
                        break

            if not img_path: continue

            itk_img = sitk.ReadImage(img_path)
            itk_mask = sitk.ReadImage(mask_path)
            vol_img = sitk.GetArrayFromImage(itk_img)
            vol_mask = sitk.GetArrayFromImage(itk_mask)

            if vol_img.shape != vol_mask.shape: continue

            vendor = "RET"
            if 'cirrus' in mask_path.lower():
                vendor = "Cirrus"
            elif 'spectralis' in mask_path.lower():
                vendor = "Spectralis"
            elif 'topcon' in mask_path.lower():
                vendor = "Topcon"
            case_id = os.path.basename(os.path.dirname(mask_path))

            for i in range(vol_img.shape[0]):
                slice_mask = vol_mask[i]
                if slice_mask.max() > 0 or np.random.rand() < 0.05:
                    split = 'train' if np.random.rand() > args.val_ratio else 'val'
                    save_dir = os.path.join(OUTPUT_ROOT, split)

                    pil_img = Image.fromarray(vol_img[i]).convert('L').resize((args.image_size, args.image_size))
                    pil_mask = Image.fromarray(slice_mask.astype(np.uint8)).resize((args.image_size, args.image_size),
                                                                                   Image.Resampling.NEAREST)

                    filename = f"{vendor}_{case_id}_slice{i:03d}"
                    save_image_mask(np.array(pil_img), np.array(pil_mask), save_dir, "RET", filename)
                    count_total += 1

        except Exception as e:
            continue

    print(f"    -> RETOUCH 处理完成: 共生成 {count_total} 张切片。")


def main():
    args = get_args()
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

    print(f"=== 终极数据整合: RETOUCH + Duke (修复版) ===")

    # 1. 处理 Duke
    process_duke(args)

    # 2. 处理 RETOUCH
    process_retouch(args)

    print(f"\n🎉 整合完成！所有数据已保存至: {OUTPUT_ROOT}")
    print(f"现在运行 train.py，它会自动加载这两个数据集的所有图片进行训练。")


if __name__ == '__main__':
    main()