import os
import importlib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

DATA_ROOT = r"D:\Graduation_Design\Dataset_Unified"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
INFERENCE_SIZE = 256
BATCH_SIZE = 8
SPLIT = "val"

CLASS_NAMES = {1: "IRF", 2: "SRF", 3: "PED"}


class EvalDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "val"):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.mask_dir = os.path.join(self.split_dir, "masks")
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]) if os.path.exists(self.img_dir) else []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L").resize((INFERENCE_SIZE, INFERENCE_SIZE))
        mask = Image.open(mask_path).resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.Resampling.NEAREST)

        img_tensor = transforms.ToTensor()(image)
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np < 0] = 0
        mask_np[mask_np >= NUM_CLASSES] = 0

        return img_tensor, torch.from_numpy(mask_np).long(), img_name


def get_model_class(module, primary_class: str, fallback_classes=None):
    candidates = [primary_class] + (fallback_classes or [])
    for cls_name in candidates:
        if hasattr(module, cls_name):
            return getattr(module, cls_name)
    raise AttributeError(f"在模块 {module.__name__} 中找不到这些类: {candidates}")


def build_model(conf: Dict[str, Any]):
    module = importlib.import_module(conf["module"])
    model_cls = get_model_class(module, conf["class"], conf.get("fallback_classes", []))
    return model_cls(**conf["args"])


def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str):
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model.load_state_dict(ckpt)
    return model


def oldstyle_metrics_per_case(pred: np.ndarray, gt: np.ndarray):
    """
    OldStyle 口径：
    1. 不计算背景；
    2. 对每张图分别计算 IRF / SRF / PED；
    3. 某类在预测和 GT 中都不存在时，该类 Dice / IoU 记为 1.0；
    4. 每张图先对三类平均，再对验证集平均。
    """
    metric_lists = {"Dice": [], "IoU": [], "Precision": [], "Recall": []}
    per_class = {}

    for cls in range(1, NUM_CLASSES):
        p = (pred == cls).astype(float)
        t = (gt == cls).astype(float)

        inter = (p * t).sum()
        union = p.sum() + t.sum()
        pred_sum = p.sum()
        gt_sum = t.sum()

        dice = (2.0 * inter) / (union + 1e-6) if union > 0 else 1.0
        iou = inter / (union - inter + 1e-6) if union > 0 else 1.0
        precision = inter / (pred_sum + 1e-6) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
        recall = inter / (gt_sum + 1e-6) if gt_sum > 0 else 1.0

        name = CLASS_NAMES[cls]
        per_class[f"{name}_Dice"] = float(dice)
        per_class[f"{name}_IoU"] = float(iou)
        per_class[f"{name}_Precision"] = float(precision)
        per_class[f"{name}_Recall"] = float(recall)

        metric_lists["Dice"].append(dice)
        metric_lists["IoU"].append(iou)
        metric_lists["Precision"].append(precision)
        metric_lists["Recall"].append(recall)

    out = {k: float(np.mean(v)) for k, v in metric_lists.items()}
    out.update(per_class)
    return out


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def evaluate_model(conf: Dict[str, Any], dataloader: DataLoader) -> Optional[Dict[str, float]]:
    print(f"\n🔄 正在评估模型: {conf['name']}")

    if not os.path.exists(conf["path"]):
        print(f"⚠️ 跳过：找不到权重文件 {conf['path']}")
        return None

    try:
        model = build_model(conf).to(DEVICE)
        params_m = count_params_m(model)
    except Exception as e:
        print(f"⚠️ 跳过：模型导入失败 {conf['module']} -> {e}")
        return None

    try:
        model = load_state_dict_safely(model, conf["path"])
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return None

    model.eval()

    total = {"Dice": 0.0, "IoU": 0.0, "Precision": 0.0, "Recall": 0.0}
    for name in CLASS_NAMES.values():
        total[f"{name}_Dice"] = 0.0
        total[f"{name}_IoU"] = 0.0
        total[f"{name}_Precision"] = 0.0
        total[f"{name}_Recall"] = 0.0

    count = 0
    with torch.no_grad():
        for imgs, masks, _ in tqdm(dataloader, desc=conf["name"]):
            imgs = imgs.to(DEVICE)
            masks_np = masks.numpy()

            outputs = model(imgs)
            if isinstance(outputs, dict):
                outputs = outputs.get("segmentation", list(outputs.values())[0])

            preds_np = torch.argmax(outputs, dim=1).cpu().numpy()

            for i in range(len(preds_np)):
                res = oldstyle_metrics_per_case(preds_np[i], masks_np[i])
                for k in total:
                    total[k] += res[k]
                count += 1

    row = {k: v / max(count, 1) for k, v in total.items()}
    row["Model"] = conf["name"]
    row["Params_M"] = params_m
    row["Mean_3Fluid_Dice"] = float(np.mean([row["IRF_Dice"], row["SRF_Dice"], row["PED_Dice"]]))
    row["Mean_3Fluid_IoU"] = float(np.mean([row["IRF_IoU"], row["SRF_IoU"], row["PED_IoU"]]))

    print(
        f"   -> Dice: {row['Dice']:.4f} | IoU: {row['IoU']:.4f} | "
        f"3类Dice均值: {row['Mean_3Fluid_Dice']:.4f} | 3类IoU均值: {row['Mean_3Fluid_IoU']:.4f}"
    )
    return row


def add_delta_vs_base(df: pd.DataFrame, base_name: str = "nnU-Net Base") -> pd.DataFrame:
    df = df.copy()
    if base_name not in df["Model"].values:
        return df

    base = df[df["Model"] == base_name].iloc[0]
    for metric in ["Dice", "IoU", "Mean_3Fluid_Dice", "Mean_3Fluid_IoU", "Precision", "Recall"]:
        if metric in df.columns:
            df[f"Delta_{metric}_vs_Base"] = df[metric] - base[metric]
            df[f"Rel_{metric}_vs_Base_%"] = (df[metric] - base[metric]) / max(abs(base[metric]), 1e-8) * 100

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "Model",
        "Dice", "IoU", "Mean_3Fluid_Dice", "Mean_3Fluid_IoU",
        "Precision", "Recall",
        "IRF_Dice", "SRF_Dice", "PED_Dice",
        "IRF_IoU", "SRF_IoU", "PED_IoU",
        "IRF_Precision", "SRF_Precision", "PED_Precision",
        "IRF_Recall", "SRF_Recall", "PED_Recall",
        "Delta_Dice_vs_Base", "Rel_Dice_vs_Base_%",
        "Delta_IoU_vs_Base", "Rel_IoU_vs_Base_%",
        "Delta_Mean_3Fluid_Dice_vs_Base", "Rel_Mean_3Fluid_Dice_vs_Base_%",
        "Delta_Mean_3Fluid_IoU_vs_Base", "Rel_Mean_3Fluid_IoU_vs_Base_%",
        "Params_M",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    return df[[c for c in preferred_cols if c in df.columns] + other_cols]


def plot_mean_dice_iou(df: pd.DataFrame, title: str, output_name: str):
    plot_df = df.sort_values("Dice", ascending=True).copy()
    models = plot_df["Model"].tolist()
    y = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(11, max(5.5, len(models) * 0.65)))
    bars1 = ax.barh(y - height / 2, plot_df["Dice"], height, label="Mean Dice")
    bars2 = ax.barh(y + height / 2, plot_df["IoU"], height, label="Mean IoU")

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    for bars in [bars1, bars2]:
        for rect in bars:
            w = rect.get_width()
            ax.text(w + 0.005, rect.get_y() + rect.get_height() / 2,
                    f"{w:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"📊 图表已保存: {output_name}")


def plot_per_fluid(df: pd.DataFrame, metric: str, title: str, output_name: str):
    cols = [f"IRF_{metric}", f"SRF_{metric}", f"PED_{metric}"]
    plot_df = df.sort_values("Mean_3Fluid_Dice", ascending=False).copy()
    x = np.arange(len(plot_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 5.8))
    labels = ["IRF", "SRF", "PED"]

    for i, col in enumerate(cols):
        bars = ax.bar(x + (i - 1) * width, plot_df[col], width, label=labels[i])
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.006, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"📊 图表已保存: {output_name}")


def run_evaluation(models_to_evaluate, out_csv, chart_prefix, title):
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 数据集目录不存在: {DATA_ROOT}")
        return

    dataset = EvalDataset(DATA_ROOT, split=SPLIT)
    if len(dataset) == 0:
        print("❌ 验证集为空")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"📊 准备评估 {len(dataset)} 张图像，共 {len(models_to_evaluate)} 个模型配置。")

    results = []
    for conf in models_to_evaluate:
        row = evaluate_model(conf, dataloader)
        if row is not None:
            results.append(row)

    if not results:
        print("❌ 没有成功评估任何模型。")
        return

    df = pd.DataFrame(results)
    df = add_delta_vs_base(df, base_name="nnU-Net Base")
    df = df.sort_values("Dice", ascending=False).reset_index(drop=True)
    df = reorder_columns(df)

    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))
    print("=" * 140)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"💾 表格已保存: {out_csv}")

    plot_mean_dice_iou(df, title, f"{chart_prefix}_mean_dice_iou.png")
    plot_per_fluid(df, "Dice", f"{title}：IRF / SRF / PED Dice 对比", f"{chart_prefix}_per_fluid_dice.png")
    plot_per_fluid(df, "IoU", f"{title}：IRF / SRF / PED IoU 对比", f"{chart_prefix}_per_fluid_iou.png")


MODELS_TO_EVALUATE = [
    {
        "name": "SegNet",
        "module": "model_segnet",
        "class": "SegNet",
        "fallback_classes": [],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_dicece_segnet/best_model.pth",
    },
    {
        "name": "U-Net",
        "module": "model",
        "class": "UNet",
        "fallback_classes": [],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_dicece_unet/best_model.pth",
    },
    {
        "name": "Attention U-Net",
        "module": "model_attention",
        "class": "AttentionUNet",
        "fallback_classes": [],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_dicece_attention/best_model.pth",
    },
    {
        "name": "DeepLabV3+",
        "module": "model_deeplabv3plus",
        "class": "DeepLabV3Plus",
        "fallback_classes": ["DeepLabv3Plus", "DeepLabV3plus"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_dicece_deeplabv3plus/best_model.pth",
    },
    {
        "name": "nnU-Net Base",
        "module": "model_nnunet_base",
        "class": "NNUNetBase",
        "fallback_classes": ["nnUNet"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_base/best_model.pth",
    },
]


def main():
    run_evaluation(
        models_to_evaluate=MODELS_TO_EVALUATE,
        out_csv="baseline_models_results_oldstyle.csv",
        chart_prefix="baseline_models",
        title="五个基线模型对比实验"
    )


if __name__ == "__main__":
    main()
