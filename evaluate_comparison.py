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


# ================= 配置区域 =================
DATA_ROOT = r"D:\Graduation_Design\Dataset_Unified"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
INFERENCE_SIZE = 256
BATCH_SIZE = 8
SPLIT = "val"

CLASS_NAMES = {
    1: "IRF",
    2: "SRF",
    3: "PED",
}

# 不包含 ASPP-Bridge
# 如果某个模型文件或权重不存在，会自动跳过。
MODELS_TO_EVALUATE = [
    {
        "name": "Standard U-Net (Base)",
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
        "name": "nnU-Net Base",
        "module": "model_nnunet_base",
        "class": "NNUNetBase",
        "fallback_classes": ["nnUNet"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_base/best_model.pth",
    },
    {
        "name": "nnU-Net + ECA-Encoder",
        "module": "model_nnunet_eca_encoder",
        "class": "nnUNet",
        "fallback_classes": ["NNUNetBase"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_eca_encoder/best_model.pth",
    },
    {
        "name": "nnU-Net + CBAM-Decoder",
        "module": "model_nnunet_cbam_decoder",
        "class": "nnUNet",
        "fallback_classes": ["NNUNetBase"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_cbam_decoder/best_model.pth",
    },
    {
        "name": "nnU-Net + ECA/CBAM",
        "module": "model_nnunet_eca_cbam",
        "class": "nnUNet",
        "fallback_classes": ["NNUNetBase"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_eca_cbam/best_model.pth",
    },
    {
        "name": "nnU-Net + SDA",
        "module": "model_nnunet_sda",
        "class": "nnUNet",
        "fallback_classes": ["NNUNetBase"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_sda/best_model.pth",
    },
    {
        "name": "nnU-Net + SASC",
        "module": "model_nnunet_sasc",
        "class": "nnUNet",
        "fallback_classes": ["NNUNetBase"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_sasc/best_model.pth",
    },
    {
        "name": "nnU-Net + SDA/SASC",
        "module": "model_nnunet_sda_sasc",
        "class": "nnUNet",
        "fallback_classes": ["NNUNetBase"],
        "args": {"n_channels": 1, "n_classes": NUM_CLASSES},
        "path": "checkpoints_nnunet_sda_sasc/best_model.pth",
    },
]


class EvalDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "val"):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.mask_dir = os.path.join(self.split_dir, "masks")

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]) if os.path.exists(self.img_dir) else []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path)

        image = image.resize((INFERENCE_SIZE, INFERENCE_SIZE))
        mask = mask.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.Resampling.NEAREST)

        img_tensor = transforms.ToTensor()(image)
        mask_np = np.array(mask, dtype=np.int64)

        # 保持旧 evaluate_comparison.py 的处理方式：
        # 0~3 之外全部映射为背景。
        mask_np[mask_np < 0] = 0
        mask_np[mask_np >= NUM_CLASSES] = 0

        return img_tensor, torch.from_numpy(mask_np).long(), img_name


def get_model_class(module, primary_class: str, fallback_classes=None):
    fallback_classes = fallback_classes or []
    candidates = [primary_class] + fallback_classes
    for cls_name in candidates:
        if hasattr(module, cls_name):
            return getattr(module, cls_name)
    raise AttributeError(f"在模块 {module.__name__} 中找不到这些类: {candidates}")


def build_model(conf: Dict[str, Any]):
    module = importlib.import_module(conf["module"])
    model_cls = get_model_class(
        module,
        conf["class"],
        conf.get("fallback_classes", [])
    )
    return model_cls(**conf["args"])


def load_state_dict_safely(model: torch.nn.Module, ckpt_path: str):
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    return model


def oldstyle_metrics_per_case(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    """
    完全沿用旧 evaluate_comparison.py 的核心逻辑：
    每张图、每个前景类别单独计算，然后取平均。
    空类别 union=0 时 Dice/IoU=1.0。
    """
    metrics = {
        "Dice": [],
        "IoU": [],
        "Precision": [],
        "Recall": [],
    }

    per_class = {}

    for cls in range(1, num_classes):
        p = (pred == cls).astype(float)
        t = (gt == cls).astype(float)

        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        pred_sum = p.sum()
        gt_sum = t.sum()

        dice = (2.0 * intersection) / (union + 1e-6) if union > 0 else 1.0
        iou = intersection / (union - intersection + 1e-6) if union > 0 else 1.0
        precision = intersection / (pred_sum + 1e-6) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
        recall = intersection / (gt_sum + 1e-6) if gt_sum > 0 else 1.0

        name = CLASS_NAMES.get(cls, f"Class_{cls}")
        per_class[f"{name}_Dice"] = dice
        per_class[f"{name}_IoU"] = iou
        per_class[f"{name}_Precision"] = precision
        per_class[f"{name}_Recall"] = recall

        metrics["Dice"].append(dice)
        metrics["IoU"].append(iou)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)

    out = {k: float(np.mean(v)) for k, v in metrics.items()}
    out.update(per_class)
    return out


def evaluate_model(conf: Dict[str, Any], dataloader: DataLoader) -> Optional[Dict[str, float]]:
    print(f"\n🔄 正在评估模型: {conf['name']}")

    if not os.path.exists(conf["path"]):
        print(f"⚠️ 跳过：找不到权重文件 {conf['path']}")
        return None

    try:
        model = build_model(conf).to(DEVICE)
    except Exception as e:
        print(f"⚠️ 跳过：模型导入失败 {conf['module']} -> {e}")
        return None

    try:
        model = load_state_dict_safely(model, conf["path"])
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return None

    model.eval()

    total = {
        "Dice": 0.0,
        "IoU": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
    }

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
                outputs = outputs["segmentation"]

            preds_np = torch.argmax(outputs, dim=1).cpu().numpy()

            for i in range(len(preds_np)):
                res = oldstyle_metrics_per_case(preds_np[i], masks_np[i], NUM_CLASSES)
                for k in total:
                    total[k] += res[k]
                count += 1

    row = {k: v / max(count, 1) for k, v in total.items()}
    row["Model"] = conf["name"]

    print(f"   -> Dice: {row['Dice']:.4f} | IoU: {row['IoU']:.4f} | Recall: {row['Recall']:.4f} | Precision: {row['Precision']:.4f}")
    return row


def add_delta_vs_base(df: pd.DataFrame, base_name: str = "nnU-Net Base") -> pd.DataFrame:
    df = df.copy()
    if base_name not in df["Model"].values:
        return df

    base = df[df["Model"] == base_name].iloc[0]
    for metric in ["Dice", "IoU", "Precision", "Recall"]:
        df[f"Delta_{metric}_vs_Base"] = df[metric] - base[metric]
        df[f"Rel_{metric}_vs_Base_%"] = (df[metric] - base[metric]) / max(abs(base[metric]), 1e-8) * 100

    return df


def plot_oldstyle_comparison(df: pd.DataFrame):
    if df.empty:
        return

    plot_df = df.sort_values("Dice", ascending=True).copy()
    models = plot_df["Model"].tolist()

    y = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(11, max(5.5, len(models) * 0.65)))

    bars1 = ax.barh(y - height / 2, plot_df["Dice"], height, label="Dice")
    bars2 = ax.barh(y + height / 2, plot_df["IoU"], height, label="IoU")

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Score")
    ax.set_title("统一评估口径下模型性能对比（不含 ASPP）")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    for bars in [bars1, bars2]:
        for rect in bars:
            w = rect.get_width()
            ax.text(w + 0.005, rect.get_y() + rect.get_height() / 2,
                    f"{w:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("comparison_chart_oldstyle_no_aspp.png", dpi=300)
    plt.savefig("comparison_chart_model_performance.png", dpi=300)
    print("📊 图表已保存: comparison_chart_oldstyle_no_aspp.png")
    print("📊 图表已保存: comparison_chart_model_performance.png")


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 数据集目录不存在: {DATA_ROOT}")
        return

    dataset = EvalDataset(DATA_ROOT, split=SPLIT)
    if len(dataset) == 0:
        print("❌ 验证集为空")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"📊 准备评估 {len(dataset)} 张图像，共 {len(MODELS_TO_EVALUATE)} 个模型配置。")

    results = []
    for conf in MODELS_TO_EVALUATE:
        row = evaluate_model(conf, dataloader)
        if row is not None:
            results.append(row)

    if not results:
        print("❌ 没有成功评估任何模型。")
        return

    df = pd.DataFrame(results)
    df = add_delta_vs_base(df, base_name="nnU-Net Base")
    df = df.sort_values("Dice", ascending=False).reset_index(drop=True)

    preferred_cols = [
        "Model",
        "Dice", "IoU", "Recall", "Precision",
        "Delta_Dice_vs_Base", "Rel_Dice_vs_Base_%",
        "IRF_Dice", "SRF_Dice", "PED_Dice",
        "IRF_IoU", "SRF_IoU", "PED_IoU",
        "Delta_IoU_vs_Base", "Rel_IoU_vs_Base_%",
        "Delta_Recall_vs_Base", "Rel_Recall_vs_Base_%",
        "Delta_Precision_vs_Base", "Rel_Precision_vs_Base_%",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[[c for c in preferred_cols if c in df.columns] + other_cols]

    print("\n" + "=" * 120)
    print("🏆 统一评估口径模型对比报告（不含 ASPP）")
    print("=" * 120)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))
    print("=" * 120)

    out_csv = "model_comparison_results_oldstyle_no_aspp.csv"
    df.to_csv(out_csv, index=False)
    print(f"💾 表格已保存: {out_csv}")

    # 为了兼容 GUI，如果你希望 GUI 直接读取它，可以取消下面这行注释。
    # df.to_csv("model_comparison_results.csv", index=False)

    plot_oldstyle_comparison(df)


if __name__ == "__main__":
    main()
