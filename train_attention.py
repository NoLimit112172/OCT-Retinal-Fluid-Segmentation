import os
import csv
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model_attention import AttentionUNet

# ================= 配置区域 =================
DATA_ROOT = r'D:\Graduation_Design\Dataset_Unified'
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4  # 0:背景, 1:IRF, 2:SRF, 3:PED
SEED = 42

# 建议换损失函数后从头开始训练，避免沿用旧的 CE-only 权重
RESUME = False
CHECKPOINT_DIR = 'checkpoints_dicece_attention'
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
LOG_CSV_PATH = os.path.join(CHECKPOINT_DIR, 'train_log.csv')

# 组合损失权重
CE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
# ===========================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OCTDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, 'images')
        self.mask_dir = os.path.join(self.split_dir, 'masks')

        self.images = []
        if os.path.exists(self.img_dir):
            self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path)

        img_tensor = transforms.ToTensor()(image)
        mask_np = np.array(mask, dtype=np.int64)

        if mask_np.max() >= NUM_CLASSES:
            mask_np[mask_np >= NUM_CLASSES] = 0

        mask_tensor = torch.from_numpy(mask_np).long()
        return img_tensor, mask_tensor


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5, smooth=1e-6, ignore_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        start_cls = 1 if self.ignore_background else 0
        dice_loss = 0.0
        class_count = 0

        for cls in range(start_cls, self.num_classes):
            prob_c = probs[:, cls]
            target_c = target_one_hot[:, cls]

            intersection = (prob_c * target_c).sum(dim=(1, 2))
            denominator = prob_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_loss += (1.0 - dice_score).mean()
            class_count += 1

        dice_loss = dice_loss / max(class_count, 1)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss, ce_loss.detach(), dice_loss.detach()


def calculate_dice(pred, target, num_classes):
    dice_scores = []
    pred_mask = torch.argmax(pred, dim=1)
    for cls in range(1, num_classes):
        p = (pred_mask == cls).float()
        t = (target == cls).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice = (2.0 * intersection) / (union + 1e-6)
            dice_scores.append(dice.item())
    return float(np.mean(dice_scores))


def train():
    set_seed(SEED)
    print(f"🔥 [Attention U-Net] Dice + CE 训练开始... 设备: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not RESUME or not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'val_dice'])

    if not os.path.exists(DATA_ROOT):
        print(f"❌ 错误: 找不到数据目录 {DATA_ROOT}")
        return

    train_ds = OCTDataset(DATA_ROOT, 'train')
    val_ds = OCTDataset(DATA_ROOT, 'val')

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("❌ 错误: 训练集或验证集为空，请检查 Dataset_Unified/train 与 val 目录。")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == 'cuda')
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == 'cuda')
    )

    model = AttentionUNet(n_channels=1, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = DiceCELoss(
        num_classes=NUM_CLASSES,
        ce_weight=CE_WEIGHT,
        dice_weight=DICE_WEIGHT,
        ignore_background=True
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    start_epoch = 0
    best_dice = 0.0

    if RESUME and os.path.exists(LAST_CKPT_PATH):
        print(f"🔄 发现中断训练进度: {LAST_CKPT_PATH}")
        checkpoint = torch.load(LAST_CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"✅ 恢复成功！从 Epoch {start_epoch + 1} 继续")
    else:
        print("🆕 从头开始训练（推荐，因为损失函数已更改）。")

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            epoch_total_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_dice_loss = 0.0

            current_lr = optimizer.param_groups[0]['lr']
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [LR={current_lr:.1e}]")

            for imgs, masks in pbar:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(imgs)
                total_loss, ce_loss, dice_loss = criterion(outputs, masks)
                total_loss.backward()
                optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_dice_loss += dice_loss.item()
                pbar.set_postfix({
                    'total': f"{total_loss.item():.4f}",
                    'ce': f"{ce_loss.item():.4f}",
                    'dice': f"{dice_loss.item():.4f}"
                })

            avg_train_loss = epoch_total_loss / len(train_loader)
            avg_ce_loss = epoch_ce_loss / len(train_loader)
            avg_dice_loss = epoch_dice_loss / len(train_loader)

            model.eval()
            val_dice_total = 0.0
            with torch.no_grad():
                for v_imgs, v_masks in val_loader:
                    v_imgs = v_imgs.to(DEVICE, non_blocking=True)
                    v_masks = v_masks.to(DEVICE, non_blocking=True)
                    v_out = model(v_imgs)
                    val_dice_total += calculate_dice(v_out, v_masks, NUM_CLASSES)

            avg_val_dice = val_dice_total / len(val_loader)
            scheduler.step(avg_val_dice)

            print(
                f"✅ [Att-UNet] Epoch {epoch + 1} | Total Loss: {avg_train_loss:.4f} "
                f"(CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f}) | Val Dice: {avg_val_dice:.4f}"
            )

            with open(LOG_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_dice])

            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice
            }
            torch.save(checkpoint_dict, LAST_CKPT_PATH)

            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"🏆 [Att-UNet] 新纪录！最佳模型已保存 (Dice: {best_dice:.4f})")

            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'epoch_{epoch + 1}.pth'))

        print("🎉 Attention U-Net 训练全部结束！")

    except KeyboardInterrupt:
        print("\n⏸️ 训练已暂停。")


if __name__ == '__main__':
    train()
