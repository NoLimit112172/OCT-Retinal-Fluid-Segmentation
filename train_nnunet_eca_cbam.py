
import os
import csv
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from model_nnunet_eca_cbam import nnUNet


DATA_ROOT = r'D:\Graduation_Design\Dataset_Unified'
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4
RESUME = False

BASE_CHANNELS = 32

CHECKPOINT_DIR = 'checkpoints_nnunet_eca_cbam'
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
LOG_CSV_PATH = os.path.join(CHECKPOINT_DIR, 'train_log.csv')

LAMBDA_CE = 0.5
LAMBDA_DICE = 0.5
SEED = 2024


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class OCTDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, 'images')
        self.mask_dir = os.path.join(self.split_dir, 'masks')

        self.images = []
        if os.path.exists(self.img_dir):
            self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith('.png')]
            self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('L').resize((256, 256))
        mask = Image.open(mask_path).resize((256, 256), Image.Resampling.NEAREST)

        img_tensor = transforms.ToTensor()(image)
        mask_np = np.array(mask, dtype=np.int64)

        if mask_np.max() >= NUM_CLASSES or mask_np.min() < 0:
            mask_np[mask_np < 0] = 0
            mask_np[mask_np >= NUM_CLASSES] = 0

        return img_tensor, torch.from_numpy(mask_np).long()


class DiceLoss(nn.Module):
    def __init__(self, num_classes, include_background=False, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        start_class = 0 if self.include_background else 1
        dice_loss = 0.0
        valid_classes = 0

        for c in range(start_class, self.num_classes):
            p = probs[:, c]
            t = targets_onehot[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_score)
            valid_classes += 1

        return dice_loss / max(valid_classes, 1)


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes, include_background=False)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        total = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        return total, loss_ce.detach(), loss_dice.detach()


def calculate_val_dice(pred_logits, target, num_classes):
    """
    用于训练阶段保存 best_model。
    保持你前面实验的 OldStyle 风格：空类别记为 1.0。
    最终论文结果用 evaluate_comparison_oldstyle_no_aspp.py 统一评估。
    """
    dice_scores = []
    pred_mask = torch.argmax(pred_logits, dim=1)

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
    print(f'🔥 [nnU-Net + ECA-Encoder + CBAM-Decoder] 准备训练... 设备: {DEVICE}')
    print(f'📌 base_channels = {BASE_CHANNELS}')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not RESUME or not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'ce', 'dice_loss', 'val_dice', 'lr'])

    if not os.path.exists(DATA_ROOT):
        print(f'❌ 错误: 找不到数据目录 {DATA_ROOT}')
        return

    train_ds = OCTDataset(DATA_ROOT, 'train')
    val_ds = OCTDataset(DATA_ROOT, 'val')

    if len(train_ds) == 0:
        print('❌ 错误: 训练集为空，请检查 DATA_ROOT/train/images')
        return
    if len(val_ds) == 0:
        print('❌ 错误: 验证集为空，请检查 DATA_ROOT/val/images')
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = nnUNet(n_channels=1, n_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'📦 模型参数量: {total_params / 1e6:.2f} M')

    class_weights = torch.tensor([0.2, 1.0, 1.0, 1.0], dtype=torch.float32).to(DEVICE)
    criterion = DiceCELoss(
        num_classes=NUM_CLASSES,
        ce_weight=LAMBDA_CE,
        dice_weight=LAMBDA_DICE,
        class_weights=class_weights
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    start_epoch = 0
    best_dice = 0.0

    if RESUME and os.path.exists(LAST_CKPT_PATH):
        print(f'🔄 发现中断的训练进度: {LAST_CKPT_PATH}')
        ckpt = torch.load(LAST_CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice = ckpt.get('best_dice', 0.0)
        print(f'✅ 恢复成功！从 Epoch {start_epoch + 1} 继续')
    else:
        print('🆕 从头开始训练新的 nnU-Net + ECA/CBAM 实验。')

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            epoch_loss = 0.0
            epoch_ce = 0.0
            epoch_dice_loss = 0.0

            current_lr = optimizer.param_groups[0]['lr']
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} [LR={current_lr:.1e}]')

            for imgs, masks in pbar:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                optimizer.zero_grad()
                logits = model(imgs)
                total_loss, loss_ce, loss_dice = criterion(logits, masks)
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_ce += loss_ce.item()
                epoch_dice_loss += loss_dice.item()

                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'ce': f'{loss_ce.item():.4f}',
                    'dice': f'{loss_dice.item():.4f}'
                })

            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            avg_ce = epoch_ce / max(len(train_loader), 1)
            avg_dice_loss = epoch_dice_loss / max(len(train_loader), 1)

            model.eval()
            val_dice_total = 0.0
            with torch.no_grad():
                for v_imgs, v_masks in val_loader:
                    v_imgs = v_imgs.to(DEVICE)
                    v_masks = v_masks.to(DEVICE)
                    v_logits = model(v_imgs)
                    val_dice_total += calculate_val_dice(v_logits, v_masks, NUM_CLASSES)

            avg_val_dice = val_dice_total / max(len(val_loader), 1)

            scheduler.step(avg_val_dice)
            lr_after = optimizer.param_groups[0]['lr']

            print(
                f'✅ Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | '
                f'CE: {avg_ce:.4f} | DiceLoss: {avg_dice_loss:.4f} | '
                f'Val Dice: {avg_val_dice:.4f} | LR: {lr_after:.2e}'
            )

            with open(LOG_CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_ce, avg_dice_loss, avg_val_dice, lr_after])

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice
            }
            torch.save(ckpt, LAST_CKPT_PATH)

            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f'🏆 新纪录！ECA/CBAM 最佳模型已保存: {BEST_MODEL_PATH} (Dice: {best_dice:.4f})')

            if (epoch + 1) % 5 == 0:
                epoch_path = os.path.join(CHECKPOINT_DIR, f'epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), epoch_path)

        print('🎉 训练全部结束！')

    except KeyboardInterrupt:
        print('\n⏸️ 训练已暂停。当前进度已保存到 last_checkpoint。')


if __name__ == '__main__':
    train()
