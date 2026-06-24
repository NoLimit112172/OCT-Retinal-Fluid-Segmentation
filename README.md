# 基于深度学习的视网膜 OCT 图像积液分割系统

本科毕业设计项目。面向视网膜 OCT（光学相干断层扫描）图像中的**积液病灶分割**任务，构建了一套完整的医学影像智能分析流程：从数据预处理、多模型训练与对比、消融实验，到桌面端推理可视化。

分割目标为三类视网膜积液病灶：

- **IRF** — 视网膜内液（Intraretinal Fluid）
- **SRF** — 视网膜下液（Subretinal Fluid）
- **PED** — 色素上皮脱离（Pigment Epithelial Detachment）

---

## 主要特性

- **多模型对比**：实现并统一对比 U-Net、Attention U-Net、SegNet、DeepLabV3+ 与 nnU-Net 系列。
- **注意力机制消融**：在 nnU-Net 基线上系统性消融 CBAM、ECA、SDA、SASC 等多种注意力模块及其组合。
- **统一损失函数**：采用 CrossEntropy + Dice 组合损失（DiceCE），缓解病灶类别不平衡。
- **完整评估体系**：基于 Dice、IoU、Precision、Recall，支持整图评估、仅病灶切片评估与按数据来源分组评估。
- **桌面端 GUI**：基于 Tkinter 实现一键推理、掩膜彩色叠加与结果展示（约 1700 行）。

---

## 实验结果

在统一测试集上的整图分割结果（按 Dice 排序，**加粗为本文方法**）：

| 模型 | Dice | IoU | 参数量 (M) |
|------|:----:|:---:|:---------:|
| **nnU-Net + SDA/SASC（Ours）** | **0.9016** | **0.8592** | **8.36** |
| DeepLabV3+ | 0.8946 | 0.8542 | 12.21 |
| nnU-Net + ECA | 0.8944 | 0.8512 | 7.56 |
| nnU-Net + ECA/CBAM | 0.8928 | 0.8496 | 7.57 |
| nnU-Net + CBAM | 0.8924 | 0.8488 | 7.57 |
| nnU-Net + SASC | 0.8916 | 0.8477 | 8.36 |
| nnU-Net + SDA | 0.8902 | 0.8464 | 7.56 |
| nnU-Net Base | 0.8712 | 0.8246 | 1.89 |
| U-Net | 0.8656 | 0.8209 | 17.26 |
| Attention U-Net | 0.8610 | 0.8151 | 34.88 |
| SegNet | 0.8088 | 0.7597 | 15.31 |

本文方法 **nnU-Net + SDA/SASC** 相比 nnU-Net 基线 Dice 提升约 **3.0 个百分点（+3.5%）**，且参数量（8.36M）显著低于 U-Net（17.26M）与 Attention U-Net（34.88M），在精度与轻量化之间取得较好平衡。

> 完整逐病灶（IRF / SRF / PED）指标、按数据集分组结果与可视化图表见 [`results/`](results/) 目录。

---

## 项目结构

```
OCT-Retinal-Fluid-Segmentation/
├── model.py                      # U-Net 基线
├── model_attention.py            # Attention U-Net
├── model_segnet.py               # SegNet
├── model_nnunet_base.py          # nnU-Net 基线
├── model_nnunet_cbam_decoder.py  # nnU-Net + CBAM（解码器）
├── model_nnunet_eca_encoder.py   # nnU-Net + ECA（编码器）
├── model_nnunet_eca_cbam.py      # nnU-Net + ECA/CBAM
├── model_nnunet_sda.py           # nnU-Net + SDA
├── model_nnunet_sasc.py          # nnU-Net + SASC
├── model_nnunet_sda_sasc.py      # nnU-Net + SDA/SASC（本文方法）
├── train*.py                     # 各模型对应的训练脚本
├── evaluate*.py                  # 评估与对比脚本（整图 / 仅病灶 / 分数据集）
├── preprocess_data.py            # 数据预处理
├── extract_*.py                  # DUKE / RETOUCH 原始切片抽取
├── gui.py                        # Tkinter 推理可视化界面
├── results/                      # 实验结果 CSV 与图表
├── requirements.txt
└── README.md
```

---

## 环境配置

```bash
# 建议使用 Python 3.10+，并新建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` 中的 PyTorch 为 CUDA 11.8 版本。若使用 CPU 或其他 CUDA 版本，请前往 [pytorch.org](https://pytorch.org) 选择对应安装命令。

---

## 使用说明

代码中的数据路径以常量形式写在各脚本顶部（如 `DATA_ROOT`、`CHECKPOINT_DIR`），运行前请按本地实际路径修改。

```bash
# 1. 数据预处理（将原始 OCT 数据整理为统一格式）
python preprocess_data.py

# 2. 训练（以本文方法为例）
python train_nnunet_sda_sasc.py

# 3. 评估与多模型对比
python evaluate_all_models_oldstyle.py

# 4. 启动可视化推理界面
python gui.py
```

---

## 数据集

实验使用两个公开视网膜 OCT 数据集：

- **DUKE**（Chiu et al., 2015 BOE）
- **RETOUCH** 挑战赛数据集

> 出于数据使用规范与隐私合规考虑，本仓库**不包含任何原始医学影像数据与训练权重**，请从各数据集官方渠道自行获取。

---

## 作者

吴贤（Wu Xian） · 合肥工业大学 电子信息工程
