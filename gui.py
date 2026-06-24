import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import os
import csv
import importlib

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from model import UNet
from model_attention import AttentionUNet

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# ================= Matplotlib 配置 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STHeiti', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================= 全局配置 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4
INFERENCE_SIZE = 256
IMG_DISPLAY_SIZE = (430, 430)

COLORS = np.array([
    [0, 0, 0],
    [231, 76, 60],   # IRF
    [46, 204, 113],  # SRF
    [52, 152, 219]   # PED
], dtype=np.uint8)

# ================= 主题色 =================
APP_BG = "#EEF2F7"
SIDEBAR_BG = "#0F172A"
CARD_BG = "#FFFFFF"
CARD_ALT_BG = "#F8FAFC"
BORDER = "#D8E1EC"
PRIMARY = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
SUCCESS = "#16A34A"
WARNING = "#D97706"
DANGER = "#DC2626"
TEXT = "#0F172A"
TEXT_MUTED = "#64748B"
PANEL_BG = "#E5EAF2"

MODEL_DISPLAY_NAMES = [
    "SegNet",
    "Standard U-Net (Base)",
    "Attention U-Net",
    "nnU-Net Base",
    "nnU-Net + ECA-Encoder",
    "nnU-Net + CBAM-Decoder",
    "nnU-Net + ECA/CBAM",
    "nnU-Net + SDA",
    "nnU-Net + SASC",
    "nnU-Net + SDA/SASC",
]

MODEL_REGISTRY = {
    "SegNet": {
        "kind": "dynamic",
        "module": "model_segnet",
        "class": "SegNet",
        "checkpoint": "checkpoints_dicece_segnet/best_model.pth",
        "log": "checkpoints_dicece_segnet/train_log.csv",
        "color": "#64748B",
    },
    "Standard U-Net (Base)": {
        "kind": "static",
        "factory": UNet,
        "checkpoint": "checkpoints_dicece_unet/best_model.pth",
        "log": "checkpoints_dicece_unet/train_log.csv",
        "color": "#3B82F6",
    },
    "Attention U-Net": {
        "kind": "static",
        "factory": AttentionUNet,
        "checkpoint": "checkpoints_dicece_attention/best_model.pth",
        "log": "checkpoints_dicece_attention/train_log.csv",
        "color": "#8B5CF6",
    },
    "nnU-Net Base": {
        "kind": "dynamic",
        "module": "model_nnunet_base",
        "class": "NNUNetBase",
        "checkpoint": "checkpoints_nnunet_base/best_model.pth",
        "log": "checkpoints_nnunet_base/train_log.csv",
        "color": "#10B981",
    },
    "nnU-Net + ECA-Encoder": {
        "kind": "dynamic",
        "module": "model_nnunet_eca_encoder",
        "class": "nnUNet",
        "checkpoint": "checkpoints_nnunet_eca_encoder/best_model.pth",
        "log": "checkpoints_nnunet_eca_encoder/train_log.csv",
        "color": "#F59E0B",
    },
    "nnU-Net + CBAM-Decoder": {
        "kind": "dynamic",
        "module": "model_nnunet_cbam_decoder",
        "class": "nnUNet",
        "checkpoint": "checkpoints_nnunet_cbam_decoder/best_model.pth",
        "log": "checkpoints_nnunet_cbam_decoder/train_log.csv",
        "color": "#A16207",
    },
    "nnU-Net + ECA/CBAM": {
        "kind": "dynamic",
        "module": "model_nnunet_eca_cbam",
        "class": "nnUNet",
        "checkpoint": "checkpoints_nnunet_eca_cbam/best_model.pth",
        "log": "checkpoints_nnunet_eca_cbam/train_log.csv",
        "color": "#0EA5E9",
    },
    "nnU-Net + SDA": {
        "kind": "dynamic",
        "module": "model_nnunet_sda",
        "class": "nnUNet",
        "checkpoint": "checkpoints_nnunet_sda/best_model.pth",
        "log": "checkpoints_nnunet_sda/train_log.csv",
        "color": "#EC4899",
    },
    "nnU-Net + SASC": {
        "kind": "dynamic",
        "module": "model_nnunet_sasc",
        "class": "nnUNet",
        "checkpoint": "checkpoints_nnunet_sasc/best_model.pth",
        "log": "checkpoints_nnunet_sasc/train_log.csv",
        "color": "#6B7280",
    },
    "nnU-Net + SDA/SASC": {
        "kind": "dynamic",
        "module": "model_nnunet_sda_sasc",
        "class": "nnUNet",
        "checkpoint": "checkpoints_nnunet_sda_sasc/best_model.pth",
        "log": "checkpoints_nnunet_sda_sasc/train_log.csv",
        "color": "#EF4444",
    },
}

COMPARISON_KEEP_ORDER = MODEL_DISPLAY_NAMES.copy()


class OCTSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DR 智能分析工作站 | 糖尿病视网膜病变检测系统")
        self.root.geometry("1540x980")
        self.root.minsize(1320, 860)
        self.root.configure(bg=APP_BG)

        self.setup_styles()
        self.placeholder = self.create_placeholder()

        self.model = None
        self.current_image_path = None
        self.current_mask_path = None
        self.origin_img_pil = None          # 用于界面显示的原图
        self.origin_full_pil = None         # 原始分辨率原图，用于导出
        self.pred_mask_pil = None           # 用于界面显示的彩色预测图
        self.pred_mask_full_pil = None      # 原始分辨率彩色预测图
        self.pred_label_256 = None          # 256x256 标签图，值为 0/1/2/3
        self.pred_label_full = None         # 原始分辨率标签图，值为 0/1/2/3
        self.overlay_full_pil = None        # 原始分辨率叠加图
        self.current_right_pil = None       # 当前右侧显示图

        self.view_mode = tk.StringVar(value="overlay")
        self.opacity = tk.DoubleVar(value=0.45)
        self.selected_model = tk.StringVar(value="nnU-Net Base")
        self.log_model_selection = tk.StringVar(value="nnU-Net Base")
        self.compare_metric = tk.StringVar(value="Dice")
        self.compare_scope = tk.StringVar(value="All")
        self.comparison_df = None

        self.create_main_layout()
        self.root.after(100, self.load_model)

    # ================= 基础样式 =================
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TNotebook', background=APP_BG, borderwidth=0)
        style.configure('TNotebook.Tab', background='#DCE6F2', foreground=TEXT,
                        font=('Microsoft YaHei', 11, 'bold'), padding=(18, 10), borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', CARD_BG)],
                  foreground=[('selected', PRIMARY)])

        style.configure('Primary.TButton', font=('Microsoft YaHei', 10, 'bold'),
                        background=PRIMARY, foreground='white', borderwidth=0, padding=(14, 8))
        style.map('Primary.TButton', background=[('active', PRIMARY_HOVER)])

        style.configure('Light.TButton', font=('Microsoft YaHei', 10),
                        background=CARD_ALT_BG, foreground=TEXT, borderwidth=1,
                        relief='solid', padding=(12, 8))
        style.map('Light.TButton', background=[('active', '#E8EEF8')])

        style.configure('Sidebar.TCombobox', fieldbackground=CARD_BG, background=CARD_BG,
                        foreground=TEXT, arrowsize=14)

        style.configure('TScale', background=CARD_BG)
        style.configure('TRadiobutton', background=CARD_BG, foreground=TEXT, font=('Microsoft YaHei', 10))
        style.map('TRadiobutton', background=[('active', CARD_BG)])

        style.configure('Treeview', font=('Microsoft YaHei', 10), rowheight=30,
                        background='white', fieldbackground='white', foreground=TEXT, borderwidth=0)
        style.configure('Treeview.Heading', font=('Microsoft YaHei', 10, 'bold'),
                        background='#EDF2F7', foreground=TEXT, relief='flat')
        style.map('Treeview', background=[('selected', '#DBEAFE')], foreground=[('selected', TEXT)])

    def create_placeholder(self):
        img = Image.new('RGB', IMG_DISPLAY_SIZE, PANEL_BG)
        draw = ImageDraw.Draw(img)
        w, h = IMG_DISPLAY_SIZE
        draw.rounded_rectangle([16, 16, w - 16, h - 16], radius=18, outline='#CBD5E1', width=2)
        draw.rectangle([w // 2 - 34, h // 2 - 3, w // 2 + 34, h // 2 + 3], fill='#94A3B8')
        draw.rectangle([w // 2 - 3, h // 2 - 34, w // 2 + 3, h // 2 + 34], fill='#94A3B8')
        return ImageTk.PhotoImage(img)

    def create_card(self, parent, title=None, subtitle=None, pad=18):
        outer = tk.Frame(parent, bg=APP_BG, highlightbackground=BORDER, highlightthickness=1)
        outer.configure(bd=0)
        inner = tk.Frame(outer, bg=CARD_BG)
        inner.pack(fill='both', expand=True)
        if title:
            title_frame = tk.Frame(inner, bg=CARD_BG)
            title_frame.pack(fill='x', padx=pad, pady=(pad, 8))
            tk.Label(title_frame, text=title, bg=CARD_BG, fg=TEXT,
                     font=('Microsoft YaHei', 13, 'bold')).pack(anchor='w')
            if subtitle:
                tk.Label(title_frame, text=subtitle, bg=CARD_BG, fg=TEXT_MUTED,
                         font=('Microsoft YaHei', 9)).pack(anchor='w', pady=(4, 0))
        return outer, inner

    def create_section_label(self, parent, text):
        tk.Label(parent, text=text, bg=CARD_BG, fg=TEXT_MUTED,
                 font=('Microsoft YaHei', 9, 'bold')).pack(anchor='w', pady=(0, 6))

    # ================= 布局 =================
    def create_main_layout(self):
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=300)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        self._create_sidebar_content(sidebar)

        main = tk.Frame(self.root, bg=APP_BG)
        main.pack(side='right', fill='both', expand=True, padx=22, pady=18)

        header = tk.Frame(main, bg=APP_BG)
        header.pack(fill='x', pady=(0, 14))
        tk.Label(header, text="糖尿病视网膜病变检测系统", bg=APP_BG, fg=TEXT,
                 font=('Microsoft YaHei', 20, 'bold')).pack(anchor='w')
        tk.Label(header, text="智能诊断 / 实验对比 / 病例报告", bg=APP_BG, fg=TEXT_MUTED,
                 font=('Microsoft YaHei', 10)).pack(anchor='w', pady=(4, 0))

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill='both', expand=True)

        self.tab_inference = tk.Frame(self.notebook, bg=APP_BG)
        self.tab_compare = tk.Frame(self.notebook, bg=APP_BG)
        self.tab_report = tk.Frame(self.notebook, bg=APP_BG)

        self.notebook.add(self.tab_inference, text=" 智能诊断 ")
        self.notebook.add(self.tab_compare, text=" 实验对比 ")
        self.notebook.add(self.tab_report, text=" 病例报告 ")

        self._init_inference_tab(self.tab_inference)
        self._init_comparison_tab(self.tab_compare)
        self._init_report_tab(self.tab_report)


    def _create_sidebar_content(self, parent):
        top = tk.Frame(parent, bg=SIDEBAR_BG)
        top.pack(fill='x', padx=22, pady=(26, 18))

        badge = tk.Canvas(top, width=44, height=44, bg=SIDEBAR_BG, highlightthickness=0)
        badge.create_oval(4, 4, 40, 40, fill=PRIMARY, outline='')
        badge.create_text(22, 22, text='DR', fill='white', font=('Arial', 12, 'bold'))
        badge.pack(anchor='w')

        tk.Label(top, text='DR 智能分析工作站', bg=SIDEBAR_BG, fg='white',
                 font=('Microsoft YaHei', 16, 'bold')).pack(anchor='w', pady=(12, 4))
        tk.Label(top, text='面向 OCT 视网膜积液分割实验', bg=SIDEBAR_BG, fg='#94A3B8',
                 font=('Microsoft YaHei', 9)).pack(anchor='w')

        sep = tk.Frame(parent, bg='#1E293B', height=1)
        sep.pack(fill='x', padx=20, pady=10)

        card = tk.Frame(parent, bg='#111C31', highlightbackground='#1F2A44', highlightthickness=1)
        card.pack(fill='x', padx=18, pady=10)

        inner = tk.Frame(card, bg='#111C31')
        inner.pack(fill='x', padx=14, pady=14)

        tk.Label(inner, text='当前模型', bg='#111C31', fg='#CBD5E1',
                 font=('Microsoft YaHei', 10, 'bold')).pack(anchor='w')
        ttk.Combobox(inner, textvariable=self.selected_model, values=MODEL_DISPLAY_NAMES,
                     state='readonly', width=28, style='Sidebar.TCombobox').pack(fill='x', pady=(8, 6))
        inner.children[list(inner.children.keys())[-1]].bind("<<ComboboxSelected>>", lambda e: self.load_model())

        tk.Label(inner, text=f'设备: {DEVICE.upper()}', bg='#111C31', fg='#94A3B8',
                 font=('Microsoft YaHei', 9)).pack(anchor='w', pady=(6, 2))

        self.model_badge = tk.Label(inner, text='未加载模型', bg='#172554', fg='#BFDBFE',
                                    font=('Microsoft YaHei', 9, 'bold'), padx=10, pady=5)
        self.model_badge.pack(anchor='w', pady=(8, 0))

        info_card = tk.Frame(parent, bg='#111C31', highlightbackground='#1F2A44', highlightthickness=1)
        info_card.pack(fill='x', padx=18, pady=10)
        info_inner = tk.Frame(info_card, bg='#111C31')
        info_inner.pack(fill='x', padx=14, pady=14)
        tk.Label(info_inner, text='实验说明', bg='#111C31', fg='#CBD5E1',
                 font=('Microsoft YaHei', 10, 'bold')).pack(anchor='w')
        tips = [
            '• 支持 OCT 图像分割与叠加显示',
            '• 自动统计 IRF / SRF / PED 面积占比',
            '• 实验对比展示整体指标与逐类 Dice',
            '• 病例报告页可直接用于答辩展示',
        ]
        for tip in tips:
            tk.Label(info_inner, text=tip, bg='#111C31', fg='#94A3B8',
                     font=('Microsoft YaHei', 9), justify='left').pack(anchor='w', pady=2)

        bottom = tk.Frame(parent, bg=SIDEBAR_BG)
        bottom.pack(side='bottom', fill='x', padx=18, pady=18)
        self.status_label = tk.Label(bottom, text='系统就绪', bg=SIDEBAR_BG, fg='#E2E8F0',
                                     font=('Microsoft YaHei', 10), justify='left', wraplength=250)
        self.status_label.pack(anchor='w')

    # ================= 模型加载 =================
    def _build_model_instance(self, model_type):
        config = MODEL_REGISTRY.get(model_type)
        if config is None:
            raise ValueError(f"未知模型类型: {model_type}")

        if config['kind'] == 'static':
            model_cls = config['factory']
            return model_cls(n_channels=1, n_classes=NUM_CLASSES)

        module = importlib.import_module(config['module'])
        class_candidates = [config['class']] + config.get('fallback_classes', [])
        model_cls = None
        for cls_name in class_candidates:
            if hasattr(module, cls_name):
                model_cls = getattr(module, cls_name)
                break
        if model_cls is None:
            raise AttributeError(f"模块 {config['module']} 中找不到模型类: {class_candidates}")
        return model_cls(n_channels=1, n_classes=NUM_CLASSES)

    def load_model(self):
        model_type = self.selected_model.get()
        config = MODEL_REGISTRY.get(model_type)
        if config is None:
            self.status_label.config(text=f'❌ 未知模型: {model_type}', fg=DANGER)
            return

        ckpt_path = config['checkpoint']
        try:
            model_instance = self._build_model_instance(model_type)
        except Exception as e:
            self.model = None
            self.status_label.config(text=f'❌ 模型文件导入失败\n{e}', fg=DANGER)
            if hasattr(self, 'lbl_model_info'):
                self.lbl_model_info.config(text='模型导入失败', fg=DANGER)
            self.model_badge.config(text='导入失败', bg='#7F1D1D', fg='#FECACA')
            return

        if not os.path.exists(ckpt_path):
            self.model = None
            self.status_label.config(text=f'⚠️ 未找到权重\n{ckpt_path}', fg=WARNING)
            if hasattr(self, 'lbl_model_info'):
                self.lbl_model_info.config(text='未加载模型权重（请先训练）', fg=TEXT_MUTED)
            self.model_badge.config(text='未加载权重', bg='#78350F', fg='#FDE68A')
            return

        try:
            try:
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            except TypeError:
                state = torch.load(ckpt_path, map_location=DEVICE)

            if isinstance(state, dict) and 'model_state_dict' in state:
                model_instance.load_state_dict(state['model_state_dict'])
            else:
                model_instance.load_state_dict(state)

            model_instance.to(DEVICE)
            model_instance.eval()
            self.model = model_instance
            self.status_label.config(text=f'✅ 已加载模型\n{model_type}', fg=SUCCESS)
            self.model_badge.config(text='模型已就绪', bg='#14532D', fg='#BBF7D0')
            if hasattr(self, 'lbl_model_info'):
                self.lbl_model_info.config(text=f'当前模型: {model_type}', fg=SUCCESS)
        except Exception as e:
            self.model = None
            self.status_label.config(text=f'❌ 权重加载失败\n{e}', fg=DANGER)
            self.model_badge.config(text='加载失败', bg='#7F1D1D', fg='#FECACA')
            if hasattr(self, 'lbl_model_info'):
                self.lbl_model_info.config(text='模型加载失败', fg=DANGER)

    # ================= TAB 1: 智能诊断 =================
    def _init_inference_tab(self, parent):
        topbar_card, topbar_inner = self.create_card(parent)
        topbar_card.pack(fill='x', pady=(0, 14))
        action_bar = tk.Frame(topbar_inner, bg=CARD_BG)
        action_bar.pack(fill='x', padx=18, pady=14)

        ttk.Button(action_bar, text='加载图像', style='Primary.TButton', command=self.load_image).pack(side='left', padx=(0, 10))
        ttk.Button(action_bar, text='运行推理', style='Primary.TButton', command=self.segment_image).pack(side='left', padx=10)
        ttk.Button(action_bar, text='导出十模型预测', style='Light.TButton', command=self.save_result).pack(side='left', padx=10)

        content = tk.Frame(parent, bg=APP_BG)
        content.pack(fill='both', expand=True)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # 左侧影像卡片
        vis_card, vis_inner = self.create_card(content, '影像分析视图', '左侧显示原图，右侧显示预测/叠加/GT')
        vis_card.grid(row=0, column=0, sticky='nsew', padx=(0, 14), pady=0)

        image_area = tk.Frame(vis_inner, bg=CARD_BG)
        image_area.pack(fill='both', expand=True, padx=18, pady=(0, 16))
        image_area.grid_columnconfigure(0, weight=1)
        image_area.grid_columnconfigure(1, weight=1)
        image_area.grid_rowconfigure(1, weight=1)

        tk.Label(image_area, text='原始图像', bg=CARD_BG, fg=TEXT, font=('Microsoft YaHei', 11, 'bold')).grid(row=0, column=0, sticky='w', pady=(4, 8))
        tk.Label(image_area, text='分割结果', bg=CARD_BG, fg=TEXT, font=('Microsoft YaHei', 11, 'bold')).grid(row=0, column=1, sticky='w', pady=(4, 8), padx=(12, 0))

        left_wrap = tk.Frame(image_area, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        left_wrap.grid(row=1, column=0, sticky='nsew', padx=(0, 10))
        right_wrap = tk.Frame(image_area, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        right_wrap.grid(row=1, column=1, sticky='nsew', padx=(10, 0))

        self.panel_left = tk.Label(left_wrap, bg=PANEL_BG, image=self.placeholder)
        self.panel_left.pack(fill='both', expand=True, padx=8, pady=8)
        self.panel_right = tk.Label(right_wrap, bg=PANEL_BG, image=self.placeholder)
        self.panel_right.pack(fill='both', expand=True, padx=8, pady=8)

        ctrl_bar = tk.Frame(vis_inner, bg=CARD_ALT_BG, highlightbackground=BORDER, highlightthickness=1)
        ctrl_bar.pack(fill='x', padx=18, pady=(0, 18))
        ctrl_inner = tk.Frame(ctrl_bar, bg=CARD_ALT_BG)
        ctrl_inner.pack(fill='x', padx=14, pady=12)

        ttk.Radiobutton(ctrl_inner, text='纯色预测', variable=self.view_mode, value='prediction', command=self.update_right_image).pack(side='left', padx=(0, 14))
        ttk.Radiobutton(ctrl_inner, text='叠加显示', variable=self.view_mode, value='overlay', command=self.update_right_image).pack(side='left', padx=14)
        self.rb_gt = ttk.Radiobutton(ctrl_inner, text='专家真值 GT', variable=self.view_mode, value='gt', command=self.update_right_image, state='disabled')
        self.rb_gt.pack(side='left', padx=14)
        tk.Label(ctrl_inner, text='透明度', bg=CARD_ALT_BG, fg=TEXT, font=('Microsoft YaHei', 10)).pack(side='left', padx=(20, 10))
        ttk.Scale(ctrl_inner, from_=0.1, to=1.0, variable=self.opacity, command=lambda x: self.update_right_image()).pack(side='left', fill='x', expand=True)

        # 右侧指标看板
        right_col = tk.Frame(content, bg=APP_BG)
        right_col.grid(row=0, column=1, sticky='nsew')
        right_col.grid_rowconfigure(0, weight=1)

        stat_card, stat_inner = self.create_card(right_col, '定量评估看板', '病灶负荷、三类积液面积与逐类指标')
        stat_card.grid(row=0, column=0, sticky='nsew')
        stat_inner.grid_columnconfigure(0, weight=1)
        stat_inner.grid_rowconfigure(3, weight=1)

        self.lbl_model_info = tk.Label(
            stat_inner,
            text='初始化中...',
            bg='#F8FAFC',
            fg=TEXT_MUTED,
            font=('Microsoft YaHei', 10, 'bold'),
            anchor='w',
            justify='left',
            padx=12,
            pady=8,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        self.lbl_model_info.pack(fill='x', padx=18, pady=(0, 10))

        # KPI 卡片
        kpi_frame = tk.Frame(stat_inner, bg=CARD_BG)
        kpi_frame.pack(fill='x', padx=18, pady=(0, 10))
        kpi_frame.grid_columnconfigure((0, 1), weight=1)

        self.kpi_labels = {}
        kpi_defs = [
            ('lesion_area', '病灶面积占比', '0.00%'),
            ('dominant', '主要积液类型', '--'),
            ('mean_dice', 'Mean Dice', '--'),
            ('mean_iou', 'Mean IoU', '--'),
        ]
        for idx, (key, title, value) in enumerate(kpi_defs):
            box = tk.Frame(kpi_frame, bg='#F8FAFC', highlightbackground=BORDER, highlightthickness=1)
            box.grid(row=idx // 2, column=idx % 2, sticky='nsew',
                     padx=(0 if idx % 2 == 0 else 8, 0),
                     pady=(0 if idx < 2 else 8, 0))
            tk.Label(box, text=title, bg='#F8FAFC', fg=TEXT_MUTED,
                     font=('Microsoft YaHei', 8, 'bold')).pack(anchor='w', padx=10, pady=(8, 2))
            lbl = tk.Label(box, text=value, bg='#F8FAFC', fg=TEXT,
                           font=('Microsoft YaHei', 13, 'bold'))
            lbl.pack(anchor='w', padx=10, pady=(0, 8))
            self.kpi_labels[key] = lbl

        # 面积占比图：用横向条形图替代原始饼图，读数更清晰
        chart_box = tk.Frame(stat_inner, bg=CARD_ALT_BG, highlightbackground=BORDER, highlightthickness=1)
        chart_box.pack(fill='x', padx=18, pady=(0, 10))
        self.fig_pie = Figure(figsize=(3.8, 2.15), dpi=100)
        self.ax_pie = self.fig_pie.add_subplot(111)
        self.fig_pie.patch.set_facecolor(CARD_ALT_BG)
        self.canvas_pie = FigureCanvasTkAgg(self.fig_pie, master=chart_box)
        self.canvas_pie.get_tk_widget().pack(fill='x', padx=6, pady=6)

        # 逐类指标表
        table_box = tk.Frame(stat_inner, bg=CARD_BG)
        table_box.pack(fill='x', padx=18, pady=(0, 10))

        cols = ('Category', 'Area', 'Pixels', 'Dice', 'IoU', 'Precision', 'Recall')
        self.inference_tree = ttk.Treeview(table_box, columns=cols, show='headings', height=3)
        headers = {
            'Category': '类别',
            'Area': '面积%',
            'Pixels': '像素',
            'Dice': 'Dice',
            'IoU': 'IoU',
            'Precision': 'Prec.',
            'Recall': 'Rec.',
        }
        widths = {
            'Category': 58,
            'Area': 58,
            'Pixels': 68,
            'Dice': 58,
            'IoU': 58,
            'Precision': 58,
            'Recall': 58,
        }
        for col in cols:
            self.inference_tree.heading(col, text=headers[col])
            self.inference_tree.column(col, width=widths[col], anchor='center')

        self.inference_tree.tag_configure('irf', foreground='#E11D48')
        self.inference_tree.tag_configure('srf', foreground='#16A34A')
        self.inference_tree.tag_configure('ped', foreground='#2563EB')
        self.inference_tree.pack(fill='x')

        # 自动结论提示
        self.lbl_insight = tk.Label(
            stat_inner,
            text='等待推理后生成自动结论。',
            bg='#F8FAFC',
            fg=TEXT,
            font=('Microsoft YaHei', 9),
            justify='left',
            anchor='nw',
            wraplength=355,
            padx=12,
            pady=10,
            highlightbackground=BORDER,
            highlightthickness=1
        )
        self.lbl_insight.pack(fill='both', expand=True, padx=18, pady=(0, 18))

        # 兼容旧代码中对 lbl_perf 的调用
        self.lbl_perf = self.lbl_insight

        self._reset_inference_table()
        self.update_pie_chart([0, 0, 0])

    def _reset_inference_table(self):
        if hasattr(self, 'inference_tree'):
            for item in self.inference_tree.get_children():
                self.inference_tree.delete(item)
            self.inference_tree.insert('', 'end', values=('IRF', '0.00', '0', '--', '--', '--', '--'), tags=('irf',))
            self.inference_tree.insert('', 'end', values=('SRF', '0.00', '0', '--', '--', '--', '--'), tags=('srf',))
            self.inference_tree.insert('', 'end', values=('PED', '0.00', '0', '--', '--', '--', '--'), tags=('ped',))
        if hasattr(self, 'kpi_labels'):
            self.kpi_labels['lesion_area'].config(text='0.00%')
            self.kpi_labels['dominant'].config(text='--')
            self.kpi_labels['mean_dice'].config(text='--')
            self.kpi_labels['mean_iou'].config(text='--')
        if hasattr(self, 'lbl_insight'):
            self.lbl_insight.config(text='等待推理后生成自动结论。')
        self._update_report_tab(None)


    def _init_report_tab(self, parent):
        card, inner = self.create_card(parent, '病例报告', '自动汇总当前图像、模型、分割面积与逐类指标')
        card.pack(fill='both', expand=True)

        wrap = tk.Frame(inner, bg=CARD_BG)
        wrap.pack(fill='both', expand=True, padx=18, pady=(0, 18))
        wrap.grid_columnconfigure(0, weight=2)
        wrap.grid_columnconfigure(1, weight=3)
        wrap.grid_rowconfigure(0, weight=1)

        left = tk.Frame(wrap, bg='#F8FAFC', highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        tk.Label(left, text='自动报告摘要', bg='#F8FAFC', fg=TEXT,
                 font=('Microsoft YaHei', 12, 'bold')).pack(anchor='w', padx=14, pady=(14, 6))

        self.report_text = tk.Text(left, height=18, bg='#F8FAFC', fg=TEXT,
                                   font=('Microsoft YaHei', 10), relief='flat', wrap='word')
        self.report_text.pack(fill='both', expand=True, padx=14, pady=(0, 14))
        self.report_text.insert('1.0', '完成一次推理后，这里会自动生成病例级分析摘要。')
        self.report_text.config(state='disabled')

        right = tk.Frame(wrap, bg='#F8FAFC', highlightbackground=BORDER, highlightthickness=1)
        right.grid(row=0, column=1, sticky='nsew', padx=(10, 0))

        self.fig_report = Figure(figsize=(7, 4.8), dpi=100)
        self.ax_report = self.fig_report.add_subplot(111)
        self.fig_report.patch.set_facecolor('#F8FAFC')
        self.canvas_report = FigureCanvasTkAgg(self.fig_report, master=right)
        self.canvas_report.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)
        self._update_report_tab(None)

    def _update_metric_cards(self, ratios, counts, dices=None, ious=None):
        total_lesion = sum(ratios)
        names = ['IRF', 'SRF', 'PED']

        if sum(counts) > 0:
            max_idx = int(np.argmax(counts))
            dominant = names[max_idx]
        else:
            dominant = '无明显积液'

        if hasattr(self, 'kpi_labels'):
            self.kpi_labels['lesion_area'].config(text=f'{total_lesion * 100:.2f}%')
            self.kpi_labels['dominant'].config(text=dominant)
            self.kpi_labels['mean_dice'].config(text='--' if dices is None else f'{np.mean(dices):.4f}')
            self.kpi_labels['mean_iou'].config(text='--' if ious is None else f'{np.mean(ious):.4f}')

    def _build_insight_text(self, ratios, counts, dices=None, ious=None, precs=None, recs=None):
        names = ['IRF', 'SRF', 'PED']
        total_ratio = sum(ratios)
        if sum(counts) > 0:
            dominant_idx = int(np.argmax(counts))
            dominant = names[dominant_idx]
            dominant_ratio = ratios[dominant_idx] * 100
        else:
            dominant = '无明显积液'
            dominant_ratio = 0.0

        lines = [
            f'病灶总体占比约 {total_ratio * 100:.2f}%，主要区域为 {dominant}（{dominant_ratio:.2f}%）。'
        ]

        if dices is not None:
            best_idx = int(np.argmax(dices))
            weak_idx = int(np.argmin(dices))
            lines.append(
                f'逐类分割中，{names[best_idx]} Dice 最高（{dices[best_idx]:.4f}），'
                f'{names[weak_idx]} 相对较弱（{dices[weak_idx]:.4f}）。'
            )
            lines.append(
                f'平均 Dice={np.mean(dices):.4f}，平均 IoU={np.mean(ious):.4f}。'
            )
        else:
            lines.append('当前图像未检测到配套真值标签，因此仅展示预测面积分布。')

        return '\n'.join(lines)

    def _update_report_tab(self, report):
        if not hasattr(self, 'report_text'):
            return

        self.ax_report.clear()
        self.ax_report.set_facecolor('#F8FAFC')

        self.report_text.config(state='normal')
        self.report_text.delete('1.0', 'end')

        if report is None:
            self.report_text.insert('1.0', '完成一次推理后，这里会自动生成病例级分析摘要。')
            self.ax_report.text(0.5, 0.5, 'No Case Report', ha='center', va='center',
                                transform=self.ax_report.transAxes, color=TEXT_MUTED)
        else:
            text = (
                f"图像文件：{report['image']}\n"
                f"使用模型：{report['model']}\n"
                f"设备：{DEVICE.upper()}\n\n"
                f"预测病灶面积占比：{report['total_ratio'] * 100:.2f}%\n"
                f"主要积液类型：{report['dominant']}\n\n"
                f"IRF：{report['ratios'][0] * 100:.2f}%  /  {report['counts'][0]} px\n"
                f"SRF：{report['ratios'][1] * 100:.2f}%  /  {report['counts'][1]} px\n"
                f"PED：{report['ratios'][2] * 100:.2f}%  /  {report['counts'][2]} px\n\n"
                f"自动结论：\n{report['insight']}\n"
            )
            self.report_text.insert('1.0', text)

            labels = ['IRF', 'SRF', 'PED']
            y = np.arange(3)
            vals = [r * 100 for r in report['ratios']]
            self.ax_report.barh(y, vals, color=['#EF4444', '#22C55E', '#3B82F6'])
            self.ax_report.set_yticks(y)
            self.ax_report.set_yticklabels(labels)
            self.ax_report.set_xlabel('Area Ratio (%)')
            self.ax_report.set_title('当前病例三类积液面积占比', fontsize=12, fontweight='bold')
            self.ax_report.grid(axis='x', linestyle='--', alpha=0.30)

            for i, v in enumerate(vals):
                self.ax_report.text(v + 0.03, i, f'{v:.2f}%', va='center', fontsize=9)

            self.ax_report.set_xlim(0, max(1.0, max(vals) * 1.25))

        self.report_text.config(state='disabled')
        self.fig_report.tight_layout()
        self.canvas_report.draw()

    # ================= TAB 2: 训练监控 =================
    def _init_training_tab(self, parent):
        top_card, top_inner = self.create_card(parent, '训练日志监控', '支持切换模型查看 Train Loss / Val Dice 曲线')
        top_card.pack(fill='both', expand=True)

        bar = tk.Frame(top_inner, bg=CARD_BG)
        bar.pack(fill='x', padx=18, pady=(0, 10))
        tk.Label(bar, text='选择日志模型', bg=CARD_BG, fg=TEXT, font=('Microsoft YaHei', 10, 'bold')).pack(side='left')
        combo = ttk.Combobox(bar, textvariable=self.log_model_selection, values=MODEL_DISPLAY_NAMES,
                             state='readonly', width=30)
        combo.pack(side='left', padx=12)
        combo.bind("<<ComboboxSelected>>", self.refresh_training_plot)
        ttk.Button(bar, text='刷新', style='Light.TButton', command=self.refresh_training_plot).pack(side='left', padx=8)
        self.lbl_log_status = tk.Label(bar, text='', bg=CARD_BG, fg=TEXT_MUTED, font=('Microsoft YaHei', 9))
        self.lbl_log_status.pack(side='left', padx=12)

        chart_holder = tk.Frame(top_inner, bg=CARD_ALT_BG, highlightbackground=BORDER, highlightthickness=1)
        chart_holder.pack(fill='both', expand=True, padx=18, pady=(6, 18))

        self.fig_train = Figure(figsize=(9, 6.6), dpi=100)
        self.ax_loss = self.fig_train.add_subplot(211)
        self.ax_dice = self.fig_train.add_subplot(212)
        self.fig_train.subplots_adjust(hspace=0.42, left=0.08, right=0.97, top=0.95, bottom=0.08)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=chart_holder)
        self.canvas_train.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas_train.mpl_connect('motion_notify_event', self.on_plot_hover)
        self.refresh_training_plot()

    def refresh_training_plot(self, event=None):
        selected = self.log_model_selection.get()
        config = MODEL_REGISTRY.get(selected)
        log_path = config['log'] if config else ''
        color = config['color'] if config else '#2563EB'

        self.ax_loss.clear()
        self.ax_dice.clear()
        self.annot_loss = self.ax_loss.annotate('', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
                                                bbox=dict(boxstyle='round', fc='w', ec='gray', alpha=0.95),
                                                arrowprops=dict(arrowstyle='->'))
        self.annot_loss.set_visible(False)
        self.annot_dice = self.ax_dice.annotate('', xy=(0, 0), xytext=(15, 15), textcoords='offset points',
                                                bbox=dict(boxstyle='round', fc='w', ec='gray', alpha=0.95),
                                                arrowprops=dict(arrowstyle='->'))
        self.annot_dice.set_visible(False)

        epochs, losses, dices = [], [], []
        data_loaded = False

        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if row and len(row) >= 3:
                            epochs.append(int(row[0]))
                            losses.append(float(row[1]))
                            dices.append(float(row[2]))
                data_loaded = len(epochs) > 0
                self.lbl_log_status.config(text=f'已加载: {log_path}', fg=SUCCESS)
            except Exception as e:
                self.lbl_log_status.config(text=f'日志读取失败: {e}', fg=DANGER)
        else:
            self.lbl_log_status.config(text=f'未找到日志: {log_path}', fg=WARNING)

        for ax in [self.ax_loss, self.ax_dice]:
            ax.set_facecolor('#FBFDFF')
            ax.grid(True, linestyle='--', alpha=0.35)

        if not data_loaded:
            self.ax_loss.text(0.5, 0.5, 'Log Not Found', ha='center', va='center', transform=self.ax_loss.transAxes, color=TEXT_MUTED)
            self.ax_dice.text(0.5, 0.5, 'Log Not Found', ha='center', va='center', transform=self.ax_dice.transAxes, color=TEXT_MUTED)
        else:
            self.line_loss, = self.ax_loss.plot(epochs, losses, color='#EF4444', marker='o', linewidth=1.8, markersize=4,
                                                label=f'{selected} Train Loss')
            self.ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.legend(loc='upper right')

            self.line_dice, = self.ax_dice.plot(epochs, dices, color=color, marker='s', linewidth=1.8, markersize=4,
                                                label=f'{selected} Val Dice')
            self.ax_dice.set_title('Validation Dice', fontsize=12, fontweight='bold')
            self.ax_dice.set_xlabel('Epochs')
            self.ax_dice.set_ylabel('Dice Score')
            self.ax_dice.set_ylim(0, 1.05)
            self.ax_dice.legend(loc='lower right')

            max_dice = max(dices)
            max_epoch = epochs[dices.index(max_dice)]
            self.ax_dice.annotate(f'Max: {max_dice:.4f}', xy=(max_epoch, max_dice),
                                  xytext=(max_epoch, max(0.05, max_dice - 0.14)),
                                  arrowprops=dict(facecolor='black', shrink=0.05),
                                  ha='center', fontsize=9, fontweight='bold')

        self.canvas_train.draw()

    def on_plot_hover(self, event):
        if event.inaxes == self.ax_loss and hasattr(self, 'line_loss'):
            self.update_annot(self.annot_loss, self.line_loss, event)
        elif event.inaxes == self.ax_dice and hasattr(self, 'line_dice'):
            self.update_annot(self.annot_dice, self.line_dice, event)
        else:
            if hasattr(self, 'annot_loss') and self.annot_loss.get_visible():
                self.annot_loss.set_visible(False)
                self.canvas_train.draw_idle()
            if hasattr(self, 'annot_dice') and self.annot_dice.get_visible():
                self.annot_dice.set_visible(False)
                self.canvas_train.draw_idle()

    def update_annot(self, annot, line, event):
        x, y = line.get_data()
        if event.xdata is None or len(x) == 0:
            return
        idx = min(range(len(x)), key=lambda i: abs(x[i] - event.xdata))
        if abs(x[idx] - event.xdata) > 0.5:
            if annot.get_visible():
                annot.set_visible(False)
                self.canvas_train.draw_idle()
            return
        annot.xy = (x[idx], y[idx])
        annot.set_text(f'Epoch: {int(x[idx])}\nValue: {y[idx]:.4f}')
        annot.set_visible(True)
        self.canvas_train.draw_idle()

    # ================= TAB 3: 实验对比 =================
    def _init_comparison_tab(self, parent):
        card, inner = self.create_card(parent, '实验结果对比', '整体指标与 IRF / SRF / PED 三类积液 Dice 分开展示')
        card.pack(fill='both', expand=True)

        top = tk.Frame(inner, bg=CARD_BG)
        top.pack(fill='x', padx=18, pady=(0, 8))

        ttk.Button(top, text='刷新对比数据', style='Primary.TButton',
                   command=self.load_comparison_results).pack(side='left')

        tk.Label(top, text='主指标', bg=CARD_BG, fg=TEXT,
                 font=('Microsoft YaHei', 10, 'bold')).pack(side='left', padx=(16, 6))
        metric_combo = ttk.Combobox(
            top,
            textvariable=self.compare_metric,
            values=['Dice', 'IoU', 'Mean_3Fluid_Dice', 'Mean_3Fluid_IoU', 'IRF_Dice', 'SRF_Dice', 'PED_Dice', 'IRF_IoU', 'SRF_IoU', 'PED_IoU', 'Recall', 'Precision'],
            state='readonly',
            width=12
        )
        metric_combo.pack(side='left')
        metric_combo.bind('<<ComboboxSelected>>', self.refresh_comparison_view)

        tk.Label(top, text='范围', bg=CARD_BG, fg=TEXT,
                 font=('Microsoft YaHei', 10, 'bold')).pack(side='left', padx=(16, 6))
        scope_combo = ttk.Combobox(
            top,
            textvariable=self.compare_scope,
            values=['All', 'Top 5'],
            state='readonly',
            width=8
        )
        scope_combo.pack(side='left')
        scope_combo.bind('<<ComboboxSelected>>', self.refresh_comparison_view)

        self.lbl_comp_status = tk.Label(top, text='', bg=CARD_BG, fg=TEXT_MUTED,
                                        font=('Microsoft YaHei', 9))
        self.lbl_comp_status.pack(side='left', padx=12)

        self.lbl_comp_summary = tk.Label(
            inner,
            text='尚未加载实验结果',
            bg='#F8FAFC',
            fg=TEXT,
            anchor='w',
            font=('Microsoft YaHei', 10, 'bold'),
            highlightbackground=BORDER,
            highlightthickness=1
        )
        self.lbl_comp_summary.pack(fill='x', padx=18, pady=(0, 10), ipady=8, ipadx=10)

        charts_area = tk.Frame(inner, bg=CARD_BG)
        charts_area.pack(fill='both', expand=True, padx=18, pady=(0, 10))
        charts_area.grid_rowconfigure(0, weight=1)
        charts_area.grid_rowconfigure(1, weight=1)
        charts_area.grid_columnconfigure(0, weight=1)

        plot_holder_main = tk.Frame(charts_area, bg=CARD_ALT_BG, highlightbackground=BORDER, highlightthickness=1)
        plot_holder_main.grid(row=0, column=0, sticky='nsew', pady=(0, 8))

        self.fig_cmp_main = Figure(figsize=(11.5, 3.4), dpi=100)
        self.ax_cmp_main = self.fig_cmp_main.add_subplot(111)
        self.canvas_cmp_main = FigureCanvasTkAgg(self.fig_cmp_main, master=plot_holder_main)
        self.canvas_cmp_main.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)

        plot_holder_class = tk.Frame(charts_area, bg=CARD_ALT_BG, highlightbackground=BORDER, highlightthickness=1)
        plot_holder_class.grid(row=1, column=0, sticky='nsew')

        self.fig_cmp_radar = Figure(figsize=(11.5, 3.4), dpi=100)
        self.ax_cmp_radar = self.fig_cmp_radar.add_subplot(111)
        self.canvas_cmp_radar = FigureCanvasTkAgg(self.fig_cmp_radar, master=plot_holder_class)
        self.canvas_cmp_radar.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)

        table_holder = tk.Frame(inner, bg=CARD_ALT_BG, highlightbackground=BORDER, highlightthickness=1)
        table_holder.pack(fill='x', padx=18, pady=(0, 18))

        cols = ('Rank', 'Model', 'Dice', 'IoU', 'IRF Dice', 'SRF Dice', 'PED Dice', 'Recall', 'Precision', 'ΔDice')
        self.tree = ttk.Treeview(table_holder, columns=cols, show='headings', height=5)

        widths = {
            'Rank': 55,
            'Model': 230,
            'Dice': 78,
            'IoU': 78,
            'IRF Dice': 82,
            'SRF Dice': 82,
            'PED Dice': 82,
            'Recall': 78,
            'Precision': 88,
            'ΔDice': 82,
        }
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor='center')

        self.tree.tag_configure('best', background='#ECFDF5')
        self.tree.tag_configure('second', background='#EFF6FF')
        self.tree.tag_configure('third', background='#FFF7ED')
        self.tree.tag_configure('normal', background='white')

        yscrollbar = ttk.Scrollbar(table_holder, orient='vertical', command=self.tree.yview)
        xscrollbar = ttk.Scrollbar(table_holder, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscroll=yscrollbar.set, xscroll=xscrollbar.set)

        yscrollbar.pack(side='right', fill='y')
        xscrollbar.pack(side='bottom', fill='x')
        self.tree.pack(fill='x', expand=False, padx=6, pady=6)

        self.load_comparison_results()


    def _normalize_model_name(self, name):
        name = str(name).strip()
        mapping = {
            'SegNet': 'SegNet',
            'segnet': 'SegNet',
            'Standard U-Net (Baseline)': 'Standard U-Net (Base)',
            'Standard U-Net (Base)': 'Standard U-Net (Base)',
            'U-Net': 'Standard U-Net (Base)',
            'UNet': 'Standard U-Net (Base)',
            'U-Net (Base)': 'Standard U-Net (Base)',
            'Attention U-Net': 'Attention U-Net',
            'AttentionUNet': 'Attention U-Net',
            'Multi-Task SDA-UNet': 'nnU-Net + SDA/SASC',
            'nnU-Net': 'nnU-Net + SDA/SASC',
            'nnUnet': 'nnU-Net + SDA/SASC',
            'nnUNet': 'nnU-Net + SDA/SASC',
            'nnU-Net Base': 'nnU-Net Base',
            'nnU-Net (Base)': 'nnU-Net Base',
            'nnUNet Base': 'nnU-Net Base',
            'nnU-Net + ECA': 'nnU-Net + ECA-Encoder',
            'nnU-Net + ECA-Encoder': 'nnU-Net + ECA-Encoder',
            'nnU-Net + CBAM': 'nnU-Net + CBAM-Decoder',
            'nnU-Net + CBAM-Decoder': 'nnU-Net + CBAM-Decoder',
            'nnU-Net + ECA/CBAM': 'nnU-Net + ECA/CBAM',
            'nnU-Net + ECA-Encoder + CBAM-Decoder': 'nnU-Net + ECA/CBAM',
            'nnU-Net + ECA-CBAM': 'nnU-Net + ECA/CBAM',
            'nnU-Net + ECA_CBAM': 'nnU-Net + ECA/CBAM',
            'nnU-Net + SDA': 'nnU-Net + SDA',
            'nnU-Net + SASC': 'nnU-Net + SASC',
            'nnU-Net + SDA/SASC': 'nnU-Net + SDA/SASC',
        }
        return mapping.get(name, name)

    def _short_model_name(self, name):
        mapping = {
            'SegNet': 'SegNet',
            'Standard U-Net (Base)': 'U-Net',
            'Attention U-Net': 'Attention U-Net',
            'nnU-Net Base': 'nnU-Net Base',
            'nnU-Net + ECA-Encoder': '+ECA Encoder',
            'nnU-Net + CBAM-Decoder': '+CBAM Decoder',
            'nnU-Net + ECA/CBAM': '+ECA/CBAM',
            'nnU-Net + SDA': '+SDA',
            'nnU-Net + SASC': '+SASC',
            'nnU-Net + SDA/SASC': '+SDA/SASC',
        }
        return mapping.get(str(name), str(name))

    def _update_summary_cards(self, df):
        if not hasattr(self, 'lbl_comp_summary'):
            return

        metric = self.compare_metric.get()
        if df is None or df.empty:
            self.lbl_comp_summary.config(text='尚未加载实验结果')
            return

        best_row = df.sort_values(metric, ascending=False).iloc[0]
        best_iou_row = df.sort_values('IoU', ascending=False).iloc[0] if 'IoU' in df.columns else best_row

        if 'nnU-Net Base' in df['Model'].astype(str).values:
            base_dice = float(df[df['Model'].astype(str) == 'nnU-Net Base']['Dice'].iloc[0])
            delta = float(best_row['Dice'] - base_dice)
            delta_text = f"；相对 nnU-Net Base 的 Dice 变化：{delta:+.4f}"
        else:
            delta_text = ""

        text = (
            f"最佳模型：{best_row['Model']}    "
            f"{metric} = {best_row[metric]:.4f}    "
            f"最高 IoU：{best_iou_row['IoU']:.4f}"
            f"{delta_text}"
        )
        self.lbl_comp_summary.config(text=text)

    def _draw_comparison_dashboard(self, df):
        metric = self.compare_metric.get()
        scope = self.compare_scope.get()

        plot_df = df.copy().sort_values(metric, ascending=False)
        if scope == 'Top 5':
            plot_df = plot_df.head(5)

        # 图 1：主指标排名图
        self.ax_cmp_main.clear()
        self.ax_cmp_main.set_facecolor('#FBFDFF')
        self.ax_cmp_main.grid(axis='x', linestyle='--', alpha=0.30)

        if plot_df.empty:
            self.ax_cmp_main.text(
                0.5, 0.5, 'No Comparison Data',
                ha='center', va='center',
                transform=self.ax_cmp_main.transAxes,
                color=TEXT_MUTED
            )
        else:
            labels = [self._short_model_name(m) for m in plot_df['Model']]
            values = plot_df[metric].astype(float).values
            colors = ['#2563EB'] * len(plot_df)
            colors[0] = '#16A34A'

            y = np.arange(len(labels))
            bars = self.ax_cmp_main.barh(y, values, color=colors, edgecolor='none', height=0.58)
            self.ax_cmp_main.set_yticks(y)
            self.ax_cmp_main.set_yticklabels(labels, fontsize=9)
            self.ax_cmp_main.invert_yaxis()
            self.ax_cmp_main.set_xlim(0, 1.05)
            self.ax_cmp_main.set_xlabel(metric)
            self.ax_cmp_main.set_title(f'{metric} 排名对比', fontsize=12, fontweight='bold')

            for idx, (bar, val) in enumerate(zip(bars, values)):
                suffix = '  ★' if idx == 0 else ''
                self.ax_cmp_main.text(
                    val + 0.006,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}{suffix}',
                    va='center',
                    fontsize=9,
                    color=TEXT
                )

        self.fig_cmp_main.tight_layout()
        self.canvas_cmp_main.draw()

        # 图 2：三类积液 Dice 分组柱状图
        self.ax_cmp_radar.clear()
        self.ax_cmp_radar.set_facecolor('#FBFDFF')
        class_cols = ['IRF_Dice', 'SRF_Dice', 'PED_Dice']
        class_labels = ['IRF', 'SRF', 'PED']

        if all(col in df.columns for col in class_cols):
            lesion_df = df.copy().sort_values('Dice', ascending=False)
            if scope == 'Top 5':
                lesion_df = lesion_df.head(5)

            labels = [self._short_model_name(m) for m in lesion_df['Model']]
            x = np.arange(len(lesion_df))
            width = 0.24

            for i, (col, label) in enumerate(zip(class_cols, class_labels)):
                vals = lesion_df[col].astype(float).values
                bars = self.ax_cmp_radar.bar(x + (i - 1) * width, vals, width, label=label)

                for rect, val in zip(bars, vals):
                    self.ax_cmp_radar.text(
                        rect.get_x() + rect.get_width() / 2,
                        val + 0.004,
                        f'{val:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=7
                    )

            self.ax_cmp_radar.set_xticks(x)
            self.ax_cmp_radar.set_xticklabels(labels, rotation=0, ha='center', fontsize=8)
            self.ax_cmp_radar.set_ylim(0.75, 0.96)
            self.ax_cmp_radar.set_ylabel('Dice')
            self.ax_cmp_radar.set_title('IRF / SRF / PED 三类积液 Dice 对比', fontsize=12, fontweight='bold')
            self.ax_cmp_radar.grid(axis='y', linestyle='--', alpha=0.30)
            self.ax_cmp_radar.legend(loc='upper center', ncol=3, fontsize=8, frameon=False)
        else:
            self.ax_cmp_radar.text(
                0.5, 0.5,
                'CSV 中未找到 IRF_Dice / SRF_Dice / PED_Dice',
                ha='center',
                va='center',
                transform=self.ax_cmp_radar.transAxes,
                color=TEXT_MUTED,
                wrap=True
            )

        self.fig_cmp_radar.tight_layout()
        self.canvas_cmp_radar.draw()


    def refresh_comparison_view(self, event=None):
        if self.comparison_df is None:
            return
        df = self.comparison_df.copy()
        if df.empty:
            return

        metric = self.compare_metric.get()
        if metric not in df.columns:
            metric = 'Dice'
            self.compare_metric.set(metric)

        for item in self.tree.get_children():
            self.tree.delete(item)

        rank_df = df.sort_values(metric, ascending=False).reset_index(drop=True).copy()

        if 'nnU-Net Base' in df['Model'].astype(str).values:
            base_dice = float(df[df['Model'].astype(str) == 'nnU-Net Base']['Dice'].iloc[0])
        else:
            base_dice = np.nan

        for idx, (_, row) in enumerate(rank_df.iterrows(), start=1):
            delta = row['Dice'] - base_dice if pd.notna(row['Dice']) and pd.notna(base_dice) else np.nan
            values = [
                idx,
                self._short_model_name(row['Model']),
                f"{row['Dice']:.4f}" if pd.notna(row['Dice']) else 'N/A',
                f"{row['IoU']:.4f}" if pd.notna(row['IoU']) else 'N/A',
                f"{row['IRF_Dice']:.4f}" if 'IRF_Dice' in row and pd.notna(row['IRF_Dice']) else 'N/A',
                f"{row['SRF_Dice']:.4f}" if 'SRF_Dice' in row and pd.notna(row['SRF_Dice']) else 'N/A',
                f"{row['PED_Dice']:.4f}" if 'PED_Dice' in row and pd.notna(row['PED_Dice']) else 'N/A',
                f"{row['Recall']:.4f}" if pd.notna(row['Recall']) else 'N/A',
                f"{row['Precision']:.4f}" if pd.notna(row['Precision']) else 'N/A',
                f"{delta:+.4f}" if pd.notna(delta) else 'N/A',
            ]
            tag = 'normal'
            if idx == 1:
                tag = 'best'
            elif idx == 2:
                tag = 'second'
            elif idx == 3:
                tag = 'third'
            self.tree.insert('', 'end', values=values, tags=(tag,))

        self._update_summary_cards(df)
        self._draw_comparison_dashboard(df)

    def load_comparison_results(self):
        candidate_csvs = [
            'all_models_results_oldstyle.csv',
            'model_comparison_results_oldstyle_no_aspp.csv',
            'baseline_models_results_oldstyle.csv',
            'ablation_nnunet_results_oldstyle.csv',
            'model_comparison_results.csv',
            'model_comparison_results_corrected.csv',
            'model_comparison_results_detailed.csv'
        ]
        csv_path = next((p for p in candidate_csvs if os.path.exists(p)), None)

        self.ax_cmp_main.clear()
        self.ax_cmp_radar.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

        if csv_path is None:
            self.lbl_comp_status.config(text='未找到实验对比 CSV', fg=WARNING)
            self.ax_cmp_main.text(0.5, 0.5, 'No Comparison Data Found', ha='center', va='center', transform=self.ax_cmp_main.transAxes, color=TEXT_MUTED)
            self.ax_cmp_radar.text(0.5, 0.5, 'No Radar Data', ha='center', va='center', transform=self.ax_cmp_radar.transAxes, color=TEXT_MUTED)
            self.canvas_cmp_main.draw()
            self.canvas_cmp_radar.draw()
            self.comparison_df = pd.DataFrame()
            self._update_summary_cards(self.comparison_df)
            return

        try:
            df = pd.read_csv(csv_path)
            if 'Model' not in df.columns:
                raise ValueError('CSV 缺少 Model 列')

            df['Model'] = df['Model'].apply(self._normalize_model_name)
            aliases = {
                'mDice': 'Dice', 'Mean Dice': 'Dice', 'mean_dice': 'Dice', 'val_dice': 'Dice',
                'mIoU': 'IoU', 'Mean IoU': 'IoU', 'mean_iou': 'IoU',
                'Acc': 'Accuracy', 'Pixel Accuracy': 'Accuracy', 'PA': 'Accuracy',
            }
            df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
            df = df[df['Model'].isin(COMPARISON_KEEP_ORDER)].copy()

            required = ['Model', 'Dice', 'IoU', 'Recall', 'Precision']
            for col in required:
                if col not in df.columns:
                    if col in ['Recall', 'Precision']:
                        df[col] = np.nan
                    else:
                        raise ValueError(f'CSV 缺少 {col} 列')

            for col in ['IRF_Dice', 'SRF_Dice', 'PED_Dice']:
                if col not in df.columns:
                    df[col] = np.nan

            # 同名模型保留最后一条，避免 CSV 多次追加导致重复展示
            df = df.drop_duplicates(subset=['Model'], keep='last')

            df['Model'] = pd.Categorical(df['Model'], categories=COMPARISON_KEEP_ORDER, ordered=True)
            df = df.sort_values('Model')
            self.comparison_df = df.reset_index(drop=True)
            self.lbl_comp_status.config(text=f'已加载 {len(df)} 个模型结果 | 来源: {os.path.basename(csv_path)}', fg=SUCCESS)
            self.refresh_comparison_view()
        except Exception as e:
            self.lbl_comp_status.config(text=f'读取错误: {e}', fg=DANGER)
            self.ax_cmp_main.text(0.5, 0.5, 'CSV Read Error', ha='center', va='center', transform=self.ax_cmp_main.transAxes, color=TEXT_MUTED)
            self.ax_cmp_radar.text(0.5, 0.5, 'CSV Read Error', ha='center', va='center', transform=self.ax_cmp_radar.transAxes, color=TEXT_MUTED)
            self.canvas_cmp_main.draw()
            self.canvas_cmp_radar.draw()
            self.comparison_df = pd.DataFrame()
            self._update_summary_cards(self.comparison_df)
    # ================= 推理业务逻辑 =================
    def update_pie_chart(self, ratios):
        self.ax_pie.clear()
        self.ax_pie.set_facecolor(CARD_ALT_BG)

        names = ['IRF', 'SRF', 'PED']
        values = [float(r) * 100 for r in ratios]
        colors = ['#EF4444', '#22C55E', '#3B82F6']
        y = np.arange(len(names))

        self.ax_pie.barh(y, values, color=colors, height=0.55)
        self.ax_pie.set_yticks(y)
        self.ax_pie.set_yticklabels(names, fontsize=9)
        self.ax_pie.set_xlabel('Area (%)', fontsize=8)
        self.ax_pie.set_title('三类积液预测面积占比', fontsize=10, fontweight='bold')
        self.ax_pie.grid(axis='x', linestyle='--', alpha=0.25)
        self.ax_pie.set_xlim(0, max(1.0, max(values) * 1.25 if values else 1.0))

        for i, v in enumerate(values):
            self.ax_pie.text(v + 0.02, i, f'{v:.2f}%', va='center', fontsize=8)

        self.fig_pie.tight_layout()
        self.canvas_pie.draw()


    def _resize_img_to_display(self, pil_img):
        ratio = min(IMG_DISPLAY_SIZE[0] / pil_img.size[0], IMG_DISPLAY_SIZE[1] / pil_img.size[1])
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

    def load_image(self):
        initial_dir = r'D:\Graduation_Design\Dataset_Unified\val\images'
        if not os.path.exists(initial_dir):
            initial_dir = '/'
        file_path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp')])
        if not file_path:
            return

        self.current_image_path = file_path
        possible = file_path.replace('images', 'masks')
        if os.path.exists(possible):
            self.current_mask_path = possible
        else:
            parent = os.path.dirname(os.path.dirname(file_path))
            masks_dir = os.path.join(parent, 'masks')
            base = os.path.basename(file_path)
            cand = os.path.join(masks_dir, base)
            self.current_mask_path = cand if os.path.exists(cand) else None

        img = Image.open(file_path).convert('RGB')
        self.origin_full_pil = img.copy()
        self.origin_img_pil = self._resize_img_to_display(img)
        tk_img = ImageTk.PhotoImage(self.origin_img_pil)
        self.panel_left.config(image=tk_img)
        self.panel_left.image = tk_img

        self.pred_mask_pil = None
        self.pred_mask_full_pil = None
        self.pred_label_256 = None
        self.pred_label_full = None
        self.overlay_full_pil = None
        self.current_right_pil = None
        self.panel_right.config(image=self.placeholder)
        self.panel_right.image = self.placeholder
        self._reset_inference_table()
        self.update_pie_chart([0, 0, 0])

        if self.current_mask_path:
            self.rb_gt.config(state='normal')
            status = f'已加载图像\n{os.path.basename(file_path)}\n检测到配套标签'
        else:
            self.rb_gt.config(state='disabled')
            if self.view_mode.get() == 'gt':
                self.view_mode.set('overlay')
            status = f'已加载图像\n{os.path.basename(file_path)}\n未检测到标签'
        self.status_label.config(text=status, fg='#E2E8F0')

    def segment_image(self):
        if not self.model or not self.current_image_path:
            messagebox.showinfo('提示', '请先加载模型和图像')
            return

        origin_gray = Image.open(self.current_image_path).convert('L')
        img_in = origin_gray.resize((INFERENCE_SIZE, INFERENCE_SIZE))
        tensor = transforms.ToTensor()(img_in).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, dict):
                output = output['segmentation']
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # 保存 256x256 标签预测，以及原始分辨率预测
        self.pred_label_256 = pred.astype(np.uint8)

        if self.origin_full_pil is None:
            self.origin_full_pil = Image.open(self.current_image_path).convert('RGB')

        pred_label_full_img = Image.fromarray(self.pred_label_256, mode='L').resize(
            self.origin_full_pil.size,
            Image.Resampling.NEAREST
        )
        self.pred_label_full = np.array(pred_label_full_img, dtype=np.uint8)

        # 原始分辨率彩色预测图
        self.pred_mask_full_pil = Image.fromarray(COLORS[self.pred_label_full])

        # 界面显示用预测图
        self.pred_mask_pil = self.pred_mask_full_pil.resize(
            self.origin_img_pil.size,
            Image.Resampling.NEAREST
        )

        # 原始分辨率叠加图，用于导出
        self.overlay_full_pil = self._make_overlay_image(
            self.origin_full_pil,
            self.pred_mask_full_pil,
            alpha=self.opacity.get()
        )

        pixels = pred.size
        counts = [(pred == i).sum() for i in range(1, 4)]
        ratios = [c / pixels for c in counts]

        names = ['IRF', 'SRF', 'PED']
        tags = ['irf', 'srf', 'ped']

        dices = ious = precs = recs = None

        if self.current_mask_path:
            gt = np.array(Image.open(self.current_mask_path).convert('L').resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.Resampling.NEAREST), dtype=np.int64)
            gt[gt < 0] = 0
            gt[gt >= NUM_CLASSES] = 0
            pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            dices, ious, precs, recs = [], [], [], []
            for c in range(1, NUM_CLASSES):
                p = (pred_256 == c).astype(float)
                t = (gt == c).astype(float)
                inter = (p * t).sum()
                union = p.sum() + t.sum()
                pred_sum = p.sum()
                gt_sum = t.sum()

                dice = (2 * inter) / (union + 1e-6) if union > 0 else 1.0
                iou = inter / (p.sum() + t.sum() - inter + 1e-6) if (p.sum() + t.sum() - inter) > 0 else 1.0
                prec = inter / (pred_sum + 1e-6) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
                rec = inter / (gt_sum + 1e-6) if gt_sum > 0 else 1.0

                dices.append(dice)
                ious.append(iou)
                precs.append(prec)
                recs.append(rec)

        for item in self.inference_tree.get_children():
            self.inference_tree.delete(item)

        for i, name in enumerate(names):
            values = (
                name,
                f'{ratios[i] * 100:.2f}',
                f'{int(counts[i])}',
                '--' if dices is None else f'{dices[i]:.4f}',
                '--' if ious is None else f'{ious[i]:.4f}',
                '--' if precs is None else f'{precs[i]:.4f}',
                '--' if recs is None else f'{recs[i]:.4f}',
            )
            self.inference_tree.insert('', 'end', values=values, tags=(tags[i],))

        self.update_pie_chart(ratios)
        self._update_metric_cards(ratios, counts, dices, ious)

        insight = self._build_insight_text(ratios, counts, dices, ious, precs, recs)
        self.lbl_insight.config(text=insight)

        dominant = names[int(np.argmax(counts))] if sum(counts) > 0 else '无明显积液'
        report = {
            'image': os.path.basename(self.current_image_path),
            'model': self.selected_model.get(),
            'ratios': ratios,
            'counts': [int(c) for c in counts],
            'total_ratio': float(sum(ratios)),
            'dominant': dominant,
            'insight': insight,
        }
        self._update_report_tab(report)
        self.update_right_image()
        self.status_label.config(text='✅ 推理分析完成', fg='#E2E8F0')


    def _make_overlay_image(self, origin_pil, mask_pil, alpha=0.45):
        """生成叠加图：背景区域保持原图，病灶区域按 alpha 叠加彩色 mask。"""
        origin = np.array(origin_pil.convert('RGB'))
        mask = np.array(mask_pil.convert('RGB'))

        if mask.shape != origin.shape:
            mask_pil = mask_pil.resize(origin_pil.size, Image.Resampling.NEAREST)
            mask = np.array(mask_pil.convert('RGB'))

        is_bg = np.all(mask == [0, 0, 0], axis=-1)
        blended = origin.copy()
        blended[~is_bg] = (
            origin[~is_bg] * (1 - alpha) + mask[~is_bg] * alpha
        ).astype(np.uint8)

        return Image.fromarray(blended)

    def _get_current_display_image(self):
        """返回当前右侧模式对应的显示图，用于单图导出。"""
        if not self.pred_mask_pil:
            return None

        mode = self.view_mode.get()
        if mode == 'prediction':
            return self.pred_mask_pil.copy()
        if mode == 'overlay':
            return self._make_overlay_image(
                self.origin_img_pil,
                self.pred_mask_pil,
                alpha=self.opacity.get()
            )
        if mode == 'gt' and self.current_mask_path:
            gt = Image.open(self.current_mask_path).convert('L').resize(
                self.origin_img_pil.size,
                Image.Resampling.NEAREST
            )
            gt_np = np.array(gt, dtype=np.int64)
            gt_np[gt_np < 0] = 0
            gt_np[gt_np >= NUM_CLASSES] = 0
            return Image.fromarray(COLORS[gt_np])
        return self.pred_mask_pil.copy()


    def _safe_filename(self, name):
        """Windows 文件名安全化。"""
        mapping = {
            "Standard U-Net (Base)": "01_Standard_UNet_Base",
            "Attention U-Net": "02_Attention_UNet",
            "nnU-Net Base": "03_nnUNet_Base",
            "nnU-Net + ECA-Encoder": "04_nnUNet_ECA_Encoder",
            "nnU-Net + CBAM-Decoder": "05_nnUNet_CBAM_Decoder",
            "nnU-Net + ECA/CBAM": "06_nnUNet_ECA_CBAM",
            "nnU-Net + SDA": "07_nnUNet_SDA",
            "nnU-Net + SASC": "08_nnUNet_SASC",
            "nnU-Net + SDA/SASC": "09_nnUNet_SDA_SASC",
        }
        if name in mapping:
            return mapping[name]
        safe = str(name)
        for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ', '+']:
            safe = safe.replace(ch, '_')
        while '__' in safe:
            safe = safe.replace('__', '_')
        return safe.strip('_')

    def _load_model_for_export(self, model_name):
        """批量导出时单独加载模型，不改变当前 GUI 中 self.model 的状态。"""
        config = MODEL_REGISTRY.get(model_name)
        if config is None:
            raise ValueError(f"未知模型: {model_name}")

        ckpt_path = config['checkpoint']
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到权重: {ckpt_path}")

        model = self._build_model_instance(model_name)

        try:
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location=DEVICE)

        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

        model.to(DEVICE)
        model.eval()
        return model

    def _predict_one_model_for_export(self, model):
        """对当前切片执行一次模型预测，返回 label/full-color/overlay/统计。"""
        origin_gray = Image.open(self.current_image_path).convert('L')
        img_in = origin_gray.resize((INFERENCE_SIZE, INFERENCE_SIZE))
        tensor = transforms.ToTensor()(img_in).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            if isinstance(output, dict):
                output = output['segmentation']
            pred_256 = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        if self.origin_full_pil is None:
            self.origin_full_pil = Image.open(self.current_image_path).convert('RGB')

        pred_label_full_img = Image.fromarray(pred_256, mode='L').resize(
            self.origin_full_pil.size,
            Image.Resampling.NEAREST
        )
        pred_label_full = np.array(pred_label_full_img, dtype=np.uint8)
        pred_color_full = Image.fromarray(COLORS[pred_label_full])
        overlay_full = self._make_overlay_image(
            self.origin_full_pil,
            pred_color_full,
            alpha=self.opacity.get()
        )

        total_pixels = max(pred_label_full.size, 1)
        counts = [int((pred_label_full == c).sum()) for c in range(1, NUM_CLASSES)]
        ratios = [c / total_pixels for c in counts]

        return pred_256, pred_label_full, pred_color_full, overlay_full, counts, ratios

    def update_right_image(self):
        if not self.pred_mask_pil:
            return

        final = self._get_current_display_image()
        if final is None:
            return

        self.current_right_pil = final.copy()
        tk_img = ImageTk.PhotoImage(final)
        self.panel_right.config(image=tk_img)
        self.panel_right.image = tk_img


    def save_result(self):
        """
        批量导出当前切片在全部已配置模型下的预测结果。
        固定保存到：
            D:\Graduation_Design\预测\<当前切片文件名>\
        每个模型导出：
            *_pred_mask_color.png
            *_pred_mask_label.png
            *_overlay.png
        同时导出 original.png、gt_mask_color.png（如果有标签）、export_summary.csv。
        """
        if not self.current_image_path:
            messagebox.showinfo('提示', '请先加载一张 OCT 切片图像')
            return

        if self.origin_full_pil is None:
            self.origin_full_pil = Image.open(self.current_image_path).convert('RGB')

        image_stem = os.path.splitext(os.path.basename(self.current_image_path))[0]
        export_root = r"D:\Graduation_Design\预测"
        export_dir = os.path.join(export_root, image_stem)
        os.makedirs(export_dir, exist_ok=True)

        # 原图只保存一次
        self.origin_full_pil.save(os.path.join(export_dir, "original.png"))

        # GT 彩色图：如果存在配套标签，也保存一次
        if self.current_mask_path and os.path.exists(self.current_mask_path):
            gt = Image.open(self.current_mask_path).convert('L').resize(
                self.origin_full_pil.size,
                Image.Resampling.NEAREST
            )
            gt_np = np.array(gt, dtype=np.int64)
            gt_np[gt_np < 0] = 0
            gt_np[gt_np >= NUM_CLASSES] = 0
            Image.fromarray(COLORS[gt_np]).save(os.path.join(export_dir, "gt_mask_color.png"))

        summary_rows = []
        success_count = 0
        failed = []

        self.status_label.config(text='正在导出十模型预测结果...', fg='#E2E8F0')
        self.root.update_idletasks()

        for model_name in MODEL_DISPLAY_NAMES:
            safe_name = self._safe_filename(model_name)

            try:
                model = self._load_model_for_export(model_name)
                pred_256, pred_label_full, pred_color_full, overlay_full, counts, ratios = \
                    self._predict_one_model_for_export(model)

                pred_color_full.save(os.path.join(export_dir, f"{safe_name}_pred_mask_color.png"))
                Image.fromarray(pred_label_full.astype(np.uint8), mode='L').save(
                    os.path.join(export_dir, f"{safe_name}_pred_mask_label.png")
                )
                overlay_full.save(os.path.join(export_dir, f"{safe_name}_overlay.png"))

                summary_rows.append({
                    'Model': model_name,
                    'Status': 'OK',
                    'Pred_Color': f"{safe_name}_pred_mask_color.png",
                    'Pred_Label': f"{safe_name}_pred_mask_label.png",
                    'Overlay': f"{safe_name}_overlay.png",
                    'IRF_Pixels': counts[0],
                    'SRF_Pixels': counts[1],
                    'PED_Pixels': counts[2],
                    'IRF_Ratio_%': ratios[0] * 100,
                    'SRF_Ratio_%': ratios[1] * 100,
                    'PED_Ratio_%': ratios[2] * 100,
                    'Checkpoint': MODEL_REGISTRY[model_name]['checkpoint'],
                })

                success_count += 1

                # 如果导出的模型正好是当前选择的模型，同步更新 GUI 里的当前预测
                if model_name == self.selected_model.get():
                    self.pred_label_256 = pred_256
                    self.pred_label_full = pred_label_full
                    self.pred_mask_full_pil = pred_color_full
                    self.overlay_full_pil = overlay_full
                    self.pred_mask_pil = pred_color_full.resize(self.origin_img_pil.size, Image.Resampling.NEAREST)
                    self.update_right_image()

            except Exception as e:
                failed.append(f"{model_name}: {e}")
                summary_rows.append({
                    'Model': model_name,
                    'Status': f'FAILED: {e}',
                    'Pred_Color': '',
                    'Pred_Label': '',
                    'Overlay': '',
                    'IRF_Pixels': '',
                    'SRF_Pixels': '',
                    'PED_Pixels': '',
                    'IRF_Ratio_%': '',
                    'SRF_Ratio_%': '',
                    'PED_Ratio_%': '',
                    'Checkpoint': MODEL_REGISTRY.get(model_name, {}).get('checkpoint', ''),
                })

        # 汇总 CSV
        try:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(export_dir, "export_summary.csv"),
                index=False,
                encoding='utf-8-sig'
            )
        except Exception:
            # pandas 异常时兜底写 txt
            pass

        # 导出说明 txt
        info_lines = [
            "OCT Segmentation Batch Export",
            f"Image: {os.path.basename(self.current_image_path)}",
            f"Export directory: {export_dir}",
            "",
            "Files:",
            "  original.png                         原始切片图像",
            "  gt_mask_color.png                    专家标签彩色图，如果存在",
            "  *_pred_mask_color.png                各模型彩色预测图",
            "  *_pred_mask_label.png                各模型单通道预测标签图，0=背景,1=IRF,2=SRF,3=PED",
            "  *_overlay.png                        各模型预测结果叠加图",
            "  export_summary.csv                   各模型导出状态与三类积液面积统计",
            "",
            f"Success: {success_count}/{len(MODEL_DISPLAY_NAMES)}",
        ]

        if failed:
            info_lines.append("")
            info_lines.append("Failed models:")
            info_lines.extend([f"  {x}" for x in failed])

        with open(os.path.join(export_dir, "export_info.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(info_lines))

        if failed:
            msg = (
                f"已导出 {success_count}/{len(MODEL_DISPLAY_NAMES)} 个模型结果。\n"
                f"保存路径：\n{export_dir}\n\n"
                f"有模型失败，详情见 export_info.txt"
            )
            self.status_label.config(text=f'⚠️ 批量导出完成，但部分模型失败\n{export_dir}', fg=WARNING)
            messagebox.showwarning('导出完成', msg)
        else:
            msg = f"全部模型预测图和叠加图已导出到：\n{export_dir}"
            self.status_label.config(text=f'✅ 全部模型预测结果已导出\n{export_dir}', fg='#E2E8F0')
            messagebox.showinfo('导出完成', msg)



if __name__ == '__main__':
    root = tk.Tk()
    app = OCTSystemGUI(root)
    root.mainloop()
