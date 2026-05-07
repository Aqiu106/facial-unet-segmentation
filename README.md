# Facial U-Net Segmentation（面部语义分割）

基于 **Attention Gate U-Net** 的多类别面部语义分割课程 / 实验项目，支持残差块、瓶颈上下文模块（ASPP 风格）、深监督与断点续训。

远程仓库：<https://github.com/Aqiu106/facial-unet-segmentation>

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **模型** | `unet_pkg.models.UNet`：Attention Gate、可选 **Residual / DoubleConv**、`batch` / `group` 归一化、瓶颈 **`bottleneck_context`**、**深监督** 与辅助损失 |
| **训练** | `scripts/train.py`：AMP、`resume_from`（含与新结构不完全匹配时的**部分加载**提示）、验证指标与可视化 |
| **评估** | `scripts/evaluate.py`：按 checkpoint 内 **`config`** 重建同构模型；输出 `test_results.json`、`test_metrics.png`、`test_visualization.png` |
| **数据** | HELEN 转换、划分 train/val、目录约定见下文 |

---

## 环境要求

- **Python** ≥ 3.10  
- **PyTorch**（建议与 CUDA 版本匹配；无 GPU 可将配置中 `device` 设为 `cpu`）  
- 推荐使用虚拟环境（`venv` / `conda`）

### 依赖安装示例

在仓库根目录执行：

```bash
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按你的 CUDA 调整
pip install albumentations opencv-python-headless pyyaml tqdm matplotlib numpy
```

说明：`requirements.txt` 若为空或不全，请以实际环境与上述包为准。若遇 **NumPy 2.x** 与旧版 PyTorch 二进制不兼容，可固定 `numpy<2`。

---

## 数据集目录结构

默认数据根目录由配置项 **`data_dir`** 指定（示例为仓库下的 `data/`）。

```
data/
├── class_info.yaml          # 类别数、类别名称等（训练/评估会读取）
├── train/
│   ├── images/              # .jpg / .png ...
│   └── masks/               # 与图像 stem 同名的单通道 png，像素值为类别 id
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

示例 `class_info.yaml` 字段：`num_classes`、`class_names`；可与划分脚本生成的统计信息一并维护。

---

## 训练

**须在仓库根目录运行**，以保证 `configs/`、`data/` 等相对路径正确。

```bash
python scripts/train.py --config configs/train_config.yaml
```

常用覆盖示例：

```bash
python scripts/train.py --config configs/train_config.yaml --resume_from auto
python scripts/train.py --config configs/train_config.yaml --resume_from unet_model/checkpoints/latest.pth
```

### `configs/train_config.yaml` 摘要

| 配置项 | 含义 |
|--------|------|
| `data_dir` | 数据根目录 |
| `output_root` | 输出根路径（如 `unet_model`，其下会生成 `checkpoints/`、`results/`） |
| `image_size` / `batch_size` / `num_workers` | 输入尺寸与 DataLoader |
| `base_channels` | U-Net 起点通道（宽度） |
| `block_type` | `double_conv` 或 `residual` |
| `norm_type` | `batch` 或 `group` |
| `use_attention` / `attention_mode` / `attention_dsample` | Attention Gate |
| `bottleneck_context` | 瓶颈多尺度上下文 |
| `deep_supervision` / `aux_loss_weight` | 深监督与辅助损失权重 |
| `resume_from` | `auto`：尝试最新 checkpoint；也可指定 `.pth` 路径 |
| `use_amp` | 混合精度（CUDA 上生效） |

权重与日志默认位于：`{output_root}/checkpoints/`、`{output_root}/results/`。

---

## 评估与可视化

```bash
python scripts/evaluate.py \
  --model_path unet_model/checkpoints/best_model.pth \
  --data_dir data \
  --results_dir unet_model/test_results \
  --batch_size 8 \
  --image_size 256
```

产出示例：

- `test_results.json`：总体与各类 IoU / Dice 等  
- `test_metrics.png`：各类 IoU、Dice 柱状对比  
- `test_visualization.png`：Input / GT / Pred / Overlay 及类别配色示意  

模型加载会使用 checkpoint 中保存的 **`config`** 字段构造网络，请保证评估所用的 **`data_dir`** 下存在正确的 **`class_info.yaml`**（或与训练一致的类别定义）。

---

## 辅助脚本

| 脚本 | 说明 |
|------|------|
| `scripts/convert_helen.py` | HELEN → 本项目标注与目录格式 |
| `scripts/split_dataset.py` | 从 train 划分出 val |
| `scripts/download_helen_dataset.py` | 下载 HELEN（依赖 `kagglehub` 等，按脚本内说明配置） |

更细的目录说明见 **`PROJECT_STRUCTURE.md`**。

---

## 常见问题

1. **`albumentations` 未安装** → `pip install albumentations`。  
2. **`resume` 后 shape 不匹配** → 训练脚本会尝试部分加载兼容层并提示从较早 epoch 继续；或与旧 checkpoint 结构完全一致时再全开权重。  
3. **评估报错结构与权重不符** → 使用本次训练保存的 `.pth`（内含 `config`），勿混用旧版纯 `state_dict` 且无配置的权重。

---

## 开源协议

若仓库未包含 `LICENSE` 文件，默认版权归原作者所有；如需开源请注明许可证类型并补充 `LICENSE`。
