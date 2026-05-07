# 项目结构（目录拆分版）

在仓库根目录 `unet_model/` 下：

## 代码包 `unet_pkg/`

- `unet_pkg/models/attention_unet.py`：带 Attention Gate 的 U-Net
- `unet_pkg/datasets/dataset.py`：数据集与 `create_dataloaders`
- `unet_pkg/metrics/core.py`：IoU / Dice / 像素精度及各类损失

## 可执行脚本 `scripts/`

在**仓库根目录**执行（保证相对路径如 `unet_model/data` 正确）：

- `python scripts/train.py`：训练
- `python scripts/evaluate.py`：测试与可视化
- `python scripts/convert_helen.py`：HELEN → 本项目数据格式
- `python scripts/split_dataset.py`：从 train 划分 val
- `python scripts/download_helen_dataset.py`：下载 HELEN（需 kagglehub）

脚本开头会把仓库根目录加入 `sys.path`，以便 `import unet_pkg`。

## 训练配置 `configs/`

- `configs/train_config.yaml`：训练超参数、数据路径、输出目录、AMP、断点续训等配置。
- 支持命令行覆盖：`python scripts/train.py --config configs/train_config.yaml --resume_from auto`

## 可选：可编辑安装

在仓库根目录执行一次后，可在任意工作目录 `import unet_pkg`：

```bash
pip install -e .
```

## 数据与输出（与代码分离）

- `data/`：示例 `class_info.yaml`（你的图像 train/val/test 可放在配置的 `data_dir` 下）
- `unet_model/results/`、`unet_model/test_results/`、`unet_model/checkpoints/`：训练与测试结果（路径由脚本内配置决定）
