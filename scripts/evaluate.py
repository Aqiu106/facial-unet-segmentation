"""
U-Net evaluation script.
Run from repo root: python scripts/evaluate.py --model_path ...
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from unet_pkg.datasets import create_dataloaders
from unet_pkg.metrics import calculate_dice, calculate_iou, calculate_pixel_accuracy
from unet_pkg.models import UNet


class Tester:
    def __init__(self, model_path, data_dir, results_dir, image_size=256):
        self.model_path = model_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.image_size = image_size
        os.makedirs(self.results_dir, exist_ok=True)

        class_info_path = os.path.join(data_dir, "class_info.yaml")
        if os.path.exists(class_info_path):
            with open(class_info_path, "r", encoding="utf-8") as f:
                class_info = yaml.safe_load(f)
            self.class_names = class_info.get("class_names", [])
            self.num_classes = int(class_info.get("num_classes", len(self.class_names)))
        else:
            self.num_classes = 7
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        # 显式声明 weights_only，避免未来版本行为变化带来的 warning
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        self.model = UNet(
            n_channels=int(model_cfg.get("n_channels", 3)),
            n_classes=self.num_classes,
            bilinear=bool(model_cfg.get("bilinear", True)),
            use_attention=bool(model_cfg.get("use_attention", True)),
            attention_mode=model_cfg.get("attention_mode", "concatenation"),
            attention_dsample=tuple(model_cfg.get("attention_dsample", (2, 2))),
            base_channels=int(model_cfg.get("base_channels", 64)),
            block_type=model_cfg.get("block_type", "double_conv"),
            norm_type=model_cfg.get("norm_type", "batch"),
            bottleneck_context=bool(model_cfg.get("bottleneck_context", False)),
            deep_supervision=bool(model_cfg.get("deep_supervision", False)),
        ).to(self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        print(f"✅ 成功加载模型: {model_path}")

    @staticmethod
    def _metric_value(x):
        return x[0] if isinstance(x, tuple) else x

    def _save_metrics_plot(self, results):
        per_class_iou = results.get("per_class", {}).get("iou", [])
        per_class_dice = results.get("per_class", {}).get("dice", [])
        if not per_class_iou or not per_class_dice:
            return

        x_labels = self.class_names if self.class_names else [f"class_{i}" for i in range(len(per_class_iou))]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        bars_iou = axes[0].bar(x_labels, per_class_iou, color="#8fbcd1", alpha=0.95)
        axes[0].set_title("Per-Class IoU Metrics", fontweight="bold")
        axes[0].set_ylabel("IoU")
        axes[0].set_ylim(0, 1.0)
        axes[0].tick_params(axis="x", rotation=40)
        axes[0].grid(axis="y", alpha=0.2)
        for bar, v in zip(bars_iou, per_class_iou):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        bars_dice = axes[1].bar(x_labels, per_class_dice, color="#e7a0a0", alpha=0.95)
        axes[1].set_title("Per-Class Dice Coefficient", fontweight="bold")
        axes[1].set_ylabel("Dice Coefficient")
        axes[1].set_ylim(0, 1.0)
        axes[1].tick_params(axis="x", rotation=40)
        axes[1].grid(axis="y", alpha=0.2)
        for bar, v in zip(bars_dice, per_class_dice):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(self.results_dir, "test_metrics.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 指标图已保存: {out_path}")

    @staticmethod
    def _denormalize_image(img):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return np.clip(img * std + mean, 0.0, 1.0)

    def _save_visualizations(self, vis_samples, results):
        if not vis_samples:
            return
        num_samples = len(vis_samples)
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = np.array([axes])
        cmap = plt.get_cmap("viridis")
        vmax = max(self.num_classes - 1, 1)
        im_for_colorbar = None
        for i, (img, gt, pred) in enumerate(vis_samples):
            img = self._denormalize_image(img)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(gt, cmap=cmap, vmin=0, vmax=vmax)
            axes[i, 1].set_title("GT Mask")
            axes[i, 1].axis("off")
            im_for_colorbar = axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=vmax)
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")
            axes[i, 3].imshow(img)
            axes[i, 3].imshow(pred, cmap=cmap, vmin=0, vmax=vmax, alpha=0.45)
            axes[i, 3].set_title("Overlay")
            axes[i, 3].axis("off")

        miou = results.get("overall", {}).get("mean_iou", 0.0)
        mdice = results.get("overall", {}).get("mean_dice", 0.0)
        fig.suptitle(f"U-Net Test Results (mIoU: {miou:.4f}, Dice: {mdice:.4f})", fontsize=13, fontweight="bold")

        if im_for_colorbar is not None:
            cbar = fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
            cbar.set_label("Class")
            tick_values = list(range(self.num_classes))
            cbar.set_ticks(tick_values)
            tick_labels = self.class_names if len(self.class_names) == self.num_classes else [f"class_{i}" for i in tick_values]
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=7)

        plt.tight_layout(rect=[0, 0, 0.94, 0.95])
        out_path = os.path.join(self.results_dir, "test_visualization.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ 可视化图已保存: {out_path}")

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()
        total_iou = total_dice = total_acc = 0.0
        per_class_iou_sum = np.zeros(self.num_classes, dtype=np.float64)
        per_class_dice_sum = np.zeros(self.num_classes, dtype=np.float64)
        per_class_iou_count = np.zeros(self.num_classes, dtype=np.int64)
        per_class_dice_count = np.zeros(self.num_classes, dtype=np.int64)
        vis_samples = []
        max_vis_samples = 3
        pbar = tqdm(test_loader, desc="Testing")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            outputs = self.model(images)
            # 兼容新版模型输出: {"logits": ..., "aux_logits": ...}
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=1)

            iou_result = calculate_iou(preds, masks, self.num_classes)
            dice_result = calculate_dice(preds, masks, self.num_classes)
            miou = self._metric_value(iou_result)
            mdice = self._metric_value(dice_result)
            macc = calculate_pixel_accuracy(preds, masks)
            total_iou += miou
            total_dice += mdice
            total_acc += macc
            pbar.set_postfix(iou=f"{miou:.4f}", dice=f"{mdice:.4f}", acc=f"{macc:.4f}")

            if isinstance(iou_result, tuple) and len(iou_result) > 1:
                iou_list = iou_result[1]
                for idx, value in enumerate(iou_list):
                    if idx >= self.num_classes:
                        break
                    if not np.isnan(value):
                        per_class_iou_sum[idx] += float(value)
                        per_class_iou_count[idx] += 1
            if isinstance(dice_result, tuple) and len(dice_result) > 1:
                dice_list = dice_result[1]
                for idx, value in enumerate(dice_list):
                    if idx >= self.num_classes:
                        break
                    if not np.isnan(value):
                        per_class_dice_sum[idx] += float(value)
                        per_class_dice_count[idx] += 1
            if len(vis_samples) < max_vis_samples:
                b = min(images.size(0), max_vis_samples - len(vis_samples))
                images_cpu = images[:b].detach().cpu().permute(0, 2, 3, 1).numpy()
                masks_cpu = masks[:b].detach().cpu().numpy()
                preds_cpu = preds[:b].detach().cpu().numpy()
                for j in range(b):
                    vis_samples.append((images_cpu[j], masks_cpu[j], preds_cpu[j]))

        n = len(test_loader)
        mean_per_class_iou = np.divide(
            per_class_iou_sum,
            np.maximum(per_class_iou_count, 1),
        ).tolist()
        mean_per_class_dice = np.divide(
            per_class_dice_sum,
            np.maximum(per_class_dice_count, 1),
        ).tolist()
        results = {
            "overall": {
                "mean_iou": float(total_iou / n),
                "mean_dice": float(total_dice / n),
                "pixel_accuracy": float(total_acc / n),
            },
            "per_class": {
                "iou": [float(x) for x in mean_per_class_iou],
                "dice": [float(x) for x in mean_per_class_dice],
            },
            "class_names": self.class_names,
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": self.model_path,
        }
        out_json = os.path.join(self.results_dir, "test_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 测试结果已保存: {out_json}")
        self._save_metrics_plot(results)
        self._save_visualizations(vis_samples, results)
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net 评估")
    parser.add_argument("--model_path", type=str, default="unet_model/checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="unet_model/test_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    tester = Tester(
        model_path=args.model_path,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        image_size=args.image_size,
    )
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        visualize_train_samples=False,
    )
    results = tester.test(test_loader)
    print(
        f"完成: IoU={results['overall']['mean_iou']:.4f}, "
        f"Dice={results['overall']['mean_dice']:.4f}, "
        f"Acc={results['overall']['pixel_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()


#bash
#python scripts/evaluate.py --model_path unet_model/checkpoints/best_model.pth --data_dir data --results_dir unet_model/test_results --batch_size 8 --image_size 256