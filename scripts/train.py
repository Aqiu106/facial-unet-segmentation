"""
U-Net training script.
Run from repo root: python scripts/train.py --config configs/train_config.yaml
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from unet_pkg.datasets import create_dataloaders
from unet_pkg.metrics import DiceLoss, FocalDiceLoss, FocalLoss, calculate_dice, calculate_iou
from unet_pkg.models import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_root = cfg.get("output_root", "unet_model")
    cfg["checkpoint_dir"] = cfg.get("checkpoint_dir") or os.path.join(output_root, "checkpoints")
    cfg["results_dir"] = cfg.get("results_dir") or os.path.join(output_root, "results")
    cfg.setdefault("seed", 42)
    cfg.setdefault("use_amp", True)
    cfg.setdefault("resume_from", "auto")
    cfg.setdefault("save_interval", 5)
    cfg.setdefault("visualize_interval", 10)
    cfg.setdefault("visualize_predictions", True)
    cfg.setdefault("visualize_train_samples", False)
    cfg.setdefault("visualize_num_samples", 3)
    cfg.setdefault("early_stop_patience", 10)
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    return cfg


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(config.get("use_amp", True) and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.checkpoint_dir = config["checkpoint_dir"]
        self.results_dir = config["results_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(config["data_dir"], "class_info.yaml"), "r", encoding="utf-8") as f:
            class_info = yaml.safe_load(f)
        self.num_classes = class_info["num_classes"]
        self.class_names = class_info["class_names"]

        self.model = self._init_model()
        self.criterion = self._init_criterion()
        if hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.start_epoch = 0
        self.train_history = {
            "train_loss": [], "val_loss": [],
            "train_iou": [], "val_iou": [],
            "train_dice": [], "val_dice": [],
            "learning_rate": [],
        }

    def _init_model(self):
        # 旧实现（保留）:
        # model = UNet(
        #     n_channels=3,
        #     n_classes=self.num_classes,
        #     bilinear=self.config.get("bilinear", True),
        #     use_attention=self.config.get("use_attention", True),
        #     attention_mode=self.config.get("attention_mode", "concatenation"),
        #     attention_dsample=tuple(self.config.get("attention_dsample", (2, 2))),
        # ).to(self.device)
        model = UNet(
            n_channels=3,
            n_classes=self.num_classes,
            bilinear=self.config.get("bilinear", True),
            use_attention=self.config.get("use_attention", True),
            attention_mode=self.config.get("attention_mode", "concatenation"),
            attention_dsample=tuple(self.config.get("attention_dsample", (2, 2))),
            base_channels=int(self.config.get("base_channels", 64)),
            block_type=self.config.get("block_type", "double_conv"),
            norm_type=self.config.get("norm_type", "batch"),
            bottleneck_context=bool(self.config.get("bottleneck_context", False)),
            deep_supervision=bool(self.config.get("deep_supervision", False)),
        ).to(self.device)
        pretrained = self.config.get("pretrained_weights")
        if pretrained and os.path.exists(pretrained):
            model.load_state_dict(torch.load(pretrained, map_location=self.device))
            print(f"加载预训练权重: {pretrained}")
        return model

    @staticmethod
    def _extract_logits(model_outputs):
        if isinstance(model_outputs, dict):
            return model_outputs["logits"], model_outputs.get("aux_logits", [])
        return model_outputs, []

    def _compute_loss(self, model_outputs, masks):
        logits, aux_logits = self._extract_logits(model_outputs)
        loss = self.criterion(logits, masks)
        if aux_logits:
            aux_weight = float(self.config.get("aux_loss_weight", 0.2))
            aux_loss = sum(self.criterion(aux, masks) for aux in aux_logits) / len(aux_logits)
            loss = loss + aux_weight * aux_loss
        return loss, logits

    def _init_criterion(self):
        name = self.config.get("criterion", "cross_entropy")
        if name == "cross_entropy":
            weight = self.config.get("class_weights")
            if weight:
                weight = torch.tensor(weight, device=self.device)
            return nn.CrossEntropyLoss(weight=weight)
        if name == "dice":
            return DiceLoss(num_classes=self.num_classes)
        if name == "focal":
            alpha = self.config.get("focal_alpha")
            if alpha is not None and isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
            return FocalLoss(self.num_classes, alpha=alpha, gamma=self.config.get("focal_gamma", 2.0))
        if name == "focal_dice":
            alpha = self.config.get("focal_alpha")
            if alpha is not None and isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
            return FocalDiceLoss(
                num_classes=self.num_classes,
                alpha=alpha,
                gamma=self.config.get("focal_gamma", 2.0),
                dice_weight=self.config.get("dice_weight", 0.5),
                focal_weight=self.config.get("focal_weight", 0.5),
            )
        if name == "combined":
            ce = nn.CrossEntropyLoss()
            dice = DiceLoss(num_classes=self.num_classes)
            return lambda pred, target: 0.5 * ce(pred, target) + 0.5 * dice(pred, target)
        raise ValueError(f"未知损失函数: {name}")

    def _init_optimizer(self):
        name = self.config.get("optimizer", "adam")
        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config.get("weight_decay", 1e-4))
        if name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9, weight_decay=self.config.get("weight_decay", 1e-4))
        raise ValueError(f"未知优化器: {name}")

    def _init_scheduler(self):
        return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)

    @staticmethod
    def _load_state_dict_compatible(model, state_dict):
        model_state = model.state_dict()
        matched = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)
        model_state.update(matched)
        model.load_state_dict(model_state)
        return len(matched), len(skipped)

    def maybe_resume(self):
        resume_from = self.config.get("resume_from")
        if resume_from == "auto":
            latest = os.path.join(self.checkpoint_dir, "latest_model.pth")
            resume_from = latest if os.path.exists(latest) else None
        if not resume_from:
            return
        if not os.path.exists(resume_from):
            print(f"警告: resume 文件不存在 {resume_from}")
            return
        ckpt = torch.load(resume_from, map_location=self.device)
        model_state = ckpt.get("model_state_dict", ckpt)
        try:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.train_history = ckpt.get("train_history", self.train_history)
            self.start_epoch = int(ckpt.get("epoch", -1)) + 1
            if self.use_amp and ckpt.get("scaler_state_dict"):
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"断点续训: {resume_from}，从 epoch {self.start_epoch + 1} 开始")
        except RuntimeError as e:
            loaded, skipped = self._load_state_dict_compatible(self.model, model_state)
            self.start_epoch = 0
            print(
                f"警告: checkpoint 与当前模型结构不兼容，已切换为部分加载权重。"
                f"加载参数 {loaded} 个，跳过 {skipped} 个。"
            )
            print(f"详细信息: {e}")
            print("优化器/调度器/历史记录不会恢复，将从 epoch 1 重新训练。")

    @staticmethod
    def _metric_value(metric_result):
        return metric_result[0] if isinstance(metric_result, tuple) else metric_result

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = total_iou = total_dice = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['num_epochs']} [Train]")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss, logits = self._compute_loss(outputs, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            preds = torch.argmax(logits, dim=1)
            miou = self._metric_value(calculate_iou(preds, masks, self.num_classes))
            mdice = self._metric_value(calculate_dice(preds, masks, self.num_classes))
            total_loss += loss.item()
            total_iou += miou
            total_dice += mdice
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{miou:.4f}", dice=f"{mdice:.4f}")
        n = len(train_loader)
        return total_loss / n, total_iou / n, total_dice / n

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = total_iou = total_dice = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{self.config['num_epochs']} [Val]")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss, logits = self._compute_loss(outputs, masks)
            preds = torch.argmax(logits, dim=1)
            miou = self._metric_value(calculate_iou(preds, masks, self.num_classes))
            mdice = self._metric_value(calculate_dice(preds, masks, self.num_classes))
            total_loss += loss.item()
            total_iou += miou
            total_dice += mdice
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{miou:.4f}", dice=f"{mdice:.4f}")
        n = len(val_loader)
        return total_loss / n, total_iou / n, total_dice / n

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "train_history": self.train_history,
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1:03d}.pth"))
        torch.save(ckpt, os.path.join(self.checkpoint_dir, "latest_model.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best_model.pth"))

    def plot_training_history(self):
        if not self.train_history["train_loss"]:
            return
        epochs = range(1, len(self.train_history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(epochs, self.train_history["train_loss"], label="Train")
        axes[0, 0].plot(epochs, self.train_history["val_loss"], label="Val")
        axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True)
        axes[0, 1].plot(epochs, self.train_history["train_iou"], label="Train")
        axes[0, 1].plot(epochs, self.train_history["val_iou"], label="Val")
        axes[0, 1].set_title("IoU"); axes[0, 1].legend(); axes[0, 1].grid(True)
        axes[1, 0].plot(epochs, self.train_history["train_dice"], label="Train")
        axes[1, 0].plot(epochs, self.train_history["val_dice"], label="Val")
        axes[1, 0].set_title("Dice"); axes[1, 0].legend(); axes[1, 0].grid(True)
        axes[1, 1].plot(epochs, self.train_history["learning_rate"])
        axes[1, 1].set_yscale("log"); axes[1, 1].set_title("Learning Rate"); axes[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_history.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def train(self, train_loader, val_loader):
        best_iou = max(self.train_history["val_iou"]) if self.train_history["val_iou"] else 0.0
        patience = int(self.config.get("early_stop_patience", 10))
        no_improve = 0
        final_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            final_epoch = epoch
            train_loss, train_iou, train_dice = self.train_epoch(train_loader, epoch)
            val_loss, val_iou, val_dice = self.validate(val_loader, epoch)
            self.scheduler.step(val_loss)
            self.train_history["train_loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["train_iou"].append(train_iou)
            self.train_history["val_iou"].append(val_iou)
            self.train_history["train_dice"].append(train_dice)
            self.train_history["val_dice"].append(val_dice)
            self.train_history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % int(self.config.get("save_interval", 5)) == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            if patience > 0 and no_improve >= patience:
                print(f"早停触发，best IoU={best_iou:.4f}")
                break

        self.plot_training_history()
        self.save_checkpoint(final_epoch, is_best=False)
        with open(os.path.join(self.results_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net 训练")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="训练配置文件")
    parser.add_argument("--resume_from", type=str, default=None, help="断点文件路径；auto 自动从 latest 恢复")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from
    set_seed(int(cfg.get("seed", 42)))
    trainer = Trainer(cfg)
    trainer.maybe_resume()
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        image_size=cfg["image_size"],
        num_workers=cfg["num_workers"],
        visualize_train_samples=cfg.get("visualize_train_samples", False),
        visualize_num_samples=cfg.get("visualize_num_samples", 3),
    )
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
