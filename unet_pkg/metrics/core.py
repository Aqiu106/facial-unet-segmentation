"""
unet_pkg.metrics.core
评估指标与损失函数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_iou(pred, target, num_classes, ignore_index=255):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            iou_value = intersection / union
            if isinstance(iou_value, torch.Tensor):
                iou_value = iou_value.cpu().item() if iou_value.is_cuda else iou_value.item()
            ious.append(iou_value)

    if len(ious) == 0:
        return 0.0, [0.0] * num_classes
    mean_iou = np.nanmean(ious)
    return mean_iou, ious


def calculate_dice(pred, target, num_classes, ignore_index=255):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    dices = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = pred_inds.sum().float() + target_inds.sum().float()
        if union == 0:
            dices.append(float('nan'))
        else:
            dice_value = (2.0 * intersection) / union
            if isinstance(dice_value, torch.Tensor):
                dice_value = dice_value.cpu().item() if dice_value.is_cuda else dice_value.item()
            dices.append(dice_value)

    dices_array = np.array(dices, dtype=np.float32)
    dices_array = dices_array[~np.isnan(dices_array)]
    if len(dices_array) == 0:
        return 0.0, [0.0] * num_classes
    mean_dice = np.nanmean(dices_array)
    return mean_dice, dices_array.tolist()


def calculate_pixel_accuracy(pred, target, ignore_index=255):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    mask = (target != ignore_index)
    correct = (pred[mask] == target[mask]).sum().float()
    total = mask.sum().float()
    if total == 0:
        return 0.0
    return float(correct) / float(total)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(
            target.long(),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        return dice_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer('alpha', torch.ones(num_classes))
        elif isinstance(alpha, (list, np.ndarray)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha.float())
        else:
            self.register_buffer('alpha', torch.ones(num_classes))

    def forward(self, pred, target):
        alpha = self.alpha.to(pred.device)
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=alpha)
        log_pt = -ce_loss
        pt = torch.exp(log_pt)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, dice_weight=0.5, focal_weight=0.5, dice_smooth=1e-6):
        super().__init__()
        self.focal_loss = FocalLoss(num_classes, alpha, gamma)
        self.dice_loss = DiceLoss(num_classes, dice_smooth)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice


def calculate_confusion_matrix(pred, target, num_classes, normalize=True):
    pred_flat = pred.flatten().cpu().numpy()
    target_flat = target.flatten().cpu().numpy()
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    return cm


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if cm.max() <= 1 else 'd',
                cmap='Blues', cbar=True,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_class_metrics(pred, target, num_classes, class_names):
    metrics = {}
    for cls in range(num_classes):
        if cls >= len(class_names):
            continue
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        metrics[class_names[cls]] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }
    return metrics


def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor)


def calculate_dice_safe(pred, target, num_classes):
    pred_np = tensor_to_numpy(pred)
    target_np = tensor_to_numpy(target)
    dices = []
    for cls in range(1, num_classes):
        pred_cls = (pred_np == cls)
        target_cls = (target_np == cls)
        intersection = np.sum(pred_cls & target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)
        if union == 0:
            dices.append(float('nan'))
        else:
            dices.append(2.0 * intersection / union)
    dices_array = np.array(dices)
    valid_dices = dices_array[~np.isnan(dices_array)]
    if len(valid_dices) == 0:
        return 0.0
    return np.mean(valid_dices)
