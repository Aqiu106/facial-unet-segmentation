"""
unet_pkg.datasets.dataset
数据集与 DataLoader。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import matplotlib.pyplot as plt

class FacialFeaturesDataset(Dataset):
    """面部特征分割数据集"""
    
    def __init__(self, data_dir, split='train', transform=None, image_size=256):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据增强变换
            image_size: 图像大小
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # 设置图像和掩码路径
        self.images_dir = self.data_dir / split / 'images'
        self.masks_dir = self.data_dir / split / 'masks'
        
        # 获取所有图像文件
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
        
        print(f"数据集 '{split}': 找到 {len(self.image_files)} 张图像")
        
        if not self.image_files:
            raise ValueError(f"在 {self.images_dir} 中没有找到图像文件")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像路径
        img_path = self.image_files[idx]
        
        # 读取图像
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # 如果没有掩码，创建全零掩码
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # 确保掩码是long类型
            if isinstance(mask, torch.Tensor):
                mask = mask.long()  # 转换为long类型
            elif isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()  # 转换为long类型
        else:
            # 手动转换
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()  # 转换为long类型
        
        # 最终确保掩码是long类型
        if isinstance(mask, torch.Tensor) and mask.dtype != torch.int64:
            mask = mask.long()
        
        return image, mask
    
    def visualize_sample(self, idx=0, num_samples=3):
        """可视化数据集样本"""
        if idx >= len(self):
            idx = 0
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            sample_idx = (idx + i) % len(self)
            image, mask = self[sample_idx]
            
            # 转换为numpy用于显示
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()
            
            # 创建彩色掩码
            unique_classes = np.unique(mask_np)
            colored_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            
            for class_id in unique_classes:
                if class_id > 0:  # 跳过背景
                    color = plt.cm.tab20(class_id / max(1, mask_np.max()))[:3]
                    color = (np.array(color) * 255).astype(np.uint8)
                    colored_mask[mask_np == class_id] = color
            
            # 叠加显示
            overlay = cv2.addWeighted((image_np * 255).astype(np.uint8), 0.7, 
                                      colored_mask, 0.3, 0)
            
            # 绘制
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'图像 {sample_idx}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='viridis')
            axes[i, 1].set_title('掩码')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('叠加')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'数据集样本可视化 ({self.split}集)', fontsize=16)
        plt.tight_layout()
        plt.show()

def get_transforms(split='train', image_size=256):
    """获取数据增强变换
    
    Args:
        split: 数据集分割 ('train', 'val', 'test')
        image_size: 目标图像大小
    """
    if split == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
                              rotate_limit=15, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)
    
    else:  # val/test
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)

def create_dataloaders(
    data_dir,
    batch_size=8,
    image_size=256,
    num_workers=4,
    visualize_train_samples=False,
    visualize_num_samples=3,
):
    """创建训练、验证、测试数据加载器"""
    
    # 创建数据集
    train_dataset = FacialFeaturesDataset(
        data_dir=data_dir,
        split='train',
        transform=get_transforms('train', image_size),
        image_size=image_size
    )
    
    val_dataset = FacialFeaturesDataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val', image_size),
        image_size=image_size
    )
    
    test_dataset = FacialFeaturesDataset(
        data_dir=data_dir,
        split='test',
        transform=get_transforms('test', image_size),
        image_size=image_size
    )
    
    # 可选可视化训练样本，默认关闭避免训练脚本产生副作用
    if visualize_train_samples:
        print("可视化训练集样本...")
        train_dataset.visualize_sample(num_samples=visualize_num_samples)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n数据加载器统计:")
    print(f"  训练集: {len(train_dataset)} 个样本, {len(train_loader)} 个批次")
    print(f"  验证集: {len(val_dataset)} 个样本, {len(val_loader)} 个批次")
    print(f"  测试集: {len(test_dataset)} 个样本, {len(test_loader)} 个批次")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    data_dir = "unet_model/data"
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            image_size=256
        )
        
        # 测试一个批次
        for images, masks in train_loader:
            print(f"批次图像尺寸: {images.shape}")
            print(f"批次掩码尺寸: {masks.shape}")
            print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
            print(f"掩码类别: {torch.unique(masks)}")
            break
            
    except Exception as e:
        print(f"数据集测试失败: {e}")
        print("请确保已运行数据转换脚本")