import os
import shutil
import random
from pathlib import Path
import yaml


def split_dataset(data_dir='unet_model/data', val_ratio=0.2, seed=42):
    """
    从训练集自动划分验证集
    """
    random.seed(seed)

    train_img_dir = Path(data_dir) / 'train' / 'images'
    train_mask_dir = Path(data_dir) / 'train' / 'masks'
    val_img_dir = Path(data_dir) / 'val' / 'images'
    val_mask_dir = Path(data_dir) / 'val' / 'masks'

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(train_img_dir.glob('*.*'))
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = [img for img in all_images if img.suffix.lower() in image_extensions]

    if not all_images:
        print(f"错误: 在 {train_img_dir} 中没有找到图片文件")
        return

    print(f"找到 {len(all_images)} 张训练图片")

    num_val = int(len(all_images) * val_ratio)
    if num_val < 1:
        num_val = 1

    val_images = random.sample(all_images, num_val)
    print(f"从训练集随机选择 {num_val} 张图片作为验证集 ({val_ratio*100:.1f}%)")

    moved_count = 0
    for img_path in val_images:
        mask_path = train_mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            mask_path = train_mask_dir / f"{img_path.stem}.jpg"

        if mask_path.exists():
            shutil.move(str(img_path), str(val_img_dir / img_path.name))
            shutil.move(str(mask_path), str(val_mask_dir / mask_path.name))
            moved_count += 1
        else:
            print(f"警告: 找不到 {img_path.stem} 对应的掩码文件")

    update_class_info(data_dir)

    print(f"✓ 成功移动 {moved_count} 对图片/掩码到验证集")
    print(f"✓ 训练集剩余: {len(list(train_img_dir.glob('*.*')))} 张图片")
    print(f"✓ 验证集现有: {len(list(val_img_dir.glob('*.*')))} 张图片")


def update_class_info(data_dir='unet_model/data'):
    """更新类别信息文件，包含数据集划分统计"""
    class_info_path = Path(data_dir) / 'class_info.yaml'

    if class_info_path.exists():
        with open(class_info_path, 'r') as f:
            class_info = yaml.safe_load(f)
    else:
        class_info = {
            'num_classes': 6,
            'class_names': ['background', 'face', 'left_eye', 'right_eye', 'nose', 'mouth']
        }

    train_count = len(list((Path(data_dir) / 'train' / 'images').glob('*.*')))
    val_count = len(list((Path(data_dir) / 'val' / 'images').glob('*.*')))
    test_count = len(list((Path(data_dir) / 'test' / 'images').glob('*.*')))

    class_info['dataset_stats'] = {
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'total': train_count + val_count + test_count
    }

    with open(class_info_path, 'w') as f:
        yaml.dump(class_info, f, default_flow_style=False)

    print(f"✓ 更新类别信息文件: {class_info_path}")
    print(f"  训练集: {train_count} 张, 验证集: {val_count} 张, 测试集: {test_count} 张")


def verify_dataset(data_dir='unet_model/data'):
    """验证数据集结构"""
    print("\n验证数据集结构...")

    for split in ['train', 'val', 'test']:
        img_dir = Path(data_dir) / split / 'images'
        mask_dir = Path(data_dir) / split / 'masks'

        images = list(img_dir.glob('*.*'))
        masks = list(mask_dir.glob('*.*'))

        print(f"\n{split.upper()}集:")
        print(f"  图片数量: {len(images)}")
        print(f"  掩码数量: {len(masks)}")

        img_names = {img.stem for img in images}
        mask_names = {mask.stem for mask in masks}

        missing_masks = img_names - mask_names
        extra_masks = mask_names - img_names

        if missing_masks:
            print(f"  警告: {len(missing_masks)} 张图片没有对应的掩码")
        if extra_masks:
            print(f"  警告: {len(extra_masks)} 个掩码没有对应的图片")


if __name__ == "__main__":
    split_dataset('unet_model/data', val_ratio=0.2)
    verify_dataset('unet_model/data')
