"""
convert_helen.py
将 HELEN 数据集转换为 U-Net 格式
"""
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import random

def convert_helen_dataset(helen_dir, unet_dir, val_ratio=0.15, seed=114514):
    """
    转换 HELEN 数据集为 U-Net 格式
    
    Args:
        helen_dir: HELEN 数据集根目录
        unet_dir: 输出 U-Net 数据目录
        val_ratio: 从训练集划分验证集的比例
        seed: 随机种子
    """
    helen_dir = Path(helen_dir)
    unet_dir = Path(unet_dir)
    
    # HELEN 原始类别定义（11个类别）
    helen_classes = {
        0: 'background',
        1: 'facial_skin',
        2: 'left_brow',
        3: 'right_brow',
        4: 'left_eye',
        5: 'right_eye',
        6: 'nose',
        7: 'upper_lip',
        8: 'inner_mouth',
        9: 'lower_lip',
        10: 'hair'
    }
    
    # 类别映射：将 HELEN 的 11 个类别映射到 7 个类别
    # 映射规则：
    # 0: background -> 0: background
    # 1: facial_skin -> 5: face (面部皮肤)
    # 2: left_brow, 3: right_brow -> 1: eye_brown (眉毛)
    # 4: left_eye, 5: right_eye -> 2: eye (眼睛)
    # 6: nose -> 3: nose (鼻子)
    # 7: upper_lip, 8: inner_mouth, 9: lower_lip -> 4: mouth (嘴巴)
    # 10: hair -> 6: hair (头发)
    class_mapping = {
        0: 0,   # background -> background
        1: 5,   # facial_skin -> face
        2: 1,   # left_brow -> eye_brown
        3: 1,   # right_brow -> eye_brown
        4: 2,   # left_eye -> eye
        5: 2,   # right_eye -> eye
        6: 3,   # nose -> nose
        7: 4,   # upper_lip -> mouth
        8: 4,   # inner_mouth -> mouth
        9: 4,   # lower_lip -> mouth
        10: 6   # hair -> hair
    }
    
    # 目标类别定义（7个类别）
    target_classes = {
        0: 'background',
        1: 'eye_brown',
        2: 'eye',
        3: 'nose',
        4: 'mouth',
        5: 'face',
        6: 'hair'
    }
    
    def remap_mask(mask):
        """将掩码从 11 个类别映射到 7 个类别"""
        remapped_mask = np.zeros_like(mask)
        for old_class, new_class in class_mapping.items():
            remapped_mask[mask == old_class] = new_class
        return remapped_mask
    
    print("="*60)
    print("HELEN 数据集转换工具")
    print("="*60)
    print(f"源目录: {helen_dir}")
    print(f"目标目录: {unet_dir}")
    print(f"验证集比例: {val_ratio}")
    print("\n类别映射 (HELEN 11类 -> U-Net 7类):")
    print("  0: background <- [0:background]")
    print("  1: eye_brown <- [2:left_brow, 3:right_brow]")
    print("  2: eye <- [4:left_eye, 5:right_eye]")
    print("  3: nose <- [6:nose]")
    print("  4: mouth <- [7:upper_lip, 8:inner_mouth, 9:lower_lip]")
    print("  5: face <- [1:facial_skin]")
    print("  6: hair <- [10:hair]")
    print("="*60)
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        (unet_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (unet_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {unet_dir / split / 'images'}")
        print(f"创建目录: {unet_dir / split / 'masks'}")
    
    # 转换训练集
    train_dir = helen_dir / 'train'
    if not train_dir.exists():
        print(f"错误: 训练集目录不存在: {train_dir}")
        return
    
    train_images = sorted(train_dir.glob('*_image.jpg'))
    print(f"\n找到 {len(train_images)} 张训练图像")
    
    if len(train_images) == 0:
        print("错误: 没有找到训练图像")
        return
    
    # 划分训练集和验证集
    random.seed(seed)
    train_images_shuffled = train_images.copy()
    random.shuffle(train_images_shuffled)
    val_split = int(len(train_images_shuffled) * val_ratio)
    val_images = train_images_shuffled[:val_split]
    train_images_final = train_images_shuffled[val_split:]
    
    print(f"训练集: {len(train_images_final)} 张")
    print(f"验证集: {len(val_images)} 张")
    
    # 转换训练集
    train_count = 0
    for img_path in tqdm(train_images_final, desc="转换训练集"):
        # 获取对应的掩码路径
        mask_path = train_dir / f"{img_path.stem.replace('_image', '_label')}.png"
        
        if mask_path.exists():
            # 生成新文件名（去掉 _image 后缀）
            new_img_name = img_path.stem.replace('_image', '') + '.jpg'
            new_mask_name = img_path.stem.replace('_image', '') + '.png'
            
            # 复制图像
            shutil.copy(img_path, unet_dir / 'train' / 'images' / new_img_name)
            
            # 读取掩码并重新映射类别
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                remapped_mask = remap_mask(mask)
                # 保存重新映射后的掩码
                cv2.imwrite(str(unet_dir / 'train' / 'masks' / new_mask_name), remapped_mask)
                train_count += 1
            else:
                print(f"警告: 无法读取掩码文件 {mask_path}")
        else:
            print(f"警告: 找不到掩码文件 {mask_path}")
    
    # 转换验证集
    val_count = 0
    for img_path in tqdm(val_images, desc="转换验证集"):
        mask_path = train_dir / f"{img_path.stem.replace('_image', '_label')}.png"
        
        if mask_path.exists():
            new_img_name = img_path.stem.replace('_image', '') + '.jpg'
            new_mask_name = img_path.stem.replace('_image', '') + '.png'
            
            shutil.copy(img_path, unet_dir / 'val' / 'images' / new_img_name)
            
            # 读取掩码并重新映射类别
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                remapped_mask = remap_mask(mask)
                cv2.imwrite(str(unet_dir / 'val' / 'masks' / new_mask_name), remapped_mask)
                val_count += 1
            else:
                print(f"警告: 无法读取掩码文件 {mask_path}")
        else:
            print(f"警告: 找不到掩码文件 {mask_path}")
    
    # 转换测试集
    test_dir = helen_dir / 'test'
    test_count = 0
    if test_dir.exists():
        test_images = sorted(test_dir.glob('*_image.jpg'))
        print(f"\n找到 {len(test_images)} 张测试图像")
        
        for img_path in tqdm(test_images, desc="转换测试集"):
            mask_path = test_dir / f"{img_path.stem.replace('_image', '_label')}.png"
            
            if mask_path.exists():
                new_img_name = img_path.stem.replace('_image', '') + '.jpg'
                new_mask_name = img_path.stem.replace('_image', '') + '.png'
                
                shutil.copy(img_path, unet_dir / 'test' / 'images' / new_img_name)
                
                # 读取掩码并重新映射类别
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    remapped_mask = remap_mask(mask)
                    cv2.imwrite(str(unet_dir / 'test' / 'masks' / new_mask_name), remapped_mask)
                    test_count += 1
                else:
                    print(f"警告: 无法读取掩码文件 {mask_path}")
            else:
                print(f"警告: 找不到掩码文件 {mask_path}")
    else:
        print(f"警告: 测试集目录不存在: {test_dir}")
    
    # 验证掩码格式
    print("\n验证掩码格式...")
    train_masks = list((unet_dir / 'train' / 'masks').glob('*.png'))
    if train_masks:
        sample_mask_path = train_masks[0]
        mask = cv2.imread(str(sample_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_values = np.unique(mask)
            print(f"样本掩码唯一值: {unique_values}")
            print(f"掩码值范围: [{mask.min()}, {mask.max()}]")
            print(f"掩码尺寸: {mask.shape}")
            print(f"类别映射: HELEN 11类 -> U-Net 7类")
            print(f"  0: background")
            print(f"  1: eye_brown (left_brow + right_brow)")
            print(f"  2: eye (left_eye + right_eye)")
            print(f"  3: nose")
            print(f"  4: mouth (upper_lip + inner_mouth + lower_lip)")
            print(f"  5: face (facial_skin)")
            print(f"  6: hair")
    
    # 创建类别信息文件（7个类别）
    class_info = {
        'num_classes': 7,  # 0-6
        'class_names': [target_classes[i] for i in range(7)],
        'dataset_stats': {
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'total': train_count + val_count + test_count
        }
    }
    
    class_info_path = unet_dir / 'class_info.yaml'
    with open(class_info_path, 'w', encoding='utf-8') as f:
        yaml.dump(class_info, f, default_flow_style=False, allow_unicode=True)
    
    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)
    print(f"训练集: {train_count} 张")
    print(f"验证集: {val_count} 张")
    print(f"测试集: {test_count} 张")
    print(f"总计: {train_count + val_count + test_count} 张")
    print(f"类别数: {class_info['num_classes']}")
    print(f"类别信息已保存: {class_info_path}")
    print("="*60)
    
    return class_info


def verify_conversion(unet_dir):
    """验证转换后的数据集"""
    unet_dir = Path(unet_dir)
    
    print("\n" + "="*60)
    print("验证转换后的数据集")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        images_dir = unet_dir / split / 'images'
        masks_dir = unet_dir / split / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            print(f"警告: {split} 集目录不存在")
            continue
        
        images = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))
        masks = sorted(masks_dir.glob('*.png'))
        
        print(f"\n{split.upper()}集:")
        print(f"  图像数量: {len(images)}")
        print(f"  掩码数量: {len(masks)}")
        
        # 检查图像和掩码是否匹配
        img_names = {img.stem for img in images}
        mask_names = {mask.stem for mask in masks}
        
        missing_masks = img_names - mask_names
        extra_masks = mask_names - img_names
        
        if missing_masks:
            print(f"  警告: {len(missing_masks)} 张图像没有对应的掩码")
            if len(missing_masks) <= 5:
                print(f"    示例: {list(missing_masks)[:5]}")
        if extra_masks:
            print(f"  警告: {len(extra_masks)} 个掩码没有对应的图像")
            if len(extra_masks) <= 5:
                print(f"    示例: {list(extra_masks)[:5]}")
        
        # 检查掩码格式
        if masks:
            sample_mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            unique_vals = np.unique(sample_mask)
            print(f"  掩码值范围: [{sample_mask.min()}, {sample_mask.max()}]")
            print(f"  掩码唯一值数量: {len(unique_vals)}")
    
    print("="*60)


def rebalance_test_set(unet_dir, min_samples_per_class=5, seed=42):
    """
    重新平衡测试集，确保包含所有类别
    
    Args:
        unet_dir: U-Net 数据目录
        min_samples_per_class: 每个类别在测试集中的最小样本数
        seed: 随机种子
    """
    unet_dir = Path(unet_dir)
    random.seed(seed)
    
    print("\n" + "="*60)
    print("重新平衡测试集，确保包含所有类别")
    print("="*60)
    
    # 读取类别信息
    class_info_path = unet_dir / 'class_info.yaml'
    if class_info_path.exists():
        with open(class_info_path, 'r', encoding='utf-8') as f:
            class_info = yaml.safe_load(f)
        class_names = class_info.get('class_names', [])
        num_classes = class_info.get('num_classes', len(class_names))
    else:
        class_names = ['background', 'eye_brown', 'eye', 'nose', 'mouth', 'face', 'hair']
        num_classes = 7
    
    train_images_dir = unet_dir / 'train' / 'images'
    train_masks_dir = unet_dir / 'train' / 'masks'
    test_images_dir = unet_dir / 'test' / 'images'
    test_masks_dir = unet_dir / 'test' / 'masks'
    
    # 检查当前测试集中各类别的分布
    print("\n检查当前测试集类别分布...")
    test_masks = list(test_masks_dir.glob('*.png'))
    test_class_counts = {i: 0 for i in range(num_classes)}
    
    for mask_path in test_masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_vals = np.unique(mask)
            for cls_id in unique_vals:
                if cls_id < num_classes:
                    test_class_counts[cls_id] += 1
    
    print("当前测试集各类别样本数:")
    for cls_id in range(num_classes):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        count = test_class_counts[cls_id]
        status = "✓" if count >= min_samples_per_class else "✗"
        print(f"  {status} {cls_id}: {cls_name:12s} - {count:4d} 个样本")
    
    # 找出缺失的类别
    missing_classes = [cls_id for cls_id in range(num_classes) 
                      if test_class_counts[cls_id] < min_samples_per_class]
    
    if not missing_classes:
        print("\n✓ 测试集已包含所有类别，无需重新平衡")
        return
    
    print(f"\n需要补充的类别: {[class_names[i] for i in missing_classes]}")
    
    # 从训练集中找出包含缺失类别的样本
    train_images = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png'))
    train_masks = list(train_masks_dir.glob('*.png'))
    
    # 创建图像名到掩码路径的映射
    train_mask_dict = {mask.stem: mask for mask in train_masks}
    
    # 找出包含缺失类别的训练样本
    samples_by_class = {cls_id: [] for cls_id in missing_classes}
    
    print("\n从训练集中查找包含缺失类别的样本...")
    for img_path in tqdm(train_images, desc="分析训练集"):
        mask_path = train_mask_dict.get(img_path.stem)
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_vals = np.unique(mask)
                for cls_id in missing_classes:
                    if cls_id in unique_vals:
                        samples_by_class[cls_id].append((img_path, mask_path))
    
    # 随机选择样本移动到测试集
    moved_count = 0
    for cls_id in missing_classes:
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        available_samples = samples_by_class[cls_id]
        needed = min_samples_per_class - test_class_counts[cls_id]
        
        if len(available_samples) < needed:
            print(f"  警告: {cls_name} 类别在训练集中只有 {len(available_samples)} 个样本，需要 {needed} 个")
            needed = len(available_samples)
        
        if needed > 0:
            selected = random.sample(available_samples, needed)
            print(f"\n移动 {needed} 个包含 {cls_name} 的样本到测试集:")
            
            for img_path, mask_path in selected:
                # 移动图像
                new_img_path = test_images_dir / img_path.name
                shutil.move(str(img_path), str(new_img_path))
                
                # 移动掩码
                new_mask_path = test_masks_dir / mask_path.name
                shutil.move(str(mask_path), str(new_mask_path))
                
                moved_count += 1
                print(f"  → {img_path.name}")
    
    print(f"\n✓ 共移动 {moved_count} 个样本到测试集")
    
    # 更新统计信息
    train_count = len(list(train_images_dir.glob('*.jpg'))) + len(list(train_images_dir.glob('*.png')))
    test_count = len(list(test_images_dir.glob('*.jpg'))) + len(list(test_images_dir.glob('*.png')))
    
    # 更新 class_info.yaml
    if class_info_path.exists():
        with open(class_info_path, 'r', encoding='utf-8') as f:
            class_info = yaml.safe_load(f)
        class_info['dataset_stats']['train'] = train_count
        class_info['dataset_stats']['test'] = test_count
        class_info['dataset_stats']['total'] = train_count + class_info['dataset_stats']['val'] + test_count
        
        with open(class_info_path, 'w', encoding='utf-8') as f:
            yaml.dump(class_info, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✓ 已更新类别信息文件")
        print(f"  训练集: {train_count} 张")
        print(f"  测试集: {test_count} 张")
    
    # 验证最终结果
    print("\n验证重新平衡后的测试集...")
    test_masks = list(test_masks_dir.glob('*.png'))
    final_class_counts = {i: 0 for i in range(num_classes)}
    
    for mask_path in test_masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_vals = np.unique(mask)
            for cls_id in unique_vals:
                if cls_id < num_classes:
                    final_class_counts[cls_id] += 1
    
    print("重新平衡后测试集各类别样本数:")
    all_present = True
    for cls_id in range(num_classes):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        count = final_class_counts[cls_id]
        status = "✓" if count >= min_samples_per_class else "✗"
        if count < min_samples_per_class:
            all_present = False
        print(f"  {status} {cls_id}: {cls_name:12s} - {count:4d} 个样本")
    
    if all_present:
        print("\n✓ 测试集现在包含所有类别！")
    else:
        print("\n⚠ 部分类别仍然不足，可能需要增加训练集样本或降低 min_samples_per_class")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将 HELEN 数据集转换为 U-Net 格式')
    parser.add_argument('--helen_dir', type=str, 
                       default='/root/.cache/kagglehub/datasets/abtahimajeed/helen-dataset/versions/1/helenstar_release',
                       help='HELEN 数据集根目录')
    parser.add_argument('--unet_dir', type=str, 
                       default='/root/autodl-tmp/unet_model/data',
                       help='输出 U-Net 数据目录')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='从训练集划分验证集的比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--verify', action='store_true',
                       help='转换后验证数据集')
    parser.add_argument('--rebalance', action='store_true',
                       help='重新平衡测试集，确保包含所有类别')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='每个类别在测试集中的最小样本数（用于rebalance）')
    
    args = parser.parse_args()
    
    # 执行转换或重新平衡
    if args.rebalance:
        # 重新平衡测试集
        rebalance_test_set(args.unet_dir, min_samples_per_class=args.min_samples, seed=args.seed)
    else:
        # 执行转换
        class_info = convert_helen_dataset(
            helen_dir=args.helen_dir,
            unet_dir=args.unet_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        
        # 验证转换结果
        if args.verify:
            verify_conversion(args.unet_dir)

