import os
from glob import glob
import torch
import numpy as np
from typing import Tuple, List
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityd, RandCropByPosNegLabeld,
    Lambdad, EnsureTyped, CastToTyped, CopyItemsd
)
from monai.data import Dataset, DataLoader, CacheDataset


def split_dataset(total_samples: int = 25,
                  train_ratio: float = 0.6,
                  val_ratio: float = 0.2,
                  test_ratio: float = 0.2,
                  seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    划分数据集为训练集、验证集和测试集

    Args:
        total_samples: 总样本数 (默认 25)
        train_ratio: 训练集比例 (默认 0.6 -> 15 samples)
        val_ratio: 验证集比例 (默认 0.2 -> 5 samples)
        test_ratio: 测试集比例 (默认 0.2 -> 5 samples)
        seed: 随机种子

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(seed)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()

    print(f"数据集划分 (总计 {total_samples} 样本):")
    print(f"  训练集: {len(train_indices)} 样本 - {train_indices}")
    print(f"  验证集: {len(val_indices)} 样本 - {val_indices}")
    print(f"  测试集: {len(test_indices)} 样本 - {test_indices}")

    return train_indices, val_indices, test_indices


def get_vessel_loader(image_dir: str,
                      label_dir_42: str,
                      label_dir_5: str = None,
                      label_dir_binary: str = None,
                      batch_size: int = 2,
                      roi_size: Tuple[int, int, int] = (96, 96, 96),
                      indices: List[int] = None,
                      use_cache: bool = True,
                      num_workers: int = 4,
                      is_train: bool = True):
    """
    创建脑血管分割数据加载器

    Args:
        image_dir: 图像目录路径
        label_dir_42: 42类标签目录路径
        label_dir_5: 5类标签目录路径 (可选,如果为None则动态生成)
        label_dir_binary: 二值标签目录路径 (可选,如果为None则动态生成)
        batch_size: 批次大小
        roi_size: 随机裁剪尺寸
        indices: 样本索引列表 (用于划分训练/验证/测试集)
        use_cache: 是否使用缓存加速
        num_workers: 数据加载线程数
        is_train: 是否为训练模式 (训练模式会进行随机裁剪)

    Returns:
        DataLoader
    """
    # 1. 获取文件列表
    images = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
    labels_42 = sorted(glob(os.path.join(label_dir_42, "*.nii.gz")))

    # 根据 indices 筛选样本
    if indices is not None:
        images = [images[i] for i in indices]
        labels_42 = [labels_42[i] for i in indices]

    # 构建数据字典
    data_dicts = []
    for img, lbl_42 in zip(images, labels_42):
        data_dict = {"image": img, "label_fine": lbl_42}

        # 如果提供了预生成的标签,直接加载
        if label_dir_5 is not None:
            lbl_5_path = os.path.join(label_dir_5, os.path.basename(lbl_42))
            if os.path.exists(lbl_5_path):
                data_dict["label_reg"] = lbl_5_path

        if label_dir_binary is not None:
            lbl_bin_path = os.path.join(label_dir_binary, os.path.basename(lbl_42))
            if os.path.exists(lbl_bin_path):
                data_dict["label_bin"] = lbl_bin_path

        data_dicts.append(data_dict)

    # 2. 定义解剖映射逻辑 (用于动态生成 5 系统标签)
    def create_regional_label(label):
        # label shape: [1, H, W, D]
        reg_label = torch.zeros_like(label)
        # 系统 1: 后循环
        for i in [1, 2, 3, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
            reg_label[label == i] = 1
        # 系统 2: 前循环-颈内
        for i in [4, 5, 6, 7, 17, 18, 19, 20, 31, 32, 33, 34]:
            reg_label[label == i] = 2
        # 系统 3: 前循环-前交通
        for i in [11, 12, 13, 14, 15, 16]:
            reg_label[label == i] = 3
        # 系统 4: 交通动脉
        for i in [8, 9, 10]:
            reg_label[label == i] = 4
        # 系统 5: 颈外系统
        for i in [35, 36, 37, 38, 39, 40, 41, 42]:
            reg_label[label == i] = 5
        return reg_label

    # 3. 构建变换流水线
    # 判断是否需要加载预生成的标签
    load_keys = ["image", "label_fine"]
    if "label_reg" in data_dicts[0]:
        load_keys.append("label_reg")
    if "label_bin" in data_dicts[0]:
        load_keys.append("label_bin")

    # 为每个 key 设置对应的插值模式 (image用双线性, 标签用最近邻)
    spacing_modes = ["bilinear"] + ["nearest"] * (len(load_keys) - 1)

    transforms_list = [
        LoadImaged(keys=load_keys),
        EnsureChannelFirstd(keys=load_keys),
        # 统一分辨率到 1.0mm
        Spacingd(keys=load_keys, pixdim=(1.0, 1.0, 1.0), mode=spacing_modes),
        Orientationd(keys=load_keys, axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
    ]

    # 如果没有预生成的标签,动态生成
    if "label_bin" not in data_dicts[0]:
        transforms_list.extend([
            CopyItemsd(keys=["label_fine"], times=1, names=["label_bin"]),
            Lambdad(keys=["label_bin"], func=lambda x: (x > 0).float()),
        ])

    if "label_reg" not in data_dicts[0]:
        transforms_list.extend([
            CopyItemsd(keys=["label_fine"], times=1, names=["label_reg"]),
            Lambdad(keys=["label_reg"], func=create_regional_label),
        ])

    # 训练模式:随机裁剪
    if is_train:
        transforms_list.append(
            RandCropByPosNegLabeld(
                keys=["image", "label_fine", "label_bin", "label_reg"],
                label_key="label_bin",
                spatial_size=roi_size,
                pos=1, neg=1, num_samples=4
            )
        )

    # 类型转换
    transforms_list.extend([
        CastToTyped(keys=["label_fine", "label_reg"], dtype=torch.long),
        EnsureTyped(keys=["image", "label_fine", "label_bin", "label_reg"]),
    ])

    train_transforms = Compose(transforms_list)

    # 4. 创建 Dataset 和 DataLoader
    if use_cache:
        ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0)
    else:
        ds = Dataset(data=data_dicts, transform=train_transforms)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

    return loader


# ============ 测试代码 ============
if __name__ == "__main__":
    print("=" * 70)
    print("脑血管分割 DataLoader 测试")
    print("=" * 70)

    # 数据路径
    IMAGE_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_mr"
    LABEL_42_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr"
    LABEL_5_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_5groups"
    LABEL_BIN_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_binary"

    # 1. 划分数据集
    print("\n[1] 数据集划分")
    train_idx, val_idx, test_idx = split_dataset(total_samples=25, seed=42)

    # 2. 创建训练集 DataLoader
    print("\n[2] 创建训练集 DataLoader")
    train_loader = get_vessel_loader(
        image_dir=IMAGE_DIR,
        label_dir_42=LABEL_42_DIR,
        label_dir_5=LABEL_5_DIR,
        label_dir_binary=LABEL_BIN_DIR,
        batch_size=2,
        roi_size=(96, 96, 96),
        indices=train_idx,
        is_train=True
    )

    # 3. 创建验证集 DataLoader
    print("\n[3] 创建验证集 DataLoader")
    val_loader = get_vessel_loader(
        image_dir=IMAGE_DIR,
        label_dir_42=LABEL_42_DIR,
        label_dir_5=LABEL_5_DIR,
        label_dir_binary=LABEL_BIN_DIR,
        batch_size=1,
        roi_size=(96, 96, 96),
        indices=val_idx,
        is_train=False
    )

    print("\n" + "-" * 70)
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    # 4. 测试加载一个批次
    print("\n[4] 测试加载一个训练批次")
    batch = next(iter(train_loader))
    print(f"  - image shape: {batch['image'].shape}")
    print(f"  - label_fine (42类) shape: {batch['label_fine'].shape}")
    print(f"  - label_reg (5系统) shape: {batch['label_reg'].shape}")
    print(f"  - label_bin (二值) shape: {batch['label_bin'].shape}")

    print("\n" + "=" * 70)
    print("✓ DataLoader 测试通过!")
    print("=" * 70)
