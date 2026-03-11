"""
简化的 DataLoader 测试脚本
用于验证数据集划分和加载逻辑
"""
import os
from glob import glob

# 数据路径
IMAGE_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_mr"
LABEL_42_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr"
LABEL_5_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_5groups"
LABEL_BIN_DIR = "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_binary"

print("=" * 70)
print("数据集文件检查")
print("=" * 70)

# 检查文件数量
images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii.gz")))
labels_42 = sorted(glob(os.path.join(LABEL_42_DIR, "*.nii.gz")))
labels_5 = sorted(glob(os.path.join(LABEL_5_DIR, "*.nii.gz")))
labels_bin = sorted(glob(os.path.join(LABEL_BIN_DIR, "*.nii.gz")))

print(f"\n图像文件数: {len(images)}")
print(f"42类标签数: {len(labels_42)}")
print(f"5类标签数: {len(labels_5)}")
print(f"二值标签数: {len(labels_bin)}")

# 显示前5个文件名
print(f"\n前5个图像文件:")
for i, img in enumerate(images[:5]):
    print(f"  {i}: {os.path.basename(img)}")

# 验证文件名匹配
print(f"\n文件名匹配检查:")
all_match = True
for img, l42, l5, lbin in zip(images[:5], labels_42[:5], labels_5[:5], labels_bin[:5]):
    img_name = os.path.basename(img).replace("_0000.nii.gz", "")
    l42_name = os.path.basename(l42).replace(".nii.gz", "")
    l5_name = os.path.basename(l5).replace(".nii.gz", "")
    lbin_name = os.path.basename(lbin).replace(".nii.gz", "")

    match = (img_name == l42_name == l5_name == lbin_name)
    status = "OK" if match else "MISMATCH"
    print(f"  {img_name}: {status}")
    if not match:
        all_match = False

print(f"\n所有文件名匹配: {'OK' if all_match else 'MISMATCH'}")

# 数据集划分模拟
import numpy as np
np.random.seed(42)
indices = np.arange(25)
np.random.shuffle(indices)

train_size = int(25 * 0.6)
val_size = int(25 * 0.2)

train_indices = indices[:train_size].tolist()
val_indices = indices[train_size:train_size + val_size].tolist()
test_indices = indices[train_size + val_size:].tolist()

print(f"\n" + "=" * 70)
print("数据集划分 (总计 25 样本)")
print("=" * 70)
print(f"训练集 ({len(train_indices)} 样本): {train_indices}")
print(f"验证集 ({len(val_indices)} 样本): {val_indices}")
print(f"测试集 ({len(test_indices)} 样本): {test_indices}")

# 显示划分后的文件
print(f"\n训练集文件示例:")
for idx in train_indices[:3]:
    print(f"  [{idx}] {os.path.basename(images[idx])}")

print(f"\n验证集文件示例:")
for idx in val_indices[:3]:
    print(f"  [{idx}] {os.path.basename(images[idx])}")

print(f"\n" + "=" * 70)
print("done")
print("=" * 70)
