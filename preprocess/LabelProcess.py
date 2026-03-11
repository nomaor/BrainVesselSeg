# 标签处理
import os
from glob import glob
import nibabel as nib
import numpy as np


# def remap_vessel_labels(input_path, output_path):
#     """
#     将 42 类脑血管标签重映射为 5 个解剖系统分组
#     """
#     nifti_img = nib.load(input_path)
#     label_data = nifti_img.get_fdata().astype(np.int32)
#     affine = nifti_img.affine
#     header = nifti_img.header

#     mapping = {0: 0}

#     # 系统 1: 后循环系统 (Posterior)
#     for i in [1, 2, 3, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
#         mapping[i] = 1

#     # 系统 2: 前循环-颈内系统 (Anterior-ICA)
#     for i in [4, 5, 6, 7, 17, 18, 19, 20, 31, 32, 33, 34]:
#         mapping[i] = 2

#     # 系统 3: 前循环-前交通系统 (Anterior-ACA)
#     for i in [11, 12, 13, 14, 15, 16]:
#         mapping[i] = 3

#     # 系统 4: 交通动脉 (Communicating)
#     for i in [8, 9, 10]:
#         mapping[i] = 4

#     # 系统 5: 颈外/头皮系统 (External)
#     for i in [35, 36, 37, 38, 39, 40, 41, 42]:
#         mapping[i] = 5

#     new_label_data = np.zeros_like(label_data)
#     for old_val, new_val in mapping.items():
#         new_label_data[label_data == old_val] = new_val

#     new_img = nib.Nifti1Image(new_label_data.astype(np.uint8), affine, header)
#     nib.save(new_img, output_path)
#     print(f"转换完成！保存路径: {output_path}")


# def batch_remap_vessel_labels(input_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     label_paths = sorted(glob(os.path.join(input_dir, '*.nii.gz')))

#     if not label_paths:
#         raise FileNotFoundError(f'在目录中未找到 .nii.gz 标签文件: {input_dir}')

#     for input_path in label_paths:
#         file_name = os.path.basename(input_path)
#         output_path = os.path.join(output_dir, file_name)
#         remap_vessel_labels(input_path, output_path)

#     print(f'批量处理完成，共处理 {len(label_paths)} 个文件。')


# # 使用示例
# input_label_dir = './Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr'
# output_label_dir = './Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_5groups'
# batch_remap_vessel_labels(input_label_dir, output_label_dir)


def remap_to_binary_label(input_path, output_path):
    """
    将 42 类脑血管标签重映射为 1 个解剖系统分组 (即全血管二值化)
    0: 背景
    1: 所有血管 (原始标签 1-42)
    """
    # 1. 加载数据
    nifti_img = nib.load(input_path)
    label_data = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header

    # 2. 执行二值化映射
    # 只要原始值大于 0，全部设为 1
    new_label_data = np.where(label_data > 0, 1, 0).astype(np.uint8)

    # 3. 保存新文件
    new_img = nib.Nifti1Image(new_label_data, affine, header)
    nib.save(new_img, output_path)
    print(f"二值化转换完成！保存路径: {output_path}")

def batch_remap_to_binary(input_dir, output_dir):
    """
    批量将目录下的 42 类标签转换为二值化标签
    """
    os.makedirs(output_dir, exist_ok=True)
    label_paths = sorted(glob(os.path.join(input_dir, '*.nii.gz')))

    if not label_paths:
        raise FileNotFoundError(f'在目录中未找到 .nii.gz 标签文件: {input_dir}')

    for input_path in label_paths:
        file_name = os.path.basename(input_path)
        # 为了区分，建议在文件名中加入 binary 后缀，或者存放在不同文件夹
        output_path = os.path.join(output_dir, file_name)
        remap_to_binary_label(input_path, output_path)

    print(f'批量处理完成，共处理 {len(label_paths)} 个文件。')

# 使用示例
if __name__ == "__main__":
    # 输入路径 (原始 42 类标签)
    input_label_dir = './Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr'
    
    # 输出路径 (全血管二值化标签)
    output_label_dir = './Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_binary'
    
    batch_remap_to_binary(input_label_dir, output_label_dir)
