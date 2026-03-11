# 可视化 3D 医学影像数据 .nii.gz
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, patches


def build_label_colormap(label_values):
    label_values = sorted(int(v) for v in label_values if v != 0)
    if not label_values:
        return {}, colors.ListedColormap([]), colors.BoundaryNorm([], 0)

    if len(label_values) <= 20:
        base_cmap = plt.get_cmap('tab20', len(label_values))
    else:
        base_cmap = plt.get_cmap('nipy_spectral', len(label_values))

    color_list = [base_cmap(i) for i in range(len(label_values))]
    color_map = {label: color_list[idx] for idx, label in enumerate(label_values)}
    listed_cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(np.arange(len(label_values) + 1) - 0.5, listed_cmap.N)
    return color_map, listed_cmap, norm


def extract_surface_voxels(label_volume):
    occupied = label_volume > 0
    if not np.any(occupied):
        return occupied

    interior = occupied.copy()
    interior[1:-1, 1:-1, 1:-1] &= occupied[:-2, 1:-1, 1:-1]
    interior[1:-1, 1:-1, 1:-1] &= occupied[2:, 1:-1, 1:-1]
    interior[1:-1, 1:-1, 1:-1] &= occupied[1:-1, :-2, 1:-1]
    interior[1:-1, 1:-1, 1:-1] &= occupied[1:-1, 2:, 1:-1]
    interior[1:-1, 1:-1, 1:-1] &= occupied[1:-1, 1:-1, :-2]
    interior[1:-1, 1:-1, 1:-1] &= occupied[1:-1, 1:-1, 2:]
    return occupied & ~interior

def plot_multiclass_vessels(img_path, label_path, slice_idx=None):
    # 加载数据
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata().astype(np.int32)

    # 默认展示中心切片
    if slice_idx is None:
        slice_idx = img.shape[2] // 2

    slice_label = label[:, :, slice_idx]
    unique_labels = np.unique(slice_label)
    unique_labels = unique_labels[unique_labels != 0]
    color_map, listed_cmap, norm = build_label_colormap(unique_labels)

    remapped_slice = np.full(slice_label.shape, -1, dtype=np.int32)
    for mapped_idx, label_value in enumerate(unique_labels):
        remapped_slice[slice_label == label_value] = mapped_idx

    masked_label = np.ma.masked_where(remapped_slice < 0, remapped_slice)

    plt.figure(figsize=(14, 6))

    # 原始影像
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, slice_idx].T, cmap='gray', origin='lower')
    plt.title('Original MRA (Axial)')
    plt.axis('off')

    # 叠加标签
    plt.subplot(1, 2, 2)
    plt.imshow(img[:, :, slice_idx].T, cmap='gray', origin='lower')
    if unique_labels.size > 0:
        plt.imshow(masked_label.T, cmap=listed_cmap, norm=norm, origin='lower', alpha=0.75)
        handles = [
            patches.Patch(color=color_map[label_value], label=f'Label {label_value}')
            for label_value in unique_labels
        ]
        plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=8)
    plt.title('Label Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_label_3d(label_path, downsample=4, max_points=50000, point_size=2):
    label = nib.load(label_path).get_fdata().astype(np.int32)

    if downsample > 1:
        label = label[::downsample, ::downsample, ::downsample]

    surface_mask = extract_surface_voxels(label)
    unique_labels = np.unique(label[surface_mask])
    color_map, _, _ = build_label_colormap(unique_labels)

    if unique_labels.size == 0:
        raise ValueError('标签文件中没有前景类别，无法进行 3D 可视化。')

    coords = np.argwhere(surface_mask)
    point_labels = label[surface_mask]

    if coords.shape[0] > max_points:
        sampled_idx = np.linspace(0, coords.shape[0] - 1, max_points, dtype=np.int32)
        coords = coords[sampled_idx]
        point_labels = point_labels[sampled_idx]

    point_colors = np.array([color_map[int(label_value)] for label_value in point_labels])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=point_colors,
        s=point_size,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_title('3D Label Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect(label.shape)

    handles = [
        patches.Patch(color=color_map[label_value], label=f'Label {label_value}')
        for label_value in unique_labels
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)

    plt.tight_layout()
    plt.show()


# 调用示例
image_path = './Data/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_mr/topcow_mr_001_0000.nii.gz'
label_path = './Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr/topcow_mr_001.nii.gz'
plot_multiclass_vessels(image_path, label_path)
plot_label_3d(label_path, downsample=2, max_points=5000000, point_size=2)
