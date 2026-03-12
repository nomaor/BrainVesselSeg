"""
TopBrain 2025 脑血管分割训练主入口
整合 MultiHeadVesselUNet + 三标签训练策略
"""
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import DiceLoss
import time
from pathlib import Path

from models.unet_base import MultiHeadVesselUNet
from data_utils import split_dataset, get_vessel_loader
from train import train_one_epoch


def validate(model, loader, device):
    """
    验证循环
    """
    model.eval()

    dice_loss_func = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, to_onehot_y=True, softmax=True)
    ce_loss_func = torch.nn.CrossEntropyLoss()

    total_loss = 0
    total_loss_bin = 0
    total_loss_reg = 0
    total_loss_fine = 0

    with torch.no_grad():
        for batch_data in loader:
            inputs = batch_data["image"].to(device)
            target_bin = batch_data["label_bin"].to(device)
            target_reg = batch_data["label_reg"].to(device)
            target_fine = batch_data["label_fine"].to(device)

            # 前向传播
            out_bin, out_reg, out_fine = model(inputs)

            # 计算各分支损失
            loss_bin = F.binary_cross_entropy(out_bin, target_bin.float())
            loss_reg = ce_loss_func(out_reg, target_reg.squeeze(1).long())
            loss_fine = dice_loss_func(out_fine, target_fine.long())

            # 固定权重 (验证时不需要动态调整)
            loss = 0.5 * loss_bin + 0.5 * loss_reg + 1.0 * loss_fine

            total_loss += loss.item()
            total_loss_bin += loss_bin.item()
            total_loss_reg += loss_reg.item()
            total_loss_fine += loss_fine.item()

    n_batches = len(loader)
    return {
        "total": total_loss / n_batches,
        "bin": total_loss_bin / n_batches,
        "reg": total_loss_reg / n_batches,
        "fine": total_loss_fine / n_batches
    }


def main():
    # ==================== 配置参数 ====================
    CONFIG = {
        # 数据路径
        "image_dir": "Data/TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_mr",
        "label_42_dir": "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr",
        "label_5_dir": "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_5groups",
        "label_bin_dir": "Data/TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr_binary",

        # 训练参数
        "batch_size": 2,
        "roi_size": (96, 96, 96),
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "num_workers": 4,

        # 数据集划分
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "seed": 42,

        # 模型参数
        "in_channels": 1,
        "out_channels_bin": 2,    # 整体血管 + 背景
        "out_channels_reg": 6,    # 5系统 + 背景
        "out_channels_fine": 43,  # 42类 + 背景

        # 保存路径
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
    }

    # 创建保存目录
    Path(CONFIG["checkpoint_dir"]).mkdir(exist_ok=True)
    Path(CONFIG["log_dir"]).mkdir(exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # ==================== 数据集划分 ====================
    print("\n" + "=" * 70)
    print("数据集划分")
    print("=" * 70)
    train_idx, val_idx, test_idx = split_dataset(
        total_samples=25,
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        test_ratio=CONFIG["test_ratio"],
        seed=CONFIG["seed"]
    )

    # ==================== 创建 DataLoader ====================
    print("\n" + "=" * 70)
    print("创建 DataLoader")
    print("=" * 70)

    print("加载训练集...")
    train_loader = get_vessel_loader(
        image_dir=CONFIG["image_dir"],
        label_dir_42=CONFIG["label_42_dir"],
        label_dir_5=CONFIG["label_5_dir"],
        label_dir_binary=CONFIG["label_bin_dir"],
        batch_size=CONFIG["batch_size"],
        roi_size=CONFIG["roi_size"],
        indices=train_idx,
        use_cache=True,
        num_workers=CONFIG["num_workers"],
        is_train=True
    )

    print("加载验证集...")
    val_loader = get_vessel_loader(
        image_dir=CONFIG["image_dir"],
        label_dir_42=CONFIG["label_42_dir"],
        label_dir_5=CONFIG["label_5_dir"],
        label_dir_binary=CONFIG["label_bin_dir"],
        batch_size=1,
        roi_size=CONFIG["roi_size"],
        indices=val_idx,
        use_cache=True,
        num_workers=CONFIG["num_workers"],
        is_train=False
    )

    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    # ==================== 初始化模型 ====================
    print("\n" + "=" * 70)
    print("初始化模型")
    print("=" * 70)

    model = MultiHeadVesselUNet(
        in_channels=CONFIG["in_channels"],
        out_channels_fine=CONFIG["out_channels_fine"],
        out_channels_reg=CONFIG["out_channels_reg"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024**2:.2f} MB (FP32)")

    # ==================== 优化器和调度器 ====================
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    # ==================== 训练循环 ====================
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        epoch_start_time = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 验证
        val_losses = validate(model, val_loader, device)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 计算耗时
        epoch_time = time.time() - epoch_start_time

        # 打印日志
        print(f"\nEpoch [{epoch}/{CONFIG['num_epochs']}] - {epoch_time:.2f}s - LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_losses['total']:.4f} "
              f"(Bin: {val_losses['bin']:.4f}, "
              f"Reg: {val_losses['reg']:.4f}, "
              f"Fine: {val_losses['fine']:.4f})")

        # 保存最优模型
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch

            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_losses['total'],
                'config': CONFIG
            }, checkpoint_path)
            print(f"  -> 保存最优模型 (Val Loss: {best_val_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_losses['total'],
                'config': CONFIG
            }, checkpoint_path)
            print(f"  -> 保存 checkpoint: epoch_{epoch}.pth")

    # ==================== 训练完成 ====================
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"最优模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
    print(f"模型保存路径: {CONFIG['checkpoint_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
