from torch.optim import Adam
from monai.losses import DiceCELoss

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    # 统一使用 DiceCELoss，但各输出头权重可配置
    loss_bin_func = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=0.5,  # Binary 任务 Dice 权重
        lambda_ce=0.5     # Binary 任务 CE 权重
    )

    loss_reg_func = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=0.6,  # Regional 任务 Dice 权重
        lambda_ce=0.4     # Regional 任务 CE 权重
    )

    loss_fine_func = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=0.7,  # Fine-grained 任务 Dice 权重（更重视重叠度）
        lambda_ce=0.3     # Fine-grained 任务 CE 权重
    )

    # 动态调整权重 (调度策略)
    # 前 20 个 epoch 重点练二值化，之后重点练 42 类
    if epoch < 20:
        w_bin, w_reg, w_fine = 1.0, 0.5, 0.2
    else:
        w_bin, w_reg, w_fine = 0.3, 0.6, 1.0

    total_epoch_loss = 0

    for batch_data in loader:
        inputs = batch_data["image"].to(device)
        # 获取三种标签
        target_bin = batch_data["label_bin"].to(device)   # [B, 1, H, W, D]
        target_reg = batch_data["label_reg"].to(device)   # [B, 1, H, W, D]
        target_fine = batch_data["label_fine"].to(device) # [B, 1, H, W, D]

        optimizer.zero_grad()

        # 模型前向传播
        out_bin, out_reg, out_fine = model(inputs)

        # 统一使用 DiceCELoss
        loss_bin = loss_bin_func(out_bin, target_bin)
        loss_reg = loss_reg_func(out_reg, target_reg)
        loss_fine = loss_fine_func(out_fine, target_fine)

        # 总损失同步回传
        total_loss = w_bin * loss_bin + w_reg * loss_reg + w_fine * loss_fine

        total_loss.backward()
        optimizer.step()

        total_epoch_loss += total_loss.item()

    return total_epoch_loss / len(loader)

# 提示：DataLoader 需要预处理出三种 label key
# 可以通过 Lambdad 变换实时生成，参考之前的重映射代码