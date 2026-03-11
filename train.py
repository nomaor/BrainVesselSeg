import torch
from torch.optim import Adam
from monai.losses import DiceLoss, DiceCELoss
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    
    # 定义基础损失函数
    dice_loss_func = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, to_onehot_y=True, softmax=True)
    ce_loss_func = torch.nn.CrossEntropyLoss()
    
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

        # 1. Binary Loss (使用 BCE 或 Dice)
        # 这里的 target_bin 通常是 [B, 1, H, W, D]，值为 0 或 1
        loss_bin = F.binary_cross_entropy(out_bin, target_bin.float())

        # 2. Regional Loss
        # target_reg 值为 0-5
        loss_reg = ce_loss_func(out_reg, target_reg.squeeze(1).long())

        # 3. Fine-grained Loss (42类)
        # 推荐使用 DiceCELoss 或在 Dice 中加入对 Side-road 类别的加权
        loss_fine = dice_loss_func(out_fine, target_fine)

        # 总损失同步回传
        total_loss = w_bin * loss_bin + w_reg * loss_reg + w_fine * loss_fine
        
        total_loss.backward()
        optimizer.step()

        total_epoch_loss += total_loss.item()

    return total_epoch_loss / len(loader)

# 提示：DataLoader 需要预处理出三种 label key
# 可以通过 Lambdad 变换实时生成，参考之前的重映射代码