# 基础 3D U-Net
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.blocks import Convolution

class MultiHeadVesselUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels_fine=43, out_channels_reg=6, out_channels_bin=2):
        super().__init__()
        # 使用 MONAI 的基础 UNet 作为 Backbone
        # 我们只取其内部的 encoder 和 decoder 逻辑
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=32,  # 最后一层特征维度
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        # 获取 UNet 的核心特征提取部分（去除原有的输出层）
        self.model_base = self.unet.model

        # 1. 二值化输出头 (Binary Head): 判断血管 vs 背景
        self.head_bin = Convolution(
            spatial_dims=3, in_channels=32, out_channels=out_channels_bin,
            kernel_size=1, act=None, bias=True
        )

        # 2. 区域系统输出头 (Regional Head): 判断 5 大系统 + 背景
        # 级联监督：拼接 shared_features + binary 预测
        self.head_reg = Convolution(
            spatial_dims=3, in_channels=32 + out_channels_bin, out_channels=out_channels_reg,
            kernel_size=1, act=None, bias=True
        )

        # 3. 精细分类输出头 (Fine Head): 42 类血管分类
        # 级联监督：拼接 shared_features + binary 预测 + regional 预测
        self.head_fine = Convolution(
            spatial_dims=3, in_channels=32 + out_channels_bin + out_channels_reg,
            out_channels=out_channels_fine,
            kernel_size=1, act=None, bias=True
        )

    def forward(self, x):
        # 提取共享的 Decoder 特征 [B, 32, H, W, D]
        shared_features = self.model_base(x)

        # 分支 1：生成二值预测 [B, 2, H, W, D]
        out_bin = self.head_bin(shared_features)

        # 分支 2：生成区域预测 (级联：拼接 binary 信息)
        reg_input = torch.cat([shared_features, out_bin], dim=1)  # [B, 32+2, H, W, D]
        out_reg = self.head_reg(reg_input)

        # 分支 3：精细分类 (级联：拼接 binary + regional 信息)
        fine_input = torch.cat([shared_features, out_bin, out_reg], dim=1)  # [B, 32+2+6, H, W, D]
        out_fine = self.head_fine(fine_input)

        return out_bin, out_reg, out_fine


if __name__ == "__main__":
    # 简单测试代码
    print("=" * 60)
    print("MultiHeadVesselUNet 级联监督结构测试")
    print("=" * 60)

    # 创建模型
    model = MultiHeadVesselUNet(
        in_channels=1,
        out_channels_fine=43,
        out_channels_reg=6,
        out_channels_bin=2
    )
    model.eval()

    # 模拟输入：[Batch=2, Channel=1, H=96, W=96, D=96]
    dummy_input = torch.randn(2, 1, 96, 96, 96)

    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输入内存: {dummy_input.element_size() * dummy_input.nelement() / 1024**2:.2f} MB")

    # 前向传播
    with torch.no_grad():
        out_bin, out_reg, out_fine = model(dummy_input)

    print("\n" + "-" * 60)
    print("输出形状:")
    print(f"  - Binary Head (2类):   {out_bin.shape}")
    print(f"  - Regional Head (6类): {out_reg.shape}")
    print(f"  - Fine Head (43类):    {out_fine.shape}")

    print("\n" + "-" * 60)
    print("级联监督信息流:")
    print(f"  - Binary Head 输入:   [32] 共享特征")
    print(f"  - Regional Head 输入: [32+2] 共享特征 + Binary预测")
    print(f"  - Fine Head 输入:     [32+2+6] 共享特征 + Binary预测 + Regional预测")

    print("\n" + "-" * 60)
    print("参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小: {total_params * 4 / 1024**2:.2f} MB (FP32)")

    print("\n" + "=" * 60)
    print("✓ 测试通过！级联监督网络结构正常")
    print("=" * 60)