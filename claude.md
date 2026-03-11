# Claude.md - TopBrain 2025 脑血管分割项目说明书

## 1. 项目背景 (Project Context)
- **目标**：实现 TopBrain 2025 (MICCAI) 挑战赛 MRA 轨道 42 类全脑血管精细化分割。
- **核心难点**：42类极度不平衡；细小分支（Side roads）易断裂；要求解剖学拓扑一致性。
- **研究方案**：空间先验（CoordConv）+ 结构约束（clDice/PH Loss）+ 多头输出架构。

## 2. 技术栈 (Technical Stack)
- **核心框架**：PyTorch 2.x, MONAI (必选).
- **开发工具**：VS Code
- **数据格式**：NIfTI (.nii.gz), 3D 体素数据。
- **预处理标准**：N4 偏场校正、1.0mm 各向同性重采样、颅骨剥离 (Skull Stripping)。

<!-- ## 3. 标签体系与解剖映射 (Anatomy Knowledge Base) -->

<!-- ### A. 分级任务 (Hierarchy)
- **Level 1 (Binary)**: 全血管 [1-42] vs 背景 [0].
- **Level 2 (Regional)**: 
  - Willis 环区域 (Labels: 1-12, 15)
  - 远端分支区域 (Labels: 13-34)
  - 颈外/头皮系统 (Labels: 35-42)

### B. 重要性权重 (Weighting Strategy)
- **Highway (干道)**: 1, 2, 3, 4, 5, 6, 11, 12, 35-40 (侧重 Dice).
- **Side-road (支路)**: 8, 9, 10, 13-34, 41, 42 (侧重拓扑一致性与 F1 Score). -->

## 4. 网络架构规则 (Model Architecture)
- **主干网络**：3D U-Net / ERNet 变体。
<!-- - **输入层**：集成 CoordConv (拼入 x, y, z 归一化坐标张量)。
- **输出头 (Heads)**：
  - `head_bin`: [B, 1, H, W, D] (Vessel mask)
  - `head_reg`: [B, 7, H, W, D] (Regional groups)
  - `head_fine`: [B, 43, H, W, D] (42 classes + BG) -->

## 5. 损失函数策略 (Loss Strategy)
- **复合损失公式**：$L = \alpha L_{Dice} + \beta L_{CE} + \gamma L_{clDice}$.
- **拓扑约束**：对于 Side-road 类别，必须引入 Soft-clDice。
<!-- - **PH Loss**：仅在微调阶段 (Fine-tuning) 启用，默认代码中预留接口但设为 `requires_grad=False` 以节省资源。 -->

## 6. 编码与协作规范 (Coding Guidelines)
- **模块化**：损失函数写在 `losses.py`，网络结构写在 `models/`。
- **类型检查**：所有函数需包含 Type Hints，明确 `torch.Tensor` 的形状。
- **内存优化**：3D 训练 Patch 限制在 `[96, 96, 96]`，默认使用 `torch.cuda.amp` 混合精度。
- **拒绝冗余**：除非显式要求，否则不要重写已存在的 MONAI 基础组件。

<!-- ## 7. 当前任务状态 (Current Status)
- [ ] 环境初始化与数据路径配置
- [ ] 42类标签映射脚本编写
- [ ] 多头输出 3D U-Net 模型定义
- [ ] 训练 Pipeline (Trainer) 开发 -->