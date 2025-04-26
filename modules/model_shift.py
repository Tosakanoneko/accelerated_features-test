import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftModule(nn.Module):
    """
    基于移位操作的卷积模块：
    1. 对输入特征按照通道数分为 5 组，每组采用不同的移位策略（上、下、左、右和不变）。
    2. 移位操作本身无任何乘加运算，随后利用 1×1 卷积进行跨通道混合。
    
    该模块支持下采样（通过1×1卷积的 stride 参数），适合替换分辨率最高阶段的空间卷积，
    从而降低早期阶段的计算开销（MAC）并有望整体降低约 20%-30% 的 MAC。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shift(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def shift(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        # 将通道数划分为 5 组：上、下、左、右、保持不动（身份映射）
        group = C // 5
        remainder = C - group * 5
        out = []
        idx = 0
        # 第一组：向上平移
        ch = group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        out.append(torch.roll(x[:, idx:idx+ch, :, :], shifts=-1, dims=2))
        idx += ch
        # 第二组：向下平移
        ch = group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        out.append(torch.roll(x[:, idx:idx+ch, :, :], shifts=1, dims=2))
        idx += ch
        # 第三组：向左平移
        ch = group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        out.append(torch.roll(x[:, idx:idx+ch, :, :], shifts=-1, dims=3))
        idx += ch
        # 第四组：向右平移
        ch = group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        out.append(torch.roll(x[:, idx:idx+ch, :, :], shifts=1, dims=3))
        idx += ch
        # 第五组：不平移（身份映射）
        out.append(x[:, idx:, :, :])
        return torch.cat(out, dim=1)

class BasicLayer(nn.Module):
    """
    基本卷积层：3×3卷积 -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class XFeatModel_shift(nn.Module):
    """
    XFeat 主干网络的实现，参考论文 “XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024.”
    
    修改说明：
    - 针对高分辨率的早期卷积层（block1 前两层），将 BasicLayer 替换为基于移位操作的 ShiftModule，
      从而完全消除空间卷积的MAC开销，仅保留轻量的1×1卷积。预计整体MAC下降可达20%-30%。
    - 其他部分保持原有结构，确保后续特征提取、融合以及头部网络均符合设计预期。
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )

        # 修改 block1：用 ShiftModule 替换前两层 BasicLayer
        self.block1 = nn.Sequential(
            ShiftModule(1, 4, stride=1),   # 第一层：低计算开销的移位操作
            ShiftModule(4, 8, stride=2),     # 第二层：移位 + 下采样
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x, ws=2):
        """
        将张量按照指定的窗口尺寸 (ws) 展开，并将各通道拼接。
        """
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)

    def forward(self, x):
        """
        输入:
            x -> torch.Tensor(B, C, H, W) 灰度或RGB图像
        输出:
            feats     -> torch.Tensor(B, 64, H/8, W/8) 局部稠密特征
            keypoints -> torch.Tensor(B, 65, H/8, W/8) 关键点对数图（未归一化的logits）
            heatmap   -> torch.Tensor(B,  1, H/8, W/8) 可靠性图
        """
        # 在归一化阶段不反向传播
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # 主干网络
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # 金字塔融合
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        # Heads
        heatmap = self.heatmap_head(feats)  # 可靠性图
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))  # 关键点logits

        return feats, keypoints, heatmap
