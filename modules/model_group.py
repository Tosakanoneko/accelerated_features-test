import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def channel_shuffle(x, groups):
    """
    对输入张量 x 进行通道 shuffle 操作。
    将通道维度先分为 groups 组，再在组内交换通道，实现跨组信息混合。
    """
    B, C, H, W = x.size()
    assert C % groups == 0, "channels must be divisible by groups"
    channels_per_group = C // groups
    # 重塑为 (B, groups, channels_per_group, H, W)
    x = x.view(B, groups, channels_per_group, H, W)
    # 转置 groups 与 channels_per_group 维度
    x = torch.transpose(x, 1, 2).contiguous()
    # 恢复原始形状
    x = x.view(B, C, H, W)
    return x

class BasicLayer(nn.Module):
    """
      基本卷积层，支持可选的分组卷积和通道 shuffle：
      Conv2d -> BatchNorm -> ReLU -> (如果启用则进行 channel shuffle)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, groups=1, shuffle=False):
        super().__init__()
        self.groups = groups
        self.shuffle_flag = shuffle
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.shuffle_flag and self.groups > 1:
            x = channel_shuffle(x, self.groups)
        return x

class XFeatModel(nn.Module):
    """
       XFeat 主干网络的修改版：
       在中后期卷积层（block3, block4, block5 以及 block_fusion 部分）中采用 4 组分组卷积，
       并搭配通道 shuffle 以保证跨组信息流通，从而降低 MAC 而对精度影响有限。
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )
        
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=1),
            BasicLayer(4, 8, stride=2),
            BasicLayer(8, 8, stride=1),
            BasicLayer(8, 24, stride=2),
        )
        
        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )
        
        # 中期卷积层：采用 4 组分组卷积及通道 shuffle
        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2, groups=4, shuffle=True),
            BasicLayer(64, 64, stride=1, groups=4, shuffle=True),
            BasicLayer(64, 64, kernel_size=1, padding=0, groups=4, shuffle=True),
        )
        # 后期卷积层
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2, groups=4, shuffle=True),
            BasicLayer(64, 64, stride=1, groups=4, shuffle=True),
            BasicLayer(64, 64, stride=1, groups=4, shuffle=True),
        )
        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2, groups=4, shuffle=True),
            BasicLayer(128, 128, stride=1, groups=4, shuffle=True),
            BasicLayer(128, 128, stride=1, groups=4, shuffle=True),
            BasicLayer(128, 64, kernel_size=1, padding=0, groups=4, shuffle=True),
        )
        
        # 融合层：降低通道数同时采用分组卷积帮助减少计算
        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1, groups=4, shuffle=True),
            BasicLayer(64, 64, stride=1, groups=4, shuffle=True),
            nn.Conv2d(64, 64, 1, padding=0)
        )
        
        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, kernel_size=1, padding=0),
            BasicLayer(64, 64, kernel_size=1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, kernel_size=1, padding=0),
            BasicLayer(64, 64, kernel_size=1, padding=0),
            BasicLayer(64, 64, kernel_size=1, padding=0),
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
            对二维张量进行 unfold 操作，窗口大小 ws，
            并将卷展后的特征通道拼接。
        """
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape(B, C, H // ws, W // ws, ws ** 2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)
    
    def forward(self, x):
        """
            输入:
                x -> torch.Tensor(B, C, H, W) 灰度或 RGB 图像
            输出:
                feats     -> torch.Tensor(B, 64, H/8, W/8) 稠密局部特征
                keypoints -> torch.Tensor(B, 65, H/8, W/8) 关键点 logits 图
                heatmap   -> torch.Tensor(B, 1, H/8, W/8) 可靠性热图
        """
        # 不对归一化反向传播梯度
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)
        
        # 主干网络
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        
        # 金字塔式融合
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)
        
        # 预测头
        heatmap = self.heatmap_head(feats)  # 可靠性热图
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))  # 关键点 logits 图
        
        return feats, keypoints, heatmap
