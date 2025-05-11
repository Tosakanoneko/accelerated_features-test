"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    Modified with ShuffleNet V2‑style backbone units.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------  ShuffleNet V2 基础组件  ---------- #
class ChannelShuffle(nn.Module):
    """Channel shuffle for G=2."""
    def __init__(self, groups: int = 2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        g = self.groups
        # (B, g, C/g, H, W) -> transpose -> (B, C, H, W)
        x = x.view(b, g, c // g, h, w).transpose(1, 2).contiguous()
        return x.view(b, c, h, w)


class InvertedResidual(nn.Module):
    """
    ShuffleNet V2 单元
        stride = 1 :  输入通道数 == 输出通道数
        stride = 2 :  空间下采样 & 通道数增加
    """
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        branch_features = oup // 2

        if stride == 1:
            assert inp == oup, "When stride=1, inp should equal oup"
            # 前一半直接跳过，后一半走分支‑2
            self.branch1 = nn.Identity()
            self.branch2 = nn.Sequential(
                # pw
                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(branch_features, branch_features, 3, 1, 1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                # pw‑linear
                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                nn.ReLU(inplace=True),
            )
        else:  # stride == 2
            # 分支‑1：仅产生特征，负责下采样
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 2, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, affine=False),
                # pw‑linear
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                nn.ReLU(inplace=True),
            )
            # 分支‑2
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, 2, 1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features, affine=False),
                nn.ReLU(inplace=True),
            )

        self.shuffle = ChannelShuffle(groups=2)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:  # stride == 2
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return self.shuffle(out)


# ----------  其他通用层  ---------- #
class BasicLayer(nn.Module):
    """Conv‑BN‑ReLU wrapper (沿用原实现，供首层 / heads / fusion 使用)"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, d=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p,
                      dilation=d, bias=bias),
            nn.BatchNorm2d(out_c, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # noqa: D401
        return self.layer(x)


# ----------  XFeat 主体  ---------- #
class XFeatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)   # 灰度归一化（与原版一致）

        # ---- CNN Backbone ---- #
        # (1) stem & skip‑connection‑path  (分辨率 /4, 通道 24)
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1, stride=1, padding=0)
        )
        self.block1 = nn.Sequential(
            BasicLayer(1, 4, 3, 1),
            BasicLayer(4, 8, 3, 2),    # /2
            BasicLayer(8, 8, 3, 1),
            BasicLayer(8, 24, 3, 2),   # /4
        )                              # out: 24 ch, /4

        # (2) ShuffleNet V2 stages
        # -------------  /4 -> /4 (24 ch) -------------
        self.block2 = nn.Sequential(
            InvertedResidual(24, 24, 1),
            InvertedResidual(24, 24, 1),
        )
        # -------------  /4 -> /8 (64 ch) -------------
        self.block3 = nn.Sequential(
            InvertedResidual(24, 64, 2),   # /8
            InvertedResidual(64, 64, 1),
            InvertedResidual(64, 64, 1),
        )
        # -------------  /8 -> /16 (64 ch) -------------
        self.block4 = nn.Sequential(
            InvertedResidual(64, 64, 2),   # /16
            InvertedResidual(64, 64, 1),
            InvertedResidual(64, 64, 1),
        )
        # -------------  /16 -> /32 (128 ch → 再压缩到 64) -------------
        self.block5 = nn.Sequential(
            InvertedResidual(64, 128, 2),  # /32
            InvertedResidual(128, 128, 1),
            InvertedResidual(128, 128, 1),
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),  # 对齐通道
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
        )

        # Pyramid fusion (保持原版设计)
        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, 3, 1),
            BasicLayer(64, 64, 3, 1),
            nn.Conv2d(64, 64, 1, 1, 0)
        )

        # Heads
        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            nn.Conv2d(64, 65, 1)
        )

        # Fine matcher MLP（原封不动）
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    # ----------  util: unfold‑2D (保留)  ---------- #
    @staticmethod
    def _unfold2d(x, ws: int = 2):
        """窗口展开并拼接通道，保持与原实现一致。"""
        b, c, h, w = x.shape
        x = (x.unfold(2, ws, ws)
               .unfold(3, ws, ws)
               .reshape(b, c, h // ws, w // ws, ws ** 2))
        return x.permute(0, 1, 4, 2, 3).reshape(b, -1, h // ws, w // ws)

    # ----------  forward ---------- #
    def forward(self, img):
        """
        Args:
            img (tensor): (B, C, H, W)  RGB / 灰度 (自动转灰度)
        Returns:
            feats     (B, 64, H/8, W/8)   dense descriptors
            keypoints (B, 65, H/8, W/8)   keypoint logits
            heatmap   (B, 1,  H/8, W/8)   reliability map
        """
        # 灰度 + InstanceNorm (不反向传播)
        with torch.no_grad():
            x = img.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # Backbone
        x1 = self.block1(x)                 # /4 , 24c
        x2 = self.block2(x1 + self.skip1(x))  # /4 , 24c
        x3 = self.block3(x2)                # /8 , 64c
        x4 = self.block4(x3)                # /16, 64c
        x5 = self.block5(x4)                # /32, 64c

        # Pyramid fusion  (所有特征对齐到 /8)
        x4_up = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x5_up = F.interpolate(x5, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        feats = self.block_fusion(x3 + x4_up + x5_up)   # (B,64,H/8,W/8)

        # Heads
        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))

        return feats, keypoints, heatmap
