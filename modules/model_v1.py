import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):
    """
    Basic Conv->BN->ReLU with BN affine=False to match original training.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride=stride,
                      padding=padding, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class DepthwiseSeparableFused(nn.Module):
    """
    Fused Depthwise Separable Conv for inference:
      - depthwise_conv (bias=True) -> ReLU
      - pointwise_conv (bias=True) -> ReLU
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, stride=stride,
            padding=padding, groups=in_channels,
            bias=True
        )
        self.depthwise_relu = nn.ReLU(inplace=True)
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=True
        )
        self.pointwise_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_relu(x)
        x = self.pointwise_conv(x)
        return self.pointwise_relu(x)
# class DepthwiseSeparableLayer(nn.Module):
#     """
#     Depthwise Separable Convolution Layer:
#     1) depthwise conv (groups=in_channels) + BN + ReLU
#     2) pointwise   conv (1×1)            + BN + ReLU
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
#         super().__init__()
#         self.ds = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size,
#                       stride=stride, padding=padding, groups=in_channels, bias=bias),
#             nn.BatchNorm2d(in_channels, affine=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
#             nn.BatchNorm2d(out_channels, affine=False),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.ds(x)

class XFeatModel(nn.Module):
    """
    Inference-only XFeat model with fused DepthwiseSeparableLayer.
    Load 'fused_weights.pth' directly.
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)

        # skip1 & block1 (standard convs)
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
            nn.Conv2d(1, 24, 1)
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

        # block3: use fused DS
        self.block3 = nn.Sequential(
            DepthwiseSeparableFused(24, 64, stride=2),
            DepthwiseSeparableFused(64, 64, stride=1),
            BasicLayer(64, 64, kernel_size=1, padding=0)
        )
        # block4: all fused DS
        self.block4 = nn.Sequential(
            DepthwiseSeparableFused(64, 64, stride=2),
            DepthwiseSeparableFused(64, 64, stride=1),
            DepthwiseSeparableFused(64, 64, stride=1),
        )
        # block5: first three fused, last is 1x1 conv
        self.block5 = nn.Sequential(
            DepthwiseSeparableFused(64, 128, stride=2),
            DepthwiseSeparableFused(128, 128, stride=1),
            DepthwiseSeparableFused(128, 128, stride=1),
            BasicLayer(128, 64, kernel_size=1, padding=0)
        )

        # fusion head
        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1)
        )

        # heads
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

        # fine matcher
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.BatchNorm1d(512, affine=False), nn.ReLU(inplace=True),
            nn.Linear(512, 64)
        )

    def _unfold2d(self, x, ws=2):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws)
        x = x.reshape(B, C, H//ws, W//ws, ws**2)
        x = x.permute(0, 1, 4, 2, 3)
        return x.reshape(B, -1, H//ws, W//ws)

    def forward(self, x):
        # normalize
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        # multi-scale
        x4_up = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear')
        x5_up = F.interpolate(x5, size=x3.shape[-2:], mode='bilinear')
        feats = self.block_fusion(x3 + x4_up + x5_up)
        heatmap = self.heatmap_head(feats)
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8))
        return feats, keypoints, heatmap

# Usage:
# model = XFeatModelFused()
# model.load_state_dict(torch.load('fused_weights.pth', map_location='cpu'))
# model.eval()
