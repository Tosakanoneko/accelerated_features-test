import torch
import torch.nn as nn
import time
from modules.model_shufflenet import InvertedResidual

class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class DepthwiseSeparableLayer(nn.Module):
    """
    Depthwise Separable Convolution Layer:
    1) depthwise conv (groups=in_channels) + BN + ReLU
    2) pointwise   conv (1×1)            + BN + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class block4_ori(nn.Module):
    """
       Implementation of architecture described in 
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            BasicLayer(in_channels, out_channels, stride=2),
			BasicLayer(in_channels, out_channels, stride=1),
			BasicLayer(in_channels, out_channels, stride=1),
        )
    def forward(self, x):
        return self.block(x)
    
class block4_ds(nn.Module):
    """
       Implementation of architecture described in 
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableLayer(in_channels, out_channels, stride=2),
			DepthwiseSeparableLayer(in_channels, out_channels, stride=1),
			DepthwiseSeparableLayer(in_channels, out_channels, stride=1),
        )
    def forward(self, x):
        return self.block(x)
    
class block4_sf(nn.Module):
    """
       Implementation of architecture described in 
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            InvertedResidual(in_channels, out_channels, 2),   
            InvertedResidual(in_channels, out_channels, 1),
            InvertedResidual(in_channels, out_channels, 1),
        )
    def forward(self, x):
        return self.block(x)

def measure_time(module, runs=100):
    """测量 module 在 input_tensor 上的平均前向推理时间 (ms)。"""
    # 如果使用 GPU，需要在计时前后同步
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            x = torch.randn(1, in_channels, H, W, device=device)
            _ = module(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / runs * 1000.0

# 测试参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size  = 1
in_channels = 512
out_channels= 512
H = 640 // 8
W = 480 // 8

if __name__ == '__main__':
   
    # basic = BasicLayer(in_channels, out_channels).to(device).eval()
    # ds    = DepthwiseSeparableLayer(in_channels, out_channels).to(device).eval()

    basic = block4_ori().to(device).eval()
    ds    = block4_sf().to(device).eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            x = torch.randn(in_channels, in_channels, H, W, device=device)
            _ = basic(x)
            _ = ds(x)

    # 测时
    runs = 1000
    t_basic = measure_time(basic, runs)
    t_ds    = measure_time(ds,    runs)

    # 输出结果
    print(f"平均每次前向推理耗时（{runs} 次取平均）:")
    print(f"  BasicLayer:                {t_basic:7.3f} ms")
    print(f"  DepthwiseSeparableLayer:   {t_ds:7.3f} ms")

