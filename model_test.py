import torch
import torch.nn as nn
import time

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

def measure_time(module, runs=100):
    """测量 module 在 input_tensor 上的平均前向推理时间 (ms)。"""
    # device = input_tensor.device
    device = torch.device('cuda')
    # 如果使用 GPU，需要在计时前后同步
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            x = torch.randn(1, 128, 960, 1280, device=device)
            _ = module(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / runs * 1000.0

def main():
    # 测试参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size  = 1
    in_channels = 128
    out_channels= 128
    H = 640
    W = 480

    # 随机输入


    # 实例化并切换到评估模式
    basic = BasicLayer(in_channels, out_channels).to(device).eval()
    ds    = DepthwiseSeparableFused(in_channels, out_channels).to(device).eval()

    # # 预热
    # with torch.no_grad():
    #     for _ in range(10):
    #         _ = basic(x)
    #         _ = ds(x)

    # 测时
    runs = 100
    t_basic = measure_time(basic, runs)
    t_ds    = measure_time(ds,    runs)

    # 输出结果
    print(f"平均每次前向推理耗时（{runs} 次取平均）:")
    print(f"  BasicLayer:                {t_basic:7.3f} ms")
    print(f"  DepthwiseSeparableLayer:   {t_ds:7.3f} ms")

if __name__ == '__main__':
    main()
