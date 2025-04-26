import torch
import torch.nn.functional as F
import time

# 模拟输入数据，kpts的形状是 (1, 65, 60, 80)
kpts = torch.randn(1, 65, 60, 80)
softmax_temp = 1.0  # 假设 softmax_temp 为1.0

# 传统 softmax 计算方法
start_time = time.time()
scores_traditional = F.softmax(kpts * softmax_temp, dim=1)[:, :64]
end_time = time.time()
traditional_time = end_time - start_time

# 自定义 softmax 计算方法（改进版）
def custom_softmax(kpts):
    # 首先计算每个元素的最大值，用来稳定计算
    max_x = torch.max(kpts, dim=1, keepdim=True)[0]
    # 应用软最大方法，减去最大值，避免溢出
    exp_x = torch.exp(kpts - max_x)
    # 归一化
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x / sum_exp_x

start_time = time.time()
scores_custom = custom_softmax(kpts)[:, :64]
end_time = time.time()
custom_time = end_time - start_time

# 打印出计算时间
print(f"传统 Softmax 计算时间: {traditional_time:.6f}秒")
print(f"自定义 Softmax 计算时间: {custom_time:.6f}秒")

# 计算精度差异
difference = torch.abs(scores_traditional - scores_custom).mean().item()
print(f"两者之间的精度差异: {difference:.6e}")
