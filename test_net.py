import torch
import statistics
from modules.xfeat_v1 import XFeatModel as XFeatModel_v1
from modules.xfeat   import XFeatModel as XFeatModel_ori

def benchmark_model(model, 
                    input_size=(1,3,288,384),
                    num_warmup=10, 
                    num_tests=100, 
                    discard=10, 
                    device=None):
    """
    对 model 进行基准测试：
      1) 先用 num_warmup 次前向推理“预热”GPU
      2) 正式用 torch.cuda.Event 记录 num_tests 次运行时间（单位毫秒）
      3) 丢弃前 discard 次结果后，计算剩余的平均值和中位数
    返回 (avg_ms, median_ms)
    """
    model.eval().to(device)
    b, c, h, w = input_size

    # 1) 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            x = torch.randn(b, c, h, w, device=device)
            model(x)

    # 2) 正式测试
    times = []
    with torch.no_grad():
        for _ in range(num_tests):
            x = torch.randn(b, c, h, w, device=device)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            model(x)
            end_evt.record()
            # 等待 GPU 完成
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)
            times.append(elapsed_ms)

    # 3) 丢弃前 discard 次
    if 0 < discard < len(times):
        times = times[discard:]

    avg_ms    = sum(times) / len(times)
    median_ms = statistics.median(times)
    return avg_ms, median_ms

if __name__ == "__main__":
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs       = 10000      # 正式测试次数
    warmup     = 10       # 预热次数
    discard    = 10       # 丢弃前 N 次
    input_size = (1,3,288,384)

    print("runs:",runs)
    # 测试 XFeatModel_v1（深度可分离卷积）
    net_v1 = XFeatModel_v1()
    avg_v1, med_v1 = benchmark_model(
        net_v1, input_size, warmup, runs, discard, device
    )
    print(f"XFeatModel_v1 —— 平均: {avg_v1:7.3f} ms， 中位数: {med_v1:7.3f} ms")

    # 测试 XFeatModel_ori（传统卷积）
    net_ori = XFeatModel_ori()
    avg_ori, med_ori = benchmark_model(
        net_ori, input_size, warmup, runs, discard, device
    )
    print(f"XFeatModel_ori —— 平均: {avg_ori:7.3f} ms， 中位数: {med_ori:7.3f} ms")

    # 对比
    print(f"平均时间差 (ori - v1): {avg_ori - avg_v1:7.3f} ms")
    print(f"中位数差     (ori - v1): {med_ori- med_v1:7.3f} ms")

    print(f"平均时间加快 : {(avg_ori - avg_v1)/avg_ori*100:7.3f} %")
    print(f"中位数差加快 : {(med_ori- med_v1)/med_ori*100:7.3f} %")
    
