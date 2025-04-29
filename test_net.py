import torch
import statistics
from modules.xfeat_v1 import XFeatModel as XFeatModel_v1, ds_time
from modules.xfeat   import XFeatModel as XFeatModel_ori, basic_time

def benchmark_model(model1, 
                    model2,
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
    model1.eval().to(device)
    model2.eval().to(device)
    b, c, h, w = input_size

    # 1) 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            x = torch.randn(b, c, h, w, device=device)
            model1(x)
            model2(x)

    # 2) 正式测试
    times1 = []
    times2 = []
    with torch.no_grad():
        for _ in range(num_tests):
            x = torch.randn(b, c, h, w, device=device)
            if device.type == "cuda":
                start_evt1 = torch.cuda.Event(enable_timing=True)
                end_evt1   = torch.cuda.Event(enable_timing=True)
                start_evt1.record()
                model1(x)
                end_evt1.record()
                # 等待 GPU 完成
                torch.cuda.synchronize()
                elapsed_ms1 = start_evt1.elapsed_time(end_evt1)

                start_evt2 = torch.cuda.Event(enable_timing=True)
                end_evt2   = torch.cuda.Event(enable_timing=True)
                start_evt2.record()
                model2(x)
                end_evt1.record()
                # 等待 GPU 完成
                torch.cuda.synchronize()
                elapsed_ms2 = start_evt2.elapsed_time(end_evt2)
            else:
                import time
                start_time = time.perf_counter()
                model1(x)
                elapsed_ms1 = (time.perf_counter() - start_time) * 1000

                start_time = time.perf_counter()
                model2(x)
                elapsed_ms2 = (time.perf_counter() - start_time) * 1000
            times1.append(elapsed_ms1)
            times2.append(elapsed_ms2)


    avg_ms1    = sum(times1) / len(times1)
    median_ms1 = statistics.median(times1)
    avg_ms2    = sum(times2) / len(times2)
    median_ms2 = statistics.median(times2)
    return avg_ms1, median_ms1, avg_ms2, median_ms2

if __name__ == "__main__":
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs       = 100      # 正式测试次数
    warmup     = 100       # 预热次数
    discard    = 0       # 丢弃前 N 次
    input_size = (1,1,244,244)

    print("runs:",runs)
    # 测试 XFeatModel_ori（传统卷积）
    net_ori = XFeatModel_ori()
    net_v1 = XFeatModel_v1()
    avg_ori, med_ori, avg_v1, med_v1 = benchmark_model(
        net_ori, net_v1, input_size, warmup, runs, discard, device
    )
    print(f"XFeatModel_ori —— 平均: {avg_ori:7.3f} ms， 中位数: {med_ori:7.3f} ms")
    print(f"XFeatModel_v1 —— 平均: {avg_v1:7.3f} ms， 中位数: {med_v1:7.3f} ms")

    print(f"basic_time:{sum(basic_time)}")
    print(f"ds_time   :{sum(ds_time)}")

    
    # 对比
    print(f"平均时间差 (ori - v1): {avg_ori - avg_v1:7.3f} ms")
    print(f"中位数差     (ori - v1): {med_ori- med_v1:7.3f} ms")

    print(f"平均时间加快 : {(avg_ori - avg_v1)/avg_ori*100:7.3f} %")
    print(f"中位数差加快 : {(med_ori- med_v1)/med_ori*100:7.3f} %")
