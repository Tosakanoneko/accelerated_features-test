import torch
import torch.nn as nn
import time
import numpy as np

def original_nms(x, threshold=0.05, kernel_size=5):
    """
    原始 NMS：利用 2D 最大池化找局部最大值。
    输入:
        x: Tensor，形状为 (B, 1, H, W)，每个元素为置信度
        threshold: 置信度阈值
        kernel_size: 池化窗口大小
    输出:
        pos_out: Tensor，形状为 (B, N, 2)，每个候选点的 (col, row) 坐标，
                 注意不同 batch 中候选点个数可能不同，程序中采用 pad 补齐
    """
    start_time = time.time()
    B, C, H, W = x.shape
    pad = kernel_size // 2
    maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    local_max = maxpool(x)
    # 找到局部最大且大于阈值的位置
    pos = (x == local_max) & (x > threshold)
    # pos: (B, 1, H, W)，nonzero 得到 (B, num, 4)，[batch, channel, row, col]
    # 这里我们只取 row 和 col 信息（注意：flip 顺序变为 (col, row)）
    pos_batched = [p.nonzero()[..., 2:].flip(-1) for p in pos]
    
    # 找出每个 batch 中候选点数目的最大值，便于后续 pad
    pad_val = max([p.shape[0] for p in pos_batched]) if pos_batched else 0
    pos_out = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)
    for b in range(B):
        if pos_batched[b].shape[0] > 0:
            pos_out[b, :pos_batched[b].shape[0], :] = pos_batched[b]
    elapsed = time.time() - start_time
    print("Original NMS 耗时: {:.6f}s".format(elapsed))
    return pos_out

def matrix_nms_point(x, threshold=0.05, kernel_size=5, sigma=0.1):
    """
    借鉴 Matrix NMS 思想的 NMS：将每个候选点视为一个固定大小的检测框，然后通过
    框与框之间的 IoU 计算衰减系数，对置信度进行软抑制，最终过滤低分候选点。
    
    输入:
        x: Tensor，形状为 (B, 1, H, W)
        threshold: 置信度阈值
        kernel_size: 检测框大小，与原始 NMS 中池化窗口对应，检测框中心为候选点，
                     框尺寸为 kernel_size×kernel_size（这里采用简单的对称扩展）
        sigma: 高斯核参数，用于计算衰减系数
    输出:
        result_list: list，每个元素为当前 batch 中保留的候选点坐标，形状 (N, 2)，
                     坐标格式为 (col, row)
    """
    start_time = time.time()
    B, _, H, W = x.shape
    pad = kernel_size // 2  # 用于构造检测框
    result_list = []
    for b in range(B):
        scores = x[b, 0]  # (H, W)
        pos = scores > threshold
        if pos.sum() == 0:
            result_list.append(torch.empty((0, 2), dtype=torch.long, device=x.device))
            continue
        # 获取候选点 (row, col) 坐标
        coords = pos.nonzero()  # shape: (N, 2)
        cand_scores = scores[pos]  # shape: (N,)
        
        # 构造每个候选点对应的检测框，格式 [x1, y1, x2, y2]
        # 注意：这里将 col 视为 x 坐标，row 视为 y 坐标
        boxes = torch.zeros((coords.shape[0], 4), dtype=torch.float, device=x.device)
        boxes[:, 0] = coords[:, 1].float() - pad  # x1
        boxes[:, 1] = coords[:, 0].float() - pad  # y1
        boxes[:, 2] = coords[:, 1].float() + pad  # x2
        boxes[:, 3] = coords[:, 0].float() + pad  # y2
        
        # 计算每个检测框的面积
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        
        # 按候选分数从高到低排序
        cand_scores, order = cand_scores.sort(descending=True)
        boxes = boxes[order]
        coords = coords[order]  # 对应的 (row, col)
        
        N = boxes.shape[0]
        if N == 0:
            result_list.append(torch.empty((0, 2), dtype=torch.long, device=x.device))
            continue
        
        # 计算候选检测框之间的 IoU
        boxes1 = boxes.unsqueeze(1).expand(N, N, 4)
        boxes2 = boxes.unsqueeze(0).expand(N, N, 4)
        inter_x1 = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
        inter_y1 = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
        inter_x2 = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
        inter_y2 = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])
        inter_w = (inter_x2 - inter_x1 + 1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1 + 1).clamp(min=0)
        inter_area = inter_w * inter_h
        union_area = areas.unsqueeze(1) + areas.unsqueeze(0) - inter_area
        iou_matrix = inter_area / union_area  # shape: (N, N)
        # 只取上三角部分，对角线以上的部分对应分数较低的候选点与分数较高的候选点的 IoU
        iou_matrix = iou_matrix.triu(diagonal=1)
        
        # 对每个候选点，找出与分数更高候选点（排序在前）的最大 IoU（补偿因子）
        compensate_iou, _ = iou_matrix.max(0)  # shape: (N,)
        # 为了后续计算方便，将其扩展到 (N, N)
        compensate_iou = compensate_iou.expand(N, N).transpose(0, 1)
        
        decay_iou = iou_matrix
        # 计算高斯衰减矩阵
        decay_matrix = torch.exp(-sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-sigma * (compensate_iou ** 2))
        # 对每个候选点，取所有对应比率的最小值作为衰减系数
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        
        # 更新候选点得分
        updated_scores = cand_scores * decay_coefficient
        
        # 根据更新后的得分进行过滤
        keep = updated_scores > threshold
        # 注意：这里为了与 original_nms 保持一致，输出坐标格式为 (col, row)
        kept_coords = coords[keep][:, [1, 0]]  # 将 (row, col) 转换为 (col, row)
        result_list.append(kept_coords)
    
    elapsed = time.time() - start_time
    print("Matrix NMS 耗时: {:.6f}s".format(elapsed))
    return result_list

if __name__ == '__main__':
    # 固定随机种子，方便对比
    torch.manual_seed(0)
    # 生成一个形状为 (B, 1, 64, 48) 的随机置信度矩阵，取值范围 [0, 1)
    B, C, H, W = 1, 1, 64, 48
    x = torch.rand(B, C, H, W)
    
    # 设置阈值和 kernel_size
    threshold = 0.05
    kernel_size = 5
    sigma = 0.5
    
    print("输入矩阵形状:", x.shape)
    
    # 运行原始 NMS
    pos_original = original_nms(x, threshold=threshold, kernel_size=kernel_size)
    print("Original NMS 检测到的候选点数量 (batch 0):", (pos_original[0].sum(dim=1) != 0).sum().item())
    # 输出候选点坐标（可能含有 pad 的零点）
    print("Original NMS 候选点 (batch 0):")
    print(pos_original[0][(pos_original[0].sum(dim=1) != 0)])
    
    # 运行 Matrix NMS（对每个 batch 分别处理，返回 list）
    pos_matrix_list = matrix_nms_point(x, threshold=threshold, kernel_size=kernel_size, sigma=sigma)
    pos_matrix = pos_matrix_list[0]
    print("Matrix NMS 检测到的候选点数量 (batch 0):", pos_matrix.shape[0])
    print("Matrix NMS 候选点 (batch 0):")
    print(pos_matrix)
    
    # 对比两种方法结果的差异
    # 注意：由于两种方法思想不同（一个是严格的局部极大值，一个是软抑制后阈值过滤），
    # 因此候选点的位置和数量可能会有一定差别。
    
    # 计算两种方法候选点坐标之间的欧氏距离均值（仅在候选点数目较多时有意义）
    if pos_matrix.shape[0] > 0 and (pos_original[0][(pos_original[0].sum(dim=1) != 0)].shape[0] > 0):
        # 为了简单起见，取原始 NMS 与 Matrix NMS 检测结果的交集个数
        set_original = set([tuple(p.tolist()) for p in pos_original[0][(pos_original[0].sum(dim=1) != 0)]])
        set_matrix = set([tuple(p.tolist()) for p in pos_matrix])
        common = set_original & set_matrix
        print("两种方法检测到的共同候选点数量:", len(common))
    else:
        print("候选点数量太少，无法比较共同候选点。")
    
    # 若需要进一步比较耗时和结果差异，可多次重复测试并统计平均耗时。
