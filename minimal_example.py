"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm
import time

from modules.xfeat_v1 import XFeat as XFeat_v1
from modules.xfeat import XFeat as c

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat_v1 = XFeat_v1()
xfeat_ori = XFeat_v1()

#Random input
x = torch.randn(1,3,480,640)
gap = 0

#Simple inference with batch = 1
for i in range(0,1000):
	x = torch.randn(1,3,480,640)
	start = time.time()
	output_v1 = xfeat_v1.detectAndComputeDense(x, top_k = 4096)
	end1 = time.time()
	output_ori = xfeat_ori.detectAndComputeDense(x, top_k = 4096)
	end2 = time.time()
	gap_once = (end2 - end1) - (end1 - start)
	print("NO:",i,"gap time:",gap_once)
	gap += gap_once

print("ori - v1 time:", gap)
# print("----------------")
# print("keypoints: ", output['keypoints'].shape)
# print("descriptors: ", output['descriptors'].shape)
# print("scores: ", output['scales'].shape)
# print("----------------\n")

# x = torch.randn(1,3,480,640)
# output = xfeat.detectAndComputeDense(x, top_k = 4096)

# # Stress test
# for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
# 	output = xfeat.detectAndCompute(x, top_k = 4096)

# # Batched mode
# x = torch.randn(4,3,480,640)
# outputs = xfeat.detectAndCompute(x, top_k = 4096)
# print("# detected features on each batch item:", [len(o['keypoints']) for o in outputs])

# # Match two images with sparse features
# x1 = torch.randn(1,3,480,640)
# x2 = torch.randn(1,3,480,640)
# mkpts_0, mkpts_1 = xfeat.match_xfeat(x1, x2)

# # Match two images with semi-dense approach -- batched mode with batch size 4
# x1 = torch.randn(1,3,480,640)
# x2 = torch.randn(1,3,480,640)
# t1 = time.perf_counter()
# matches_list = xfeat.match_xfeat_star(x1, x2)
# t2 = time.perf_counter()
# print("total time: ", t2-t1)
# print(matches_list[0].shape)
