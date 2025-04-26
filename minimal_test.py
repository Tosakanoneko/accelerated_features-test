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

from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU
DRIVER = '---------------'

xfeat = XFeat()

#Random input
x = torch.randn(1,3,480,640)

#Simple inference with batch = 1
output = xfeat.detectAndCompute1(x, top_k = 4096)[0]
num_keypoints = output['keypoints'].shape[0]
avg_score = output['scores'].mean().item() if num_keypoints > 0 else 0.0
descriptors_shape = output['descriptors'].shape
print(DRIVER+'1'+DRIVER)
print("keypoints: ", num_keypoints)
print("avg_score: ", avg_score)
print("descriptors_shape: ", descriptors_shape)
print(DRIVER+DRIVER)

output = xfeat.detectAndCompute2(x, top_k = 4096)[0]
num_keypoints = output['keypoints'].shape[0]
avg_score = output['scores'].mean().item() if num_keypoints > 0 else 0.0
descriptors_shape = output['descriptors'].shape
print(DRIVER+'2'+DRIVER)
print("keypoints: ", num_keypoints)
print("avg_score: ", avg_score)
print("descriptors_shape: ", descriptors_shape)
print(DRIVER+DRIVER)

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
