import torch
import torch.nn.functional as F
import numpy as np
import time

# Parameters
W = 8  # Number of bits for quantization
LUT_SIZE = 7  # x_q ≈ 6 for W=8
SCALE = (1 << W) - 1  # 255 for W=8
softmax_temp = 1.0  # Softmax temperature
kpts = torch.randn(1, 65, 60, 80)  # Example tensor, shape (1, 65, 60, 80)

# Build LUT
def build_LUT():
    LUT = np.zeros(LUT_SIZE, dtype=np.uint8)
    for i in range(LUT_SIZE):
        val = (1.0 / np.exp(i)) * SCALE
        LUT[i] = round(val)
    return LUT

# Using LUT-based softmax
def softmax_lut(kpts, LUT):
    # Find max value in the input tensor for numerical stability
    max_val = torch.max(kpts, dim=1, keepdim=True)[0]
    
    # Calculate the indices for LUT using the formula (max_val - kpts)
    indices = torch.round(max_val - kpts).long()

    # Apply LUT, making sure the index doesn't exceed LUT_SIZE-1
    indices = torch.clamp(indices, 0, LUT_SIZE - 1)
    
    # Get the corresponding LUT values and calculate the sum
    lut_values = torch.tensor(LUT, dtype=torch.float32)[indices]
    sum_lut = torch.sum(lut_values, dim=1, keepdim=True)

    # Normalize and return the result
    return lut_values / sum_lut

# Traditional softmax using exp
def softmax_traditional(kpts):
    # Flatten and apply softmax using exp
    max_val = torch.max(kpts, dim=1, keepdim=True)[0]
    exp_values = torch.exp(kpts - max_val)
    sum_exp = torch.sum(exp_values, dim=1, keepdim=True)
    return exp_values / sum_exp

# Build LUT table
LUT = build_LUT()

# Measure the time for both methods
start_time = time.time()
scores_traditional = softmax_traditional(kpts * softmax_temp)[:, :, :, :64]
end_time = time.time()
traditional_time = end_time - start_time

start_time = time.time()
scores_lut = softmax_lut(kpts * softmax_temp, LUT)[:, :, :, :64]
end_time = time.time()
lut_time = end_time - start_time

# Print times
print(f"Traditional Softmax Time: {traditional_time:.6f} seconds")
print(f"LUT Softmax Time: {lut_time:.6f} seconds")

# Calculate and print the error
error = torch.abs(scores_traditional - scores_lut).mean().item()
print(f"Average Absolute Error: {error:.6e}")
