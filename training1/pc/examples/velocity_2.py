import torch

# Define dummy dimensions
B, H, W = 2, 3, 4

# 1. Create Tensor 'a' with shape [B, 1, H, W]
# Filled with random attention weights between 0 and 1
a = torch.rand(B, 1, H, W)

# 2. Create Tensor 'u' with shape [B, 2, H, W]
# Filled with sequential numbers to easily track slicing
u = torch.arange(B * 2 * H * W, dtype=torch.float32).reshape(B, 2, H, W)

print(f"Tensor 'a' shape: {a.shape}")
print(f"Tensor 'u' shape: {u.shape}\n")

# ==========================================
# SLICING THE 4D TENSORS
# ==========================================

# Example A: Extract only the 1st channel (X-vector component) from 'u'
# Syntax: [all batches, channel 0, all heights, all widths]
ux = u[:, 0:1, :, :] 
print("Shape of ux (u[:, 0:1, :, :]):", ux.shape) # Keeps 4D shape: [B, 1, H, W]

# Example B: Extract a 2x2 spatial patch from the center of all channels/batches
# Extracts rows 1 to 3 (exclusive) and columns 1 to 3 (exclusive)
patch = u[:, :, 1:3, 1:3]
print("Shape of center patch:", patch.shape) # Shape: [B, 2, 2, 2]

# Example C: Broadened interaction (Multiplying a and u)
# Because 'a' has 1 channel, it automatically duplicates (broadcasts) 
# itself to multiply across both channels of 'u'.
result = a * u
print("Shape after element-wise multiplication (a * u):", result.shape) # Shape: [B, 2, H, W]


# Key Takeaway for 4D Slicing
    
#     When slicing u[:, 0:1, :, :]

# The first : keeps all batches (B).
# 0:1 targets the first channel but forces it to stay 4D ([B, 1, H, W]), meaning it matches the exact shape structure of a.
# The remaining two : keep the full spatial image height (H) and width (W)



