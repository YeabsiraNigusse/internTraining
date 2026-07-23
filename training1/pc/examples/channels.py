import numpy as np

# ==========================================
# 1. CREATING CHANNELS
# ==========================================
print("--- 1. CREATING CHANNELS ---")

# Spatial base setup (Height=3, Width=4)
height, width = 3, 4

# 1 Channel: Grayscale (Height, Width)
channel_1 = np.ones((height, width)) * 10
# creates 3 by 4 matrix with all valued 10
print(f"1 Channel Shape: {channel_1.shape}\n{channel_1}\n")

# 2 Channels: e.g., Optical Flow (Channels, Height, Width)
ch_x = np.ones((height, width)) * 1
ch_y = np.ones((height, width)) * 2
channel_2 = np.stack([ch_x, ch_y], axis=0) 
print(f"2 Channels Shape: {channel_2.shape}\n{channel_2}\n")

# 3 Channels: RGB Image (Channels, Height, Width)
r_channel = np.ones((height, width)) * 255  # Red
g_channel = np.zeros((height, width)) * 255      # Green
b_channel = np.zeros((height, width))       # Blue
channel_3 = np.stack([r_channel, g_channel, b_channel], axis=0)
print(f"3 Channels Shape: {channel_3.shape}\n {channel_3}")


# ==========================================
# 2. SLICING ACROSS DIMENSIONS
# ==========================================
print("--- 2. SLICING ACROSS DIMENSIONS ---")

# Create a sample 3D dataset with shape (3 channels, 4 rows, 4 columns)
# Each channel will hold values corresponding to its channel index for clarity
data = np.zeros((3, 4, 4), dtype=int)
data[0, :, :] = 10  # Channel 0 filled with 10s
data[1, :, :] = 20  # Channel 1 filled with 20s
data[2, :, :] = 30  # Channel 2 filled with 30s

print("Original 3D Data Shape:", data.shape)

# Step A: Slice a single channel entirely (Extracting a 2D matrix from 3D)
first_channel = data[0, :, :]
print("\n[Slice A] Entire First Channel (Shape: {}):".format(first_channel.shape))
print(first_channel)

# Step B: Slice a specific patch across ALL channels
# Extracts a 2x2 spatial patch from rows 1-2 and columns 1-2 for every channel
spatial_patch = data[:, 1:3, 1:3]
print("\n[Slice B] 2x2 Patch Across All Channels (Shape: {}):".format(spatial_patch.shape))
print(spatial_patch)

# Step C: Subsampling / Strided slicing
# Takes every 2nd row and every 2nd column from channel 2
strided_slice = data[2, ::2, ::2]
print("\n[Slice C] Strided Slice from Channel 2 (Shape: {}):".format(strided_slice.shape))
print(strided_slice)


