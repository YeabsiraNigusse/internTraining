import numpy as np

# Create a sample 3x3 matrix
u = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Extract columns using 2D slicing
ux = u[:, 0:1]
uy = u[:, 1:2]

print("Original Array u:\n", u)
print("Shape of u:", u.shape)

print("\nux = u[:, 0:1] (First column as 2D):\n", ux)
print("Shape of ux:", ux.shape)

print("\nuy = u[:, 1:2] (Second column as 2D):\n", uy)
print("Shape of uy:", uy.shape)
