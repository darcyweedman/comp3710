import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# For pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Default
# Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
# The butt of mandlebrot
Y, X = np.mgrid[-0.7:0.7:0.001, -0.1:1.3:0.001]

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)
zs = z.clone()
ns = torch.zeros_like(z, dtype=torch.float32)

# Transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# Mandelbrot Set
start_time = time.time()
max_iterations = 1000  # Increase iterations for more detail

for i in range(max_iterations):
    # Compute the new values of z: z^2 + x
    zs_ = zs*zs + z
    # Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    # Update variables to compute
    ns += not_diverged.float()
    zs = torch.where(not_diverged, zs_, zs)

    if (i+1) % 100 == 0:
        print(f"Completed {i+1} iterations")

end_time = time.time()
print(f"Computation time: {end_time - start_time:.2f} seconds")

# Plot
plt.figure(figsize=(20,20))

def processFractal(a: torch.Tensor) -> torch.Tensor:
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()), cmap='hot', interpolation='nearest')
plt.title("High-Resolution Mandelbrot Set")
plt.tight_layout(pad=0)
plt.savefig("high_res_mandelbrot.png", dpi=1000)