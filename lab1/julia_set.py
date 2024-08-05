import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

# For pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For julia set
c = complex(-0.4, 0.6)

Y, X = np.mgrid[-1.5:1.5:0.0005, -1.5:1.5:0.0005]

# Load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)

# Transfer to the GPU device
z = z.to(device)
ns = torch.zeros_like(z, dtype=torch.float32).to(device)

start_time = time.time()
max_iterations = 600

for i in range(max_iterations):
    z = z**2 + c
    not_diverged = torch.abs(z) < 4.0
    ns += not_diverged.float()
    
    if (i+1) % 50 == 0:
        print(f"Completed {i+1} iterations")

end_time = time.time()
print(f"Computation time: {end_time - start_time:.2f} seconds")

# Colors to make it look like the one on wikipedia
colors = ['darkviolet', 'yellowgreen', 'cyan', 'yellowgreen', 'yellow']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Plot
plt.figure(figsize=(12,12))
plt.imshow(ns.cpu().numpy(), cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5])
plt.tight_layout()
plt.savefig("julia_set.png", dpi=500)
