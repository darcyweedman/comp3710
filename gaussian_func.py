import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X)
y = torch.Tensor(Y)

x = x.to(device)
y = y.to(device)

# Compute Gaussian
z = (torch.sin(x) * torch.sin(y)) * (torch.exp(-(x**2+y**2)/2.0))

plt.imshow(z.cpu().numpy())#Updated!
plt.tight_layout()
plt.savefig("2d_gaussian_multiplied")