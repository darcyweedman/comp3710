import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to perform one iteration of the Koch curve transformation
def koch_iteration(points, device):
    new_points = []
    for i in range(len(points) - 1):
        start = torch.tensor(points[i], device=device)
        end = torch.tensor(points[i + 1], device=device)
        diff = (end - start) / 3.0

        # Compute the four new points for this segment
        p0 = start
        p1 = start + diff
        p2 = p1 + torch.tensor([diff[0] * 0.5 - diff[1] * np.sqrt(3) / 2,
                                diff[0] * np.sqrt(3) / 2 + diff[1] * 0.5], device=device)
        p3 = start + 2.0 * diff

        new_points.extend([p0.cpu().numpy(), p1.cpu().numpy(), p2.cpu().numpy(), p3.cpu().numpy()])
        
    new_points.append(points[-1]) # Add the last point
    return new_points

# Function to generate the Koch snowflake fractal
def koch_snowflake(iterations, device):
    # Initial equilateral triangle
    points = np.array([[0, 0],
                       [1, 0],
                       [0.5, np.sqrt(3) / 2],
                       [0, 0]], dtype='float64')

    # Apply the Koch curve transformation iteratively
    for _ in range(iterations):
        points = koch_iteration(points, device)

    return np.array(points)


# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute the Koch snowflake fractal
iterations = 6
points = koch_snowflake(iterations, device)

# Plot the fractal
plt.plot(points[:, 0], points[:, 1], 'b-')
plt.axis('equal')
plt.axis('off')
plt.savefig('koch_snowflake.png', dpi=300, bbox_inches='tight')
