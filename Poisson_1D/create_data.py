import torch
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate x values in the domain [0, 1]
xmin, xmax = 0, 1
N = 1000  # Number of points
x = torch.linspace(xmin, xmax, steps=N).view(-1, 1).to(device)

# Compute the exact solution u(x)
u_x = torch.sin(np.pi * x)

# Optionally, compute the source term f(x) = pi^2 * sin(pi * x)
f_x = (np.pi**2) * torch.sin(np.pi * x)

# Prepare the data for testing and validation (here, using the same data for simplicity)
X_test = x
y_test = u_x
X_val = x
y_val = u_x  # In practice, you might have separate validation data

# Save the data to a file
torch.save((X_test, y_test, X_val, y_val), 'Poisson_test')
