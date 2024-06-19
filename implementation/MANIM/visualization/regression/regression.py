import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LIM = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the DNN model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Create the model instance
model = DNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=0)


# Example z value grid: 15x15 grid (original)
z_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 3, 1, 0, 1, 5, 8, 5, 1, 0, 1, 3, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# Create input (x, y) coordinates for the grid
grid_size = len(z_grid)
x_values = np.linspace(-LIM, LIM, grid_size)
y_values = np.linspace(-LIM, LIM, grid_size)
xx, yy = np.meshgrid(x_values, y_values)
xy_grid = np.c_[xx.ravel(), yy.ravel()]

# Flatten z_grid to match input shape
z_values = np.array(z_grid).flatten()

# Create a finer grid with higher resolution
x_fine = np.linspace(-LIM, LIM, 500)  # 100 points from -10 to 10
y_fine = np.linspace(-LIM, LIM, 500)  # 100 points from -10 to 10
xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)

# Interpolate the z values on the finer grid
z_fine = griddata((xx.ravel(), yy.ravel()), z_values, (xx_fine, yy_fine), method='cubic')

# Convert to tensors
inputs_train = torch.tensor(np.c_[xx_fine.ravel(), yy_fine.ravel()], dtype=torch.float32)
z_train = torch.tensor(z_fine.ravel(), dtype=torch.float32).view(-1, 1)

# Training loop
num_epochs = 10000
losses = []
model = model.to(device)
inputs_train = inputs_train.to(device)
z_train = z_train.to(device)
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(inputs_train)
    loss = criterion(outputs, z_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step the scheduler
    scheduler.step()
    
    losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'dnn_model.pth')
print("Model saved as 'dnn_model.pth'")

# Plot loss over epochs
plt.figure()
plt.plot(losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

model = model.cpu()
inputs_train = inputs_train.cpu()

# Test the model with an example input
model.eval()
test_input = torch.tensor([[3.0, 3.0]])
predicted_output = model(test_input)
print(f'Input: {test_input.tolist()}, Predicted Output: {predicted_output.item()}')

# Visualize the predictions on the grid
with torch.no_grad():
    z_pred = model(inputs_train).cpu().numpy().reshape(xx_fine.shape)

# Create the figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(xx_fine, yy_fine, z_pred, cmap='viridis')

# Add a color bar which maps values to colors
fig.colorbar(surf, ax=ax, label='z value')

# Add labels and title
ax.set_title('3D Visualization of Predicted Z Grid')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Value')

# Show plot
plt.show()
