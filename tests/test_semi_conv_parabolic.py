import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from semifield_integration.semi_convolutions import SemiConv2dParabolic

# define two layer model
class TwoLayerModel(nn.Module):
    def __init__(self, semifield, c_in, c_out, ks, device, initial_scale=1.0):
        super(TwoLayerModel, self).__init__()
        self.conv1 = SemiConv2dParabolic(semifield, c_in, c_out,
                                         ks, device, initial_scale)
        self.conv2 = SemiConv2dParabolic(semifield, c_out, c_out,
                                         ks, device, initial_scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# set seed for reproducibility
torch.manual_seed(1)

# function for dilation
def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


def plot_3d(input_tensor):
    x = np.linspace(0, 1, input_tensor.shape[2])
    y = np.linspace(0, 1, input_tensor.shape[3])
    xv, yv = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 5))  # Wider figure to accommodate all subplots
    for i in range(input_tensor.shape[1]):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot_surface(xv, yv, input_tensor[0, i].cpu().numpy(), cmap='viridis')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Amplitude')
        ax.set_title(f'Channel {i + 1}')

    plt.suptitle('3D Visualization of Waved Surface Across Channels')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
    plt.show()


def generate_linear_gradient(batch_size, c_in, image_size, device):
    gradient = torch.zeros(batch_size, c_in, image_size, image_size,
                           device=device)

    for b in range(batch_size):
        for c in range(c_in):
            slope = 2.0 / image_size * (1 + 0.5 * c + 0.1 * b)
            intercept = -1 + 0.5 * c + 0.1 * b
            x = torch.linspace(intercept, slope * (image_size - 1) + intercept,
                               steps=image_size)
            y = torch.linspace(-1, 1, steps=image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            gradient[b, c] = X

    return gradient


def generate_radial_gradient(batch_size, c_in, image_size, device):
    gradient = torch.zeros(batch_size, c_in, image_size, image_size)
    center_x, center_y = image_size // 2, image_size // 2

    for b in range(batch_size):
        for c in range(c_in):
            scale = 1 + 0.1 * c + 0.05 * b
            x = torch.linspace(-scale, scale, steps=image_size)
            y = torch.linspace(-scale, scale, steps=image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')

            radial_gradient = torch.sqrt((X - center_x * 0.01 * c)**2 +
                                         (Y - center_y * 0.01 * b)**2)
            gradient[b, c] = radial_gradient

    return gradient.to(device)


def generate_wave(batch_size, c_in, image_size, device):
    gradient = torch.zeros(batch_size, c_in, image_size, image_size,
                           device=device)

    for b in range(batch_size):
        for c in range(c_in):
            x = torch.linspace(-100, 100, steps=image_size)
            y = torch.linspace(-100, 100, steps=image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            sine_wave = torch.sin(0.1 * X + c) + torch.cos(0.05 * X + b) + \
                        torch.sin(0.1 * Y + b) + torch.cos(0.05 * Y + c)
            gradient[b, c] = sine_wave

    return gradient


def test_semi_conv_parabolic(semifield, c_in, c_out, ks, device):
    # Generate input tensor
    batch_size = 10
    image_size = 32
    input_tensor = generate_wave(batch_size, c_in, image_size, device)

    # Initialze the target tensor
    params = (semifield, c_in, c_out, ks, device)
    target_tensor = TwoLayerModel(*params).to(device)(input_tensor).detach()

    # Initialize the model, optimizer and criterion
    model = TwoLayerModel(*params, initial_scale=3.0).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Check gradients and parameters before training
    print("Gradients and scales before training:")
    for name, param in model.named_parameters():
        print(f"{name}")
        print(f"Values: {param.data}\n")

    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()  # gradients are computed here
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Check gradients and parameters after training
    print("\nGradients and scales after training:")
    for name, param in model.named_parameters():
        print(f"{name}")
        print(f"Values: {param.data}")
        print(f"Gradient: {param.grad.data}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # field = (maxvalues, torch.add, -1 * torch.inf, 0)
    semifield = (torch.sum, torch.mul, 0, 1)
    input_channels = 3
    output_channels = 3
    kernel_size = 5

    test_semi_conv_parabolic(semifield, input_channels, output_channels,
                             kernel_size, device)


if __name__ == "__main__":
    main()
