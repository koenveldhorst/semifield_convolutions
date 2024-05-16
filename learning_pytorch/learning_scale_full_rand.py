import torch.nn.functional as F
import torch.nn as nn
import torch

# set seed for reproducibility
torch.manual_seed(0)


class SemiConvModel(nn.Module):
    def __init__(self, semifield, input_channels, output_channels, kernel_size, initial_scale=None):
        super(SemiConvModel, self).__init__()
        self.semifield = semifield
        self.kernel_size = kernel_size

          # Initialize scale parameters
        if initial_scale is not None:
            self.scales = nn.Parameter(torch.full((output_channels, input_channels), initial_scale))
        else:
            self.scales = nn.Parameter(torch.rand((output_channels, input_channels)))

        self.input_channels = input_channels
        self.output_channels = output_channels

    def _compute_kernel(self):
        z_i = torch.linspace(-self.kernel_size // 2 + 1, self.kernel_size // 2, self.kernel_size, dtype=torch.float32)
        z = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        h = -z / (4 * self.scales.view(-1, 1, 1))  # Reshape scales for broadcasting
        kernels = h.view(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        return kernels

    def forward(self, x):
        kernels = self._compute_kernel()

        out = semi_conv(x, kernels, self.semifield)
        return out


def semi_conv(batch_f, w, semifield):
    """
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B, C_in, H, W = batch_f.size()
    C_out, C_in, M, N = w.size()

    padded = F.pad(batch_f, (N // 2, N // 2, M // 2, M // 2), value=aggregation_id)
    unfolded = F.unfold(padded, kernel_size=(M,N)) # [B, C_in * M * N, H_out * W_out]

    unfolded = unfolded.view(B, 1, C_in, M * N, H * W) # [B, 1, C_in, M * N, H_out * W_out]
    w_flat = w.view(1, C_out, C_in, M * N, 1) # [1, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = torch.sum(added, dim=2)

    return result.view(B, C_out, H, W)


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


semifield = (maxvalues, torch.add, -1 * torch.inf, 0)
# semifield = (torch.sum, torch.mul, 0, 1)

c_output = 2
c_input = 3
ks = 3

input_tensor = torch.randint(0, 10, (2, 3, 4, 4)).float()

target_params = (semifield, c_input, c_output, ks)
target_tensor = SemiConvModel(*target_params, initial_scale=1.0)(input_tensor).clone().detach()

model = SemiConvModel(*target_params)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

print(f'Initial Scale Parameters: {model.scales.detach()}')
print(f'Target Scale Parameters: {torch.ones_like(model.scales)}')

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Mean Scale: {model.scales.mean().item()}, Mean Grad: {model.scales.grad.mean().item()}')

# Test the learned scale parameter
print(f'Learned Scale Parameters: {model.scales.detach()}')