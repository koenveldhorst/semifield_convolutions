import torch.nn.functional as F
import torch.nn as nn
import torch


def create_kernel(kernel_size, scale):
    z_i = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size, dtype=torch.float32)
    z = z_i ** 2
    w = -z / (4 * scale)
    return w.unsqueeze(0).unsqueeze(0)


class SemiConvModel(nn.Module):
    def __init__(self, semifield, input_channels, output_channels, kernel_size, s=1.0):
        super(SemiConvModel, self).__init__()
        self.semifield = semifield

        self.kernel_size = kernel_size
        self.scale = nn.Parameter(torch.tensor(s))  # Initialize scale parameter

        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x):
        kernel = create_kernel(self.kernel_size, self.scale)
        kernel = kernel.repeat(self.output_channels, self.input_channels, 1, 1)

        out = semi_conv(x, kernel, self.semifield)
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
        unfolded = F.unfold(padded, kernel_size=(M,N)).unsqueeze(1) # [B, 1, C_in * M * N, H_out * W_out]

        w_flat = w.view(1, C_out, -1, 1) # [1, C_out, C_in * M * N, 1]

        multiplied = weighting(unfolded, w_flat) # [B, C_out, C_in * M * N, H_out * W_out])
        result = aggregation(multiplied, dim=2) # [B, C_out, H_out * W_out]

        return result.view(B, C_out, H, W)


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values

semifield = (maxvalues, torch.add, -1 * torch.inf, 0)
# semifield = (torch.sum, torch.mul, 0, 1)

input_tensor = torch.tensor([2*x for x in range(5)], dtype=torch.float32).view(1, 1, 1, 5)
print(f'Input: {input_tensor}')
target_tensor = SemiConvModel(semifield=semifield, input_channels=1, output_channels=1, kernel_size=3, s=1.0)(input_tensor).clone().detach()
print(f'Target: {target_tensor}')

model = SemiConvModel(semifield=semifield, input_channels=1, output_channels=1, kernel_size=3, s=0.9)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Scale: {model.scale.item()}')

# Test the learned scale parameter
print(f'Learned Scale Parameter: {model.scale.item()}')
