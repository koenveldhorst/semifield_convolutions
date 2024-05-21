import torch.nn.functional as F
import torch.nn as nn
import torch

import math


class SemiConv2d(nn.Module):
    def __init__(self, semifield, input_channels, output_channels, kernel_size, device):
        super(SemiConv2d, self).__init__()

        self.semifield = semifield
        self.kernel_size = kernel_size

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.device = device

    def _semi_conv(self, batch_f, w, bias):
        """
        :param batch_f: [B, C_in, H, W]
        :param w: [B, C_in, M, N]
        :param bias: [C_out]
        :param semifield: (semifield addition, semifield multiplication,
                           addition identity, multiplication identity)

        :return: [B, C_out, H_out, W_out]
        """
        aggregation, weighting, aggregation_id, weighting_id = self.semifield
        B, C_in, H, W = batch_f.size()
        C_out, C_in, M, N = w.size()

        if M % 2 != 0:
            padded = F.pad(batch_f, (N // 2, N // 2, M // 2, M // 2), value=aggregation_id)
        else:
            padded = F.pad(batch_f, (N // 2 - 1, N // 2, M // 2 - 1, M // 2), value=aggregation_id)

        unfolded = F.unfold(padded, kernel_size=(M,N)) # [B, C_in * M * N, H_out * W_out]

        unfolded = unfolded.view(B, 1, C_in, M * N, H * W) # [B, 1, C_in, M * N, H_out * W_out]
        w_flat = w.view(1, C_out, C_in, M * N, 1) # [1, C_out, C_in, M * N, 1]

        multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
        added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

        result = torch.sum(added, dim=2) # [B, C_out, H_out * W_out]
        result = bias.view(1, -1, 1) + result

        return result.view(B, C_out, H, W)

    def forward(self, x):
        raise NotImplementedError("Subclasses must override forward.")


class SemiConv2dParabolic(SemiConv2d):
    def __init__(self, semifield, input_channels, output_channels, kernel_size, device, initial_scale=1.0):
        super(SemiConv2dParabolic, self).__init__(semifield, input_channels, output_channels, kernel_size, device)
        # Initialize scale parameters
        self.scales = nn.Parameter(torch.full((output_channels, input_channels), initial_scale))

        # Initialize bias parameters
        self.bias = nn.Parameter(torch.empty(output_channels, device=device))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _compute_kernel(self):
        z_i = torch.linspace(-self.kernel_size // 2 + 1, self.kernel_size // 2,
                             self.kernel_size, dtype=torch.float32, device=self.device)

        z = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        h = -z / (4 * self.scales.view(-1, 1, 1))  # Reshape scales for broadcasting
        kernels = h.view(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        return kernels

    def forward(self, x):
        kernels = self._compute_kernel()
        out = self._semi_conv(x, kernels, self.bias)
        return out


class SemiConv2dLearnable(SemiConv2d):
    def __init__(self, semifield, input_channels, output_channels, kernel_size, device):
        super(SemiConv2dLearnable, self).__init__(semifield, input_channels, output_channels, kernel_size, device)
        # Initialize weight parameters
        self.weights = nn.Parameter(torch.empty((output_channels, input_channels, kernel_size, kernel_size), device=device))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Initialize bias parameters
        self.bias = nn.Parameter(torch.empty(output_channels, device=device))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = self._semi_conv(x, self.weights, self.bias)
        return out
