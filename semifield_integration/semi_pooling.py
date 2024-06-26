import torch.nn.functional as F
import torch.nn as nn
import torch

import math


class SemiPool2d(nn.Module):
    def __init__(self, semifield, c_in, c_out, kernel_size, stride, device, padding='valid', ceil_mode=False):
        super(SemiPool2d, self).__init__()

        self.semifield = semifield
        self.input_channels = c_in
        self.output_channels = c_out
        self.ks = kernel_size
        self.s = stride
        self.device = device

        self.ceil_mode = ceil_mode

        if padding != 'valid' and padding != 'same':
            raise ValueError('Padding must be either "valid" or "same"')

        self.padding = padding

    def _semi_pool(self, batch_f, w):
        aggregation, weighting, aggregation_id, weighting_id = self.semifield
        B_f, C_in, H, W = batch_f.size()
        B_w, C_out, M, N = w.size()

        if self.padding == 'valid':
            H_out = ((H - M + (self.s - 1 if self.ceil_mode else 0)) // self.s) + 1
            W_out = ((W - N + (self.s - 1 if self.ceil_mode else 0)) // self.s) + 1
        elif self.padding == 'same':
            H_out = -(H // -self.s)
            W_out = -(W // -self.s)

        pad_h = max((H_out - 1) * self.s + M - H, 0)
        pad_w = max((W_out - 1) * self.s + N - W, 0)

        p_l = pad_w // 2
        p_r = pad_w - p_l
        p_t = pad_h // 2
        p_b = pad_h - p_t

        padded = F.pad(batch_f, (p_l, p_r, p_t, p_b), value=aggregation_id)
        unfolded = F.unfold(padded, kernel_size=(M, N), stride=self.s) # [B, C_in * M * N, H_out * W_out]
        unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

        w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

        multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
        added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

        result = added.view(B_f, C_out, H_out, W_out)

        return result


    def forward(self, x):
        raise NotImplementedError("Subclasses must override forward.")


class SemiPool2dParabolic(SemiPool2d):
    def __init__(self, semifield, c_in, c_out, kernel_size, stride, device, initial_scale=1.0, padding='valid', ceil_mode=False):
        super(SemiPool2dParabolic, self).__init__(semifield, c_in, c_out, kernel_size, stride, device, padding, ceil_mode)
        # Initialize scale parameters
        self.scales = nn.parameter.Parameter(torch.full((c_in,), initial_scale, device=device))

        # Initialize base kernel
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks,
                                     dtype=torch.float32, device=self.device)

        self.base_kernel = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        if self.semifield[0].__name__ == 'maxvalues':
            self.base_kernel = -self.base_kernel

    def _compute_kernel(self):
        h = self.base_kernel / (4 * torch.abs(self.scales.view(-1, 1, 1)))
        kernels = h.view(1, self.output_channels, self.ks, self.ks)
        return kernels

    def forward(self, x):
        kernels = self._compute_kernel()
        return self._semi_pool(x, kernels)


class SemiPool2dLearnable(SemiPool2d):
    def __init__(self, semifield, c_in, c_out, kernel_size, stride, device, padding='valid', ceil_mode=False):
        super(SemiPool2dLearnable, self).__init__(semifield, c_in, c_out, kernel_size, stride, device, padding, ceil_mode)
        # Initialize weight parameters
        self.weights = nn.Parameter(torch.empty(1, c_in, kernel_size, kernel_size, device=device))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x):
        return self._semi_pool(x, self.weights)
