import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(1)

class SemiPool1dParabolic(nn.Module):
    def __init__(self, semifield, c_in, c_out, kernel_size, stride, device, initial_scale=1.0, padding='valid', ceil_mode=False):
        super(SemiPool1dParabolic, self).__init__()

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
        self.scale = nn.Parameter(torch.full((1, ), initial_scale, device=device))

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

    def _compute_kernel(self):
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2,
                             self.ks, dtype=torch.float32, device=self.device)

        z = z_i ** 2
        if self.semifield[0].__name__ == 'maxvalues':
            z = -z
        h = z / (4 * torch.abs(self.scale.view(-1, 1, 1)))
        kernels = h.view(1, 1, 1, self.ks)
        return kernels

    def forward(self, x):
        kernels = self._compute_kernel()
        return self._semi_pool(x, kernels)


class TwoLayerModel(nn.Module):
    def __init__(self, c_in, c_out, ks, stride, device, initial_scale_min=1.0, initial_scale_max=1.0, padding='same', ceil_mode=False):
        super(TwoLayerModel, self).__init__()
        min_semifield = (minvalues, torch.add, torch.inf, 0)
        max_semifield = (maxvalues, torch.add, -1 * torch.inf, 0)
        self.minpool = SemiPool1dParabolic(min_semifield, c_in, c_out, ks, stride,
                                         device, initial_scale_min, padding, ceil_mode)
        self.maxpool = SemiPool1dParabolic(max_semifield, c_in, c_out, ks, stride,
                                         device, initial_scale_max, padding, ceil_mode)

    def forward(self, x):
        x = self.minpool(x)
        x = self.maxpool(x)
        return x


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


def minvalues(a, dim=1):
    return torch.min(a, dim=dim).values


def train_and_plot_2l(s, device):
    k_size = 201
    domain_x = torch.linspace(-10, 10, 201)
    f = torch.tensor([torch.sin(x) + torch.cos(0.5*x) for x in domain_x], dtype=torch.float32, device=device)

    t = 100.0
    g = TwoLayerModel(1, 1, k_size, 1, device, padding='same', initial_scale_min=t, initial_scale_max=t)(f.view(1, 1, 1, -1)).clone().detach()

    if s <= 100:
        n_iterations = 500
        iterations = {10, 150, 200, 250, 500}
    else:
        n_iterations = 500
        iterations = {10, 50, 100, 150, 500}

    print(f"Training with scale: {s}")
    model = TwoLayerModel(1, 1, k_size, 1, device, padding='same', initial_scale_min=s, initial_scale_max=s)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    criterion = nn.MSELoss()

    predictions = []

    start_time = time.time()
    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(f.view(1, 1, 1, -1))
        loss = criterion(output, g)
        loss.backward()

        optimizer.step()

        if i + 1 in iterations:
            predictions.append((output.clone().detach().cpu().numpy(), str(i + 1)))

    for func in output.grad_fn.next_functions:
        print(func)
    print(f"Training time: {time.time() - start_time}")
    print(f"Final loss: {loss.item()}")
    for name, param in model.named_parameters():
        print(f"{name}")
        print(f"Gradients: {param.grad}")
        print(f"Values: {param.data}\n")

    # Setup LaTeX
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

    # Plot
    plt.figure(figsize=(13, 7))
    plt.plot(domain_x, f, 'bo', markersize=3)
    plt.plot(domain_x, f, label= '$f$')
    plt.plot(domain_x, g[0,0,0], label= '$f \\boxminus q^{t0} \\boxplus q^{t1}$')

    for prediction in predictions:
        plt.plot(domain_x, prediction[0][0,0,0], label= '$f \\boxminus q^{s0} \\boxplus q^{s1}$ on i = ' + prediction[1])

    plt.legend()
    plt.title(f'Learning the scale of opening with $t = {t}$ and the starting scale being {s}.', fontdict={'fontsize': 15})
    plt.savefig(f"learning_scale_from_{s}_2l.pdf", format="pdf", bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_and_plot_2l(25.0, device)
    train_and_plot_2l(150.0, device)