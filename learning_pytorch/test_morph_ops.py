import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def semi_pool(batch_f, w, s, semifield, padding='valid', ceil_mode=False):
    """
    Version 5 of semi_pool: More flexible padding and ceil mode
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param stride: int
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :param padding: 'valid' or 'same'
    :param ceil_mode: bool
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B_f, C_in, H, W = batch_f.size()
    B_w, C_out, M, N = w.size()

    if padding == 'valid':
        H_out = ((H - M + (s - 1 if ceil_mode else 0)) // s) + 1
        W_out = ((W - N + (s - 1 if ceil_mode else 0)) // s) + 1
    elif padding == 'same':
        H_out = -(H // -s)
        W_out = -(W // -s)
    else:
        raise ValueError(f"Padding mode {padding} not supported")

    pad_h = max((H_out - 1) * s + M - H, 0)
    pad_w = max((W_out - 1) * s + N - W, 0)

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = F.pad(batch_f, (pad_left, pad_right, pad_top, pad_bottom), value=aggregation_id)
    unfolded = F.unfold(padded, kernel_size=(M, N), stride=s) # [B, C_in * M * N, H_out * W_out]
    unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

    w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = added.view(B_f, C_out, H_out, W_out)

    return result


def calculate_1d_kernel(kernel_size, scales, mode='default'):
    z_i =  torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size,
                          dtype=torch.float64)

    z = z_i ** 2
    h = z / (4 * scales)
    kernels = h.view(1, 1, 1, kernel_size)
    return kernels


def plot_2d(domain, input_tensor, output_tensor1, output_tensor2, output_tensor3):
    plt.plot(domain, input_tensor[0, 0, 0].cpu().numpy(), label='Input')
    plt.plot(domain, output_tensor1[0, 0, 0].cpu().numpy(), label='Dilated Output')
    plt.plot(domain, output_tensor2[0, 0, 0].cpu().numpy(), label='Eroded Output')
    plt.plot(domain, output_tensor3[0, 0, 0].cpu().numpy(), label='Opened Output')
    plt.legend()
    plt.title('Input and Output of Semi-Pooling. Kernel Size = 50, Scale = 100')
    plt.savefig('temp_plot.png')
    plt.show()


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


def minvalues(a, dim=1):
    return torch.min(a, dim=dim).values


def avgvalues(a, dim=1):
    return torch.mean(a, dim=dim)


def main():
    # kernel
    scale = torch.tensor(100, dtype=torch.float32, device='cpu')
    kernel = calculate_1d_kernel(50, scale)
    # input tensor
    domain_x = torch.linspace(-100, 100, 201)
    input_tensor = torch.tensor([torch.sin(0.1 * x) + torch.cos(0.05 * x) for x in domain_x], dtype=torch.float32)
    input_tensor = input_tensor.view(1, 1, 1, 201)
    # semi_pool
    max_semifield = (maxvalues, torch.add, -1 * torch.inf, 0)
    min_semifield = (minvalues, torch.add, torch.inf, 0)
    dilated_tensor = semi_pool(input_tensor, -kernel, 1, max_semifield, padding='same')
    eroded_tensor = semi_pool(input_tensor, kernel, 1, min_semifield, padding='same')
    opened_tensor = semi_pool(eroded_tensor, -kernel, 1, max_semifield, padding='same')
    # plot
    plot_2d(domain_x, input_tensor, dilated_tensor, eroded_tensor, opened_tensor)


if __name__ == "__main__":
    main()