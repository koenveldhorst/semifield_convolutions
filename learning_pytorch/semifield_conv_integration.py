import torch.nn.functional as F
import torch.nn as nn
import torch


def semi_conv_v1(batch_f, w, semifield):
    """
    semifield addition over feature maps in the input batch_f
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B, C_in, H, W = batch_f.size()
    C_out, C_in, M, N = w.size()

    if M % 2 != 0:
        padded = F.pad(batch_f, (N // 2, N // 2, M // 2, M // 2), value=aggregation_id)
    else:
        padded = F.pad(batch_f, (N // 2 - 1, N // 2, M // 2 - 1, M // 2), value=aggregation_id)

    unfolded = F.unfold(padded, kernel_size=(M,N)).unsqueeze(1) # [B, 1, C_in * M * N, H_out * W_out]

    w_flat = w.view(1, C_out, -1, 1) # [1, C_out, C_in * M * N, 1]

    multiplied = weighting(unfolded, w_flat) # [B, C_out, C_in * M * N, H_out * W_out])
    result = aggregation(multiplied, dim=2) # [B, C_out, H_out * W_out]

    return result.view(B, C_out, H, W)


def semi_conv_v2(batch_f, w, semifield):
    """
    standard addition over feature maps in the input batch_f
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
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

    result = torch.sum(added, dim=2)

    return result.view(B, C_out, H, W)


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


def test_semi_conv_v1(im, k):
    semifield = (torch.sum, torch.mul, 0, 1)
    result = semi_conv_v1(im, k, semifield)

    print((F.conv2d(im, k, padding=1) - result).abs().max())

    semifield = (maxvalues, torch.add, -1.0 * torch.inf, 0)
    result = semi_conv_v1(im, k, semifield)
    print(result)


def test_semi_conv_v2(im, k):
    semifield = (torch.sum, torch.mul, 0, 1)
    result = semi_conv_v2(im, k, semifield)

    print((F.conv2d(im, k, padding=1) - result).abs().max())

    semifield = (maxvalues, torch.add, -1.0 * torch.inf, 0)
    result = semi_conv_v2(im, k, semifield)
    print(result)


if __name__ == "__main__":
    image = torch.randint(0, 10, (1, 3, 4, 4)).float()
    image = image.repeat(2, 1, 1, 1)
    kernel = torch.randint(0, 10, (3, 3, 3, 3)).float()
    kernel = kernel.repeat(1, 1, 1, 1)

    test_semi_conv_v1(image, kernel)
    test_semi_conv_v2(image, kernel)

