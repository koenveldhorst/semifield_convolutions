import torch.nn.functional as F
import torch.nn as nn
import torch

import unittest


def is_tensor_view(x, y):
    x_ptrs = set(e.untyped_storage().data_ptr() for e in x)
    y_ptrs = set(e.untyped_storage().data_ptr() for e in y)
    return x_ptrs.symmetric_difference(y_ptrs) == set()


def semi_pool_v1(batch_f, w, s, semifield):
    """
    Version 1 of semi_pool: No padding
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param stride: int
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B_f, C_in, H, W = batch_f.size()
    B_w, C_out, M, N = w.size()
    H_out, W_out = (H - M) // s + 1, (W - N) // s + 1

    unfolded = F.unfold(batch_f, kernel_size=(M, N), stride=s) # [B, C_in * M * N, H_out * W_out]
    unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

    w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = added.view(B_f, C_out, H_out, W_out)
    return result


def semi_pool_v2(batch_f, w, s, semifield):
    """
    Version 2 of semi_pool: Ceil mode is used to pad on the right and bottom
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param stride: int
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B_f, C_in, H, W = batch_f.size()
    B_w, C_out, M, N = w.size()
    H_out, W_out = ((H - M + s - 1) // s) + 1, ((W - N + s - 1) // s) + 1
    pad_h, pad_w = (H_out - 1) * s + M - H, (W_out - 1) * s + N - W

    if (H - M) % s != 0:
        batch_f = F.pad(batch_f, (0, pad_w, 0, pad_h), value=aggregation_id)

    unfolded = F.unfold(batch_f, kernel_size=(M, N), stride=s) # [B, C_in * M * N, H_out * W_out]
    unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

    w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = added.view(B_f, C_out, H_out, W_out)
    return result


def semi_pool_v3(batch_f, w, s, semifield):
    """
    Version 3 of semi_pool: Padding is used on the right and bottom
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param stride: int
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B_f, C_in, H, W = batch_f.size()
    B_w, C_out, M, N = w.size()

    H_out, W_out = -(H // -s), -(W // -s)
    pad_h = (H_out - 1) * s + M - H
    pad_w = (W_out - 1) * s + N - W

    padded = F.pad(batch_f, (0, pad_w, 0, pad_h), value=aggregation_id)
    unfolded = F.unfold(padded, kernel_size=(M, N), stride=s) # [B, C_in * M * N, H_out * W_out]
    unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

    w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = added.view(B_f, C_out, H_out, W_out)

    return result


def semi_pool_v4(batch_f, w, s, semifield):
    """
    Version 4 of semi_pool: Padding is used and distributed evenly
    :param batch_f: [B, C_in, H, W]
    :param w: [B, C_in, M, N]
    :param stride: int
    :param semifield: (semifield addition, semifield multiplication,
                       addition identity, multiplication identity)
    :return: [B, C_out, H_out, W_out]
    """
    aggregation, weighting, aggregation_id, weighting_id = semifield
    B_f, C_in, H, W = batch_f.size()
    B_w, C_out, M, N = w.size()

    H_out, W_out = -(H // -s), -(W // -s)
    pad_h = max((H_out - 1) * s + M - H, 0)
    pad_w = max((W_out - 1) * s + N - W, 0)

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    padded = F.pad(batch_f, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=aggregation_id)
    unfolded = F.unfold(padded, kernel_size=(M, N), stride=s) # [B, C_in * M * N, H_out * W_out]
    unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

    w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = added.view(B_f, C_out, H_out, W_out)
    return result


def semi_pool_v5(batch_f, w, s, semifield, padding='valid', ceil_mode=False):
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

    padded = F.pad(batch_f, (0, pad_w, 0, pad_h), value=aggregation_id)
    unfolded = F.unfold(padded, kernel_size=(M, N), stride=s) # [B, C_in * M * N, H_out * W_out]
    unfolded = unfolded.view(B_f, 1, C_in, M * N, H_out * W_out) # [B, 1, C_in, M * N, H_out * W_out]

    w_flat = w.view(B_w, C_out, M * N, 1) # [B, C_out, C_in, M * N, 1]

    multiplied = weighting(unfolded, w_flat)  # [B, C_in, M * N, H_out * W_out]
    added = aggregation(multiplied, dim=3)  # [B, C_in, H_out * W_out]

    result = added.view(B_f, C_out, H_out, W_out)

    return result


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim, keepdim=True).values


class TestPooling(unittest.TestCase):
    def test_pooling_equality(self):
        semifield = (maxvalues, torch.add, -1 * torch.inf, 0)

        # Iterate over a range of dimensions, kernel sizes and strides
        for s in range(2, 5):
            for ks in range(3, 6):
                for size in range(5, 12):
                    image = torch.randint(0, 10, (1, 1, size, size)).float()
                    kernel = torch.zeros(1, 1, ks, ks).float()

                    # Custom pooling function
                    result_custom = semi_pool_v5(image, kernel, s, semifield, padding='valid', ceil_mode=False)

                    # PyTorch built-in pooling
                    result_pytorch = F.max_pool2d(image, kernel_size=ks, stride=s, ceil_mode=False)

                    # Check dimensions
                    self.assertEqual(result_custom.size(), result_pytorch.size(), f"Failed on size {size}, ks {ks}, s {s}")

                    # Check values
                    max_difference = (result_pytorch - result_custom).abs().max()
                    self.assertEqual(max_difference, 0, f"Max difference failed on size {size}, ks {ks}, s {s}")


if __name__ == "__main__":
    # unittest.main()

    semifield = (maxvalues, torch.add, -1 * torch.inf, 0)
    size = 7
    ks = 4
    s = 2

    image = torch.randint(0, 10, (1, 1, size, size)).float()
    kernel = torch.zeros(1, 1, ks, ks).float()

    result_custom = semi_pool_v5(image, kernel, s, semifield, padding='same', ceil_mode=True)
    result_pytorch = F.max_pool2d(image, kernel_size=ks, stride=s, ceil_mode=True)

    print(result_custom.size())
    print(result_pytorch.size())
