from time import time

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch


def is_tensor_view(x, y):
    x_ptrs = set(e.untyped_storage().data_ptr() for e in x)
    y_ptrs = set(e.untyped_storage().data_ptr() for e in y)
    return x_ptrs.symmetric_difference(y_ptrs) == set()


def one_d_strided_torch(one_d_tensor, kernel_size=(1, 5), stride=1):
    B, C, H, W = one_d_tensor.shape
    output_shape = (1 + (H - kernel_size[0]),
                    1 + (W - kernel_size[1]) // stride,
                    kernel_size[0], kernel_size[1])
    s = one_d_tensor.stride()
    strides = (s[0], s[1], s[3]  * stride, s[3])

    a_unfold = one_d_tensor.as_strided(output_shape, strides)
    return a_unfold.transpose(2, 3)


def one_d_strided_np(one_d_array, kernel_size=(1, 5), stride=1):
    B, C, H, W = one_d_array.shape
    output_shape = (1 + (H - kernel_size[0]),
                    1 + (W - kernel_size[1]) // stride,
                    kernel_size[0], kernel_size[1])
    s = one_d_array.strides
    strides = (s[0], s[1], s[3] * stride, s[3])
    a_unfold = np.lib.stride_tricks.as_strided(one_d_array, output_shape, strides)
    return a_unfold


def two_d_strided_torch(two_d_tensor, kernel_size=(2, 2), stride=1):
    B, C, H, W = two_d_tensor.shape
    output_shape = (1 + (H - kernel_size[0]) // stride,
                    1 + (W - kernel_size[1]) // stride,
                    kernel_size[0], kernel_size[1])
    s = two_d_tensor.stride()
    strides = (s[2] * stride, s[3] * stride, s[2], s[3])

    a_unfold = two_d_tensor.as_strided(output_shape, strides)
    return a_unfold


def two_d_strided_np(two_d_array, kernel_size=(2, 2), stride=1):
    B, C, H, W = two_d_array.shape
    output_shape = (1 + (H - kernel_size[0]) // stride,
                    1 + (W - kernel_size[1]) // stride,
                    kernel_size[0], kernel_size[1])
    s = two_d_array.strides
    strides = (s[2] * stride, s[3] * stride, s[2], s[3])

    b_unfold = np.lib.stride_tricks.as_strided(two_d_array, output_shape, strides)
    return b_unfold


def compare_1d_implementations():
    one_dimensional_t = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]], dtype=torch.float32)
    one_dimensional_np = np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]])

    one_d_unfolded_torch = one_d_strided_torch(one_dimensional_t)
    one_d_unfolded_np = one_d_strided_np(one_dimensional_np)

    print("1d unfolded torch:\n", one_d_unfolded_torch, "\n")
    print("1d unfolded numpy:\n", one_d_unfolded_np, "\n")

    assert torch.allclose(one_d_unfolded_torch, torch.from_numpy(one_d_unfolded_np).float())

    is_tensor_view_one_d = is_tensor_view(one_d_unfolded_torch, one_dimensional_t)
    is_np_view_one_d = np.may_share_memory(one_d_unfolded_np, one_dimensional_np)
    print("1d strided tensor is view:", is_tensor_view_one_d)
    print("1d strided numpy is view:", is_np_view_one_d)


def compare_2d_implementations():
    two_dimensional_t = torch.tensor([[[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]]], dtype=torch.float32)
    two_dimensional_np = np.array([[[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]]])

    two_d_unfolded_torch = two_d_strided_torch(two_dimensional_t)
    two_d_unfolded_np = two_d_strided_np(two_dimensional_np)

    print("2d unfolded torch:\n", two_d_unfolded_torch, "\n")
    print("2d unfolded numpy:\n", two_d_unfolded_np, "\n")

    assert torch.allclose(two_d_unfolded_torch, torch.from_numpy(two_d_unfolded_np).float())

    is_view_two_d = is_tensor_view(two_d_unfolded_torch, two_dimensional_t)
    is_np_view_two_d = np.may_share_memory(two_d_unfolded_np, two_dimensional_np)
    print("2d strided tensor is view:", is_view_two_d)
    print("2d strided numpy is view:", is_np_view_two_d)


if __name__ == "__main__":
    # compare_1d_implementations()
    compare_2d_implementations()
