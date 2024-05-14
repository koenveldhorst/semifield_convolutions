from time import time

import torch.nn.functional as F
import torch.nn as nn
import torch


def is_tensor_view(x, y):
    x_ptrs = set(e.untyped_storage().data_ptr() for e in x)
    y_ptrs = set(e.untyped_storage().data_ptr() for e in y)
    return x_ptrs.symmetric_difference(y_ptrs) == set()


def im2col(input_data, kernel):
    kernel_b, kernel_c, kernel_h, kernel_w = kernel.shape
    image_b, image_c, image_h, image_w = input_data.shape
    padding_h, padding_w = 1, 1

    height_col = 3
    width_col = 3
    channels_col = image_c * kernel_h * kernel_w

    unfolded = torch.zeros(1, channels_col, height_col,
                           width_col).view(kernel_h * kernel_w, -1)

    for c_col in range(channels_col):
        w_offset = c_col % kernel_w
        h_offset = (c_col // kernel_w) % kernel_h
        c_im = c_col // kernel_h // kernel_w

        for h_col in range(height_col):
            h_im = h_col * 1 - padding_h + h_offset * 1

            for w_col in range(width_col):
                w_im = w_col * 1 - padding_w + w_offset * 1

                if h_im >= 0 and w_im >= 0 and h_im < image_h and w_im < image_w:
                    unfolded[c_col, h_col * width_col + w_col] = input_data[0, c_im, h_im, w_im]
                else:
                    unfolded[c_col, h_col * width_col + w_col] = 0

    return unfolded


def f_unfold(input_data, kernel):
    return F.unfold(input_data, kernel.shape[3], padding=1)


def nn_unfold(input_data, kernel):
    return nn.Unfold(kernel.shape[2:], padding=1)(input_data)


def compare_implementations():
    data_im = torch.tensor(
        [[[1, 2], [3, 4]]], dtype=torch.float32).unsqueeze(0)
    kernel = torch.zeros(1, 1, 2, 2)

    unfolded1 = im2col(data_im, kernel)
    unfolded2 = f_unfold(data_im, kernel)
    unfolded3 = nn_unfold(data_im, kernel)

    assert torch.allclose(unfolded1, unfolded2)
    assert torch.allclose(unfolded1, unfolded3)
    assert torch.allclose(unfolded2, unfolded3)

    assert not is_tensor_view(unfolded1, data_im)
    assert not is_tensor_view(unfolded2, data_im)
    assert not is_tensor_view(unfolded3, data_im)

    print('All implementations are correct and not views of input data.')


def time_implementations():
    data_im = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32).unsqueeze(0)
    kernel = torch.zeros(1, 1, 2, 2)

    for implementation in [im2col, f_unfold, nn_unfold]:
        times = []
        for _ in range(1000):
            start = time()
            implementation(data_im, kernel)
            times.append(time() - start)

        print(f'\nImplementation: {implementation.__name__}')
        print(f'Mean time taken: {sum(times) / len(times)}')
        print(f'Standard deviation: {sum((t - sum(times) / len(times))**2 for t in times) / len(times)}')


if __name__ == "__main__":
    compare_implementations()
    time_implementations()
