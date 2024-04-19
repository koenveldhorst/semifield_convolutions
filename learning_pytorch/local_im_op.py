"""
Local Image Operators
"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


def local_im_op(f, w, algebra):
    """Local Image Operator."""
    aggregation, weighting, neutral_aggregation, neutral_weighting = algebra
    M, N = f.shape
    H, W = w.shape
    wf = w.flatten().reshape(-1, 1)
    unfold = nn.Unfold((H, W), padding=(H//2, W//2))
    fu = unfold(f.reshape(1, 1, M, N))

    return aggregation(weighting(fu, wf), dim=1).reshape(M, N)


def conv2d(
        f, w):
    """2d Convolution."""
    aggregation = torch.sum
    weighting = torch.mul
    neutral_aggregate = 0
    neutral_weighting = 1
    return local_im_op(f, w,
                       (aggregation, weighting,
                        neutral_aggregate, neutral_weighting))

def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


def dila2d(f, w):
    """2d Dilation."""
    aggregation = maxvalues
    weighting = torch.add
    neutral_aggregate = -1.0 * torch.inf
    neutral_weighting = 0
    return local_im_op(f, w,
                       (aggregation, weighting,
                        neutral_aggregate, neutral_weighting))

def eros2d(f, w):
    return -dila2d(-f, w)

def maxmul2d(f, w):
    aggregation = maxvalues
    weighting = torch.mul
    neutral_aggregate = -1.0 * torch.inf
    neutral_weighting = 1
    return local_im_op(f, w,
                       (aggregation, weighting,
                        neutral_aggregate, neutral_weighting))



if __name__=='__main__':
    import torchvision

    f = torchvision.io.read_image('trui.png',
                                  torchvision.io.ImageReadMode.GRAY)

    _, M, N = f.shape
    f = 1.0 * f.reshape(M, N)


    w = torch.tensor([[1, 2, 1], [2, 3, 2], [1, 2, 1]])
    w = w/torch.sum(w)
    print(w)
    q = torch.tensor([[-2, -1, -2], [-1, 0, -1], [-2, -1, -2]])

    gl = conv2d(f, w)
    for i in range(19):
        gl = conv2d(gl, w)

    plt.subplot(1, 4, 1)
    plt.imshow(f)
    plt.title('Original')
    plt.subplot(1, 4, 2)
    plt.imshow(gl.numpy())
    plt.gray()

    gd = dila2d(f, q)
    for i in range(9):
        gd = dila2d(gd, q)

    plt.subplot(1, 4, 3)
    plt.imshow(gd.numpy())
    plt.gray()

    ge = eros2d(f, q)
    for i in range(19):
        ge = eros2d(ge, q)

    plt.subplot(1,4,4)
    plt.imshow(ge)
    plt.gray()

    gmm = maxmul2d(f, w)
    for i in range(19):
        gmm = maxmul2d(gmm, w)

    # plt.subplot(1,4,4)
    # plt.imshow(gmm)

    plt.gray()
    plt.show()
