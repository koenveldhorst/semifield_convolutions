import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch


transform = transforms.Compose([
    transforms.ToTensor()
])


def block_morph_op(f, w, semi_field):
    aggregation, weighting, neutral_aggregation, neutral_weighting = semi_field
    Bf, Cf, M, N = f.shape
    Bw, Cw, H, W = w.shape

    


def load_cifar_batch():
    dataset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    images, _ = next(iter(dataloader))

    return images


def calculate_kernel(c_in, ks):
    w = torch.linspace(-ks // 2 + 1, ks // 2, ks, dtype=torch.float32)
    w = w.view(-1, 1) ** 2 + w.view(1, -1) ** 2
    w = torch.repeat_interleave(w.unsqueeze(0), c_in, dim=0)

    return -w / (4 * 0.8)


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


if __name__ == "__main__":
    images = load_cifar_batch()
    c_in = images.shape[1]
    kernel = calculate_kernel(c_in, 3).unsqueeze(0)

    print(images.shape, kernel.shape)

    dilation = (maxvalues, torch.add, -1.0 * torch.inf, 0)

    block_morph_op(images, kernel, dilation)