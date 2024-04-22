import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


def batch_morph_op(batch_f, w, algebra):
    """Batch Morphological Operator."""
    aggregation, weighting, neutral_aggregation, neutral_weighting = algebra
    B, C, M, N = batch_f.shape
    H, W = w.shape

    wf = w.flatten().view(-1, 1)

    padded = F.pad(batch_f, (W//2, W//2, H//2, H//2), mode='constant', value=neutral_aggregation)
    unfolded = F.unfold(padded, (H, W))

    return aggregation(weighting(unfolded, wf), dim=1).reshape(B, C, M, N)


def show_images(input_images, output_images):
    """Show images."""
    n_images = min(len(input_images), 5)
    _, axes = plt.subplots(2, n_images, figsize=(n_images * 4, 6))
    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(input_images[i][0].detach().numpy(), cmap='gray')
            ax.set_title('Input {}'.format(i+1))
            ax.axis('off')
        else:
            ax.imshow(output_images[i-n_images][0].detach().numpy(), cmap='gray')
            ax.set_title('Output {}'.format(i-n_images+1))
            ax.axis('off')

    # plt.savefig('trui_d_erosion_gray.png')
    plt.show()


def process_cifar_batch(w, algebra):
    dataset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    images, _ = next(iter(dataloader))

    dilated_images = batch_morph_op(images.float(), w, algebra)

    return images, dilated_images


def process_trui_image(w, algebra):
    trui_image = Image.open('trui.png').convert('L')
    trui_image = transform(trui_image).unsqueeze(0).float()

    dilated_image = trui_image.clone()

    for _ in range(10):
        dilated_image = batch_morph_op(dilated_image, w, algebra)

    return trui_image, dilated_image


def maxvalues(a, dim=1):
    return torch.max(a, dim=dim).values


def minvalues(a, dim=1):
    return torch.min(a, dim=dim).values


if __name__ == "__main__":
    kernel = torch.tensor([[-2, -1, -2], [-1, 0, -1], [-2, -1, -2]], dtype=torch.float32)
    kernel = kernel / 255.0

    dilation = (maxvalues, torch.add, -1.0 * torch.inf, 0)
    erosion = (minvalues, torch.add, 1.0 * torch.inf, 0)

    original, dilated = process_trui_image(kernel, dilation)

    show_images(original, dilated)