import matplotlib.pyplot as plt
from itertools import product
from timeit import timeit
import numpy as np

import torch.nn as nn
import torch

F_H = 8
F_W = 8


def unf_pad(f, w):
    """Unfold and pad."""
    _, _, H, W = w.shape
    unfolded = nn.functional.unfold(f, (H, W), padding=(H//2, W//2))

    return unfolded


def pad_unf(f, w):
    """Pad and unfold."""
    _, _, H, W = w.shape
    f = nn.functional.pad(f, (W//2, W//2, H//2, H//2), mode='constant', value=0)
    unfolded = nn.functional.unfold(f, (H, W))

    return  unfolded


def calc_exec_time(f, w):
    """Plot execution time."""
    exec_time = {}
    funcs = [unf_pad, pad_unf]
    n_itters = [100, 1000, 10000]

    itterations = 1000
    for n, func in product(n_itters, funcs):
        exec_time[(func.__name__, n)] = [timeit(lambda: func(f, w), number=n)]
        for _ in range(itterations - 1):
            exec_time[(func.__name__, n)].append(timeit(lambda: func(f, w), number=n))

        print("Exec time for {} with n={} calculated.".format(func.__name__, n))

    # Calculate average execution time and standard deviation
    mean_std_exec_time = {}
    for key, value in exec_time.items():
        mean_std_exec_time[key] = (np.mean(value), np.std(value))

    return mean_std_exec_time, n_itters


def add_labels(bars):
    """Add value labels to bars."""
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (10**(i-4)),
                 f'{yval:.6f}', va='bottom', ha='center')


def plot_exec_time(f, w):
    """Plot execution time."""
    mean_std_exec_time, n_itters = calc_exec_time(f, w)
    unf_pad_means = [mean_std_exec_time[('unf_pad', n)][0] for n in n_itters]
    pad_unf_means = [mean_std_exec_time[('pad_unf', n)][0] for n in n_itters]
    unf_pad_stds = [mean_std_exec_time[('unf_pad', n)][1] for n in n_itters]
    pad_unf_stds = [mean_std_exec_time[('pad_unf', n)][1] for n in n_itters]

    plt.figure(figsize=(10, 6))
    width = 0.3
    indices = np.arange(len(n_itters))

    # Plot bars
    bar1 = plt.bar(indices - width/2, unf_pad_means, width, yerr=unf_pad_stds,
                   label='Unfold with Pad', capsize=5, color='blue')
    bar2 = plt.bar(indices + width/2, pad_unf_means, width, yerr=pad_unf_stds,
                   label='Pad then Unfold', capsize=5, color='green')

    # Add value labels
    add_labels(bar1)
    add_labels(bar2)

    # Label and annotate
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.title(f'Mean Execution Time Comparison for {F_H}x{F_W} Feature Map and 3x3 Filter')
    plt.xticks(indices, [str(x) for x in n_itters])
    plt.legend()

    # Show the plot
    plt.yscale('log')
    plt.tight_layout()
    # plt.savefig('padding_exec_time_comparison.png')
    plt.show()


if __name__ == "__main__":
    f = torch.arange(1, F_W**2 + 1, dtype=torch.float32).view(1, 1, F_H, F_W)
    w = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)

    first_order = unf_pad(f, w)
    second_order = pad_unf(f, w)
    print(torch.allclose(first_order, second_order))

    plot_exec_time(f, w)
