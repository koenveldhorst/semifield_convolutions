import matplotlib.pyplot as plt
import numpy as np
import json


def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def setup_latex():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


def create_figure():
    fig, axs = plt.subplots(1, 2, figsize=(17, 9), sharex=True, sharey=True)
    plt.xticks(range(0, 11))
    return fig, axs


def plot_scales(axis, layer, data, legend=False):
    for key, value in data[f'{layer}'].items():
        min_s, max_s, mean_s = [float(key)], [float(key)], [float(key)]
        for scales in value:
            min_s.append(np.min(scales))
            max_s.append(np.max(scales))
            mean_s.append(np.mean(scales))

        x_axis = range(0, 11)

        if legend:
            if key == '0.5':
                axis.fill_between(x_axis, min_s, max_s, alpha=0.3, label="Area between min and max")
            else:
                axis.fill_between(x_axis, min_s, max_s, alpha=0.3)
            axis.plot(x_axis, mean_s, marker='o', label=f"Mean $s$, $s_0 = {key}$")
        else:
            axis.fill_between(x_axis, min_s, max_s, alpha=0.3)
            axis.plot(x_axis, mean_s, marker='o')

    axis.grid()


def add_labels_and_legend(fig):
    fig.text(0.5, 0.02, 'Epoch', ha='center', fontdict={'fontsize': 15})
    fig.text(-0.01, 0.49, "$s$", va='center', rotation='vertical', fontdict={'fontsize': 20})
    fig.legend(fontsize="12", loc ="upper left")
    fig.suptitle("Scales learned on different epochs with varying starting scales using the standard SE.", fontsize=22)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.07)


if __name__ == "__main__":
    filename = 'mnist_classification.json'
    data = load_data(filename)
    setup_latex()
    fig, axs = create_figure()

    axs[0].set_title('Pool 1')
    plot_scales(axs[0], "pool1.scales", data, True)

    axs[1].set_title('Pool 2')
    plot_scales(axs[1], "pool2.scales", data)

    add_labels_and_legend(fig)
    plt.savefig('mnist_classification.pdf', format="pdf", bbox_inches="tight")
    plt.show()
