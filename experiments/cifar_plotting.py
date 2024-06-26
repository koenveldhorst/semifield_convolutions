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

data = load_data('cifar_classification.json')
data_parabolic = load_data('cifar_classification_parabolic.json')

setup_latex()

plt.figure(figsize=(10, 5))

models = ["Standard MaxPool", "Semifield Pool with Parabolic SE"]
metrics = ["Avg. Accuracy", "Avg. Precision", "Avg. Recall", "Avg. F1"]

avg_accuracy = [np.mean(data['accuracy'])]
std_accuracy = [np.std(data['accuracy'])]

avg_precision = [np.mean(data['avg precision']['macro'])]
std_precision = [np.std(data['avg precision']['macro'])]

avg_recall = [np.mean(data['avg recall']['macro'])]
std_recall = [np.std(data['avg recall']['macro'])]

avg_f1 = [np.mean(data['avg f1']['macro'])]
std_f1 = [np.std(data['avg f1']['macro'])]

avg_accuracy.append(np.mean(data_parabolic['accuracy']))
std_accuracy.append(np.std(data_parabolic['accuracy']))

avg_precision.append(np.mean(data_parabolic['avg precision']['macro']))
std_precision.append(np.std(data_parabolic['avg precision']['macro']))

avg_recall.append(np.mean(data_parabolic['avg recall']['macro']))
std_recall.append(np.std(data_parabolic['avg recall']['macro']))

avg_f1.append(np.mean(data_parabolic['avg f1']['macro']))
std_f1.append(np.std(data_parabolic['avg f1']['macro']))

x = np.arange(len(models))
width = 0.15

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
error_kw = dict(lw=1.5, capsize=3, capthick=1, ecolor='black')

bars = [
    plt.bar(x - 1.5*width, avg_accuracy, width, yerr=std_accuracy, label=metrics[0], color=colors[0], error_kw=error_kw, alpha=.99),
    plt.bar(x - width/2, avg_precision, width, yerr=std_precision, label=metrics[1], color=colors[1], error_kw=error_kw, alpha=.99),
    plt.bar(x + width/2, avg_recall, width, yerr=std_recall, label=metrics[2], color=colors[2], error_kw=error_kw, alpha=.99),
    plt.bar(x + 1.5*width, avg_f1, width, yerr=std_f1, label=metrics[3], color=colors[3], error_kw=error_kw, alpha=.99)
]

ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

ax.set_ylabel('Score', fontsize=14)
ax.set_ylim([0.78, 0.85])

plt.grid(axis='y', linestyle='--', alpha=0.99)
plt.legend(fontsize=12, loc='upper left')
plt.title('CIFAR-10 classification metrics using different pooling methods', fontsize=16)
plt.tight_layout()
plt.savefig('cifar_classification.pdf', format="pdf", bbox_inches="tight")
plt.show()
