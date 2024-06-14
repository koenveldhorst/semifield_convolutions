import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')

from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from itertools import product

from semifield_integration.semi_pooling import SemiPool2dParabolic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class mnistNet(nn.Module):
    def __init__(self, in_f = 32*7*7):
        super(mnistNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                               stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=in_f, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out, x


def import_data():
    train_dataset = datasets.MNIST(
        root='MNIST',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.MNIST(
        root='MNIST',
        train=False,
        transform=transforms.ToTensor()
    )

    return train_dataset, test_dataset


def load_data(train_dataset, test_dataset, batch_size=100):
    loaders = {
        'train': DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1),
        'test': DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=1),
    }

    return loaders


def train_model(model, loaders, criterion, optimizer, epochs=5):
    model.train()
    total_step = len(loaders['train'])

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and apply gradients
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}],\
                      Step [{i + 1}/{total_step}],\
                      Loss: {loss.item()}')

    return model


def test_model(model, loaders):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds,
                                   target_names=loaders['test'].dataset.classes,
                                   output_dict=True)
    print(report)

    show_prediction(all_labels, all_preds, report)

    return report


def show_data(train_dataset):
    figure = plt.figure(figsize=(20, 3))
    cols, rows = 10, 1
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def show_prediction(true_y, pred_y, report):
    cm = confusion_matrix(true_y, pred_y)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix, average precision: {0:0.5f}'.format(
        report['weighted avg']['precision']))

    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(10)])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True number')
    plt.xlabel('Predicted number')
    plt.tight_layout()
    plt.show()


def main(first_time=True, save=False, parabolic=False):
    train_dataset, test_dataset = import_data()
    loaders = load_data(train_dataset, test_dataset)
    show_data(train_dataset)

    in_f = 32 * 7 * 7 if not parabolic else 32 * 4 * 4
    model = mnistNet(in_f=in_f).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if parabolic:
        def maxvalues(x, dim=-1):
            return torch.max(x, dim=dim)[0]

        semifield = (maxvalues, torch.add, -1 * torch.inf, 0)
        model_name = 'models/mnist_model_parabolic.pt'
        model.pool1 = SemiPool2dParabolic(semifield, 16, 16, 5, 2, device, initial_scale=5.0)
        model.pool2 = SemiPool2dParabolic(semifield, 32, 32, 5, 2, device, initial_scale=5.0)
    else:
        model_name = 'models/mnist_model.pt'

    if first_time:
        model = train_model(model, loaders, criterion, optimizer)
        if save:
            torch.save(model.state_dict(), model_name)
    else:
        model.load_state_dict(torch.load(model_name))

    for name, param in model.named_parameters():
        if parabolic and name == 'pool1.scales' or name == 'pool2.scales':
                print(name, param, param.grad)

    report = test_model(model, loaders)


if __name__ == '__main__':
    main(True, False, False)
