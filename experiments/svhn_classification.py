import sys
sys.path.append('../')

from semifield_integration.semi_pooling import SemiPool2dParabolic
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def maxvalues(x, dim=-1):
    return torch.max(x, dim=dim)[0]


SEMIFIELD = (maxvalues, torch.add, -1 * torch.inf, 0)
PARABOLIC = False


# Credit: https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc
class svhnNet(nn.Module):
    def __init__(self, scale=1.0):
        super(svhnNet, self).__init__()
        if PARABOLIC:
            self.pool1 = SemiPool2dParabolic(SEMIFIELD, 32, 32, 5, 2, device, initial_scale=scale, padding='same')
            self.pool2 = SemiPool2dParabolic(SEMIFIELD, 64, 64, 5, 2, device, initial_scale=scale, padding='same')
            self.pool3 = SemiPool2dParabolic(SEMIFIELD, 128, 128, 5, 2, device, initial_scale=scale, padding='same')
        else:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.pool1,
            nn.Dropout2d(p=0.3),

            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.pool2,
            nn.Dropout2d(p=0.3),

            # Conv Layer block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.pool3,
            nn.Dropout2d(p=0.3),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


def import_data():
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.SVHN(
        root='SVHN',
        split='train',
        download=True,
        transform=transformation
    )

    test_dataset = datasets.SVHN(
        root='SVHN',
        split='test',
        download=True,
        transform=transformation
    )

    return train_dataset, test_dataset


def load_data(train_dataset, test_dataset, batch_size=32):
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


def train_model(model, loaders, optimizer, criterion, epochs=10):
    model.train()
    total_step = len(loaders['train'].dataset) // loaders['train'].batch_size

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
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

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds,
                                   output_dict=True)

    metrics = {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'macro_precision': report['macro avg']['precision'],
        'weighted_precision': report['weighted avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'weighted_recall': report['weighted avg']['recall']
    }

    return metrics



def main(train=True, save=False, parabolic=False):
    global PARABOLIC
    PARABOLIC = parabolic

    train_dataset, test_dataset = import_data()
    loaders = load_data(train_dataset, test_dataset)

    data = {
        'accuracy': [],
        'avg f1': {'macro': [], 'weighted': []},
        'avg precision': {'macro': [], 'weighted': []},
        'avg recall': {'macro': [], 'weighted': []}
    }

    runs = 5

    for i in range(runs):
        print(f'Run {i + 1}/{runs}')

        model = svhnNet(scale=1.0).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        if parabolic:
            model_name = f'svhn_models/svhn_model_parabolic_{i}.pt'
        else:
            model_name = f'svhn_models/svhn_model_{i}.pt'

        if train:
            model = train_model(model, loaders, optimizer, criterion, epochs=10)
            if save:
                torch.save(model.state_dict(), model_name)
        else:
            model.load_state_dict(torch.load(model_name))

        metrics = test_model(model, loaders)

        data['accuracy'].append(metrics['accuracy'])
        data['avg f1']['macro'].append(metrics['macro_f1'])
        data['avg f1']['weighted'].append(metrics['weighted_f1'])
        data['avg precision']['macro'].append(metrics['macro_precision'])
        data['avg precision']['weighted'].append(metrics['weighted_precision'])
        data['avg recall']['macro'].append(metrics['macro_recall'])
        data['avg recall']['weighted'].append(metrics['weighted_recall'])

    if parabolic:
        with open('svhn_classification_parabolic.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        with open('svhn_classification.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main(False, False, False)
    main(False, False, True)