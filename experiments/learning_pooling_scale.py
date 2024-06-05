import sys
sys.path.append('../')

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json

from models.lenet_parabolic import LeNet_Parabolic

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
LR = 0.001
EPOCHS = 5
BATCH_SIZE = 32

transform = transforms.Compose(
    [transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = datasets.KMNIST(root='./KMNIST', train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root='./KMNIST', train=False, download=True, transform=transform)

# Train-validation split
n_samples = int(len(train_dataset))
train_dataset, _ = random_split(train_dataset, [n_samples, 0], generator=torch.Generator().manual_seed(42))

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

steps_per_epoch = len(train_loader.dataset) // BATCH_SIZE

print(train_dataset[0][0].shape)
IMG_SIZE = 28

startscales = [0.5]

def run_experiment(startscale):
    data = {}

    classes = len(train_dataset.dataset.classes)
    model = LeNet_Parabolic(n_channels=1, n_classes=classes, device=device, ks=7, fc_in=800, initial_scale=startscale).to(device)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    data[startscale] = {}
    data[startscale]['scales_pool1'] = []
    data[startscale]['scales_pool2'] = []

    print(f"Training with scale: {startscale}")

    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{steps_per_epoch}], Loss: {loss.item()}")

                for name, param in model.named_parameters():
                    print(f"{name}, gradient: {param.grad.mean().item() if param.grad is not None else None}")

                print(model.pool1.scales.tolist())
                print(model.pool2.scales.tolist())

        data[startscale]['scales_pool1'].append(model.pool1.scales.tolist())
        data[startscale]['scales_pool2'].append(model.pool2.scales.tolist())

    with torch.no_grad():
        model.eval()
        preds = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            preds.extend(output.argmax(axis=1).cpu().numpy())

        report = classification_report(test_dataset.targets, np.array(preds), target_names=test_dataset.classes, output_dict=True)
        data[startscale]['accuracy'] = report['accuracy']
        print(f"report: {report}")

    return data

results = {}
for startscale in startscales:
    results.update(run_experiment(startscale))

    print(f"Results: {results}")

with open(f'/results/lenet_parabolic_learnability_scale.json', 'w+') as f:
    json.dump(results, f)
