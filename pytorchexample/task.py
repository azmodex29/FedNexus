"""MNIST task helpers adapted from Flower's PyTorch quickstart.

Modified for this project to use CPU-only MNIST training, deterministic client
partitions, a small CNN, and centralized evaluation.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NUM_CLASSES = 10
RANDOM_SEED = 20261106


class Net(nn.Module):
    """Small CNN for 28x28, single-channel MNIST images."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load one deterministic, non-overlapping MNIST client partition."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id must be within the configured client count")

    full_trainset = MNIST(
        root=DATA_DIR, train=True, download=True, transform=pytorch_transforms
    )
    partition_sizes = [len(full_trainset) // num_partitions] * num_partitions
    partition_sizes[-1] += len(full_trainset) % num_partitions
    partitions = random_split(
        full_trainset,
        partition_sizes,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    client_partition = partitions[partition_id]
    validation_size = max(1, len(client_partition) // 5)
    train_size = len(client_partition) - validation_size
    trainset, valset = random_split(
        client_partition,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED + partition_id),
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED + partition_id),
    )
    testloader = DataLoader(valset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load the held-out MNIST test set for central global evaluation."""
    testset = MNIST(root=DATA_DIR, train=False, download=True, transform=pytorch_transforms)
    return DataLoader(testset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
