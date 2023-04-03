import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

from src.test import CustomDataset

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
dataset = CustomDataset(
    root="/Users/sebastianjung/Documents/projects/AI/CHoiCe-Dataset/v0.3/data",
    transform=transform
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=61,
    shuffle=True,
)

model = nn.Sequential(
    nn.Linear(28, 16),
    nn.Sigmoid(),
    nn.Linear(16, 16),
    nn.Sigmoid(),
    nn.Linear(16, 61)
)


def train():
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.3, momentum=0.9)
    time0 = time()

    for epoch in range(3000):
        running_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels.view(1, -1))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(dataloader)))

    print("\nTraining Time (in minutes) =", (time()-time0)/60)


train()

