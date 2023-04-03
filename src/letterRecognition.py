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
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(30976, 128),
    nn.ReLU(),
    nn.Linear(128, 61)
)


def train():
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    time0 = time()

    for epoch in range(3000):
        running_loss = 0
        for images, labels in dataloader:
            #print("labels: -----------------------------------------------------")
            #print(labels)
            optimizer.zero_grad()

            output = model(images)
            #print("output: ------------------------------------------------------")
            #print(output)
            loss = criterion(output, labels.view(-1, 1, 1, 1).float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(dataloader)))

    print("\nTraining Time (in minutes) =", (time()-time0)/60)


train()

