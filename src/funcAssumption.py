import numpy as np
import torch
import matplotlib.pyplot as plt


def assume_func():
    x = np.linspace(-6, 6, 1000)
    y = np.sin(x) - 0.1 * x ** 2

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 4),
        torch.nn.Sigmoid(),
        torch.nn.Linear(4, 1)
    )

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(30000):
        inputs = torch.from_numpy(x.reshape(1000, 1)).float()
        targets = torch.from_numpy(y.reshape(1000, 1)).float()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}/500 | Loss: {loss.item():.4f}')

    plt.figure()
    plt.scatter(x, y, label='data')
    plt.plot(x, outputs.data.numpy(), label='model')
    plt.legend()
    plt.show()
