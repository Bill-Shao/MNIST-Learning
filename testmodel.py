import numpy as np
import torch as torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_dataset():
    data_path = 'testdata/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader




sizes = [784, 128, 64, 10]
model = nn.Sequential(nn.Linear(sizes[0], sizes[1]), nn.ReLU(),nn.Linear(sizes[1], sizes[2]),nn.ReLU(), nn.Linear(sizes[2], sizes[3]), nn.LogSoftmax(dim=1)).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()


testloader = load_dataset()

images, labels = next(iter(testloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))