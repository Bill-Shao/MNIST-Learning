import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

sizes = [784, 128, 64, 10]
model = nn.Sequential(nn.Linear(sizes[0], sizes[1]), nn.ReLU(),nn.Linear(sizes[1], sizes[2]),nn.ReLU(), nn.Linear(sizes[2], sizes[3]), nn.LogSoftmax(dim=1)).to(device)

criterion = nn.NLLLoss().to(device)
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

images = images.to(device)
labels = labels.to(device)

logps = model(images) #log
loss = criterion(logps, labels) #loss


loss.backward()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        
        #bp
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Loss: {}".format(e, running_loss/len(trainloader)))
        print((time()-time0))
print("\nTraining Time (in minutes) =",(time()-time0)/60)





correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img.to(device))

    
    ps = torch.exp(logps)
    cpudevice = torch.device("cpu")
    probab = list(ps.to(cpudevice).numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))