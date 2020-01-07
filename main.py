import numpy as np
import torch as torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

#DISPLAY THINGS
#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
#figure = plt.figure()
#num_of_images = 60
#for index in range(1, num_of_images + 1):
   # plt.subplot(6, 10, index)
  #  plt.axis('off')
 #   plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
#plt.show()

#MODEL

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
prevtime = time0
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
        print(("Time for Epoch: {}".format(time()-prevtime)))
        prevtime = time()
print("\nTraining Time (in minutes) =",(time()-time0)/60)






images, labels = next(iter(valloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)

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

torch.save(model.state_dict(), 'model.pt')
