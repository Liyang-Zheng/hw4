#!/usr/bin/env python
# coding: utf-8

# # **24-788: Intro to Deep Learning by Prof. Amir Barati Farimani**

# ## **Spring 2024 | HW 4: CIFAR-10 Classification via CNNs**
# 
# For this assignment, you will follow the below steps:
# 
# ```
# 1. First load and normalize the CIFAR-10 dataset.
# 2. Next, define your CNN model, loss function and optimizer.
# 3. Finally, once you have defined everything correctly, you can begin training your model.
# 4. For evaluation, you will need to test the model on test data and report your test accuracy.
# 5. Plot the model train and validation: loss and accuracy curves
# ```
# 
# Please referto the tutorial from PyTorch's official documentation: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html to get familiar with PyTorch, especially the data loading part. If you are new to PyTorch, it is recommended to follow the structure in this tutorial to build your model, loss function and training function. For this HW, $20$ out of $25$ points are graded on your code and report. And remaining $05$ points on the performance of your model. Tentative cutoffs for test accuracy:
# 1. $>$ $90\%$: $5$ points ($+10$ bonus points)
# 2. $>$ $85\%$: $5$ points ($+5$ bonus points)
# 3. $>$ $80\%$: $5$ points
# 4. $>$ $75\%$: $3$ points
# 5. $≤$ $75\%$: $0$ points
# 
# **NOTE:**
# 1. The below starter notebook follows the official PyTorch Documentation. You are free to make any changes to the sample starter notebook provided below.
# 2. You are recommended to use GPU. If you are training on GPU, please transfer your data and model to GPU Device: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu)

# In[1]:


# Import required libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt


# In[2]:


# Use the below code to get your Final Test Accuracy. DO NOT EDIT (except for changing device if needed)
def print_final_accuracy(model, testloader):
    total = 0
    correct = 0
    # we need to send the data to the same device as the data, so determine the
    # model's device
    device = next(model.parameters()).device
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    assert total == 10000, "Incorrect testloader used. There should be 10,000 test images"
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


# In[3]:


import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = AlexNet()


# In[7]:


from torch.optim import lr_scheduler
from google.colab import drive

def main():
    # TODO: load and transform dataset
    train_transform   =  transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 100
    trainset    =  torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset     =  torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    testloader  =  torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # TODO: Define your optimizer and criterion.
    lr = 0.001
    weight_decay = 1e-4

    device = torch.device("cuda")
    model = AlexNet()
    model = model.to(device)

    criterion =  nn.CrossEntropyLoss()  # TODO
    optimizer =  optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)  # TODO
    num_epoch =  30 # TODO
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # TODO: store loss over epochs for plot
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("beginning training!")
    for epoch in range(num_epoch):
        running_loss = 0.0
        correct_train = 0.0
        total_train = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(trainloader))

        # Calculate train accuracy
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Calculate validation loss and accuracy
        correct_val = 0.0
        total_val = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(valloader))
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Print statistics
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.2f}%')

        scheduler.step()

    # Plot the loss vs epoch
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot the accuracy vs epoch
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print final accuracy
    model.eval()
    print_final_accuracy(model, testloader)

    #save model
    torch.save(model.state_dict(), ' hw4_model.pkl')

    # Print out the hyperparameters
    # Dicuss details of how you found the hyperparameters (what experiments you did?) with a brief explanation
    # Include it a markdown cell at the end of the assignment
    print(f"hyperparameters: learning rate = {lr}, epochs = {num_epoch}, batch size = {batch_size}, optimizer: Adam, Loss Function: CrossEntropyLoss")


if __name__ == "__main__":
    main()


# When setting hyperparameters, I initially set the learning rate to 0.01, batch size to 20, and epoch to 20, and used CNN as the network model with Adam as the optimization function. After training, the results did not converge, so I increased the epoch to 60. Upon retraining, I observed significant fluctuations in the loss and accuracy values in the later stages of training. Consequently, I gradually increased the learning rate and batch size to 0.005 and 100, respectively. Additionally, I applied a scheduler to adjust the learning rate, with specific parameters set to step_size=10 and gamma=0.8. After retraining, slight overfitting occurred, prompting the introduction of weight decay and adjustment of the batch size to 40. Subsequently, I made modifications to the network structure. However, due to achieving a relatively low test accuracy of approximately 72%, I opted for AlexNet as the network architecture and conducted another round of training. To address overfitting and significant result fluctuations, I adjusted the batch size to 30 and the learning rate to 0.001, ultimately achieving an accuracy of 85.61%.
