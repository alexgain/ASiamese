import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomApply, Resize
import torchvision.transforms.functional as F

import sys
sys.path.append("..")

# from common_utils.imgaug import RandomAffine, RandomApply

from model import Classifier, ClassifierMLP, _prune, _prune_freeze, _adj_ind_loss, _turn_off_adj, _turn_off_weights, _turn_off_multi_weights, _adj_spars_loss, _freeze_grads

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import imageio
import glob
import os

import time

gpu_boole = torch.cuda.is_available()


train_data_aug = Compose([
    Resize(size = 28),
    RandomApply(
        [RandomAffine(degrees=(-10, 10), scale=(0.8, 1.2), translate=(0.05, 0.05))],
        p=0.5
    ),
    ToTensor(),
])

test_data_aug = Compose([
    Resize(size = 28),
    ToTensor()
])

training = torchvision.datasets.MNIST(root ='./data', transform = train_data_aug, train=True, download=True)
testing =  torchvision.datasets.MNIST(root ='./data', transform = test_data_aug, train=False, download=True)
train_loader = torch.utils.data.DataLoader(dataset=training, batch_size = 128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testing, batch_size = 128, shuffle=False)

permutations = torch.load('permutations.pt')

## model and optimizer instantiations:
net = torch.load('model_task_7.2pt')
if gpu_boole:
    net = net.cuda()

## train, test eval:
loss_metric = torch.nn.CrossEntropyLoss()

def dataset_eval(data_loader, verbose = 1, task = 0, round_=False, perm = -1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in data_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        
        if perm == -1:
            images = images.view(-1,28*28)[:,permutations[task]]
        else:
            images = images.view(-1,28*28)[:,permutations[perm]]
        # images = images.view(-1, 28*28)
        labels = labels.view(-1).cpu()
        outputs = net(images, task = task, round_=round_).cpu()
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum().cpu().data.numpy().item()

        loss_sum += loss_metric(outputs,labels).cpu().data.numpy().item()
        
        del images; del labels; del outputs; del _; del predicted;
    
    correct = np.float(correct)
    total = np.float(total)
    if verbose:
        print('Accuracy:',(100 * np.float(correct) / np.float(total)))
        print('Loss:', (loss_sum / np.float(total)))

    acc = 100.0 * (np.float(correct) / np.float(total))
    loss = (loss_sum / np.float(total))
    del total; del correct; del loss_sum
    return acc, loss
    

def dataset_eval_ens(data_loader, verbose = 1, task = 0, round_=False):
    correct = 0
    total = 0
    loss_sum = 0
    
    for images, labels in data_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        
        images = images.view(-1,28*28)[:,permutations[task]]
        # images = images.view(-1, 28*28)
        labels = labels.view(-1).cpu()
        outputs = []
        for k2 in range(8):
            outputs.append(net(images, task = k2, round_=round_).cpu())
        outputs = torch.stack(outputs).max(dim=0)[0]
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum().cpu().data.numpy().item()

        loss_sum += loss_metric(outputs,labels).cpu().data.numpy().item()
        
        del images; del labels; del outputs; del _; del predicted;
    
    correct = np.float(correct)
    total = np.float(total)
    if verbose:
        print('Accuracy:',(100 * np.float(correct) / np.float(total)))
        print('Loss:', (loss_sum / np.float(total)))

    acc = 100.0 * (np.float(correct) / np.float(total))
    loss = (loss_sum / np.float(total))
    del total; del correct; del loss_sum
    return acc, loss


print("--------------------------------")
print("Test acc for all tasks:")
total_test_acc = 0
for j2 in range(8):
    print("Task:",j2)
    test_acc, test_loss = dataset_eval(test_loader, verbose = 0, task = j2)
    print("Test acc, Test loss:",test_acc, test_loss)

    test_acc, test_loss = dataset_eval(test_loader, verbose = 0, task = j2, round_=True)
    print("Test acc, Test loss: (Rounded Adj)",test_acc, test_loss)
    
    total_test_acc += test_acc

total_test_acc /= 8
print("Total test acc:",total_test_acc)
print("--------------------------------")
print()
# print("Saving model...")
# torch.save(net,'model_task_%d'%j+'2.pt')


print("--------------------------------")
print("Test acc for all tasks (ens):")
total_test_acc = 0
for j2 in range(8):
    print("Task:",j2)
    test_acc, test_loss = dataset_eval_ens(test_loader, verbose = 0, task = j2)
    print("Test acc, Test loss:",test_acc, test_loss)

    test_acc, test_loss = dataset_eval_ens(test_loader, verbose = 0, task = j2, round_=True)
    print("Test acc, Test loss: (Rounded Adj)",test_acc, test_loss)
    
    total_test_acc += test_acc

total_test_acc /= 8
print("Total test acc:",total_test_acc)
print("--------------------------------")
print()
# print("Saving model...")
# torch.save(net,'model_task_%d'%j+'2.pt')


print("--------------------------------")
print("Test acc for all tasks (i,j):")
for j1 in range(8):
    for j2 in range(8):
        print("Task",j1,"tested on Adj",j2)
        test_acc, test_loss = dataset_eval_ens(test_loader, verbose = 0, task = j2, perm=j1)
        print("Test acc, Test loss:",test_acc, test_loss)
    
        test_acc, test_loss = dataset_eval_ens(test_loader, verbose = 0, task = j2, round_=True, perm=j1)
        print("Test acc, Test loss: (Rounded Adj)",test_acc, test_loss)
        
        total_test_acc += test_acc
    
    total_test_acc /= 8
    print("Total test acc:",total_test_acc)
    print("--------------------------------")
    print()



