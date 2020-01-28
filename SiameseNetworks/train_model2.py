import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

import torchvision
import torchvision.transforms as transforms

from model import Classifier, _prune

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import imageio
import glob
import os

gpu_boole = torch.cuda.is_available()

import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_true', help='use CUDA (default: True)')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit (default: 30)')
parser.add_argument('--lr', type=float, default=2e-3, help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25, help='number of hidden units per layer (default: 25)')
parser.add_argument('--tasks', default=50, type=int, help='no. of tasks')
parser.add_argument('--hidden_size', default=64, type=int, help='hidden neurons')
args = parser.parse_args()

## Getting Dataloaders for Omniglot:

def images_by_char(char_path):
    imgs = []
    for im_path in glob.glob(char_path+"/*.png"):
        im = imageio.imread(im_path)
        imgs.append(im)
    return np.array(imgs)
    
def xy_by_alph(alph_path):
    xs = []
    ys = []
    subdirs = [ name for name in os.listdir(alph_path) if os.path.isdir(os.path.join(alph_path, name)) ]
    for subdir in subdirs:
        subdir = alph_path+subdir
        ims = images_by_char(subdir)
        xs.append(ims)
        ys.append(ims.shape[0]*[int(subdir[-2:])])
    return np.vstack(xs), np.array(ys).reshape([-1,1])

def get_all_xy_by_alph(dataset_dir):
    xys_by_alph = []
    subdirs = [ name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name)) ]
    for subdir in subdirs:
        subdir = dataset_dir+subdir+'/'
        x, y = xy_by_alph(subdir)
        xys_by_alph.append([x,y])
        
    return xys_by_alph 

all_xy = get_all_xy_by_alph('./omniglot/python/images_background/') + get_all_xy_by_alph('./omniglot/python/images_evaluation/')

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

dataloaders = []
for i in range(len(all_xy)):

    xtrain, xtest, ytrain, ytest = train_test_split(all_xy[i][0], all_xy[i][1], test_size=0.2)        
    xtrain, xtest = xtrain / xtrain.max(), xtest / xtest.max()
    
    xtrain = torch.Tensor(xtrain).float()
    xtest = torch.Tensor(xtest).float()
    ytrain = torch.Tensor(ytrain).long()
    ytest = torch.Tensor(ytest).long()
    
    xtrain.unsqueeze_(dim=1);xtest.unsqueeze_(dim=1);
    
    # train = torch.utils.data.TensorDataset(xtrain, ytrain)
    # test = torch.utils.data.TensorDataset(xtest, ytest)
    
    train = CustomTensorDataset(tensors=(xtrain, ytrain), transform=None)
    test = CustomTensorDataset(tensors=(xtest, ytest), transform=None)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)
    
    dataloaders.append([])

## model and optimizer instantiations:
net = Classifier(image_size = 105, output_shape=60, tasks=50)
if gpu_boole:
    net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

## train, test eval:
loss_metric = torch.nn.CrossEntropyLoss()

def dataset_eval(data_loader, verbose = 1, task = 0):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in data_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        # images = images.view(-1, 28*28)
        labels = labels.view(-1).cpu()
        outputs = net(images, task = task).cpu()
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
    
## Task Loop:
for j in range(len(dataloaders)):
    
    train_loader, test_loader = dataloaders[j][0], dataloaders[j][1]
    
    for epoch in range(args.epochs):
        
        print("Task:",j,"- Epoch:",epoch)

        for i, (x,y) in enumerate(train_loader):
            
            if gpu_boole:
                x, y = x.cuda(), y.cuda()
                
            y = y.view(-1)
            
            optimizer.zero_grad()
            outputs = net(x,task=j)
            
            loss = loss_metric(outputs,y)
            loss.backward()
            optimizer.step()
            
            del loss; del x; del y; del outputs;
        
        
        train_acc, train_loss = dataset_eval(train_loader, verbose = 0, task = j)
        test_acc, test_loss= dataset_eval(test_loader, verbose = 0, task = j)
        print("Train acc, Train loss", train_acc, train_loss)
        print("Test acc, Test loss", test_acc, test_loss)
        print()
    
    print("Test acc for all tasks:")
    total_test_acc = 0
    for j2 in range(len(dataloaders)):
        print("Task:",j2)
        test_acc, test_loss = dataset_eval(dataloaders[j2][1], verbose = 0, task = j2)
        # test_loss, test_acc = dataset_eval(dataloaders[j2][1], verbose = 0, task = 1)
        # print("Train acc, Train loss", train_loss, train_acc)
        print("Test acc:",test_acc)
        print("Test loss:",test_loss)
        
        total_test_acc += test_acc
    
    total_test_acc /= len(dataloaders)
    print("Total test acc:",total_test_acc)


