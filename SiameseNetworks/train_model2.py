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

from model import Classifier, _prune, _prune_freeze, _adj_ind_loss, _turn_off_adj, _turn_off_weights, _turn_off_multi_weights, _adj_spars_loss, _freeze_grads

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
parser.add_argument('--epochs2', type=int, default=0, help='number of epochs for subsequent tasks.')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 2e-3)')
parser.add_argument('--lr2', type=float, default=-1, help='second learning rate')
parser.add_argument('--lr_adj', type=float, default=-1, help='adj learning rate')
parser.add_argument('--decay', type=float, default=0, help='adj decay rate')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25, help='number of hidden units per layer (default: 25)')
parser.add_argument('--tasks', default=50, type=int, help='no. of tasks')
parser.add_argument('--hidden_size', default=64, type=int, help='hidden neurons')
parser.add_argument('--im_size', default=28, type=int, help='image dimensions')
parser.add_argument('--prune_para', default=0.999, type=float, help='sparsity percentage pruned')
parser.add_argument('--tol', default=0.13, type=float, help='sparsity loss tolerance')
parser.add_argument('--freeze', action='store_true', help='freeze params')
parser.add_argument('--turn_off_weights', action='store_true', help='turns off weights after subsequent tasks')
parser.add_argument('--prune_epoch', default=0, type=int, help='prune epoch diff')
parser.add_argument('--prune_freq', default=0, type=int, help='prune epoch schedule')
parser.add_argument('--prune_times', default=-1, type=int, help="number of times to prune for prune_freq")
parser.add_argument('--adj_ind', default=0, type=float, help='adjacency independency loss.')
parser.add_argument('--adj_spars', default=0, type=float, help='adj sparsity loss.')
args = parser.parse_args()
prune_times_global = args.prune_times

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
            x = self.transform(F.to_pil_image(x))

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


train_data_aug = Compose([
    Resize(size = args.im_size),
    RandomApply(
        [RandomAffine(degrees=(-10, 10), scale=(0.8, 1.2), translate=(0.05, 0.05))],
        p=0.5
    ),
    ToTensor(),
])

test_data_aug = Compose([
    Resize(size = args.im_size),
    ToTensor()
])


dataloaders = []
for i in range(len(all_xy)):

    xtrain, xtest, ytrain, ytest = train_test_split(all_xy[i][0], all_xy[i][1], test_size=0.2)        
    # xtrain, xtest = xtrain / xtrain.max(), xtest / xtest.max()
    
    xtrain = torch.Tensor(xtrain).float()
    xtest = torch.Tensor(xtest).float()
    ytrain = torch.Tensor(ytrain).long()
    ytest = torch.Tensor(ytest).long()
    
    xtrain.unsqueeze_(dim=1);xtest.unsqueeze_(dim=1);
    
    # train = torch.utils.data.TensorDataset(xtrain, ytrain)
    # test = torch.utils.data.TensorDataset(xtest, ytest)
    
    train = CustomTensorDataset(tensors=(xtrain, ytrain), transform=train_data_aug)
    test = CustomTensorDataset(tensors=(xtest, ytest), transform=test_data_aug)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)
    
    dataloaders.append([train_loader,test_loader])

## model and optimizer instantiations:
net = Classifier(image_size = args.im_size, output_shape=60, tasks=50, layer_size=args.hidden_size, bn_boole=True)
if gpu_boole:
    net = net.cuda()
# optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
if args.lr_adj == -1:
    args.lr_adj = args.lr
optimizer = torch.optim.Adam([
                {'params': (param for name, param in net.named_parameters() if 'adjx' not in name), 'lr':args.lr,'momentum':0},
                {'params': (param for name, param in net.named_parameters() if 'adjx' in name), 'lr':args.lr_adj,'momentum':0,'weight_decay':args.decay}
            ])

## train, test eval:
loss_metric = torch.nn.CrossEntropyLoss()

def dataset_eval(data_loader, verbose = 1, task = 0, round_=False):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in data_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
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
    
pruned = False

## Task Loop:
for j in range(len(dataloaders)):
    
    train_loader, test_loader = dataloaders[j][0], dataloaders[j][1]
    
    args.prune_times = prune_times_global
    
    for epoch in range(args.epochs):
        
        print("Task:",j,"- Epoch:",epoch)

        for i, (x,y) in enumerate(train_loader):
            
            if gpu_boole:
                x, y = x.cuda(), y.cuda()
                
            y = y.view(-1)
            
            optimizer.zero_grad()
            outputs = net(x,task=j)
            
            loss = loss_metric(outputs,y)
            if args.adj_ind > 0 and j > 0:
                loss -= args.adj_ind *_adj_ind_loss(net,j+1)
            if args.adj_spars > 0:
                loss += args.adj_spars * _adj_spars_loss(net, j, tol = args.tol, prune_para = args.prune_para)
            
            loss.backward()
            optimizer.step()
            
            del loss; del x; del y; del outputs;
        
        train_acc, train_loss = dataset_eval(train_loader, verbose = 0, task = j)
        test_acc, test_loss= dataset_eval(test_loader, verbose = 0, task = j)
        test_acc_true, test_loss_true= dataset_eval(test_loader, verbose = 0, task = j, round_=True)
        print("Train acc, Train loss", train_acc, train_loss)
        print("Test acc, Test loss", test_acc, test_loss)
        print("Test acc, Test loss (Rounded Adj)", test_acc_true, test_loss_true)
        print()

        if (epoch >= (args.epochs - args.prune_epoch) and args.epochs>args.prune_epoch):
            print("Pruning...")
            _prune(net,task=j,prune_para=args.prune_para)
            pruned=True
        elif args.prune_freq>0 and args.prune_times==-1:
            if (epoch%args.prune_freq==0 and epoch>0):                    
                print("Pruning...")
                _prune(net,task=j,prune_para=args.prune_para)
        elif args.prune_freq>0 and args.prune_times>0:
            if (epoch%args.prune_freq==0 and epoch>0):                    
                print("Pruning...")
                _prune(net,task=j,prune_para=args.prune_para)
                args.prune_times -= 1
         
            
    if j==0:
        if args.lr2 == -1:
            args.lr2 = args.lr
        if args.lr_adj == -1:
            args.lr_adj = args.lr
            
        optimizer = torch.optim.Adam([
                {'params': (param for name, param in net.named_parameters() if 'adjx' not in name), 'lr':args.lr2},
                {'params': (param for name, param in net.named_parameters() if 'adjx' in name), 'lr':args.lr_adj,'momentum':0,'weight_decay':args.decay}
            ])
        
        if args.epochs2 > 0:
            args.epochs=args.epochs2

    print()
    print('Turn off weights?',args.turn_off_weights)
    print()

    print(args.freeze)
    if args.freeze:
        cur_hooks = _freeze_grads(net, j+1)

    if j==0:
        if args.turn_off_weights:
            _turn_off_weights(net)            
        
    _turn_off_adj(net,j)
    _turn_off_multi_weights(net,j)
    
    if args.freeze:
        if j > 0:
            for h in cur_hooks: h.remove()
    
    # if args.freeze:
    #     print("Freezing...")
    #     _prune_freeze(net,task=j,prune_para=args.prune_para)
    
    print("--------------------------------")
    print("Test acc for all tasks:")
    total_test_acc = 0
    for j2 in range(j+1):
        print("Task:",j2)
        test_acc, test_loss = dataset_eval(dataloaders[j2][1], verbose = 0, task = j2)
        print("Test acc, Test loss:",test_acc, test_loss)

        test_acc, test_loss = dataset_eval(dataloaders[j2][1], verbose = 0, task = j2, round_=True)
        print("Test acc, Test loss: (Rounded Adj)",test_acc, test_loss)
        
        total_test_acc += test_acc
    
    total_test_acc /= j+1
    print("Total test acc:",total_test_acc)
    print("--------------------------------")
    print()

