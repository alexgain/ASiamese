
import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, MaxPool2d, Sigmoid, ReLU
from torch.autograd import Variable


import torch.nn.functional as F
import torch.nn as nn
from layers import ALinear, AConv2d

class ATwoLayer(nn.Module):
    def __init__(self, input_size=28, hidden_size=100, output=10, tasks=8, s_init=False, beta=False):
        super(ATwoLayer, self).__init__()

        # A bunch of convolutions one after another
        # self.l1 = ALinear(784, hidden_size, datasets=tasks, same_init=s_init, Beta=beta)
        self.l1 = ALinear(input_size, hidden_size, datasets=tasks, same_init=s_init, Beta=beta)
        self.l2 = ALinear(hidden_size, hidden_size, datasets=tasks, same_init=s_init, Beta=beta)
        self.l3 = ALinear(hidden_size, hidden_size, datasets=tasks, same_init=s_init, Beta=beta)
        self.l4 = ALinear(hidden_size, output, datasets=tasks, same_init=s_init, Beta=beta)
        self.relu = nn.ReLU()
        self.ls = nn.LogSoftmax(dim=1)

    def forward(self, x, task=None):
        # print("task:", task)
        x = x.view(x.shape[0], -1)
        x = self.l1(x, dataset=task)
        # print(x)
        x = self.relu(x)
        # x = self.model1(x, dataset=task)
        # x = self.relu(x)
        x = self.l3(x, dataset=task)
        x = self.relu(x)
        x = self.l4(x, dataset=task)
        # print(x)
        x = self.ls(x)
        # Average pooling and flatten
        return x

    def adj_sparsity_loss(self, task=None):
        Sum = 0
        for module in list(self.children()):
            if hasattr(module,'l1_loss'):
                Sum += module.l1_loss(dataset=task)

        return Sum            

    def prune(self, p_para=0.5, task=None):
        for module in list(self.children()):
            if hasattr(module,'l1_loss'):
                mask = (module.soft_round(module.adjx[task]) > p_para).data
                l = module.adjx[task]*mask.float()
                module.adjx[task].data.copy_(l.data)

    def null_grad(self, task=None):  # needs fixing - intermediate gradients are not updated
        for module in list(self.children()):
            if hasattr(module,'l1_loss'):
                for ix in range(0, task):
                    mask = (module.soft_round(module.adjx[ix]) < 0.0001).data
                    l = module.weight.grad*mask.float()
                    module.weight.grad.data.copy_(l.data)

    def get_sparsity(self):  # needs fixing
        Sum = 0
        Total = 0
        for module in list(self.children()):
            if hasattr(module,'l1_loss'):
                Sum += module.get_nconnections()
                Total += int(np.prod(list(module.adj_net.shape)))

        return Sum.cpu().data.numpy().item()/Total

class Omni(nn.Module):
    def __init__(self, input_size=28, hidden_size=64, output=2, tasks=50, s_init=False, beta=False):
        super(Omni, self).__init__()

        # A bunch of convolutions one after another
        self.l1 = AConv2d(1, hidden_size, 3, padding=2, datasets=tasks, same_init=s_init, Beta=beta)
        self.l2 = AConv2d(hidden_size, hidden_size, 3, padding=2, datasets=tasks, same_init=s_init, Beta=beta)
        self.l3 = AConv2d(hidden_size, hidden_size, 3, padding=2, datasets=tasks, same_init=s_init, Beta=beta)
        self.l4 = AConv2d(hidden_size, hidden_size, 3, padding=2, datasets=tasks, same_init=s_init, Beta=beta)
        self.relu = nn.ReLU()
        self.ls = nn.LogSoftmax(dim=1)
        # self.mp = torch.nn.MaxPool2D(2, 2)

    def forward(self, x, task=None):
        # print("task:", task)
        x = self.l1(x, dataset=task)
        x = self.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        # x = self.model1(x, dataset=task)
        # x = self.relu(x)
        x = self.l2(x, dataset=task)
        x = self.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.l3(x, dataset=task)
        x = self.ls(x)
        # Average pooling and flatten
        return x

    def adj_sparsity_loss(self, task=None):
        Sum = 0
        for module in list(self.children()):
            if hasattr(module,'l1_loss'):
                Sum += module.l1_loss(dataset=task)

        return Sum            

    def prune(self, p_para=0.8, task=None):
        for module in list(self.children()):
            if hasattr(module,'l1_loss'):
                mask = (module.soft_round(module.adjx[task]) > p_para).data
                l = module.adjx[task]*mask.float()
                module.adjx[task].data.copy_(l.data)

def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq

def AconvLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        AConv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq

import math

class Classifier(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28, tasks = 1):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.conv1 = AConv2d(num_channels, layer_size, 3, 1, 1, datasets=tasks)
        self.conv2 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv3 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv4 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        
        self.bn1 = nn.BatchNorm2d(layer_size)
        self.bn2 = nn.BatchNorm2d(layer_size)
        self.bn3 = nn.BatchNorm2d(layer_size)
        self.bn4 = nn.BatchNorm2d(layer_size)
        
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.do = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()
        self.sm = nn.Sigmoid()
        
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

        self.linear = ALinear(self.outSize, 28*28, datasets=tasks)        

        
        # self.linear = ALinear(self.outsize,1)

    def forward(self, image_input, task = 0):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.do(self.mp1(self.bn1(self.relu(self.conv1(image_input, dataset=task)))))
        x = self.do(self.mp2(self.bn2(self.relu(self.conv2(x, dataset=task)))))
        x = self.do(self.mp3(self.bn3(self.relu(self.conv3(x, dataset=task)))))
        x = self.do(self.mp4(self.bn4(self.relu(self.conv4(x, dataset=task)))))

        x = x.view(x.size()[0], -1)
        x = self.linear(x, dataset=task)
        x = self.sm(x)
        return x

class Classifier2(nn.Module):
    def __init__(self, layer_size=64, num_channels=2, keep_prob=1.0, image_size=28, tasks = 1):
        super(Classifier2, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.conv1 = AConv2d(num_channels, layer_size, 3, 1, 1, datasets=tasks)
        self.conv2 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv3 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        self.conv4 = AConv2d(layer_size, layer_size, 3, 1, 1, datasets=tasks)
        
        self.bn1 = nn.BatchNorm2d(layer_size)
        self.bn2 = nn.BatchNorm2d(layer_size)
        self.bn3 = nn.BatchNorm2d(layer_size)
        self.bn4 = nn.BatchNorm2d(layer_size)
        
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.do = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()
        self.sm = nn.Sigmoid()
        
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

        # self.linear = ALinear(self.outSize, 1, datasets=tasks)        
        self.linear = ALinear(64, 1, datasets=tasks)        

        
        # self.linear = ALinear(self.outsize,1)

    def forward(self, image_input, task = 0):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.do(self.mp1(self.bn1(self.relu(self.conv1(image_input, dataset=task)))))
        x = self.do(self.mp2(self.bn2(self.relu(self.conv2(x, dataset=task)))))
        x = self.do(self.mp3(self.bn3(self.relu(self.conv3(x, dataset=task)))))
        x = self.do(self.mp4(self.bn4(self.relu(self.conv4(x, dataset=task)))))

        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.linear(x, dataset=task)
        x = self.sm(x)
        return x


class OmniV(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28, tasks = 1):
        super(OmniV, self).__init__()
        self.classifier = Classifier(layer_size, num_channels, keep_prob, image_size, tasks)
        # self.linear = ALinear(self.classifier.outSize,1, datasets = tasks)
        self.linear = ALinear(28*28,1, datasets = tasks)
        
    def forward(self, x1, x2, task = 0):
        x1 = self.classifier(x1, task)
        x2 = self.classifier(x2, task)
        # L1 component-wise distance between vectors:
        x = torch.pow(torch.abs(x1 - x2), 2.0)
        return self.linear(x, task)
        

class OmniV2(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28, tasks = 1):
        super(OmniV, self).__init__()
        self.classifier = Classifier(layer_size, num_channels, keep_prob, image_size, tasks)
        # self.linear = ALinear(self.classifier.outSize,1, datasets = tasks)
        self.linear = ALinear(28*28,1, datasets = tasks)
        
    def forward(self, x1, x2, task = 0):
        x1 = self.classifier(x1, task)
        x2 = self.classifier(x2, task)
        # L1 component-wise distance between vectors:
        x = torch.pow(torch.abs(x1 - x2), 2.0)
        return self.linear(x, task)


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(Net, self).__init__()

        self.features = Sequential(
            Conv2d(input_shape[-1], 64, kernel_size=10),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),

            Conv2d(64, 128, kernel_size=7),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),

            Conv2d(128, 128, kernel_size=4),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=2),

            Conv2d(128, 256, kernel_size=4),
            ReLU()
        )

        # Compute number of input features for the last fully-connected layer
        input_shape = (1,) + input_shape[::-1]
        x = Variable(torch.rand(input_shape), requires_grad=False)
        x = self.features(x)
        x = Flatten()(x)
        n = x.size()[1]

        self.classifier = Sequential(
            Flatten(),
            Linear(n, 4096),
            Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class ANet(Module):
    def __init__(self, input_shape, tasks = 1):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(ANet, self).__init__()

        self.conv1 = AConv2d(input_shape[-1], 64, kernel_size=10, datasets = tasks)
        self.mp1 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = AConv2d(64, 128, kernel_size=7, datasets = tasks)
        self.mp2 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = AConv2d(128, 128, kernel_size=4, datasets = tasks)
        self.mp3 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv4 = AConv2d(128, 256, kernel_size=4, datasets = tasks)
        
        self.relu = nn.ReLU()
        
        # Compute number of input features for the last fully-connected layer
        input_shape = (1,) + input_shape[::-1]
        x = Variable(torch.rand(input_shape), requires_grad=False)
        x = self.mp1(self.relu(self.conv1(x, 0)))
        x = self.mp2(self.relu(self.conv2(x, 0)))
        x = self.mp3(self.relu(self.conv3(x, 0)))
        x = self.relu(self.conv4(x, 0))

        x = Flatten()(x)
        n = x.size()[1]

        self.linear = ALinear(n, 4096, datasets = tasks)
        self.sm = Sigmoid()

    def forward(self, x, task = 0):
        
        x = self.mp1(self.relu(self.conv1(x, task)))
        x = self.mp2(self.relu(self.conv2(x, task)))
        x = self.mp3(self.relu(self.conv3(x, task)))
        x = self.relu(self.conv4(x, task))
        x = x.view(x.size(0), -1)
        x = self.linear(x, task)
        x = self.sm(x)
        
        return x


class SiameseNetworks(Module):
    def __init__(self, input_shape):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(SiameseNetworks, self).__init__()
        self.net = Net(input_shape)

        self.classifier = Sequential(
            Linear(4096, 1, bias=False)            
        )
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                m.weight.data.normal_(0, 1e-2)
                m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, Linear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)

    def forward(self, x1, x2):
        x1 = self.net(x1)
        x2 = self.net(x2)
        # L1 component-wise distance between vectors:
        x = torch.pow(torch.abs(x1 - x2), 2.0)
        return self.classifier(x)


class ASiameseNetworks(Module):
    def __init__(self, input_shape, tasks = 1):
        """
        :param input_shape: input image shape, (h, w, c)
        """
        super(ASiameseNetworks, self).__init__()
        self.net = ANet(input_shape)

        self.classifier = ALinear(4096, 1, bias=False, datasets = tasks)            
        
        # self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, AConv2d):
                m.weight.data.normal_(0, 1e-2)
                m.bias.data.normal_(0.5, 1e-2)
            elif isinstance(m, ALinear):
                m.weight.data.normal_(0, 2.0 * 1e-1)
                if m.bias is not None:
                    m.bias.data.normal_(0.5, 1e-2)

    def forward(self, x1, x2, task = 1):
        x1 = self.net(x1, task)
        x2 = self.net(x2, task)
        # L1 component-wise distance between vectors:
        x = torch.pow(torch.abs(x1 - x2), 2.0)
        return self.classifier(x, task)
