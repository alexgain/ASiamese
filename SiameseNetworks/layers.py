import torch
import torch.nn as nn
import torch.nn.functional as F

"""Second version of adjacency layers. Need to make it dynamic!"""

class AConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, datasets=1, same_init=False, Beta=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        self.adjx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1),requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]
        if Beta:
            self.Beta = Beta
            self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
            self.initial_beta = self.beta
        else:
            self.Beta = False
        
    def soft_round(self, x, beta = 100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))
        
    def forward(self, input, dataset):
        # try:
        if self.Beta:
            return F.conv2d(input, self.soft_round(self.adjx[dataset], self.beta)*self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return F.conv2d(input, self.soft_round(self.adjx[dataset])*self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # except:
        #     print("Conv DatasetError - input shape {}, dataset {}, adjacencies {}".format(input.shape, dataset, len(self.adjx)))

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset])>0.1).sum()
        except: return("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except: return("DatasetError")

    def beta_val(self):
        return self.initial_beta.item(), self.beta.item()

class ALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, datasets=1, same_init=False, Beta=False):
        super().__init__(in_features, out_features, bias)
        
        self.adjx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1),requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]

        # if Beta:
        #     self.Beta = Beta
        #     self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
        #     self.initial_beta = self.beta
        
    def soft_round(self, x, beta = 100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))
        
    def forward(self, input, dataset):
        try:
            # if not self.Beta:
            #     return F.linear(input, self.soft_round(self.adjx[dataset], self.beta)*self.weight, self.bias)
            return F.linear(input, self.soft_round(self.adjx[dataset])*self.weight, self.bias)
        except Exception as e:
            print("DatasetError: {}".format(e))

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset])>0.1).sum()
        except: return("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except: return("DatasetError")

    # def beta_val(self):
    #     return self.initial_beta.item(), self.beta.item()
      
class AConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, output_padding=0, datasets=1, same_init=False, Beta=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, output_padding=output_padding)
        
        self.adjx = nn.ParameterList([nn.Parameter(torch.Tensor(self.weight.shape).uniform_(0, 1),requires_grad=True) for i in range(datasets)])
        if same_init:
            for ix in range(1, datasets):
                self.adjx[ix] = self.adjx[0]
        
        if Beta:
            self.Beta = Beta
            self.beta = nn.Parameter(torch.Tensor(1).random_(70, 130))
            self.initial_beta = self.beta
        
    def soft_round(self, x, beta = 100):
        return (1 / (1 + torch.exp(-(beta * (x - 0.5)))))
        
    def forward(self, input, dataset):
        if self.Beta:
            return F.conv_transpose2d(input, self.soft_round(self.adjx[dataset], self.beta)*self.weight, bias=None, stride=self.stride, padding=self.padding,
               dilation=self.dilation, output_padding=self.output_padding)
        return F.conv_transpose2d(input, self.soft_round(self.adjx[dataset])*self.weight, bias=None, stride=self.stride, padding=self.padding,
               dilation=self.dilation, output_padding=self.output_padding)
        # try:
        #     return F.conv_transpose2d(input, self.soft_round(self.adjx[dataset])*self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, output_padding=self.output_padding)
        # except: print("Deconv DatasetError - input shape {}, dataset {}, adjacencies {}".format(input.shape, dataset, len(self.adjx)))

    def get_nconnections(self, dataset):
        try:
            return (self.soft_round(self.adjx[dataset])>0.1).sum()
        except: return("DatasetError")

    def l1_loss(self, dataset):
        try:
            return self.soft_round(self.adjx[dataset]).sum()
        except: return("DatasetError")

    def beta_val(self):
        return self.initial_beta.item(), self.beta.item()