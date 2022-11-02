import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))
            

        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))
        
    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        pass

class CNN_small(nn.Module):
    
    def __init__(self, in_dim, out_dim, p_drop = 0.5):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc_layer_neurons = 256

        self.layer1_filters = 32
        self.layer2_filters = 64

        self.layer1_kernel_size = (4,4)
        self.layer2_kernel_size = (2,2)
        self.layer3_kernel_size = (1,1)
        self.layer1_stride = 2
        self.layer2_stride = 1
        self.layer3_stride = 1
        self.layer1_padding = 0
        self.layer2_padding = 0

        #NB: these calculations assume:
        #1) padding is 0;
        #2) stride is picked such that the last step ends on the last pixel, i.e., padding is not used
        self.layer1_dim_h = (self.in_dim[1] - self.layer1_kernel_size[0]) / self.layer1_stride + 1
        self.layer1_dim_w = (self.in_dim[2] - self.layer1_kernel_size[1]) / self.layer1_stride + 1
        self.layer2_dim_h = (self.layer1_dim_h - self.layer2_kernel_size[0]) / self.layer2_stride + 1
        self.layer2_dim_w = (self.layer1_dim_w - self.layer2_kernel_size[1]) / self.layer2_stride + 1      

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=self.layer1_padding)
        self.conv2 = nn.Conv2d(self.layer1_filters, self.layer2_filters, self.layer2_kernel_size, stride=self.layer2_stride, padding=self.layer2_padding)
        self.conv3 = nn.Conv2d(self.layer2_filters, self.layer2_filters, self.layer3_kernel_size, stride = self.layer3_stride)

        self.bnorm1 = nn.BatchNorm2d(self.layer1_filters)
        self.bnorm2 = nn.BatchNorm2d(self.layer2_filters)

        self.fc_inputs = int(self.layer2_filters * self.layer2_dim_h * self.layer2_dim_w)

        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer_neurons)

        self.lin2 = nn.Linear(self.fc_layer_neurons, self.out_dim)

        # dropout layer
        self.drop = nn.Dropout(p = p_drop)

    def forward(self, x):
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.bnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = F.relu(self.conv3(x))

        # dropout
        x = self.drop(x)

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
