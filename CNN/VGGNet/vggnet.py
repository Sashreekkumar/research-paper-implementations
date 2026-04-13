import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

'''
Dimension key:
width : W
height : H 
Input Channels : I
Output Classes : O
'''

VGG_architectures = {
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGGnet(nn.Module):
    def __init__(self, variant, in_channels_I = 3 , num_classes_O =1000 ):
        super(VGGnet, self).__init__()
        self.in_channels_I = in_channels_I
        self.conv_layers = self.create_conv_layers(variant)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes_O)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  #=
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels_I = self.in_channels_I
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels_I, out_channels=out_channels, kernel_size = (3,3), stride=(1,1), padding=(1,1)), 
                        #    nn.BatchNorm2d(x),
                           nn.ReLU()
                           ]
                in_channels_I = out_channels
            
            elif x == 'M':
                layers += [(nn.MaxPool2d(kernel_size = (2,2), stride=(2,2)))]
        
        return nn.Sequential(*layers)
    
