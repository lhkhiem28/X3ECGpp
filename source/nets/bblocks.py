
import os, sys
from libs import *
from .modules import *

class SEResBlock(nn.Module):
    def __init__(self, 
        in_channels, 
        downsample = False, 
    ):
        super(SEResBlock, self).__init__()
        if downsample:
            self.out_channels = in_channels*2
            self.conv_1 = nn.Conv1d(
                in_channels, self.out_channels, 
                kernel_size = 7, padding = 3, stride = 2, 
            )
            self.identity = nn.Sequential(
                nn.Conv1d(
                    in_channels, self.out_channels, 
                    kernel_size = 1, padding = 0, stride = 2, 
                ), 
                nn.BatchNorm1d(self.out_channels), 
            )
        else:
            self.out_channels = in_channels
            self.conv_1 = nn.Conv1d(
                in_channels, self.out_channels, 
                kernel_size = 7, padding = 3, stride = 1, 
            )
            self.identity = nn.Identity()
        self.conv_2 = nn.Conv1d(
            self.out_channels, self.out_channels, 
            kernel_size = 7, padding = 3, stride = 1, 
        )

        self.convs = nn.Sequential(
            self.conv_1, 
            nn.BatchNorm1d(self.out_channels), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            self.conv_2, 
            nn.BatchNorm1d(self.out_channels), 
            SEModule(self.out_channels), 
        )
        self.act_fn = nn.ReLU()

    def forward(self, 
        input, 
    ):
        output = self.convs(input) + self.identity(input)
        output = self.act_fn(output)

        return output