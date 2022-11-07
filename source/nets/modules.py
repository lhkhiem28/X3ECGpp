
import os, sys
from libs import *

class SEModule(nn.Module):
    def __init__(self, 
        in_channels, 
        reduction = 16, 
    ):
        super(SEModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.s_conv = nn.Conv1d(
            in_channels, in_channels//reduction, 
            kernel_size = 1, 
        )
        self.act_fn = nn.ReLU()
        self.e_conv = nn.Conv1d(
            in_channels//reduction, in_channels, 
            kernel_size = 1, 
        )

    def forward(self, 
        input, 
    ):
        attention_scores = self.pool(input)

        attention_scores = self.s_conv(attention_scores)
        attention_scores = self.act_fn(attention_scores)
        attention_scores = self.e_conv(attention_scores)

        return input*torch.sigmoid(attention_scores)