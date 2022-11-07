
import os, sys
from libs import *
from .modules import *
from .bblocks import *
from .backbones import *

class X3ECGpp(nn.Module):
    def __init__(self, 
        base_channels = 64, 
        num_classes = 1, 
    ):
        super(X3ECGpp, self).__init__()
        self.backbone_0 = SEResNet18(base_channels)
        self.backbone_1 = SEResNet18(base_channels)
        self.backbone_2 = SEResNet18(base_channels)
        self.lw_attention = nn.Sequential(
            nn.Linear(
                base_channels*24, base_channels*8, 
            ), 
            nn.BatchNorm1d(base_channels*8), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(
                base_channels*8, 3, 
            ), 
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(
                base_channels*8, 1, 
            ), 
        )

        self.backbone_demogr = nn.Sequential(
            nn.Linear(
                11, base_channels*2, 
            ), 
            nn.BatchNorm1d(base_channels*2), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(
                base_channels*2, base_channels*2, 
            ), 
            nn.BatchNorm1d(base_channels*2), 
            nn.ReLU(), 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(
                base_channels*10, num_classes, 
            ), 
        )

    def forward(self, 
        input, 
        return_attention_scores = False, 
    ):
        features_0 = self.backbone_0(input[0][:, 0, :].unsqueeze(1)).squeeze(2)
        features_1 = self.backbone_1(input[0][:, 1, :].unsqueeze(1)).squeeze(2)
        features_2 = self.backbone_2(input[0][:, 2, :].unsqueeze(1)).squeeze(2)
        attention_scores = torch.sigmoid(
            self.lw_attention(
                torch.cat(
                [
                    features_0, 
                    features_1, 
                    features_2, 
                ], 
                dim = 1, 
                )
            )
        )
        merged_features = torch.sum(
            torch.stack(
            [
                features_0, 
                features_1, 
                features_2, 
            ], 
            dim = 1, 
            )*attention_scores.unsqueeze(-1), 
            dim = 1, 
        )
        sub_output = self.regressor(merged_features).squeeze(-1)

        merged_features = torch.cat(
            [
                merged_features, 
                self.backbone_demogr(input[1]), 
            ], 
            axis = 1, 
        )
        output = self.classifier(merged_features)

        if not return_attention_scores:
            return (output, sub_output)
        else:
            return (output, sub_output), attention_scores