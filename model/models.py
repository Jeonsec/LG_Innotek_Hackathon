import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

class CustomModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        drop_prob = 0.3,
        norm_ws=False,        
    ):
        super(CustomModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=drop_prob)

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_prob),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=drop_prob),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),    
            nn.Dropout(p=drop_prob),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),        
            nn.Dropout(p=drop_prob),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),          
            nn.Dropout(p=drop_prob),
            nn.Linear(256,output_dim),
        )
    
                
    def _init_weight(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.layer(x)
        return x