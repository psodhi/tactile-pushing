# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.models as models
from torchvision.models.resnet import BasicBlock

import pytorch_lightning as pl

class ConstantNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(4))                                             

    def forward(self, x, *args):
        return self.mean.expand(x.shape[0], -1)


class TfRegrLinear(pl.LightningModule):
    def __init__(
            self, input_size, output_size=4):
        super().__init__()

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x)

        return x   

class TfRegrLinearClass(pl.LightningModule):
    def __init__(
            self, input_size, output_size=4):
        super().__init__()

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x1, x2, k):
        x = torch.cat([x1, x2], dim=1)

        k.unsqueeze_(1)  # b x 1 x cl
        x.unsqueeze_(-1)  # b x dim x 1
        x = torch.mul(x, k)  # b x dim x cl
        k.squeeze_(1) # b x cl

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x

class TfRegrNonlinearClass(pl.LightningModule):
    def __init__(
            self, input_size, hidden_size=32, output_size=4):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x1, x2, k):
        x = torch.cat([x1, x2], dim=1)

        k.unsqueeze_(1)  # b x 1 x cl
        x.unsqueeze_(-1)  # b x dim x 1
        x = torch.mul(x, k)  # b x dim x cl
        k.squeeze_(1) # b x cl

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class FeatMapClassNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Linear(32, 2)
        )

    def forward(self, x):

        x = x.view(x.shape[0], -1)
        logits = self.net(x)

        return logits