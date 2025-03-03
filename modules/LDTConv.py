import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
from .LDConv import LDConv
from ultralytics.nn.modules import Conv

class LDTConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, f='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = LDConv(self.dim_conv3, self.dim_conv3, 3, 2,)
        self.conv1 = Conv(self.dim_untouched, self.dim_untouched, 1,2,0)
        self.conv = Conv(dim, ouc, k=1,s=1,p=0)

        self.forward = self.forward_split_cat


    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x2= self.conv1(x2)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x
