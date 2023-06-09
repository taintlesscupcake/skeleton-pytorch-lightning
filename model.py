import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn

class SkeletonModel(nn.Module):
    def __init__(self, args):
        super(SkeletonModel, self).__init__()
        self.args = args
        self.net = resnet18(pretrained=True)
        
    def forward(self, x):
        return self.net(x)