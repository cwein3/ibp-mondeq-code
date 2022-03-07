import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MONReLU(nn.Module):
    def forward(self, *z, alpha=1):
        return tuple(F.relu(z_) for z_ in z)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)