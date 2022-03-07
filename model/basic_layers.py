import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

import math
import itertools

class IBPLinear(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        bias=True,
        ibp=False
        ):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.M = nn.Linear(in_dim, out_dim, bias=bias)
        self.ibp = ibp

    def forward(
        self,
        *z):
        assert(len(z) == 3 if self.ibp else 1)
        if not self.ibp:
            return (self.M(z[0]),)
        else:
            z0 = self.M(z[0])

            M_abs = torch.abs(self.M.weight)

            mu = (z[1] + z[2])/2
            r = (z[1] - z[2])/2

            mu = mu @ self.M.weight.T + self.M.bias
            r = r @ M_abs.T

            return (z0, mu + r, mu - r)

class IBPBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, *z, update_stats=True):
        self._check_input_dim(z[0])

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats and update_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        inp = z[0]
        # calculate running estimates
        if self.training:
            mean = inp.mean(dim=0)
            # use biased var in train
            var = inp.var(dim=0, unbiased=False)
            n = inp.numel() / inp.size(1)
            if update_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        out = [(zi - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps)) for zi in z]
        if self.affine:
            out[0] = out[0] * self.weight[None, :] + self.bias[None, :]
            if len(z) > 1:
                assert(len(z) == 3)
                w_plus = (self.weight*(self.weight > 0))[None, :]
                w_minus = (self.weight*(self.weight <= 0))[None, :]
                out1 = out[1]*w_plus + out[2]*w_minus + self.bias[None, :]
                out2 = out[1]*w_minus + out[2]*w_plus + self.bias[None, :]
                out[1] = out1
                out[2] = out2
        return tuple(out)
        
class IBPBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, *z, update_stats=True):
        self._check_input_dim(z[0])

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats and update_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        inp = z[0]
        # calculate running estimates
        if self.training:
            mean = inp.mean([0, 2, 3])
            # use biased var in train
            var = inp.var([0, 2, 3], unbiased=False)
            n = inp.numel() / inp.size(1)
            if update_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        out = [(zi - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps)) for zi in z]
        if self.affine:
            out[0] = out[0] * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            if len(z) > 1:
                assert(len(z) == 3)
                w_plus = (self.weight*(self.weight > 0))[None, :, None, None]
                w_minus = (self.weight*(self.weight <= 0))[None, :, None, None]
                out1 = out[1]*w_plus + out[2]*w_minus + self.bias[None, :, None, None]
                out2 = out[1]*w_minus + out[2]*w_plus + self.bias[None, :, None, None]
                out[1] = out1
                out[2] = out2
        return tuple(out)

class IBPConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, ibp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.M = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.stride = stride
        self.padding = padding
        self.ibp = ibp

    def forward(
        self, 
        *z):
        assert(len(z) == 3 if self.ibp else 1)
        if not self.ibp:
            return (F.conv2d(z[0], self.M.weight, bias=self.M.bias, stride=self.stride, padding=self.padding),)
        else:
            z0 = F.conv2d(z[0], self.M.weight, bias=self.M.bias, stride=self.stride, padding=self.padding)

            M_abs = torch.abs(self.M.weight)

            mu = (z[1] + z[2])/2
            r = (z[1] - z[2])/2

            mu = F.conv2d(mu, self.M.weight, bias=self.M.bias, stride=self.stride, padding=self.padding)
            r = F.conv2d(r, M_abs, bias=None, stride=self.stride, padding=self.padding)

            return (z0, mu + r, mu - r)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# wrapper which just applies the layer to each component
class SimpleIBPWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *z):
        return tuple(self.layer(zi) for zi in z)

# wrapper to apply a bunch of layers with self.ibp option in sequence
class IBPSequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *z):
        out = z
        for ind, layer in enumerate(self.layers):
            out = layer(*out)
            
        return out

    def params_for_ibp_init(self):
        params_to_use = [l.named_parameters() for l in self.layers if type(l) not in [IBPBatchNorm1d, IBPBatchNorm2d]]
        return itertools.chain.from_iterable(params_to_use)

class GenericNet(nn.Module):

    # layers is the list of layers used by the network
    def __init__(self, layers, Wout, normalize=None, ibp=False, ibp_init=False):
        super().__init__()
        self.layers = IBPSequential(layers)
        self.Wout = Wout
        self.normalize = normalize
        self.ibp = ibp

        self.norm_names = []
        for ind, layeri in enumerate(self.layers.layers):
            if hasattr(layeri, 'norm_names'):
                self.norm_names += ['layer{}.{}'.format(ind, namei) for namei in layeri.norm_names]

        if ibp_init:
            self.ibp_init()

    def forward(self, x, eps=0.0):
        x = (x,) if not self.ibp else (x, x + eps, x - eps)
        if self.normalize:
            x = tuple(self.normalize(xi) for xi in x)

        z = self.layers(*x)
        return self.Wout(z[0]), z

    def get_norms(self):
        norms = {}
        for ind, layeri in enumerate(self.layers.layers):
            if hasattr(layeri, 'get_norms'):
                norms.update(
                    {'layer{}.{}'.format(ind, key) : val for key, val in layeri.get_norms().items()})
        return norms

    def params_for_ibp_init(self):
        return itertools.chain(self.layers.params_for_ibp_init(), self.Wout.named_parameters())

    def ibp_init(self):
        for name, val in self.params_for_ibp_init():
            if 'weight' not in name:
                continue
            if val.ndim == 1:
                continue
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(val)
            std = math.sqrt(2*math.pi)/fan_in
            torch.nn.init.normal_(val, mean=0, std=std)









