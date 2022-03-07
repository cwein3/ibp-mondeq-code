import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basic_layers import IBPConv2d, IBPBatchNorm2d

class MONConv(nn.Module):
    """ MON class with a single convolution """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        shp, 
        kernel_size=3,
        lben=True,
        lben_cond=3, 
        ibp=True, 
        m=0.5,
        bn_U=True,
        U_act=True):
        super().__init__()
        self.bn_U = bn_U
        if bn_U:
            self.bn_U_module = IBPBatchNorm2d(out_channels) 
        
        self.M = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.m = m 
        self.ibp = ibp
        self.out_dim = out_channels 
        self.shp = shp
        self.kernel_size = kernel_size
        assert(kernel_size % 2 == 1) # for simplicity, use odd kernel size
        self.padding = (kernel_size - 1)//2
        self.U = IBPConv2d(in_channels, out_channels, kernel_size, padding=self.padding, stride=1, bias=True, ibp=ibp) # weight on x

        self.lben = lben
        self.lben_cond = lben_cond
        if self.lben:
            self.lben_p = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))
            self.lben_min = 1.0/self.lben_cond
            self.lben_scale = 1 - self.lben_min

        self.norm_names = ['U_2', 'M_2', 'U_1', 'M_1']
      
        self.U_act = U_act

        if self.U_act:
            self.U_post_act_bias = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))

        assert(0 <= self.m and 1 >= self.m)

    def x_shape(self, n_batch):
        return ((n_batch, self.U.in_channels, self.shp[0], self.shp[1]),)*3 if self.ibp else \
            ((n_batch, self.U.in_channels, self.shp[0], self.shp[1]),)

    def z_shape(self, n_batch):
        return ((n_batch, self.out_dim, self.shp[0], self.shp[1]),)*3 if self.ibp else \
            ((n_batch, self.out_dim, self.shp[0], self.shp[1]),)

    def prep_model(self):
        diag = self._compute_diag(self.M.weight)
        self._comp_dict = {
            'diag' : diag
        }

    def forward(self, x, *z, update_bn=True):
        a = self.bias(x, update_bn=update_bn)
        b = self.multiply(*z)
        return [c + d for c, d in zip(a, b)]

    def bias(self, x, update_bn=True):
        out= self.U(*x) 
        if self.bn_U:
            out = self.bn_U_module(*out, update_stats=update_bn)
        if self.U_act:
            out = tuple(F.relu(outi) + self.U_post_act_bias for outi in out)
        return out

    def _row_col_sums(self, M):
        abs_M = torch.abs(M)
        one_vec = torch.ones((1, self.out_dim, self.shp[0], self.shp[1]), 
            dtype=abs_M.dtype, device=abs_M.device)
        M1 = F.conv2d(one_vec, abs_M, padding=self.padding)
        M1T = F.conv_transpose2d(one_vec, abs_M, padding=self.padding)
        return (M1 + M1T)/2

    def _compute_diag(self, M):
        return self._row_col_sums(M)/(1 - self.m) + 1e-8

    def _multiply(self, M, *z, transpose=False):
        d = self._comp_dict['diag']
        sqrt_d = 1.0/torch.sqrt(d)
        conv_func = F.conv2d if transpose else F.conv_transpose2d
        
        if self.lben:
            lben_scaling = self.lben_scale*F.sigmoid(self.lben_p) + self.lben_min
            scaling_to_use = lben_scaling if not transpose else 1.0/lben_scaling
            z = [scaling_to_use*zi for zi in z]
        
        if not self.ibp:
            out = sqrt_d*conv_func(sqrt_d*z[0], M, stride=1, padding=self.padding)
            return (out,) if not self.lben else ((1.0/scaling_to_use)*out,)
        else:
            dz = [sqrt_d*zi for zi in z]

            mu = (dz[1] + dz[2])/2
            r = (dz[1] - dz[2])/2

            mu = conv_func(mu, M, stride=1, padding=self.padding)
            r = conv_func(r, torch.abs(M), stride=1, padding=self.padding)

            z1 = sqrt_d*(mu + r)
            z2 = sqrt_d*(mu - r)
            z0 = sqrt_d*conv_func(dz[0], M, stride=1, padding=self.padding)
            
            out = (z0, z1, z2)
            return out if not self.lben else tuple((1.0/scaling_to_use)*outi for outi in out)

    def multiply(self, *z):
        return self._multiply(self.M.weight, *z, transpose=False)

    def multiply_transpose(self, *g):
        return self._multiply(self.M.weight, *g, transpose=True)

    def get_norms(self):
        with torch.no_grad():
            U_2 = torch.norm(self.U.M.weight.data).detach().item()
            U_1 = torch.sum(torch.abs(self.U.M.weight.data))
            M_2 = torch.norm(self.M.weight.data).detach().item()
            M_1 = torch.sum(torch.abs(self.M.weight.data)).detach().item()

        return {
            'U_2' : U_2, 
            'U_1' : U_1, 
            'M_2' : M_2,
            'M_1' : M_1
        } 
    
    def params_for_ibp_init(self):
        return self.U.named_parameters()