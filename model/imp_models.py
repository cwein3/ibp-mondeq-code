import torch
import torch.nn as nn 
import torch.nn.functional as F
import sys 
import itertools

from .imp_fc import MONFc
from .imp_conv import MONConv

from .basic_layers import *

from .mon_act import MONReLU
from .splitting import MONForwardBackward, MONForwardBackwardAnderson

MON_NAMES = ['DEQ_FC', 'DEQ_3', 'DEQ_7']


def expand_args(defaults, kwargs):
    d = defaults.copy()
    for k, v in kwargs.items():
        d[k] = v
    return d


MON_DEFAULTS = {
    'alpha': 1.0,
    'tol': 1e-5,
    'max_iter': 50
}

def get_splitting(spm):
    if spm == 'fb':
        return MONForwardBackward
    elif spm == 'fb_anderson':
        return MONForwardBackwardAnderson

def mon_fc_layer(
    in_dim, 
    out_dim, 
    spm, 
    lben=True,
    lben_cond=3,
    ibp=True, 
    m=0.5,
    bn_U=True,
    U_act=True,
    **spm_kwargs):
    lin_module = MONFc(
        in_dim, 
        out_dim,
        lben=lben,
        lben_cond=lben_cond,
        ibp=ibp, 
        m=m,
        bn_U=bn_U,
        U_act=U_act)
    nonlin_module = MONReLU()
    return get_splitting(spm)(lin_module, nonlin_module, **expand_args(MON_DEFAULTS, spm_kwargs))

def mon_conv_layer(
    in_channels,
    out_channels,
    shp,
    spm,
    kernel_size=3,
    lben=True,
    lben_cond=3, 
    ibp=True, 
    m=0.5,
    bn_U=True,
    U_act=True,
    **spm_kwargs):
    lin_module = MONConv(
        in_channels,
        out_channels,
        shp,
        kernel_size=kernel_size,
        lben=lben,
        lben_cond=lben_cond, 
        ibp=ibp, 
        m=m,
        bn_U=bn_U,
        U_act=U_act)
    nonlin_module = MONReLU()
    return get_splitting(spm)(lin_module, nonlin_module, **expand_args(MON_DEFAULTS, spm_kwargs))

class MONNet(GenericNet):

    # layers is the list of layers used by the network
    def __init__(self, layers, Wout, normalize=None, ibp=False, ibp_init=False):
        super(MONNet, self).__init__(layers, Wout, normalize=normalize, ibp=ibp, ibp_init=ibp_init)
    
    def _get_mon_layer(self):
        for layeri in self.layers.layers:
            if type(layeri) in [MONForwardBackward, MONForwardBackwardAnderson]:
                return layeri

    def set_mon(self, mon):
        for i, layeri in enumerate(self.layers.layers):
            if type(layeri) in [MONForwardBackward, MONForwardBackwardAnderson]:
                break
        self.layers.layers[i] = mon
        
    @property
    def mon(self):
        return self._get_mon_layer()
    
    def params_for_ibp_init(self):
        mon_layers = []
        non_bn_non_mon_params = []
        mon_params = []
        for layer in self.layers.layers:
            if type(layer) in [IBPBatchNorm1d, IBPBatchNorm2d]:
                continue
            if type(layer) in [MONForwardBackward, MONForwardBackwardAnderson]:
                mon_layers.append(layer)
            else:
                non_bn_non_mon_params.append(layer.named_parameters())
        for layer in mon_layers:
            if hasattr(layer.linear_module, 'params_for_ibp_init'):
                mon_params.append(layer.linear_module.params_for_ibp_init())
            else:
                mon_params.append(layer.named_parameters())
        return itertools.chain.from_iterable(mon_params + non_bn_non_mon_params)
        
def DEQ_FC(
    in_dim, 
    out_dim, 
    spm,
    normalize=None, 
    lben=True,
    lben_cond=3,
    ibp=True, 
    m=0.5,
    n_class=10,
    ibp_init=False,
    **spm_kwargs):
    layers = [
        SimpleIBPWrapper(Flatten()),
        mon_fc_layer(
            in_dim, 
            out_dim, 
            spm, 
            lben=lben,
            lben_cond=lben_cond,
            ibp=ibp, 
            m=m,
            **spm_kwargs)]
    Wout = nn.Linear(out_dim, n_class)
    
    return MONNet(layers, Wout, normalize=normalize, ibp=ibp, ibp_init=ibp_init)

def DEQ_3(
    in_channels,
    in_dim,
    out_channels,
    spm,
    normalize=None,
    kernel_size=3,
    lben=True,
    lben_cond=3, 
    ibp=True, 
    m=0.5,
    n_class=10,
    ibp_init=True,
    bn_U=True,
    U_act=True,
    normalize_after_mon=True,
    **spm_kwargs):
    layers = [mon_conv_layer(
        in_channels, 
        out_channels, 
        (in_dim, in_dim),
        spm,
        kernel_size=kernel_size,
        lben=lben,
        lben_cond=lben_cond, 
        ibp=ibp, 
        m=m,
        bn_U=bn_U,
        U_act=U_act,
        **spm_kwargs)]
    if normalize_after_mon:
        layers.append(IBPBatchNorm2d(out_channels))
    layers += [
        SimpleIBPWrapper(nn.AvgPool2d(4)),
        SimpleIBPWrapper(Flatten())]
    Wout = nn.Linear(((in_dim//4)**2)*out_channels, n_class)
    
    return MONNet(layers, Wout, normalize=normalize, ibp=ibp, ibp_init=ibp_init)

def DEQ_7(
    in_ch, 
    in_dim,
    spm,
    normalize=None,
    kernel_size=3,
    lben=True,
    lben_cond=3, 
    ibp=True, 
    m=0.5,
    n_class=10,
    feat_dim=512,
    ibp_init=True,
    bn_U=True,
    normalize_after_mon=True,
    U_act=True,
    **spm_kwargs):
    layers = [mon_conv_layer(
            in_ch, 
            64, 
            (in_dim, in_dim),
            spm,
            kernel_size=kernel_size,
            lben=lben,
            lben_cond=lben_cond, 
            ibp=ibp, 
            m=m,
            bn_U=bn_U,
            U_act=U_act,
            **spm_kwargs)]
    layers.append(IBPBatchNorm2d(64))
    layers.append(IBPConv2d(64, 128, 3, stride=2, padding=1, ibp=ibp))
    layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU())]
    layers.append(IBPConv2d(128, 128, 3, stride=1, padding=1, ibp=ibp))
    layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU())]
    layers.append(IBPConv2d(128, 128, 3, stride=1, padding=1, ibp=ibp))
    layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU()), SimpleIBPWrapper(Flatten())]
    layers += [
        IBPLinear((in_dim//2)*(in_dim//2)*128, feat_dim, ibp=ibp),
        IBPBatchNorm1d(512), 
        SimpleIBPWrapper(nn.ReLU())
    ]
    Wout = nn.Linear(feat_dim, n_class)

    return MONNet(layers, Wout, normalize=normalize, ibp=ibp, ibp_init=ibp_init)