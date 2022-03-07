import torch
import torch.nn as nn 
import torch.nn.functional as F

from .basic_layers import *

FF_NAMES = ['explicit_3', 'explicit_7']

def explicit_3(in_ch, in_dim, out_ch, n_class=10, normalize=None, ibp=True, ibp_init=True):

	layers = [
		IBPConv2d(in_ch, out_ch, 3, stride=1, padding=1, ibp=ibp)]
	layers += [IBPBatchNorm2d(out_ch), SimpleIBPWrapper(nn.ReLU())]	
	layers.append(IBPConv2d(out_ch, out_ch, 3, stride=1, padding=1, ibp=ibp))
	layers += [IBPBatchNorm2d(out_ch), SimpleIBPWrapper(nn.ReLU())]	

	layers += [SimpleIBPWrapper(nn.AvgPool2d(4)),
		SimpleIBPWrapper(Flatten())]

	Wout = nn.Linear(((in_dim//4)**2)*out_ch, n_class)

	return GenericNet(layers, Wout, normalize=normalize, ibp=ibp, ibp_init=ibp_init)

def explicit_7(in_ch, in_dim, n_class=10, feat_dim=512, normalize=None, ibp=True, ibp_init=True, num_additional=0):
	
	layers = [
		IBPConv2d(in_ch, 64, 3, stride=1, padding=1, ibp=ibp)]
	layers += [IBPBatchNorm2d(64), SimpleIBPWrapper(nn.ReLU())]
	layers.append(IBPConv2d(64, 64, 3, stride=1, padding=1, ibp=ibp))
	layers += [IBPBatchNorm2d(64), SimpleIBPWrapper(nn.ReLU())]
	layers.append(IBPConv2d(64, 128, 3, stride=2, padding=1, ibp=ibp))
	layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU())]
	layers.append(IBPConv2d(128, 128, 3, stride=1, padding=1, ibp=ibp))
	layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU())]
	layers.append(IBPConv2d(128, 128, 3, stride=1, padding=1, ibp=ibp))
	layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU())]

	if num_additional > 0:
		for i in range(num_additional):
			layers.append(IBPConv2d(128, 128, 3, stride=1, padding=1, ibp=ibp))
			layers += [IBPBatchNorm2d(128), SimpleIBPWrapper(nn.ReLU())]
	layers += [
		SimpleIBPWrapper(Flatten()),
		IBPLinear((in_dim//2) * (in_dim//2) * 128, feat_dim, ibp=ibp)]
	layers += [IBPBatchNorm1d(feat_dim), SimpleIBPWrapper(nn.ReLU())]
	Wout = nn.Linear(feat_dim, n_class)

	return GenericNet(layers, Wout, normalize=normalize, ibp=ibp, ibp_init=ibp_init)




