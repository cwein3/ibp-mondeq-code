import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basic_layers import IBPLinear, IBPBatchNorm1d

class MONFc(nn.Module):
	""" Simple MON linear class, just a single full multiply. """

	def __init__(
		self, 
		in_dim, 
		out_dim,
		lben=True,
		lben_cond=3,
		ibp=True, 
		m=0.5, 
		bn_U=True,
		U_act=True):
		super().__init__()
		self.U = IBPLinear(in_dim, out_dim, ibp=ibp)
		
		self.M = nn.Linear(out_dim, out_dim) # unconstrained matrix in the parameterization of IBP-MonDEQ weights

		self.m = m # should be between 0 and 1
		self.ibp = ibp
		self.out_dim = out_dim

		self.lben = lben
		self.lben_cond = lben_cond
		if self.lben:
			assert(lben_cond >= 1)
			self.lben_p = nn.Parameter(torch.zeros((out_dim,)))
			self.lben_min = 1.0/self.lben_cond
			self.lben_scale = 1 - self.lben_min

		self.norm_names = ['U_2', 'M_2', 'U_1', 'M_1'] 

		self.U_act = U_act

		self.bn_U = bn_U
		if self.bn_U:
			self.bn_U_module = IBPBatchNorm1d(out_dim)

		if self.U_act:
			self.U_post_act_bias = nn.Parameter(torch.zeros((1, out_dim)))

		assert(0 <= self.m and 1 >= self.m)

	def x_shape(self, n_batch):
		return ((n_batch, self.U.in_features),)*3 if self.ibp else ((n_batch, self.U.in_features),)

	def z_shape(self, n_batch):
		return ((n_batch, self.M.in_features),)*3 if self.ibp else ((n_batch, self.M.in_features),)

	def prep_model(self):
		diag = self._compute_diag(self.M.weight)
		self._comp_dict = {
			'diag' : diag
		}

	def forward(self, x, *z, update_bn=True): # x and z are always tuples
		a = self.bias(x) 
		b = self.multiply(*z)
		return [ai + bi for ai, bi in zip(a, b)]
		
	def bias(self, x, update_bn=True):
		out = self.U(*x)
		if self.bn_U:
			out = self.bn_U_module(*out, update_stats=update_bn)
		if self.U_act:
			out = tuple(F.relu(outi) + self.U_post_act_bias for outi in out)
		return out 

	def _row_col_sums(self, M):
		abs_M = torch.abs(M)
		one_vec = torch.ones((abs_M.size(0), 1), dtype=abs_M.dtype, device=abs_M.device)
		return ((abs_M @ one_vec + abs_M.T @ one_vec)/2).view(-1)

	def _compute_diag(self, M):
		return self._row_col_sums(M)/(1 - self.m) + 1e-8

	def _multiply(self, M, *z, transpose=False):
		d = self._comp_dict['diag']
		sqrt_d = 1.0/torch.sqrt(d)
		if self.lben:
			lben_scaling = self.lben_scale*F.sigmoid(self.lben_p) + self.lben_min
			scaling_to_use = lben_scaling if not transpose else 1.0/lben_scaling
			z = [scaling_to_use*zi for zi in z]
		if not self.ibp:
			out = sqrt_d*((sqrt_d*z[0])@ M.T)
			return (out,) if not self.lben else ((1.0/scaling_to_use)*out,)
		else:
			dz = [sqrt_d*zi for zi in z]
			z0 = sqrt_d*(dz[0] @ M.T)

			mu = (dz[1] + dz[2])/2
			r = (dz[1] - dz[2])/2

			mu = mu @ M.T
			r = r @ torch.abs(M.T)

			out = (z0, sqrt_d*(mu + r), sqrt_d*(mu - r))
			return out if not self.lben else tuple((1.0/scaling_to_use)*outi for outi in out)

	def multiply(self, *z):
		return self._multiply(self.M.weight, *z, transpose=False)

	def multiply_transpose(self, *z):
		return self._multiply(self.M.weight.T, *z, transpose=True)

	def get_norms(self):
		with torch.no_grad():
			U_2 = torch.norm(self.U.M.weight.data).detach().item()
			U_1 = torch.sum(torch.abs(self.U.M.weight.data)).detach().item()
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