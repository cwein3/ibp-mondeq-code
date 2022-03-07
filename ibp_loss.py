import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_sa(n_class=10):
	sa = np.zeros((n_class, n_class - 1), dtype = np.int32)
	for i in range(sa.shape[0]):
		for j in range(sa.shape[1]):
			if j < i:
				sa[i][j] = j
			else:
				sa[i][j] = j + 1
	sa = torch.LongTensor(sa)
	return sa 

def compute_ibp_elide_z(Wout, z_up, z_low, y, sa, n_class=10):
	c = torch.eye(n_class).type_as(z_up)[y].unsqueeze(1) - torch.eye(n_class).type_as(z_up).unsqueeze(0)
	# remove specifications to self
	I = (~(y.data.unsqueeze(1) == torch.arange(n_class).type_as(y.data).unsqueeze(0)))
	c = (c[I].view(y.size(0),n_class-1,n_class))
	# scatter matrix to avoid compute margin to self
	sa_labels = sa[y]
	# storing computed lower bounds after scatter

	sa_labels = sa_labels.to(z_low.device)
	c = c.to(z_low.device)

	# multiply Wout by C
	weight = torch.matmul(c, Wout.weight.unsqueeze(0))
	bias = torch.matmul(c, Wout.bias.view(1, -1, 1)) 
	W_plus = weight*(weight > 0)
	W_minus = weight*(weight < 0)

	lb_s = torch.zeros(y.size(0), n_class, dtype=z_low.dtype, device=z_low.device)

	z_lb = torch.matmul(W_minus, z_up.unsqueeze(2)) + torch.matmul(W_plus, z_low.unsqueeze(2)) + bias
	z_lb = z_lb.squeeze()
	return -lb_s.scatter(1, sa_labels, z_lb)
