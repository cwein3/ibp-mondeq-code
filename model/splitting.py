import torch
import torch.nn as nn
from torch.autograd import Function
import time
from functools import reduce

import torch
import numpy as np

class Meter(object):
    """Computes and stores the min, max, avg, and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)

class SplittingMethodStats(object):
    def __init__(self):
        self.fwd_iters = Meter()
        self.bkwd_iters = Meter()
        self.fwd_time = Meter()
        self.bkwd_time = Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_iters.reset()
        self.bkwd_time.reset()

    def report(self):
        print('Fwd iters: {:.2f}\tFwd Time: {:.4f}\tBkwd Iters: {:.2f}\tBkwd Time: {:.4f}\n'.format(
                self.fwd_iters.avg, self.fwd_time.avg,
                self.bkwd_iters.avg, self.bkwd_time.avg))

class Backward(Function):
    @staticmethod
    def forward(ctx, splitter, *z):
        ctx.splitter = splitter
        ctx.save_for_backward(*z)
        return z

    @staticmethod
    # assume that prep_model has already been called with some associated forward pass
    def backward(ctx, *g):
        start = time.time()
        sp = ctx.splitter
        n = len(g)
        z = ctx.saved_tensors
        j = sp.nonlin_module.derivative(*z)
        I = [j[i] == 0 for i in range(n)]
        d = [(1 - j[i]) / j[i] for i in range(n)]
        v = tuple(j[i] * g[i] for i in range(n))
        u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                  for s in sp.linear_module.z_shape(g[0].shape[0]))

        err = 1.0
        it = 0
        errs = []
        while (err > sp.tol and it < sp.max_iter):
            un = sp.linear_module.multiply_transpose(*u)
            un = tuple((1 - sp.alpha) * u[i] + sp.alpha * un[i] for i in range(n))
            un = tuple((un[i] + sp.alpha * (1 + d[i]) * v[i]) / (1 + sp.alpha * d[i]) for i in range(n))
            for i in range(n):
                un[i][I[i]] = v[i][I[i]]

            err = sum((un[i] - u[i]).norm().item() / (1e-6 + un[i].norm().item()) for i in range(n))
            errs.append(err)
            u = un
            it = it + 1

        if sp.verbose:
            print("Backward: ", it, err)

        dg = sp.linear_module.multiply_transpose(*u)
        dg = tuple(g[i] + dg[i] for i in range(n))

        sp.stats.bkwd_iters.update(it)
        sp.stats.bkwd_time.update(time.time() - start)
        sp.errs = errs
        return (None,) + dg

class MONForwardBackward(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = SplittingMethodStats()
        self.save_abs_err = False
        self.norm_names = [] if not hasattr(self.linear_module, 'norm_names') else self.linear_module.norm_names

    def forward(self, *x):
        """ Forward pass of the MON, find an equilibirum with forward-backward splitting"""

        start = time.time()
        # Run the forward pass _without_ tracking gradients
        with torch.no_grad():
            self.linear_module.prep_model()
            z = tuple(torch.zeros(s, dtype=x[0].dtype, device=x[0].device)
                      for s in self.linear_module.z_shape(x[0].shape[0]))
            n = len(z)
            bias = self.linear_module.bias(x, update_bn=False)

            err = 1.0
            it = 0
            errs = []
            
            while (err > self.tol and it < self.max_iter):
                zn = self.linear_module.multiply(*z)
                zn = tuple((1 - self.alpha) * z[i] + self.alpha * (zn[i] + bias[i]) for i in range(n))
                zn = self.nonlin_module(*zn, alpha=self.alpha)
                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn, update_bn=False), alpha=1)
                    err = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
                    errs.append(err)
                else:
                    err = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))

                z = zn
                it = it + 1

        if self.verbose:
            print("Forward: ", it, err)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        self.linear_module.prep_model()
        zn = self.linear_module(x, *z, update_bn=True)
        zn = self.nonlin_module(*zn)
        zn = Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    def get_norms(self):
        return self.linear_module.get_norms()

# helper function to assign z values for Anderson iteration
def _iter_assign(Z, z, i, bsz):
    for ind, Zi in enumerate(Z):
        Zi[:, i] = z[ind].view(bsz, -1)

class MONForwardBackwardAnderson(nn.Module):

    def __init__(self, linear_module, nonlin_module, m=5, beta=1.0, lam=1e-4, alpha=1.0, tol=1e-5, max_iter=50, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.m = m # number of past iterates for anderson iteration
        self.beta = beta # anderson iteration weighting
        self.lam = lam # lambda for anderson iteration
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = SplittingMethodStats()
        self.save_abs_err = False

    def _single_it(self, z, x, bias, errs):
        n = len(z)
        zn = self.linear_module.multiply(*z)
        zn = tuple((1 - self.alpha) * z[i] + self.alpha * (zn[i] + bias[i]) for i in range(n))
        zn = self.nonlin_module(*zn, alpha=self.alpha)
        if self.save_abs_err:
            fn = self.nonlin_module(*self.linear_module(x, *zn, update_bn=False), alpha=1)
            err = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
            errs.append(err)
        else:
            err = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))
        
        return zn, err

    def forward(self, *x):
        """ Forward pass of the MON, find an equilibirum with forward-backward splitting"""

        start = time.time()
        # Run the forward pass _without_ tracking gradients
        with torch.no_grad():
            bsz = x[0].shape[0]
            z_shape = self.linear_module.z_shape(bsz)
            self.linear_module.prep_model()
            numel = tuple(reduce(lambda x, y : x*y, zs[1:], 1) for zs in z_shape) # sizes of combined non-batch dimensions

            # initialize Z and F matrices
            Z = [torch.zeros(bsz, self.m, numeli, dtype=x[0].dtype, device=x[0].device) 
                      for numeli in numel]
            F = [torch.zeros(bsz, self.m, numeli, dtype=x[0].dtype, device=x[0].device) 
                      for numeli in numel]

            bias = self.linear_module.bias(x, update_bn=False)

            errs = []
            z0 = tuple(z[:, 0].view(z_shape[ind]) for ind, z in enumerate(Z))
            f0, err = self._single_it(z0, x, bias, errs)
            _iter_assign(F, f0, 0, bsz)
            _iter_assign(Z, f0, 1, bsz)
            f1, err = self._single_it(f0, x, bias, errs)
            _iter_assign(F, f1, 1, bsz)

            H = [torch.zeros(bsz, self.m + 1, self.m + 1, dtype=x[0].dtype, device=x[0].device)]*len(Z)
            for Hi in H:
                Hi[:, 0, 1:] = Hi[:, 1:, 0] = 1

            y = [torch.zeros(bsz, self.m + 1, 1, dtype=x[0].dtype, device=x[0].device)]*len(Z)
            for yi in y:
                yi[:, 0] = 1

            res = []
            for it in range(2, self.max_iter):
                n = min(self.m, it)
                for i in range(len(Z)):
                    G = F[i][:, :n] - Z[i][:, :n]
                    GGt = torch.bmm(G, G.transpose(1, 2))
                    GGt /= torch.norm(GGt.view(GGt.size(0), -1), dim=-1).view(-1, 1, 1) + 1e-6 # normalizing doesn't affect the direction of the solution
                    H[i][:,1:n + 1, 1:n + 1] = GGt + self.lam*torch.eye(n, dtype=x[0].dtype, device=x[0].device)[None]
                    alpha = torch.linalg.solve(
                        H[i][:, :n + 1, :n + 1],
                        y[i][:, :n + 1] 
                    )[:, 1:n + 1, 0]
                    Z[i][:, it%self.m] = self.beta*(alpha[:, None] @ F[i][:, :n])[:, 0] + (1 - self.beta)*(alpha[:, None]@Z[i][:, :n])[:, 0]
                Zin = tuple(Zi[:, it%self.m].view(z_shape[ind]) for ind, Zi in enumerate(Z))
                Zo, err = self._single_it(Zin, x, bias, errs)
                _iter_assign(F, Zo, it%self.m, bsz)
                if (err < self.tol):
                    break

        if self.verbose:
            print("Forward: ", it, err)

        z = tuple(Zi[:, it % self.m].view(z_shape[ind]) for ind, Zi in enumerate(Z))

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        self.linear_module.prep_model()
        zn = self.linear_module(x, *z, update_bn=True)
        zn = self.nonlin_module(*zn)
        zn = Backward.apply(self, *zn)
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn

    def get_norms(self):
        return self.linear_module.get_norms()
