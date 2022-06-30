import cvxpy as cp
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances

class rkhsFun():
    def __init__(self,kernelFun,gamma=None, is_torch=False):
        '''
        kernel gamma param
        '''
        # self.data=data
        self.kernelFun=kernelFun
        self.kernel_gamma=gamma
        self.is_torch = is_torch

    def eval(self, x, a, data):
        '''
        compute the rkhs fucntion
            f(x) = sum { a_i * k(data_i, x) }
        x: the location to evaluate, can be a vector
        data: the expansion points, empirical data
        '''
        if len(x.shape) == 1:
            xloc = x.reshape(-1,1)
        elif len(x.shape) == 2:
            xloc = x
        else:
            raise NotImplementedError

        k = self.kernelFun(data, xloc, gamma=self.kernel_gamma)

        if self.is_torch:
            # use torch mat multiple
            fval = torch.mm(a.reshape(1,-1), torch.from_numpy(k).float())
        else:
            fval = a @ k # RKHS function linear combination of kernel bases
        return fval

    def nor_sqr(self, a, data):
        # compute the norm of RKHS function
        return a @ self.kernelFun(data, data, gamma=self.kernel_gamma) @ a.transpose()

    def __call__(self, x, a, data):
        return self.eval(x, a, data)



# pytorch GD
def apply_gd(x, step_size=0.1):
    xnew = x - step_size * x.grad.detach()
    x.grad.data.zero_()
    return xnew