import cvxpy as cp
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances

class rkhsFun():
    def __init__(self, kernelFun, gamma=None, is_torch=True, a=None, n_x = None):
        '''
        kernel gamma param
        # n_x, number of points for kernel functions
        '''
        self.kernelFun=kernelFun
        self.kernel_gamma=gamma
        self.is_torch = is_torch

        if n_x is None:
            print("dim of x n_x can't be None!")
            raise NotImplementedError

        if a is None:
            self.a = torch.rand(n_x, requires_grad=True) #coeff of kernel bases; init: uniform weights
        else:
            raise NotImplementedError

    def eval(self, x, data):
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
            fval = torch.mm(self.a.reshape(1,-1), torch.from_numpy(k).float())
        else:
            fval = self.a @ k # RKHS function linear combination of kernel bases
        return fval

    def nor_sqr(self, data):
        # compute the norm of RKHS function
        return a @ self.kernelFun(data, data, gamma=self.kernel_gamma) @ self.a.transpose()

    def __call__(self, x, data):
        return self.eval(x, data)

    def gd_step(self, x_th=None, y_th=None, step_size=0.001, reg_coeff = 0.1):
        # this function performs 1 step of optimization
        # f_th: parameterized learning model function
        # data_th: data in torch array
        # reg_coeff: the coeff for the objective todo: extract this into a separate obj
        # kernel: use the "kernel choice" class in this file

        # construct the graph (again?)
        f_val = self.eval(x_th.reshape(-1, 1), x_th.reshape(-1, 1))
        f_minus_y_sqr = torch.norm(f_val - y_th) ** 2  # frobenius/2-norm sqr

        K = self.kernelFun(x_th.reshape(-1, 1), x_th.reshape(-1, 1))
        aKa = torch.mm(torch.mm(self.a.reshape(1, -1), torch.from_numpy(K).float()), self.a.reshape(-1, 1))

        reg_term = reg_coeff * aKa
        obj_krr = f_minus_y_sqr + reg_term

        # back prop
        obj_krr.backward()  # compute grad

        # gradient step
        self.a.data = self.a.data - step_size * self.a.grad.detach()  # gradient step
        self.a.grad.zero_()  # zero gradient

class kernel_choice():
    '''
    JZ's kernel function, wraps skleran kernels
    '''
    def __init__(self, k=None, param=None):
        self.k = k
        self.param = None
    def __call__(self, x,y):
        #todo add more general than the gamma = syntax
        return self.k(x,y, gamma=self.param)

# pytorch GD
def apply_gd(x, step_size=0.1):
    xnew = x - step_size * x.grad.detach()
    x.grad.data.zero_()
    return xnew