import cvxpy as cp
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances

class rkhsFun():
    def __init__(self, kernelFun, gamma=None, is_torch=True, a=None, x=None, n_x = None, n_dim=None):
        '''
        kernel gamma param
        # n_x, number of points for kernel functions
        '''
        self.kernelFun=kernelFun
        self.kernel_gamma=gamma
        self.is_torch = is_torch

        if n_x is None:
            print("num of samples of x can't be None!")
            raise NotImplementedError

        if n_dim is None:
            print("dim of x n_dim can't be None!")
            raise NotImplementedError

        if a is None:
            self.a = torch.rand(n_x, requires_grad=True) #coeff of kernel bases; init: uniform weights
        else:
            raise NotImplementedError

        if x is None:
            raise NotImplementedError
            # self.x = torch.rand(n_x, requires_grad=True) #coeff of kernel bases; init: uniform weights
        else:
            self.x = x

    def eval(self, x_new):
        '''
        compute the rkhs fucntion
            f(x) = sum { a_i * k(data_i, x) }
        x: the location to evaluate, can be a vector
        data: the expansion points, empirical data
        '''
        # if len(x.shape) == 1:
        #     xloc = x.reshape(-1,1)
        # elif len(x.shape) == 2:
        #     xloc = x
        # else:
        #     raise NotImplementedError

        # in case x_new is a single sample
        if len(x_new.squeeze().shape) < 1:
            x_new = x_new.reshape(1, 1)

        if len(x_new.squeeze().shape) < 2:
            k = self.kernelFun(x_new, self.x.reshape(-1,1), gamma=self.kernel_gamma)
        else:
            k = self.kernelFun(x_new, self.x, gamma=self.kernel_gamma)

        if self.is_torch:
            # use torch mat multiple
            fval = torch.mm(torch.from_numpy(k).float(), self.a.reshape(-1,1), )
        else:
            fval = self.a @ k # RKHS function linear combination of kernel bases
        return fval

    def nor_sqr(self, data):
        # compute the norm of RKHS function
        return a @ self.kernelFun(data, data, gamma=self.kernel_gamma) @ self.a.transpose()

    def __call__(self, x):
        return self.eval(x)

    def gd_step(self, x_th=None, y_th=None, step_size=0.001, reg_coeff = 0.1):
        # this function performs 1 step of optimization
        # f_th: parameterized learning model function
        # data_th: data in torch array
        # reg_coeff: the coeff for the objective todo: extract this into a separate obj
        # kernel: use the "kernel choice" class in this file

        # construct the graph (again?)
        f_val = self.eval(x_th.reshape(-1, 1))
        f_minus_y_sqr = torch.norm(f_val - y_th) ** 2  # frobenius/2-norm sqr

        K = self.kernelFun(self.x.reshape(-1, 1), self.x.reshape(-1, 1))
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

def plot_sol_rkhs(f_th, data_th):
    import matplotlib.pyplot as plt
    x_grid = torch.linspace(torch.min(data_th), torch.max(data_th), 100).reshape(-1, 1)
    f_val_plot = f_th(x_grid)
    plt.plot(x_grid.detach().numpy(),
             f_val_plot.detach().numpy().reshape(-1,1),
             c='r')  # plot the uniform weights interpolant
