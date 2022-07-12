import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances

import torch
import torch.nn as nn

class ml_model():
    def __init__(self, model_class = 'nn', n_dim=None, loss = None):
        '''
        kernel gamma param
        # n_x, number of points for kernel functions
        '''

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if loss is None: # must choose torch loss class
            raise NotImplementedError
        else:
            self.loss = loss

        if n_dim is None:
            print("dim of x n_dim can't be None!")
            raise NotImplementedError
        self.n_dim = n_dim

        if model_class=='nn':
            n_hidden = 64
            self.model = nn.Sequential(nn.Linear(n_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)).to(device)
            self.parameters = self.model.parameters()
        else:
            raise NotImplementedError

    def __call__(self, x):
        return self.model(x)

    def train_step(self, x_th=None, y_th=None, optimization =None):
        # this function performs 1 step of optimization
        # f_th: parameterized learning model function
        # data_th: data in torch array

        # construct the graph (again?)
        f_val = self.model(x_th.reshape(-1, 1)) # compute f value evaluated at all data points, math: f_val = K'a
        obj = self.loss(f_val , y_th)

        # gradient step
        if optimization is None:
            raise NotImplementedError
        else: # simple optimization for pth: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            optimization.zero_grad()
            obj.backward()  # compute grad
            optimization.step()

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
            K = self.kernelFun(x_new, self.x.reshape(-1,1), gamma=self.kernel_gamma)
        else:
            K = self.kernelFun(x_new, self.x, gamma=self.kernel_gamma)

        if self.is_torch:
            # use torch mat multiple
            fval = torch.mm(torch.from_numpy(K).float(), self.a.reshape(-1,1), )
        else:
            fval = self.a @ K # RKHS function linear combination of kernel bases
        return fval, K # return also the kernel matrix

    def nor_sqr(self, data):
        # compute the norm of RKHS function
        pass
        return a @ self.kernelFun(data, data, gamma=self.kernel_gamma) @ self.a.transpose()

    def __call__(self, x):
        return self.eval(x)

    def gd_step(self, x_th=None, y_th=None, step_size=0.001, reg_coeff = 0.1, opt_auto =None):
        # this function performs 1 step of optimization
        # f_th: parameterized learning model function
        # data_th: data in torch array
        # reg_coeff: the coeff for the objective todo: extract this into a separate obj
        # kernel: use the "kernel choice" class in this file

        # construct the graph (again?)
        f_val, K = self.eval(x_th.reshape(-1, 1)) # compute f value evaluated at all data points, math: f_val = K'a
        f_minus_y_sqr = torch.norm(f_val - y_th) ** 2  # frobenius/2-norm sqr

        # aKa = torch.mm(torch.mm(self.a.reshape(1, -1), torch.from_numpy(K).float()), self.a.reshape(-1, 1))

        # reg_term = reg_coeff * aKa
        # obj_krr = f_minus_y_sqr + reg_term
        # obj_krr = f_minus_y_sqr # no regularization
        loss = torch.nn.MSELoss()
        obj_krr = loss(f_val.squeeze().float(), y_th)

        # gradient step
        if opt_auto is None:
            self.a.grad.zero_()  # zero gradient
            obj_krr.backward()  # compute grad
            self.a.data = self.a.data - step_size * self.a.grad.detach()  # gradient step
            self.a.grad.zero_()  # zero gradient
        else: # simple optimization for pth: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            opt_auto.zero_grad()
            obj_krr.backward()  # compute grad
            opt_auto.step()

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
# not used yet
def apply_gd(x, step_size=0.1):
    xnew = x - step_size * x.grad.detach()
    x.grad.data.zero_()
    return xnew

def plot_sol_rkhs(f_th, data_th):
    import matplotlib.pyplot as plt
    x_grid = torch.linspace(torch.min(data_th), torch.max(data_th), 100).reshape(-1, 1)
    f_val_plot  = f_th(x_grid)
    plt.plot(x_grid.detach().numpy(),
             f_val_plot.detach().numpy().reshape(-1,1),
             c='r')  # plot the uniform weights interpolant
