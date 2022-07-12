# simulate the LDS for multiple steps, with stochastic noise

# append parent dir to sys path
import argparse
import os
import pickle
import random
import sys

import numpy as np

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

# posisble non optimized input
import matplotlib.pyplot as plt
from opt_util import *
import torch


if __name__ == '__main__':
    kernel_gamma = 1.0  # kernel parameter
    this_kernel = rbf_kernel

    n_x = 10  # num samples
    n_dim = 2  # dim of each point
    # a = np.ones(n_x)/n_x # uniform weights

    data = np.random.uniform(-1, 1, size=[n_x, n_dim])
    data_th = 2 * np.pi * torch.from_numpy(data[:, 0])
    target_th = torch.sin(data_th) + 0.2 * torch.rand_like(data_th)

    # initialize
    f_th = rkhsFun(this_kernel, kernel_gamma, is_torch=True, x= data_th ,n_x = n_x, n_dim=n_dim)  # create the RKHS function

    plt.close('all')
    for iter_gd in range(500):
        id_rnd = random.randint(0, n_x-1)
        f_th.gd_step(x_th=data_th[id_rnd], y_th=target_th, step_size=0.001, reg_coeff=0.05)
        if iter_gd % 10 == 0:
            plt.figure()
            plot_fitting(f_th, data_th)
            plt.scatter(x=data_th, y=target_th)
            plt.title("itertion:" + str(iter_gd))
            plt.show()

    # todo:
    '''
    the issue is that
    f_th should store the expansion points since in 
    src/opt_util.py:44
    I used usual K^T alpha expression to evaluate kernel functions
    '''