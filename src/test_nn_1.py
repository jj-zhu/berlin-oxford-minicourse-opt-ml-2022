# simulate the LDS for multiple steps, with stochastic noise

# append parent dir to sys path
import argparse
import os
import pickle
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
    n_x = 10  # num samples
    dim_x = 1

    data = np.random.uniform(-1, 1, size=[n_x, dim_x+1])
    data_th = 2 * np.pi * torch.from_numpy(data[:, 0]).float()
    target_th = (torch.sin(data_th) + 0.2 * torch.rand_like(data_th, requires_grad=False)).reshape(-1,1)

    f_th = ml_model(model_class = 'nn', n_dim=dim_x, loss = torch.nn.MSELoss())
    optimizer = torch.optim.SGD(f_th.parameters, lr=0.05)

    plt.close('all')
    for iter_gd in range(5000):
        # opt step
        f_th.train_step(x_th=data_th, y_th=target_th, optimization=optimizer)

        # for plotting
        if iter_gd % 50 == 0:
            print('iteration of GD:', iter_gd)
            plt.figure()
            plot_sol_rkhs(f_th, data_th)
            plt.scatter(x=data_th, y=target_th)
            plt.title("itertion:" + str(iter_gd))
            plt.savefig('fig/nn_gditer_'+str(iter_gd)+'.png')