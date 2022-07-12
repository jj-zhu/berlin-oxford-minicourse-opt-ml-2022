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
    sgd = 1
    n_x = 10  # num samples
    dim_x = 1

    data = np.random.uniform(-1, 1, size=[n_x, dim_x+1])
    X = 2 * np.pi * torch.from_numpy(data[:, 0]).float()
    Y = (torch.sin(X) + 0.2 * torch.rand_like(X, requires_grad=False)).reshape(-1, 1)

    f_th = ml_model(model_class = 'rf', dim_x=dim_x, loss = torch.nn.MSELoss())
    optimizer = torch.optim.SGD(f_th.parameters, lr=0.1)

    plt.close('all')
    for iter_gd in range(5000):
        # opt step
        if sgd:
            id_rnd = random.randint(0, n_x - 1)
            f_th.train_step(x_th=X[id_rnd], y_th=Y[id_rnd].reshape(-1,1), optimization=optimizer)
        else:
            f_th.train_step(x_th=X, y_th=Y, optimization=optimizer)

        # for plotting
        if iter_gd % 50 == 0:
            print('iteration of GD:', iter_gd)
            plt.figure()
            plot_fitting(f_th, X)
            plt.scatter(x=X, y=Y)
            plt.title("itertion:" + str(iter_gd))
            plt.savefig('fig/nn_gditer_'+str(iter_gd)+'.png')