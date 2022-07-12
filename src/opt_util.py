from sklearn.kernel_approximation import RBFSampler
import torch
import torch.nn as nn

class ml_model():
    def __init__(self, model_class = 'nn', dim_x=None, loss = None):
        '''
        kernel gamma param
        n_x, number of points for kernel functions
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if loss is None: # must choose torch loss class
            raise NotImplementedError
        else:
            self.loss = loss

        if dim_x is None:
            print("dim of x n_dim can't be None!")
            raise NotImplementedError
        self.n_dim = dim_x

        self.model_class = model_class
        if self.model_class=='nn':
            n_hidden = 64
            self.model = nn.Sequential(nn.Linear(dim_x, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)).to(device)
        elif self.model_class=='rf':
            n_feat = 200
            self.rbf_feature = RBFSampler(gamma=10.0, n_components=n_feat,
                                          random_state=1)  # only support Gaussian RKHS for now
            self.model = nn.Sequential(Flatten(), nn.Linear(n_feat, 1, bias=True))  # random feature. the param of this models are the weights, i.e., decision var
        else:
            raise NotImplementedError
        self.parameters = self.model.parameters()

    def __call__(self, x):
        if self.model_class=='nn':
            return self.model(x)
        elif self.model_class=='rf': # output of random feature models
            x_feat = self.make_feature(x)
            return self.model(torch.from_numpy(x_feat).float())

    def make_feature(self, x):
        x_reshaped = (x.view(x.shape[0], -1))
        x_feat = self.rbf_feature.fit_transform(x_reshaped)  # only transform during evaluation
        return x_feat

    def train_step(self, x_th=None, y_th=None, optimization =None):
        # this function performs 1 step of optimization
        # construct the graph
        f_val = self.__call__(x_th.reshape(-1, 1)) # compute f value evaluated at all data points, math: f_val = K'a
        obj = self.loss(f_val, y_th)

        # gradient step
        if optimization is None:
            raise NotImplementedError
        else: # simple optimization for pth: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            optimization.zero_grad()
            obj.backward()  # compute grad
            optimization.step()

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def plot_fitting(f_th, data_th):
    import matplotlib.pyplot as plt
    x_grid = torch.linspace(torch.min(data_th), torch.max(data_th), 100).reshape(-1, 1)
    f_val_plot  = f_th(x_grid)
    plt.plot(x_grid.detach().numpy(),
             f_val_plot.detach().numpy().reshape(-1,1),
             c='r')  # plot the uniform weights interpolant
