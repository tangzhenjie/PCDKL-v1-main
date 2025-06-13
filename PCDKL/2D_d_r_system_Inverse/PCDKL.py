import torch
import numpy as np
import scipy.io as sio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dklearn import FeatureExtractor, GPR, KernelRBF
from utils import metrics, plot_tools


def load_data(dataset_filename):
    data = sio.loadmat(dataset_filename)
    x_test, u_test, f_test = data["x_test"], data["u_test"], data["f_test"]
    x_u_train, u_train = data["x_u_train"], data["u_train"]
    x_f_train, f_train = data["x_f_train"], data["f_train"]
    return x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test


def pde_fn(x, u, k):
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    f = torch.unsqueeze(0.01 * (u_xx[:, 1] + u_xx[:, 0]), dim=-1) + k * u ** 2

    return f


class PCDKL:
    def __init__(self, cfg):
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']
        self.pde_fn = cfg['pde_fn']
        self.tau = cfg['tau']
        self.lr = cfg['lr']
        self.epoch = cfg['epoch']
        self.num_samples = cfg['num_samples']

        # data
        self.x_u_train = torch.tensor(cfg['x_u_train'], dtype=torch.float64)
        self.u_train = torch.tensor(cfg['u_train'], dtype=torch.float64)

        self.x_f_train = torch.tensor(cfg['x_f_train'], dtype=torch.float64, requires_grad=True)
        self.f_train = torch.tensor(cfg['f_train'], dtype=torch.float64)

        # define the learning parameter
        self.log_ls = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        self.feature_extractor = FeatureExtractor(cfg['layers']).double()
        self.k = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        # construct the GP for u
        self.kernel_obj = KernelRBF(jitter=1e-5)
        self.gpr = GPR(train_X=self.x_u_train, train_y=self.u_train, kernel_obj=self.kernel_obj,
                       log_ls=self.log_ls, tau=self.tau, feature_map=self.feature_extractor)

        # the optimizer
        self.params = [self.log_ls, self.k] + list(self.feature_extractor.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def train(self):
        self.feature_extractor.train()
        losses = []
        for i in range(self.epoch):
            self.optimizer.zero_grad()
            loss, loss1, loss2, loss3 = self.get_ELBO()
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.optimizer.step()

            print(f'Iter {i + 1}/{self.epoch} - Loss: {loss.item():.3f} | loss1: {loss1.item():.3f}, '
                  f'loss2: {loss2.item():.3f}, loss3: {loss3.item():.3f}')
            print(self.k)

        return losses

    def get_ELBO(self):
        loss1 = -1 * self.gpr.log_marginal_likelihood()

        # compute the Posterior Regularization Loss
        u_mean, u_std = self.gpr.predict(self.x_f_train)
        loss2 = 0
        for i in range(self.num_samples):
            eta = torch.empty_like(u_mean).normal_()
            u = u_mean + u_std * eta
            f = self.pde_fn(self.x_f_train, u, self.k)

            N_f = self.x_f_train.shape[0]
            kernel_zero = torch.eye(N_f, dtype=self.x_f_train.dtype) * self.tau ** 2
            loss2 = loss2 + 0.5 * N_f * np.log(2.0 * np.pi) + 0.5 * torch.logdet(kernel_zero) \
                    + 0.5 * (self.f_train - f).T @ torch.linalg.solve(kernel_zero, (self.f_train - f))

        loss2 = loss2 / self.num_samples

        loss3 = self.feature_extractor.l2_regularization()

        loss = loss1 + loss2 * self.alpha + loss3 * self.beta

        return loss, loss1, loss2, loss3

    def prediction_u(self, x_test):
        x_test = torch.tensor(x_test, dtype=torch.float64)
        self.feature_extractor.eval()
        u_mean, u_std = self.gpr.predict(x_test)

        return u_mean.detach().numpy(), u_std.detach().numpy()


if __name__ == '__main__':
    # dataset_filename = "./dataset0.1.mat"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 现在 base_dir 是 /code
    dataset_filename = os.path.join(base_dir, "PCDKL/2D_d_r_system_Inverse/dataset0.01.mat")
    x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test = load_data(dataset_filename)
    tau = float(dataset_filename.split('dataset')[-1].split('.mat')[0])

    layers = [2, 20, 20, 20, 20]

    cfg = {
        'layers': layers,
        'lr': 5e-3,
        'epoch': 100,
        'x_u_train': x_u_train,
        'u_train': u_train,
        'x_f_train': x_f_train,
        'f_train': f_train,
        'pde_fn': pde_fn,
        'tau': tau,
        'alpha': 1.0,    # for the Posterior Regularization Loss
        'beta': 0.0,     # l2_regularization
        'num_samples': 10
    }

    model = PCDKL(cfg)

    losses = model.train()

    u_mean, u_std = model.prediction_u(x_test)

    # compute the RL2E
    # u_RL2E = metrics.RL2E(u_mean, u_test)

    # plot the results
    u_title = 'PCDKL'
    error_u = np.abs(u_mean - u_test).reshape(50, 50)
    std_u_2 = (u_std * 2).reshape(50, 50)
    u_mean = u_mean.reshape(50, 50)
    x_co = x_test[:, 1].reshape(50, 50)
    y_co = x_test[:, 0].reshape(50, 50)

    output_mean_path = "/results/mean" + str(tau) + ".jpg"
    plot_tools.plot2d(x_co, y_co, u_mean, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], xticks_num=5, yticks_num=5,
                      title="Mean-" + u_title, bar_ticks=[1.0, 0.5, 0.0, -0.5, -1.0], save_path=output_mean_path)

    output_error_path = "/results/error" + str(tau) + ".jpg"
    plot_tools.plot2d(x_co, y_co, error_u, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], xticks_num=5, yticks_num=5,
                      title="Error",  bar_ticks=[0.12, 0.09, 0.06, 0.03], save_path=output_error_path)

    output_2stds_path = "/results/2stds" + str(tau) + ".jpg"
    plot_tools.plot2d(x_co, y_co, std_u_2, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], xticks_num=5, yticks_num=5,
                      title="2stds", bar_ticks=[0.108, 0.096, 0.084, 0.072], save_path=output_2stds_path)

    output_loss_path = "/results/loss" + str(tau) + ".jpg"
    plot_tools.plot_loss(losses, xlabel="", ylabel="", title="", save_path=output_loss_path)