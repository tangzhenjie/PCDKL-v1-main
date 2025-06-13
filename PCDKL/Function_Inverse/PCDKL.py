import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dklearn import FeatureExtractor, GPR, KernelRBF
from utils import metrics, plot_tools


def load_data(dataset_filename):
    data = sio.loadmat(dataset_filename)
    x_test, u_test, f_test, k_test = data["x_test"], data["u_test"], data["f_test"], data["k_test"]
    x_u_train, u_train = data["x_u_train"], data["u_train"]
    x_f_train, f_train = data["x_f_train"], data["f_train"]
    return x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test, k_test


class KFun(nn.Module):
    def __init__(self, act=nn.Tanh()):
        super(KFun, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self.act = act

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        return l2_reg

def pde_fn(x, u, k):
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    return 0.01 * u_xx - k * u


class PCDKL:
    def __init__(self, cfg):
        self.alpha = cfg['alpha']
        self.beta1 = cfg['beta1']
        self.beta2 = cfg['beta2']
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
        self.k_fun = KFun().double()

        # construct the GP for u
        self.kernel_obj = KernelRBF(jitter=1e-5)
        self.gpr = GPR(train_X=self.x_u_train, train_y=self.u_train, kernel_obj=self.kernel_obj,
                       log_ls=self.log_ls, tau=self.tau, feature_map=self.feature_extractor)

        # the optimizer
        self.params = [self.log_ls] + list(self.feature_extractor.parameters()) + list(self.k_fun.parameters())

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def train(self):
        self.feature_extractor.train()
        self.k_fun.train()
        losses = []
        for i in range(self.epoch):
            self.optimizer.zero_grad()
            loss, loss1, loss2, loss3, loss4 = self.get_ELBO()
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
            self.optimizer.step()

            print(f'Iter {i + 1}/{self.epoch} - Loss: {loss.item():.3f} | loss1: {loss1.item():.3f}, '
                  f'loss2: {loss2.item():.3f}, loss3: {loss3.item():.3f}, loss4: {loss4.item():.3f}')

        return losses

    def get_ELBO(self):
        loss1 = -1 * self.gpr.log_marginal_likelihood()

        # compute the Posterior Regularization Loss
        u_mean, u_std = self.gpr.predict(self.x_f_train)
        loss2 = 0
        for i in range(self.num_samples):
            eta = torch.empty_like(u_mean).normal_()
            u = u_mean + u_std * eta
            f = self.pde_fn(self.x_f_train, u, self.k_fun(self.x_f_train))

            N_f = self.x_f_train.shape[0]
            kernel_zero = torch.eye(N_f, dtype=self.x_f_train.dtype) * self.tau ** 2
            loss2 = loss2 + 0.5 * N_f * np.log(2.0 * np.pi) + 0.5 * torch.logdet(kernel_zero) \
                    + 0.5 * (self.f_train - f).T @ torch.linalg.solve(kernel_zero, (self.f_train - f))

        loss2 = loss2 / self.num_samples

        loss3 = self.feature_extractor.l2_regularization()

        loss4 = self.k_fun.l2_regularization()

        loss = loss1 + loss2 * self.alpha + loss3 * self.beta1 + loss4 * self.beta2

        return loss, loss1, loss2, loss3, loss4

    def prediction_u(self, x_test):
        x_test = torch.tensor(x_test, dtype=torch.float64)
        self.feature_extractor.eval()
        self.k_fun.eval()
        u_mean, u_std = self.gpr.predict(x_test)

        return u_mean.detach().numpy(), u_std.detach().numpy()

    def prediction_f(self, x_test, n_samples=1000):
        x_test = torch.tensor(x_test, dtype=torch.float64, requires_grad=True)
        self.feature_extractor.eval()
        self.k_fun.eval()
        u_mean, u_std = self.gpr.predict(x_test)
        f_samples = []
        for i in range(n_samples):
            eta = torch.empty_like(u_mean).normal_()
            u = u_mean + u_std * eta
            f = self.pde_fn(x_test, u, self.k_fun(x_test))

            f_samples.append(f)

        f_samples = torch.cat(f_samples, dim=1)

        f_mean = torch.mean(f_samples, dim=1, keepdim=True).detach().numpy()
        f_std = torch.std(f_samples, dim=1, keepdim=True).detach().numpy()

        return f_mean, f_std

    def prediction_k(self, x_test):
        x_test = torch.tensor(x_test, dtype=torch.float64)
        self.k_fun.eval()

        k_values = self.k_fun(x_test).detach().numpy()

        return k_values




if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 现在 base_dir 是 /code
    dataset_filename = os.path.join(base_dir, "PCDKL/Function_Inverse/dataset0.01.mat")
    x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test, k_test = load_data(dataset_filename)
    tau = float(dataset_filename.split('dataset')[-1].split('.mat')[0])

    layers = [1, 20, 20, 20, 20]

    cfg = {
        'layers': layers,
        'lr': 5e-3,
        'epoch': 200,
        'x_u_train': x_u_train,
        'u_train': u_train,
        'x_f_train': x_f_train,
        'f_train': f_train,
        'pde_fn': pde_fn,
        'tau': tau,
        'alpha': 1.0,    # for the Posterior Regularization Loss
        'beta1': 0.0,    # l2_regularization for kernel
        'beta2': 0.0,    # l2_regularization for k
        'num_samples': 10
    }

    model = PCDKL(cfg)

    losses = model.train()

    u_mean, u_std = model.prediction_u(x_test)
    k_pred = model.prediction_k(x_test)

    # # compute the RL2E
    # u_RL2E = metrics.RL2E(u_mean, u_test)

    # k_RL2E = metrics.RL2E(k_pred, k_test)

    # plot the results
    u_title = 'PCDKL'
    output_result_u_path = "/results/Fun_i_result_u" + str(tau) + ".jpg"
    plot_tools.plot1d(x_u_train, u_train, x_test, u_test, u_mean, u_std,
                      xlim=[0, 1], ylim=[-1.5, 1.5], xlabel="x", ylabel="u", title=u_title, save_path=output_result_u_path)

    k_title = 'k_prediction'
    output_result_k_path = "/results/Fun_i_result_k" + str(tau) + ".jpg"
    plot_tools.plot_predictions(x_test, k_test, k_pred,
                                xlim=[0, 1], ylim=[-1.5, 1.5], xlabel="x", ylabel="k", title=k_title, save_path=output_result_k_path)

    output_loss_path = "/results/Fun_i_loss" + str(tau) + ".jpg"
    plot_tools.plot_loss(losses, xlabel="", ylabel="", title="", save_path=output_loss_path)