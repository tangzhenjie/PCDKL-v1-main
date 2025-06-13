import torch
import numpy as np
import scipy.io as sio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dklearn import FeatureExtractor, GPRCluster, KernelRBF
from utils import metrics, plot_tools


def load_data(dataset_filename):
    data = sio.loadmat(dataset_filename)
    x_test, u1_test, u2_test, u3_test = data["x_test"], data["u1_test"], data["u2_test"], data["u3_test"]
    x_u1_train, u1_train, x_u2_train, u2_train, x_u3_train, u3_train = data["x_u1_train"], data["u1_train"], data["x_u2_train"], data["u2_train"], data["x_u3_train"], data["u3_train"]
    x_f_train, f1_train, f2_train, f3_train = data["x_f_train"], data["f1_train"], data["f2_train"], data["f3_train"]
    return (x_u1_train, u1_train, x_u2_train, u2_train, x_u3_train, u3_train,
            x_f_train, f1_train, f2_train, f3_train, x_test, u1_test, u2_test, u3_test)


def pde_fn(x, u1, u2, u3, a, b):
    u1_x = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
    u2_x = torch.autograd.grad(u2.sum(), x, create_graph=True)[0]
    u3_x = torch.autograd.grad(u3.sum(), x, create_graph=True)[0]

    f1 = u1_x - a * u2 * u3
    f2 = u2_x - b * u1 * u3
    f3 = u3_x + (a + b) * u1 * u2

    return f1, f2, f3


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
        self.x_u1_train = torch.tensor(cfg['x_u1_train'], dtype=torch.float64)
        self.u1_train = torch.tensor(cfg['u1_train'], dtype=torch.float64)
        self.f1_train = torch.tensor(cfg['f1_train'], dtype=torch.float64)

        self.x_u2_train = torch.tensor(cfg['x_u2_train'], dtype=torch.float64)
        self.u2_train = torch.tensor(cfg['u2_train'], dtype=torch.float64)
        self.f2_train = torch.tensor(cfg['f2_train'], dtype=torch.float64)

        self.x_u3_train = torch.tensor(cfg['x_u3_train'], dtype=torch.float64)
        self.u3_train = torch.tensor(cfg['u3_train'], dtype=torch.float64)
        self.f3_train = torch.tensor(cfg['f3_train'], dtype=torch.float64)

        self.x_f_train = torch.tensor(cfg['x_f_train'], dtype=torch.float64, requires_grad=True)

        # define the learning parameter
        self.log_ls_u1 = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        self.log_ls_u2 = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        self.log_ls_u3 = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        self.feature_extractor_u = FeatureExtractor(cfg['layers']).double()
        self.a = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        self.b = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        # construct the GP for u
        self.kernel_obj = KernelRBF(jitter=1e-5)
        self.gpr_u1 = GPRCluster(train_X=self.x_u1_train, train_y=self.u1_train, kernel_obj=self.kernel_obj,
                       log_ls=self.log_ls_u1, tau=self.tau, feature_map=self.feature_extractor_u, out_index=0)
        self.gpr_u2 = GPRCluster(train_X=self.x_u2_train, train_y=self.u2_train, kernel_obj=self.kernel_obj,
                          log_ls=self.log_ls_u2, tau=self.tau, feature_map=self.feature_extractor_u, out_index=1)
        self.gpr_u3 = GPRCluster(train_X=self.x_u3_train, train_y=self.u3_train, kernel_obj=self.kernel_obj,
                          log_ls=self.log_ls_u3, tau=self.tau, feature_map=self.feature_extractor_u, out_index=2)

        # the optimizer
        self.params = ([self.log_ls_u1, self.log_ls_u2, self.log_ls_u3, self.a, self.b]
                       + list(self.feature_extractor_u.parameters())
                       )

        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def train(self):
        self.feature_extractor_u.train()
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

            print(self.a.item())
            print(self.b.item())

        return losses

    def get_ELBO(self):
        loss_u1 = -1 * self.gpr_u1.log_marginal_likelihood()
        loss_u2 = -1 * self.gpr_u2.log_marginal_likelihood()
        loss_u3 = -1 * self.gpr_u3.log_marginal_likelihood()

        loss1 = loss_u1 + loss_u2 + loss_u3

        # compute the Posterior Regularization Loss
        u1_mean, u1_std = self.gpr_u1.predict(self.x_f_train)
        u2_mean, u2_std = self.gpr_u2.predict(self.x_f_train)
        u3_mean, u3_std = self.gpr_u3.predict(self.x_f_train)

        loss2 = 0
        for i in range(self.num_samples):
            eta1 = torch.empty_like(u1_mean).normal_()
            u1 = u1_mean + u1_std * eta1
            eta2 = torch.empty_like(u2_mean).normal_()
            u2 = u2_mean + u2_std * eta2
            eta3 = torch.empty_like(u3_mean).normal_()
            u3 = u3_mean + u3_std * eta3

            f1, f2, f3 = self.pde_fn(self.x_f_train, u1, u2, u3, self.a, self.b)

            N_f = self.x_f_train.shape[0]
            kernel_zero = torch.eye(N_f, dtype=self.x_f_train.dtype) * self.tau ** 2
            loss_f1 = 0.5 * N_f * np.log(2.0 * np.pi) + 0.5 * torch.logdet(kernel_zero) \
                    + 0.5 * (self.f1_train - f1).T @ torch.linalg.solve(kernel_zero, (self.f1_train - f1))

            loss_f2 = 0.5 * N_f * np.log(2.0 * np.pi) + 0.5 * torch.logdet(kernel_zero) \
                      + 0.5 * (self.f2_train - f2).T @ torch.linalg.solve(kernel_zero, (self.f2_train - f2))

            loss_f3 = 0.5 * N_f * np.log(2.0 * np.pi) + 0.5 * torch.logdet(kernel_zero) \
                      + 0.5 * (self.f3_train - f3).T @ torch.linalg.solve(kernel_zero, (self.f3_train - f3))

            loss2 = loss2 + loss_f1 + loss_f2 + loss_f3

        loss2 = loss2 / self.num_samples

        loss3 = self.feature_extractor_u.l2_regularization()

        loss = loss1 + loss2 * self.alpha + loss3 * self.beta

        return loss, loss1, loss2, loss3

    def prediction_u(self, x_test, type="u1"):
        x_test = torch.tensor(x_test, dtype=torch.float64)
        self.feature_extractor_u.eval()
        u_mean, u_std = None, None
        if type == "u1":
            u_mean, u_std = self.gpr_u1.predict(x_test)
        elif type == "u2":
            u_mean, u_std = self.gpr_u2.predict(x_test)
        elif type == "u3":
            u_mean, u_std = self.gpr_u3.predict(x_test)
        else:
            raise Exception("Error type value")

        return u_mean.detach().numpy(), u_std.detach().numpy()


if __name__ == '__main__':
    # dataset_filename = "./dataset0.1.mat"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 现在 base_dir 是 /code
    dataset_filename = os.path.join(base_dir, "PCDKL/K_O_System_Inverse/dataset0.01.mat")
    (x_u1_train, u1_train, x_u2_train, u2_train, x_u3_train, u3_train,
            x_f_train, f1_train, f2_train, f3_train, x_test, u1_test, u2_test, u3_test) = load_data(dataset_filename)
    tau = float(dataset_filename.split('dataset')[-1].split('.mat')[0])

    layers = [1, 20, 20, 20, 60]

    cfg = {
        'layers': layers,
        'lr': 5e-3,
        'epoch': 5000,
        'x_u1_train': x_u1_train,
        'u1_train': u1_train,
        'x_u2_train': x_u2_train,
        'u2_train': u2_train,
        'x_u3_train': x_u3_train,
        'u3_train': u3_train,
        'x_f_train': x_f_train,
        'f1_train': f1_train,
        'f2_train': f2_train,
        'f3_train': f3_train,
        'pde_fn': pde_fn,
        'tau': tau,
        'alpha': 1.0,    # for the Posterior Regularization Loss
        'beta': 0.0,     # l2_regularization
        'num_samples': 10
    }

    model = PCDKL(cfg)

    losses = model.train()

    u1_mean, u1_std = model.prediction_u(x_test, type="u1")
    u2_mean, u2_std = model.prediction_u(x_test, type="u2")
    u3_mean, u3_std = model.prediction_u(x_test, type="u3")

    # # compute the RL2E
    # u1_RL2E = metrics.RL2E(u1_mean, u1_test)

    # u2_RL2E = metrics.RL2E(u2_mean, u2_test)

    # u3_RL2E = metrics.RL2E(u3_mean, u3_test)

    # plot the results
    u1_title = 'PCDKL'
    output_u1_path = "result_u1" + str(tau) + ".jpg"
    plot_tools.plot1d(x_u1_train, u1_train, x_test, u1_test, u1_mean, u1_std,
                      xlim=[0.0, 10.0], ylim=[0.0, 1.4], xlabel="", ylabel="", title=u1_title, save_path=output_u1_path)

    u2_title = 'PCDKL'
    output_u2_path = "result_u2" + str(tau) + ".jpg"
    plot_tools.plot1d(x_u2_train, u2_train, x_test, u2_test, u2_mean, u2_std,
                      xlim=[0.0, 10.0], ylim=[-1.5, 1.5], xlabel="", ylabel="", title=u2_title, save_path=output_u2_path)

    u3_title = 'PCDKL'
    output_u3_path = "result_u3" + str(tau) + ".jpg"
    plot_tools.plot1d(x_u3_train, u3_train, x_test, u3_test, u3_mean, u3_std,
                      xlim=[0.0, 10.0], ylim=[-1.5, 1.5], xlabel="", ylabel="", title=u3_title, save_path=output_u3_path)

    output_loss_path = "loss" + str(tau) + ".jpg"
    plot_tools.plot_loss(losses, xlabel="", ylabel="", title="", save_path=output_loss_path)