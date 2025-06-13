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
    x_u_train = data["x_train"]
    u_train = data["u_train"]
    x_test = data["x_test"]
    u_test = data["u_test"]
    return x_u_train, u_train, x_test, u_test


class PCDKL:
    def __init__(self, cfg):
        self.lamda = cfg['lamda']
        self.tau = cfg['tau']
        self.lr = cfg['lr']
        self.epoch = cfg['epoch']

        # data
        self.x_train = torch.tensor(cfg['x_train'], dtype=torch.float64)
        self.u_train = torch.tensor(cfg['u_train'], dtype=torch.float64)

        # define the learning parameter
        self.log_ls = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        self.feature_extractor = FeatureExtractor(cfg['layers']).double()

        # construct the GP for u
        self.kernel_obj = KernelRBF(jitter=1e-5)
        self.gpr = GPR(train_X=self.x_train, train_y=self.u_train, kernel_obj=self.kernel_obj,
                       log_ls=self.log_ls, tau=self.tau, feature_map=self.feature_extractor)

        # the optimizer
        self.params = [self.log_ls] + list(self.feature_extractor.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def train(self):
        self.feature_extractor.train()
        losses = []
        for i in range(self.epoch):
            self.optimizer.zero_grad()
            loss1 = -1 * self.gpr.log_marginal_likelihood()
            loss2 = self.feature_extractor.l2_regularization()
            loss = loss1 + loss2 * self.lamda
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()

            print(f'Iter {i + 1}/{self.epoch} - Loss: {loss.item():.3f} | loss1: {loss1.item():.3f}, loss2: {loss2.item():.3f}')

        return losses

    def prediction(self, x_test):
        self.feature_extractor.eval()
        u_mean, u_std = self.gpr.predict(x_test)

        return u_mean.detach().numpy(), u_std.detach().numpy()


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 现在 base_dir 是 /code
    dataset_filename = os.path.join(base_dir, "PCDKL/FunctionRegression/dataset0.01.mat")
    x_train, u_train, x_test, u_test = load_data(dataset_filename)
    tau = float(dataset_filename.split('dataset')[-1].split('.mat')[0])

    layers = [1, 20, 20, 20, 20]

    cfg = {
        'layers': layers,
        'lr': 5e-3,
        'epoch': 20,
        'x_train': x_train,
        'u_train': u_train,
        'tau': tau,
        'lamda': 0.0
    }

    model = PCDKL(cfg)

    losses = model.train()

    u_mean, u_std = model.prediction(torch.from_numpy(x_test))

    # # compute the RL2E
    # RL2E = metrics.RL2E(u_mean, u_test) * 100

    # plot the results
    title = f'PCDKL'
    output_result_path = "/results/Function_R_result"+str(tau) + ".jpg"
    plot_tools.plot1d(x_train, u_train, x_test, u_test, u_mean, u_std,
                      xlim=[-1, 1], ylim=[-3, 3], xlabel="x", ylabel="u", title=title, save_path=output_result_path)

    output_loss_path = "/results/Function_R_loss"+str(tau) + ".jpg"
    plot_tools.plot_loss(losses, xlabel="", ylabel="", title="", xlim=[-1, 20], xticks=[0, 5, 10, 15, 20], save_path=output_loss_path)