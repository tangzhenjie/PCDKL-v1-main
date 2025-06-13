import torch
import numpy as np


# define the GaussianProcessRegression model
class GPR:
    def __init__(self, train_X, train_y, kernel_obj, log_ls, tau, feature_map=None):
        self.train_X = train_X          # [None, dim]
        self.train_y = train_y          # [None, 1]
        self.kernel_obj = kernel_obj    # the RBF kernel class
        self.log_ls = log_ls            # the parameter of RBF kernel
        self.tau = tau ** 2             # the var of noise level
        self.feature_map = feature_map  # the NN feature map

    def predict(self, test_X):
        if self.feature_map is None:
            test_X_feature = test_X
            train_X_feature = self.train_X
        else:
            test_X_feature = self.feature_map(test_X)
            train_X_feature = self.feature_map(self.train_X)
        ls = torch.exp(self.log_ls)
        Kmn = self.kernel_obj.cross(test_X_feature, train_X_feature, ls)
        Knn = self.kernel_obj.matrix(train_X_feature, ls) \
              + self.tau * torch.eye(self.train_X.shape[0], dtype=self.train_X.dtype)

        pre_mean = Kmn @ torch.linalg.solve(Knn, self.train_y)

        pred_std = 1 - (Kmn * torch.linalg.solve(Knn, Kmn.T).T).sum(1)
        pred_std = torch.sqrt(pred_std).view((-1, 1))

        return pre_mean, pred_std

    def log_marginal_likelihood(self):
        if self.feature_map is None:
            train_X_feature = self.train_X
        else:
            train_X_feature = self.feature_map(self.train_X)
        N = self.train_X.shape[0]
        Knn = self.kernel_obj.matrix(train_X_feature, torch.exp(self.log_ls)) \
              + self.tau * torch.eye(N, dtype=self.train_X.dtype)

        L = -0.5 * (N * np.log(2.0 * np.pi) + torch.logdet(Knn)) \
            - 0.5 * self.train_y.T @ torch.linalg.solve(Knn, self.train_y)

        return L

    def test(self, test_X):
        if self.feature_map is None:
            test_X_feature = test_X
            train_X_feature = self.train_X
        else:
            test_X_feature = self.feature_map(test_X)
            train_X_feature = self.feature_map(self.train_X)
        ls = torch.exp(self.log_ls)
        Kmn = self.kernel_obj.cross(test_X_feature, train_X_feature, ls)
        Knn = self.kernel_obj.matrix(train_X_feature, ls) \
              + self.tau * torch.eye(self.train_X.shape[0], dtype=self.train_X.dtype)

        pre_mean = Kmn @ torch.linalg.solve(Knn, self.train_y)

        Kmm = self.kernel_obj.matrix(test_X_feature, ls)
        pre_cov = Kmm - Kmn @ torch.linalg.solve(Knn, Kmn.T)

        return pre_mean, pre_cov