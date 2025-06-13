import torch

# define the kernel class
class KernelRBF:
    def __init__(self, jitter=1e-5):
        self.jitter = jitter

    def cross(self, X1, X2, ls):
        s1 = (X1 ** 2).sum(1).view((-1, 1))
        s2 = (X2 ** 2).sum(1).view((-1, 1))
        K = s1 - 2 * X1 @ X2.T + s2.T
        K = torch.exp(-1.0 * K / ls)
        return K

    def matrix(self, X, ls):
        K = self.cross(X, X, ls)
        K = K + self.jitter * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
        return K