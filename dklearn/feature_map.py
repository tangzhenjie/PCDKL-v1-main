import torch
import torch.nn as nn
import torch.nn.init as init


class FeatureExtractor(nn.Module):
    def __init__(self, layers, act=nn.Tanh(), init_method='normal', seed=66):
        super(FeatureExtractor, self).__init__()
        self.layers = layers
        self.fc = nn.ModuleList()
        self.act = act
        for i in range(len(self.layers) - 1):
            self.fc.append(nn.Linear(self.layers[i], self.layers[i + 1]))

        # fixed the initial value
        torch.manual_seed(seed)

        if init_method == 'xavier':
            self.apply(self.init_weights_xavier)
        elif init_method == 'he':
            self.apply(self.init_weights_he)
        elif init_method == 'uniform':
            self.apply(self.init_weights_uniform)
        elif init_method == 'normal':
            self.apply(self.init_weights_normal)
        elif init_method == 'custom':
            self.apply(self.init_weights_custom)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

    def forward(self, X):
        for i in range(len(self.fc) - 1):
            X = self.act(self.fc[i](X))
        X = self.fc[-1](X)
        return X

    def l2_regularization(self):
        l2_reg = 0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        return l2_reg

    def l1_regularization(self):
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))  # Sum of absolute values of parameters
        return l1_reg

    @staticmethod
    def init_weights_xavier(m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    @staticmethod
    def init_weights_he(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='tanh')
            if m.bias is not None:
                init.zeros_(m.bias)

    @staticmethod
    def init_weights_uniform(m):
        if isinstance(m, nn.Linear):
            init.uniform_(m.weight, a=-0.05, b=0.05)
            if m.bias is not None:
                init.zeros_(m.bias)

    @staticmethod
    def init_weights_normal(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0.0, std=0.1)
            if m.bias is not None:
                init.zeros_(m.bias)

    @staticmethod
    def init_weights_custom(m):
        if isinstance(m, nn.Linear):
            init.constant_(m.weight, 0.01)
            if m.bias is not None:
                init.zeros_(m.bias)
