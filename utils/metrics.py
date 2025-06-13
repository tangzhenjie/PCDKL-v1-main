import numpy as np


def RL2E(predictions, true):
    """
    Computes the relative L2 error.

        Args:
            predictions (array): The predictions, with shape [None, dim].
            true (array): The true values, with shape [None, dim].

        Returns:
            rl2e (scalar): The relative L2 error; a scalar
    """
    return np.sqrt(np.sum((predictions - true) ** 2)) / np.sqrt(np.sum(true ** 2))