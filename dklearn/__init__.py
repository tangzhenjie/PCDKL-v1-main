"""Package for deep kernel learning."""

__all__ = [
    "FeatureExtractor",
    "GPR",
    "KernelRBF",
    "GPRCluster"
]

from .feature_map import FeatureExtractor
from .gaussian_process import GPR
from .rbf_kernel import KernelRBF
from .gaussian_mixture import GPRCluster
