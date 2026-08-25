"""
Mamba-3: Improved Sequence Modeling using State Space Principles
PyTorch Implementation of Mamba-3 (SISO & MIMO, Exponential-Trapezoidal Discretization, Complex-Valued RoPE, BCNorm, Biases)
"""

from .config import Mamba3Config
from .block import Mamba3Block
from .model import Mamba3LMHeadModel, RMSNorm, SwiGLU
from .dataset import WikiTextDataset, get_dataloaders
from .metrics import compute_metrics, evaluate_model

__all__ = [
    "Mamba3Config",
    "Mamba3Block",
    "Mamba3LMHeadModel",
    "RMSNorm",
    "SwiGLU",
    "WikiTextDataset",
    "get_dataloaders",
    "compute_metrics",
    "evaluate_model",
]
