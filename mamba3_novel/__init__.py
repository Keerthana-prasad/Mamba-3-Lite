"""
Mamba-3 Novel Extended Architecture Package
Includes:
- Grouped-MIMO (gMIMO) State Space Models
- Constrained Learnable Neural Integration (LNI)
- Episodic Continuous Hopfield Memory Hybrid
"""

from .config import Mamba3NovelConfig
from .lni import ConstrainedLearnableNeuralIntegration
from .gmimo import GroupedMIMOSSM
from .hopfield import EpisodicContinuousHopfieldMemory
from .block import Mamba3NovelBlock
from .model import Mamba3NovelLMHeadModel, SwiGLU

__all__ = [
    "Mamba3NovelConfig",
    "ConstrainedLearnableNeuralIntegration",
    "GroupedMIMOSSM",
    "EpisodicContinuousHopfieldMemory",
    "Mamba3NovelBlock",
    "Mamba3NovelLMHeadModel",
    "SwiGLU",
]
