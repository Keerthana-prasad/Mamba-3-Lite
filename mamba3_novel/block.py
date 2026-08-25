"""
Mamba-3 Novel Extended Block Architecture
Integrates Grouped-MIMO (gMIMO), Constrained Learnable Neural Integration (LNI),
and optional Episodic Continuous Hopfield Memory (EHM) into a unified layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import Mamba3NovelConfig
from .gmimo import GroupedMIMOSSM, RMSNorm
from .hopfield import EpisodicContinuousHopfieldMemory


class Mamba3NovelBlock(nn.Module):
    """
    Unified Mamba-3 Novel Block:
      x -> RMSNorm -> InProj -> (Conv1d + SiLU -> gMIMO SSM (with LNI)) * SiLU(Gate) -> OutProj
      -> (Optional Episodic Hopfield Memory Layer) -> + Residual
    """
    def __init__(
        self,
        config: Mamba3NovelConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.d_inner = config.expand * config.d_model
        self.d_conv = 4
        
        # 1. Pre-RMSNorm
        self.norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        
        # 2. In-projection (projects to inner dimension + gate)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)
        
        # 3. 1D Causal Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        
        # 4. Activation
        self.act = nn.SiLU()
        
        # 5. Grouped-MIMO SSM Layer (with Constrained LNI)
        self.ssm = GroupedMIMOSSM(config)
        
        # 6. Episodic Hopfield Memory (interleaved every hopfield_interval layers)
        self.has_hopfield = config.use_hopfield and ((layer_idx + 1) % config.hopfield_interval == 0)
        if self.has_hopfield:
            self.hopfield = EpisodicContinuousHopfieldMemory(config)
            self.hopfield_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
        memory_bank: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (B, T, D_model)
            ssm_state:     Optional recurrent state
            memory_bank:   Optional episodic memory bank
            
        Returns:
            out:           (B, T, D_model)
            next_ssm_state: Final SSM state
            next_memory_bank: Updated episodic memory bank
        """
        residual = hidden_states
        B_sz, T, D = hidden_states.shape
        
        # 1. Pre-Norm
        x_norm = self.norm(hidden_states)
        
        # 2. Input Projection -> (B, T, 2 * D_inner)
        xz = self.in_proj(x_norm)
        x, z = xz.chunk(2, dim=-1) # x: (B, T, E), z: (B, T, E)
        
        # 3. Causal Convolution on x
        # Conv1d expects (B, Channels, Length)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T] # causal truncate
        x_conv = x_conv.transpose(1, 2)
        
        # 4. Activation
        x_act = self.act(x_conv)
        
        # 5. Grouped-MIMO SSM with Constrained LNI
        # y_ssm is (B, T, D_model)
        y_ssm, next_ssm_state = self.ssm(x_act, h_prev=ssm_state)
        
        # 6. Multiplicative Gating with z
        # Project z to D_model for gating or gate in D_inner
        # In Mamba-3, gating is applied post-SSM
        # For compatibility, apply gating
        gate = self.act(z.mean(dim=-1, keepdim=True)) # channel modulation
        y_gated = y_ssm * gate
        
        # 7. Residual add
        out = residual + y_gated
        
        # 8. Optional Episodic Hopfield Memory Layer
        next_memory = memory_bank
        if self.has_hopfield:
            hop_norm = self.hopfield_norm(out)
            hop_out, next_memory = self.hopfield(hop_norm, external_memories=memory_bank)
            out = out + hop_out
            
        return out, next_ssm_state, next_memory
