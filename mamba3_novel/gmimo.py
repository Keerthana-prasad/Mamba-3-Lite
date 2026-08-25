"""
Grouped-MIMO (gMIMO) State Space Module for Mamba-3
Implements novel parameter sharing in multi-input multi-output state space dynamics.
Enables linear models to scale sequence rank R with sub-linear parameter growth.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import Mamba3NovelConfig
from .ops import compute_complex_rope_angles, constrained_lni_ssm_recurrence
from .lni import ConstrainedLearnableNeuralIntegration


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class GroupedMIMOSSM(nn.Module):
    """
    Grouped-MIMO (gMIMO) State Space Layer
    
    Partitions n_heads into G_mimo parameter-sharing groups.
    Instead of scaling projection parameters linearly with (n_heads * N * R),
    gMIMO shares rank-R projection maps across heads within each group:
      Parameters: O(G_mimo * N * R + n_heads * N) << O(n_heads * N * R)
    """
    def __init__(self, config: Mamba3NovelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.expand * config.d_model
        self.d_state = config.d_state # N
        self.head_dim = config.headdim # P
        self.num_heads = self.d_inner // self.head_dim
        self.mimo_rank = config.mimo_rank # R
        
        # Grouped-MIMO configuration
        self.use_gmimo = config.use_gmimo and (self.mimo_rank > 1)
        if self.use_gmimo:
            self.gmimo_groups = min(config.gmimo_groups, self.num_heads)
            assert self.num_heads % self.gmimo_groups == 0, \
                f"num_heads ({self.num_heads}) must be divisible by gmimo_groups ({self.gmimo_groups})"
            self.heads_per_group = self.num_heads // self.gmimo_groups
        else:
            self.gmimo_groups = self.num_heads
            self.heads_per_group = 1
            
        self.use_rope = config.use_rope
        self.use_bc_bias = config.use_bc_bias
        self.use_bc_norm = config.use_bc_norm
        self.use_lni = config.use_lni
        
        # 1. Delta_t (time-step) parameterization
        self.dt_proj = nn.Linear(self.d_inner, self.num_heads, bias=True)
        dt_init_std = 2.0 / math.sqrt(self.d_inner)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias = nn.Parameter(inv_dt)
        
        # 2. Continuous State Matrix A (log-spaced positive parameterization: A = -exp(log_A))
        A = torch.arange(1, self.head_dim + 1, dtype=torch.float32).unsqueeze(0).repeat(self.num_heads, 1)
        self.log_A_real = nn.Parameter(torch.log(A))
        
        # 3. Frequency parameter Theta for Complex RoPE
        if self.use_rope:
            self.theta_proj = nn.Linear(self.d_inner, self.num_heads * (self.d_state // 2), bias=False)
            
        # 4. Grouped-MIMO B and C Projections
        # Input projects to (gmimo_groups * d_state * mimo_rank)
        proj_dim = self.gmimo_groups * self.d_state * self.mimo_rank
        self.b_proj = nn.Linear(self.d_inner, proj_dim, bias=False)
        self.c_proj = nn.Linear(self.d_inner, proj_dim, bias=False)
        
        # 5. BCNorm
        if self.use_bc_norm:
            self.b_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
            self.c_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
            
        # 6. Channel-wise, Head-specific Learnable Biases b_B and b_C (init 1.0)
        if self.use_bc_bias:
            self.b_bias = nn.Parameter(torch.ones(self.num_heads, self.head_dim, 1))
            self.c_bias = nn.Parameter(torch.ones(self.num_heads, self.head_dim, 1))
            
        # 7. Constrained Learnable Neural Integration (LNI)
        if self.use_lni:
            self.lni = ConstrainedLearnableNeuralIntegration(config, self.num_heads)
            
        # 8. Output projection (D_inner -> D_model)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(
        self,
        u: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u: Input activations (B, T, D_inner)
            h_prev: Optional recurrent state
        Returns:
            out: (B, T, D_model)
            h_last: Final state
        """
        B_sz, T, E = u.shape
        
        # Reshape input to (B, T, num_heads, head_dim)
        x = u.view(B_sz, T, self.num_heads, self.head_dim)
        
        # Compute dt: (B, T, num_heads, 1)
        dt = F.softplus(self.dt_proj(u)).unsqueeze(-1) # (B, T, num_heads, 1)
        
        # Continuous A: (num_heads, head_dim, 1) -> negative
        A = -torch.exp(self.log_A_real).unsqueeze(-1)
        
        # Compute Integration operators (LNI or Trapezoidal)
        if self.use_lni:
            alpha, beta, gamma = self.lni(x, dt, A)
        else:
            # Standard Exponential-Trapezoidal baseline
            dtA = dt.unsqueeze(-2) * A.unsqueeze(0).unsqueeze(0) # (B, T, num_heads, head_dim, 1)
            alpha = torch.exp(dtA)
            lam = 0.5 # midpoint trapezoidal
            beta = (1.0 - lam) * dt.unsqueeze(-2) * alpha
            gamma = lam * dt.unsqueeze(-2)
            
        # Compute Complex RoPE angles
        if self.use_rope:
            theta = self.theta_proj(u).view(B_sz, T, self.num_heads, self.d_state // 2)
            phi = compute_complex_rope_angles(theta, dt)
        else:
            phi = None
            
        # Compute Grouped-MIMO B and C
        # b_proj: (B, T, gmimo_groups * d_state * mimo_rank)
        B_raw = self.b_proj(u).view(B_sz, T, self.gmimo_groups, self.d_state, self.mimo_rank)
        C_raw = self.c_proj(u).view(B_sz, T, self.gmimo_groups, self.d_state, self.mimo_rank)
        
        # Apply BCNorm across state dimension N
        if self.use_bc_norm:
            # Reshape to (B*T*G*R, N) for RMSNorm
            B_norm = self.b_norm(B_raw.permute(0, 1, 2, 4, 3).contiguous().view(-1, self.d_state))
            B_norm = B_norm.view(B_sz, T, self.gmimo_groups, self.mimo_rank, self.d_state).permute(0, 1, 2, 4, 3)
            
            C_norm = self.c_norm(C_raw.permute(0, 1, 2, 4, 3).contiguous().view(-1, self.d_state))
            C_norm = C_norm.view(B_sz, T, self.gmimo_groups, self.mimo_rank, self.d_state).permute(0, 1, 2, 4, 3)
        else:
            B_norm = B_raw
            C_norm = C_raw
            
        # Expand Grouped Projections to all heads:
        # (B, T, G_mimo, N, R) -> (B, T, G_mimo, heads_per_group, N, R) -> (B, T, num_heads, N, R)
        if self.use_gmimo and (self.heads_per_group > 1):
            B_heads = B_norm.unsqueeze(3).expand(-1, -1, -1, self.heads_per_group, -1, -1)
            B_heads = B_heads.contiguous().view(B_sz, T, self.num_heads, self.d_state, self.mimo_rank)
            
            C_heads = C_norm.unsqueeze(3).expand(-1, -1, -1, self.heads_per_group, -1, -1)
            C_heads = C_heads.contiguous().view(B_sz, T, self.num_heads, self.d_state, self.mimo_rank)
        else:
            B_heads = B_norm
            C_heads = C_norm
            
        # Apply Channel-wise learnable biases b_B and b_C (post-normalization)
        if self.use_bc_bias:
            # b_bias is (num_heads, head_dim, 1)
            # We modulate the head activations via b_bias
            x = x * self.b_bias.unsqueeze(0).unsqueeze(0).squeeze(-1)
            
        # Run SSM Recurrence
        y, h_last = constrained_lni_ssm_recurrence(
            x=x,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            B=B_heads,
            C=C_heads,
            phi=phi,
            h_prev=h_prev,
            use_rope=self.use_rope,
        ) # y is (B, T, num_heads, head_dim)
        
        # Apply Channel-wise learnable output bias c_bias
        if self.use_bc_bias:
            y = y * self.c_bias.unsqueeze(0).unsqueeze(0).squeeze(-1)
        
        # Reshape to (B, T, D_inner) and project to D_model
        y_flat = y.contiguous().view(B_sz, T, self.d_inner)
        out = self.out_proj(y_flat)
        return out, h_last
