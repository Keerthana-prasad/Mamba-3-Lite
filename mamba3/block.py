import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from .config import Mamba3Config
from .ops import (
    apply_rotary_emb,
    compute_complex_rope_angles,
    exponential_trapezoidal_ssm_recurrence,
    ssm_step,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if normalizing over dim -2 for MIMO tensors of shape (..., N, R)
        if x.ndim >= 2 and x.shape[-2] == self.weight.shape[0] and x.shape[-1] != self.weight.shape[0]:
            var = torch.mean(x ** 2, dim=-2, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight.unsqueeze(-1)
        else:
            var = torch.mean(x ** 2, dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight


class Mamba3Block(nn.Module):
    """
    Mamba-3 Layer Block.
    Implements:
      - Exponential-Trapezoidal Discretization with data-dependent lambda gate
      - Complex-valued state transitions via Data-Dependent RoPE trick
      - BC Normalization (RMSNorm on B and C)
      - Channel-wise, head-specific learnable B and C biases (init 1.0)
      - Support for SISO (R=1) and MIMO (R>1)
      - O(1) memory recurrent step decoding
    """
    def __init__(self, config: Mamba3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.d_model = config.d_model
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = config.headdim
        self.nheads = self.d_inner // self.headdim
        self.d_state = config.d_state
        self.ngroups = config.ngroups
        self.mimo_rank = config.mimo_rank
        self.is_mimo = (self.mimo_rank > 1)
        
        assert self.d_state % 2 == 0, f"d_state ({self.d_state}) must be even for complex RoPE."
        assert self.d_inner % self.headdim == 0, f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
        assert self.nheads % self.ngroups == 0, f"nheads ({self.nheads}) must be divisible by ngroups ({self.ngroups})"
        
        # 1. In-projections
        self.in_proj_x = nn.Linear(self.d_model, self.d_inner, bias=config.bias)
        self.in_proj_z = nn.Linear(self.d_model, self.d_inner, bias=config.bias)
        
        # MIMO projections for x, z if R > 1 (Appendix C)
        if self.is_mimo:
            self.x_mimo_proj = nn.Parameter(torch.randn(self.headdim, self.mimo_rank) / math.sqrt(self.headdim))
            self.z_mimo_proj = nn.Parameter(torch.randn(self.headdim, self.mimo_rank) / math.sqrt(self.headdim))
            self.out_mimo_proj = nn.Parameter(torch.randn(self.headdim, self.mimo_rank) / math.sqrt(self.mimo_rank))
            
        # B, C projections
        bc_dim = self.ngroups * self.d_state * (self.mimo_rank if self.is_mimo else 1)
        self.in_proj_b = nn.Linear(self.d_model, bc_dim, bias=config.bias)
        self.in_proj_c = nn.Linear(self.d_model, bc_dim, bias=config.bias)
        
        # Delta (dt) projection
        self.in_proj_dt = nn.Linear(self.d_model, self.nheads, bias=True)
        # Initialize dt bias
        dt_init_std = config.dt_max - config.dt_min
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.in_proj_dt.bias.copy_(inv_dt)
            self.in_proj_dt.weight.zero_()
            
        # Lambda projection (for Exponential-Trapezoidal Gate)
        if config.use_trapezoidal:
            self.in_proj_lambda = nn.Linear(self.d_model, self.nheads, bias=True)
            nn.init.zeros_(self.in_proj_lambda.weight)
            nn.init.zeros_(self.in_proj_lambda.bias)
        else:
            self.in_proj_lambda = None
            
        # Theta projection (for Complex RoPE)
        if config.use_rope:
            self.in_proj_theta = nn.Linear(self.d_model, self.ngroups * (self.d_state // 2), bias=config.bias)
            nn.init.normal_(self.in_proj_theta.weight, std=0.02)
        else:
            self.in_proj_theta = None
            
        # Continuous time decay parameter A (H,)
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # BC Normalization (RMSNorm on B and C)
        if config.use_bc_norm:
            self.b_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
            self.c_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
        else:
            self.b_norm = nn.Identity()
            self.c_norm = nn.Identity()
            
        # Learnable B and C biases (initialized to 1.0 as discovered in Section 3.4 & App F)
        if config.use_bc_bias:
            if self.is_mimo:
                self.b_bias = nn.Parameter(torch.ones(1, 1, self.ngroups, self.d_state, self.mimo_rank))
                self.c_bias = nn.Parameter(torch.ones(1, 1, self.ngroups, self.d_state, self.mimo_rank))
            else:
                self.b_bias = nn.Parameter(torch.ones(1, 1, self.ngroups, self.d_state))
                self.c_bias = nn.Parameter(torch.ones(1, 1, self.ngroups, self.d_state))
        else:
            self.b_bias = None
            self.c_bias = None
            
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(
        self,
        u: torch.Tensor,
        inference_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        B, T, D = u.shape
        
        # Step-by-step recurrent inference mode if T == 1 and cache exists
        if inference_params is not None and "mamba_state" in inference_params:
            return self.step(u, inference_params)
            
        # 1. Projections
        x = self.in_proj_x(u)  # (B, T, E)
        z = self.in_proj_z(u)  # (B, T, E)
        
        # Reshape x, z into heads: (B, T, H, P)
        x = x.view(B, T, self.nheads, self.headdim)
        z = z.view(B, T, self.nheads, self.headdim)
        
        if self.is_mimo:
            x = x.unsqueeze(-1) * self.x_mimo_proj.view(1, 1, 1, self.headdim, self.mimo_rank)
            z = z.unsqueeze(-1) * self.z_mimo_proj.view(1, 1, 1, self.headdim, self.mimo_rank)
            
        # B, C projections
        if self.is_mimo:
            b = self.in_proj_b(u).view(B, T, self.ngroups, self.d_state, self.mimo_rank)
            c = self.in_proj_c(u).view(B, T, self.ngroups, self.d_state, self.mimo_rank)
        else:
            b = self.in_proj_b(u).view(B, T, self.ngroups, self.d_state)
            c = self.in_proj_c(u).view(B, T, self.ngroups, self.d_state)
            
        # Apply BC Normalization
        b = self.b_norm(b)
        c = self.c_norm(c)
        
        # Apply learnable biases
        if self.b_bias is not None:
            b = b + self.b_bias
            c = c + self.c_bias
            
        # dt projection: (B, T, H)
        dt = F.softplus(self.in_proj_dt(u))
        
        # Trapezoidal gate lambda_t in [0, 1]
        if self.in_proj_lambda is not None:
            lambda_gate = torch.sigmoid(self.in_proj_lambda(u))
        else:
            lambda_gate = None
            
        # Complex RoPE
        if self.in_proj_theta is not None:
            theta = self.in_proj_theta(u).view(B, T, self.ngroups, self.d_state // 2)
            if self.nheads > self.ngroups:
                dt_groups = dt.view(B, T, self.ngroups, -1).mean(dim=-1)
            else:
                dt_groups = dt
            phi = compute_complex_rope_angles(theta, dt_groups)
            
            b = apply_rotary_emb(b, phi, conjugate=True)
            c = apply_rotary_emb(c, phi, conjugate=True)
            
        # Continuous time decay parameter
        A = -torch.exp(self.A_log.float())
        
        # Core SSM Recurrence
        y, final_state = exponential_trapezoidal_ssm_recurrence(
            x=x,
            b=b,
            c=c,
            dt=dt,
            A=A,
            lambda_gate=lambda_gate,
        )
        
        # Gating and Output Projection
        if self.is_mimo:
            y_gated = y * F.silu(z)
            y_contracted = torch.einsum('bthpr,pr->bthp', y_gated, self.out_mimo_proj)
            y_flat = y_contracted.reshape(B, T, self.d_inner)
        else:
            y_gated = y * F.silu(z)
            y_flat = y_gated.reshape(B, T, self.d_inner)
            
        out = self.out_proj(y_flat)
        return out

    def step(
        self,
        u: torch.Tensor,
        inference_params: Dict[str, Any],
    ) -> torch.Tensor:
        B, T, D = u.shape
        assert T == 1, "step() supports single token generation."
        
        layer_idx = self.layer_idx
        state_dict = inference_params["mamba_state"]
        if layer_idx not in state_dict:
            state_dict[layer_idx] = {
                "ssm_state": None,
                "prev_v": None,
                "cumulative_phi": None,
            }
        cache = state_dict[layer_idx]
        
        # Projections
        x = self.in_proj_x(u).squeeze(1).view(B, self.nheads, self.headdim)
        z = self.in_proj_z(u).squeeze(1).view(B, self.nheads, self.headdim)
        
        if self.is_mimo:
            x = x.unsqueeze(-1) * self.x_mimo_proj.view(1, 1, self.headdim, self.mimo_rank)
            z = z.unsqueeze(-1) * self.z_mimo_proj.view(1, 1, self.headdim, self.mimo_rank)
            
        if self.is_mimo:
            b = self.in_proj_b(u).squeeze(1).view(B, self.ngroups, self.d_state, self.mimo_rank)
            c = self.in_proj_c(u).squeeze(1).view(B, self.ngroups, self.d_state, self.mimo_rank)
        else:
            b = self.in_proj_b(u).squeeze(1).view(B, self.ngroups, self.d_state)
            c = self.in_proj_c(u).squeeze(1).view(B, self.ngroups, self.d_state)
            
        b = self.b_norm(b)
        c = self.c_norm(c)
        
        if self.b_bias is not None:
            b = b + self.b_bias.squeeze(1)
            c = c + self.c_bias.squeeze(1)
            
        dt = F.softplus(self.in_proj_dt(u)).squeeze(1)
        
        if self.in_proj_lambda is not None:
            lambda_gate = torch.sigmoid(self.in_proj_lambda(u)).squeeze(1)
        else:
            lambda_gate = None
            
        # Complex RoPE step
        if self.in_proj_theta is not None:
            theta = self.in_proj_theta(u).squeeze(1).view(B, self.ngroups, self.d_state // 2)
            if self.nheads > self.ngroups:
                dt_groups = dt.view(B, self.ngroups, -1).mean(dim=-1, keepdim=True)
            else:
                dt_groups = dt.unsqueeze(-1)
            step_angle = dt_groups * theta
            
            if cache["cumulative_phi"] is None:
                phi = step_angle
            else:
                phi = cache["cumulative_phi"] + step_angle
            cache["cumulative_phi"] = phi
            
            b = apply_rotary_emb(b, phi, conjugate=True)
            c = apply_rotary_emb(c, phi, conjugate=True)
            
        A = -torch.exp(self.A_log.float())
        
        y, new_ssm_state, new_prev_v = ssm_step(
            x_t=x,
            b_t=b,
            c_t=c,
            dt_t=dt,
            A=A,
            lambda_gate_t=lambda_gate,
            state=cache["ssm_state"],
            prev_v=cache["prev_v"],
        )
        
        cache["ssm_state"] = new_ssm_state
        cache["prev_v"] = new_prev_v
        
        if self.is_mimo:
            y_gated = y * F.silu(z)
            y_contracted = torch.einsum('bhpr,pr->bhp', y_gated, self.out_mimo_proj)
            y_flat = y_contracted.reshape(B, self.d_inner)
        else:
            y_gated = y * F.silu(z)
            y_flat = y_gated.reshape(B, self.d_inner)
            
        out = self.out_proj(y_flat).unsqueeze(1)
        return out
