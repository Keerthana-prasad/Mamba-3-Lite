"""
Constrained Learnable Neural Integration (LNI) Module for Mamba-3
Replaces rigid trapezoidal discretization heuristics with learned neural integration operators
subject to mathematical constraints (positivity, simplex partition normalization, and contractive stability).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import Mamba3NovelConfig


class ConstrainedLearnableNeuralIntegration(nn.Module):
    """
    Learns continuous integration weights beta_t = f(x_t) and gamma_t = g(x_t)
    governing input-to-state transitions:
        h_t = alpha_t * h_{t-1} + beta_t * (B_{t-1} x_{t-1}) + gamma_t * (B_t x_t)
    
    Subject to:
      1. Positivity: beta_t > 0, gamma_t > 0
      2. Partition Normalization: beta_t * alpha_t^{-1} + gamma_t = Delta_t (Simplex Partition)
      3. Contractive Stability: ||alpha_t|| <= 1 - delta
    """
    def __init__(self, config: Mamba3NovelConfig, num_heads: int):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.head_dim = config.headdim
        self.temperature = config.lni_temperature
        self.contractive_margin = config.lni_contractive_margin
        self.constraint_mode = config.lni_constraint_mode
        
        hidden_dim = config.lni_hidden_dim
        
        # Neural projection heads to learn integration weights f(x_t), g(x_t)
        # Input: (B, T, G, P) -> Output: 2 partition logits per channel/head
        self.integration_net = nn.Sequential(
            nn.Linear(self.head_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2, bias=True),
        )
        
        # Learnable temperature parameter for adaptive stiffness
        self.log_tau = nn.Parameter(torch.zeros(1, 1, self.num_heads, 1))
        
        # Initialize final layer near zero to start close to uniform/trapezoidal prior
        nn.init.zeros_(self.integration_net[-1].weight)
        nn.init.zeros_(self.integration_net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes constrained recurrence decay (alpha_t) and integration operators (beta_t, gamma_t).
        
        Args:
            x:  Input tensor (B, T, G, P)
            dt: Discretization step-size (B, T, G, 1) or (B, T, G, P, 1)
            A:  Continuous state matrix (G, P, 1) with A < 0
            
        Returns:
            alpha: State decay (B, T, G, P, 1) strictly in [0, 1 - delta]
            beta:  Learned prior integration coefficient (B, T, G, P, 1) > 0
            gamma: Learned current integration coefficient (B, T, G, P, 1) > 0
        """
        # Ensure dt has shape (B, T, G, P, 1)
        if dt.ndim == 4:
            # (B, T, G, 1) -> unsqueeze to (B, T, G, 1, 1) -> expand to (B, T, G, P, 1)
            dt_expanded = dt.unsqueeze(-2).expand(-1, -1, -1, self.head_dim, -1)
        elif dt.ndim == 5:
            dt_expanded = dt
        else:
            raise ValueError(f"Unexpected dt shape: {dt.shape}")
            
        # 1. Compute Continuous Recurrence Decay with Contractive Stability Margin
        # A is negative: -softplus(A_param)
        # alpha = exp(dt * A)
        # Boundedness constraint: clamp alpha to [0, 1 - contractive_margin]
        dtA = dt_expanded * A.unsqueeze(0).unsqueeze(0) # (B, T, G, P, 1)
        alpha = torch.exp(dtA)
        alpha = torch.clamp(alpha, min=1e-5, max=1.0 - self.contractive_margin)
        
        # 2. Compute Neural Integration Logits f(x_t), g(x_t)
        # x is (B, T, G, P) -> integration_net -> (B, T, G, 2)
        tau = torch.exp(self.log_tau) * self.temperature + 1e-4 # (1, 1, G, 1)
        raw_logits = self.integration_net(x) # (B, T, G, 2)
        
        if self.constraint_mode == "simplex_partition":
            # Simplex Partition Constraint:
            # [pi_1, pi_2] = Softmax([logits_beta, logits_gamma] / tau)
            # Guarantee: pi_1 > 0, pi_2 > 0 and pi_1 + pi_2 = 1
            # beta = pi_1 * dt * alpha
            # gamma = pi_2 * dt
            partition_weights = F.softmax(raw_logits / tau, dim=-1) # (B, T, G, 2)
            pi_1 = partition_weights[..., 0:1].unsqueeze(-2) # (B, T, G, 1, 1)
            pi_2 = partition_weights[..., 1:2].unsqueeze(-2) # (B, T, G, 1, 1)
            
            beta = pi_1 * dt_expanded * alpha
            gamma = pi_2 * dt_expanded
            
        elif self.constraint_mode == "bounded_softplus":
            # Bounded Softplus Constraint:
            # Ensures positivity and scales by dt
            soft_weights = F.softplus(raw_logits) # (B, T, G, 2)
            # Normalize to sum to 1 per time-step to preserve energy
            sum_weights = soft_weights.sum(dim=-1, keepdim=True) + 1e-6
            norm_weights = soft_weights / sum_weights # (B, T, G, 2)
            
            pi_1 = norm_weights[..., 0:1].unsqueeze(-2) # (B, T, G, 1, 1)
            pi_2 = norm_weights[..., 1:2].unsqueeze(-2) # (B, T, G, 1, 1)
            
            beta = pi_1 * dt_expanded * alpha
            gamma = pi_2 * dt_expanded
            
        else: # Standard trapezoidal fallback with learned lambda
            # lambda = sigmoid(logit)
            lam = torch.sigmoid(raw_logits[..., 0:1]).unsqueeze(-2) # (B, T, G, 1, 1)
            beta = (1.0 - lam) * dt_expanded * alpha
            gamma = lam * dt_expanded
            
        return alpha, beta, gamma
