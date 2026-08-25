import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def apply_rotary_emb(x: torch.Tensor, angles: torch.Tensor, conjugate: bool = True) -> torch.Tensor:
    """
    Applies 2D Givens / Rotary rotation to pairs of coordinates along the state dimension N.
    Supports both SISO (..., N) and MIMO (..., N, R).
    """
    rot_angles = -angles if conjugate else angles
    cos = torch.cos(rot_angles)
    sin = torch.sin(rot_angles)
    
    is_mimo = (x.ndim > angles.ndim)
    
    if is_mimo:
        # x: (..., N, R), angles: (..., N//2)
        while cos.ndim < x.ndim - 1:
            cos = cos.unsqueeze(-1)
            sin = sin.unsqueeze(-1)
        cos = cos.unsqueeze(-1) # (..., N//2, 1)
        sin = sin.unsqueeze(-1)
        
        x0 = x[..., 0::2, :]
        x1 = x[..., 1::2, :]
        
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        
        out = torch.empty_like(x)
        out[..., 0::2, :] = out0
        out[..., 1::2, :] = out1
        return out
    else:
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        
        while cos.ndim < x0.ndim:
            cos = cos.unsqueeze(-1)
            sin = sin.unsqueeze(-1)
            
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        
        out = torch.empty_like(x)
        out[..., 0::2] = out0
        out[..., 1::2] = out1
        return out


def compute_complex_rope_angles(theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Computes cumulative data-dependent RoPE rotation angles:
    Theta_t = dt_t * theta_t
    Phi_t = cumsum_{s=0}^t Theta_s
    
    Args:
        theta: Tensor of shape (B, T, G, N//2)
        dt: Tensor of shape (B, T, G, 1) or (B, T, G)
    Returns:
        phi: Cumulative angles of shape (B, T, G, N//2)
    """
    if dt.ndim < theta.ndim:
        dt = dt.unsqueeze(-1)
    step_angles = dt * theta
    phi = torch.cumsum(step_angles, dim=1)
    return phi


def exponential_trapezoidal_ssm_recurrence(
    x: torch.Tensor,                             # (B, T, H, P) or (B, T, H, P, R)
    b: torch.Tensor,                             # (B, T, G, N) or (B, T, G, N, R)
    c: torch.Tensor,                             # (B, T, G, N) or (B, T, G, N, R)
    dt: torch.Tensor,                            # (B, T, H)
    A: torch.Tensor,                             # (H,) continuous time transition, A < 0
    lambda_gate: Optional[torch.Tensor] = None,  # (B, T, H) data-dependent trapezoidal gate in [0, 1]
    initial_state: Optional[torch.Tensor] = None,# (B, H, N, P)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Core Mamba-3 Exponential-Trapezoidal Recurrent SSM.
    """
    B, T, H, P = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    is_mimo = (x.ndim == 5)
    
    G = b.shape[2]
    N = b.shape[3]
    heads_per_group = H // G
    
    # Broadcast B, C across heads if G < H
    if G < H:
        if is_mimo:
            b = b.repeat_interleave(heads_per_group, dim=2)
            c = c.repeat_interleave(heads_per_group, dim=2)
        else:
            b = b.repeat_interleave(heads_per_group, dim=2)
            c = c.repeat_interleave(heads_per_group, dim=2)
            
    if A.ndim == 1:
        A_val = -torch.exp(A) if (A > 0).any() else A
        dt_A = dt * A_val.view(1, 1, H)
    else:
        dt_A = dt * A
        
    alpha = torch.exp(dt_A)
    
    if lambda_gate is not None:
        gamma = lambda_gate * dt
        beta = (1.0 - lambda_gate) * dt * alpha
    else:
        gamma = dt
        beta = torch.zeros_like(dt)
        
    # State-input v_t: (B, T, H, N, P)
    if is_mimo:
        v = torch.einsum('bthnr,bthpr->bthnp', b, x)
    else:
        v = torch.einsum('bthn,bthp->bthnp', b, x)
        
    if initial_state is not None:
        h = initial_state.clone()
    else:
        h = torch.zeros(B, H, N, P, device=x.device, dtype=x.dtype)
        
    y_list = []
    prev_v = torch.zeros_like(v[:, 0])
    
    for t in range(T):
        alpha_t = alpha[:, t].view(B, H, 1, 1)
        gamma_t = gamma[:, t].view(B, H, 1, 1)
        beta_t = beta[:, t].view(B, H, 1, 1)
        v_t = v[:, t]
        
        # h_t = alpha_t * h_{t-1} + beta_t * v_{t-1} + gamma_t * v_t
        h = alpha_t * h + beta_t * prev_v + gamma_t * v_t
        prev_v = v_t
        
        if is_mimo:
            y_t = torch.einsum('bhnp,bhnr->bhpr', h, c[:, t])
        else:
            y_t = torch.einsum('bhnp,bhn->bhp', h, c[:, t])
            
        y_list.append(y_t)
        
    y = torch.stack(y_list, dim=1)
    return y, h


def ssm_step(
    x_t: torch.Tensor,                            # (B, H, P) or (B, H, P, R)
    b_t: torch.Tensor,                            # (B, G, N) or (B, G, N, R)
    c_t: torch.Tensor,                            # (B, G, N) or (B, G, N, R)
    dt_t: torch.Tensor,                           # (B, H)
    A: torch.Tensor,                              # (H,) continuous time transition
    lambda_gate_t: Optional[torch.Tensor] = None, # (B, H)
    state: Optional[torch.Tensor] = None,          # (B, H, N, P)
    prev_v: Optional[torch.Tensor] = None,         # (B, H, N, P)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single autoregressive decode step for Mamba-3 SSM (O(1) memory per step).
    """
    B, H, P = x_t.shape[0], x_t.shape[1], x_t.shape[2]
    is_mimo = (x_t.ndim == 4)
    G = b_t.shape[1]
    N = b_t.shape[2]
    heads_per_group = H // G
    
    if G < H:
        if is_mimo:
            b_t = b_t.repeat_interleave(heads_per_group, dim=1)
            c_t = c_t.repeat_interleave(heads_per_group, dim=1)
        else:
            b_t = b_t.repeat_interleave(heads_per_group, dim=1)
            c_t = c_t.repeat_interleave(heads_per_group, dim=1)
            
    A_val = -torch.exp(A) if (A > 0).any() else A
    alpha_t = torch.exp(dt_t * A_val.view(1, H)).view(B, H, 1, 1)
    
    if lambda_gate_t is not None:
        gamma_t = (lambda_gate_t * dt_t).view(B, H, 1, 1)
        beta_t = ((1.0 - lambda_gate_t) * dt_t).view(B, H, 1, 1) * alpha_t
    else:
        gamma_t = dt_t.view(B, H, 1, 1)
        beta_t = torch.zeros_like(gamma_t)
        
    if is_mimo:
        v_t = torch.einsum('bhnr,bhpr->bhnp', b_t, x_t)
    else:
        v_t = torch.einsum('bhn,bhp->bhnp', b_t, x_t)
        
    if state is None:
        state = torch.zeros(B, H, N, P, device=x_t.device, dtype=x_t.dtype)
    if prev_v is None:
        prev_v = torch.zeros_like(v_t)
        
    new_state = alpha_t * state + beta_t * prev_v + gamma_t * v_t
    
    if is_mimo:
        y_t = torch.einsum('bhnp,bhnr->bhpr', new_state, c_t)
    else:
        y_t = torch.einsum('bhnp,bhn->bhp', new_state, c_t)
        
    return y_t, new_state, v_t
