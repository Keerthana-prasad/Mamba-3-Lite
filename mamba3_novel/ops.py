"""
Core Mathematical Operations for Mamba-3 Novel Extended Architectures
Includes:
- Data-Dependent Complex RoPE (SISO, MIMO, and Grouped-MIMO)
- Constrained Learnable Neural Integration (LNI) Recurrence
- Continuous Modern Hopfield Associative Memory Operations
- Constant O(1) Memory Step Operations for Decoding
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


def compute_complex_rope_angles(
    theta: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """
    Computes data-dependent cumulative rotation angles Phi_t = sum_{s=1}^t Delta_s * theta_s
    Args:
        theta: (B, T, G, N // 2) frequency modulation
        dt:    (B, T, G, 1) or (B, T, G, N // 2) step-size
    Returns:
        phi:   (B, T, G, N // 2) cumulative angles
    """
    dt_theta = dt * theta
    phi = torch.cumsum(dt_theta, dim=1)
    return phi


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Applies 2D Givens rotations (RoPE) across coordinate pairs along dimension -2 or -1.
    Supports both SISO (..., N) and MIMO / gMIMO (..., N, R) tensors.
    
    Args:
        x:   Tensor of shape (..., N) or (..., N, R) where N must be even.
        cos: Tensor of shape (..., N // 2)
        sin: Tensor of shape (..., N // 2)
    Returns:
        Rotated tensor with identical shape as x.
    """
    is_mimo = (x.ndim >= 5)  # e.g., (B, T, G, N, R) or (B, 1, G, N, R)
    
    if is_mimo:
        # x is (B, T, G, N, R) -> pair coordinates along dim -2 (N)
        N = x.shape[-2]
        x1 = x[..., 0::2, :] # (B, T, G, N//2, R)
        x2 = x[..., 1::2, :] # (B, T, G, N//2, R)
        
        c = cos.unsqueeze(-1) # (B, T, G, N//2, 1)
        s = sin.unsqueeze(-1) # (B, T, G, N//2, 1)
        
        out1 = x1 * c - x2 * s
        out2 = x1 * s + x2 * c
        
        out = torch.empty_like(x)
        out[..., 0::2, :] = out1
        out[..., 1::2, :] = out2
        return out
    else:
        # x is (..., N)
        N = x.shape[-1]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        out = torch.empty_like(x)
        out[..., 0::2] = out1
        out[..., 1::2] = out2
        return out


def constrained_lni_ssm_recurrence(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
    h_prev: Optional[torch.Tensor] = None,
    use_rope: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSM Recurrence with Constrained Learnable Neural Integration (LNI)
    
    State update formulation:
        h_t = alpha_t * h_{t-1} + beta_t * (B_{t-1} x_{t-1}) + gamma_t * (B_t x_t)
        y_t = C_t^T * h_t (or Tr(C_t^T * h_t) for MIMO)
    
    Args:
        x:     Input sequence (B, T, G, P)
        alpha: State decay (B, T, G, P, 1) in [0, 1)
        beta:  Learned prior input integration operator (B, T, G, P, 1)
        gamma: Learned current input integration operator (B, T, G, P, 1)
        B:     Input matrix (B, T, G, N) or (B, T, G, N, R)
        C:     Output matrix (B, T, G, N) or (B, T, G, N, R)
        phi:   Cumulative RoPE angles (B, T, G, N // 2) if use_rope
        h_prev: Optional initial state
        use_rope: Whether to apply complex rotation
        
    Returns:
        y:     Output sequence (B, T, G, P)
        h_last: Final state for sequential autoregressive continuation
    """
    B_sz, T, G, P = x.shape
    is_mimo = (B.ndim == 5)
    
    if is_mimo:
        R = B.shape[-1]
        N = B.shape[-2]
    else:
        R = 1
        N = B.shape[-1]
        
    # Apply complex rotation via RoPE if enabled
    if use_rope and phi is not None:
        cos = torch.cos(phi)
        sin = torch.sin(phi)
        B_rot = apply_rotary_emb(B, cos, sin)
        C_rot = apply_rotary_emb(C, cos, sin)
    else:
        B_rot = B
        C_rot = C
        
    # Initialize hidden state:
    # SISO: (B_sz, G, N, P)
    # MIMO: (B_sz, G, N, P, R)
    if is_mimo:
        state_shape = (B_sz, G, N, P, R)
    else:
        state_shape = (B_sz, G, N, P)
        
    if h_prev is not None:
        h = h_prev.clone()
    else:
        h = torch.zeros(state_shape, dtype=x.dtype, device=x.device)
        
    y_list = []
    
    # Track previous rotated input term (B_{t-1} * x_{t-1})
    if is_mimo:
        # B_rot: (B, T, G, N, R), x: (B, T, G, P)
        # bx: (B_sz, G, N, P, R) -> B_rot * x.unsqueeze(-2).unsqueeze(-1)
        bx_prev = torch.zeros_like(h)
    else:
        # B_rot: (B, T, G, N), x: (B, T, G, P)
        # bx: (B_sz, G, N, P)
        bx_prev = torch.zeros_like(h)
        
    for t in range(T):
        x_t = x[:, t, ...]         # (B, G, P)
        alpha_t = alpha[:, t, ...] # (B, G, P, 1)
        beta_t = beta[:, t, ...]   # (B, G, P, 1)
        gamma_t = gamma[:, t, ...] # (B, G, P, 1)
        
        B_t = B_rot[:, t, ...]     # (B, G, N) or (B, G, N, R)
        C_t = C_rot[:, t, ...]     # (B, G, N) or (B, G, N, R)
        
        if is_mimo:
            # B_t: (B, G, N, R), x_t: (B, G, P)
            # bx_curr: (B, G, N, P, R)
            bx_curr = B_t.unsqueeze(-2) * x_t.unsqueeze(-2).unsqueeze(-1)
            
            # State update with Constrained LNI:
            # h: (B, G, N, P, R), alpha_t: (B, G, P, 1) -> permute alpha to (B, G, 1, P, 1)
            a_perm = alpha_t.permute(0, 1, 3, 2).unsqueeze(-1) # (B, G, 1, P, 1)
            b_perm = beta_t.permute(0, 1, 3, 2).unsqueeze(-1)
            g_perm = gamma_t.permute(0, 1, 3, 2).unsqueeze(-1)
            
            h = a_perm * h + b_perm * bx_prev + g_perm * bx_curr
            bx_prev = bx_curr
            
            # Output computation: y_t = Tr(C_t^T * h_t) = sum_{n, r} C_{t, n, r} * h_{t, n, p, r}
            # C_t: (B, G, N, R) -> (B, G, N, 1, R)
            # h: (B, G, N, P, R)
            # Product: (B, G, N, P, R) -> sum over N (dim 2) and R (dim 4) -> (B, G, P)
            y_t = (C_t.unsqueeze(-2) * h).sum(dim=(2, 4))
        else:
            # SISO
            # B_t: (B, G, N), x_t: (B, G, P)
            # bx_curr: (B, G, N, P)
            bx_curr = B_t.unsqueeze(-1) * x_t.unsqueeze(-2)
            
            a_perm = alpha_t.permute(0, 1, 3, 2) # (B, G, 1, P)
            b_perm = beta_t.permute(0, 1, 3, 2)
            g_perm = gamma_t.permute(0, 1, 3, 2)
            
            h = a_perm * h + b_perm * bx_prev + g_perm * bx_curr
            bx_prev = bx_curr
            
            # y_t: sum_n C_{t, n} * h_{t, n, p} -> (B, G, P)
            y_t = (C_t.unsqueeze(-1) * h).sum(dim=2)
            
        y_list.append(y_t)
        
    y = torch.stack(y_list, dim=1) # (B, T, G, P)
    return y, h


def continuous_hopfield_retrieve(
    query: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_values: torch.Tensor,
    beta: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Modern Continuous Hopfield Network Associative Energy Retrieval (Ramsauer et al., 2020)
    
    Given queries Q and stored episodic memory patterns (K, V):
    Retrieval follows the continuous associative memory fixed-point update:
        z_retrieved = Softmax(beta * Q * K^T / sqrt(D_k)) * V
    
    Args:
        query:         (B, T, H, D_k) Query patterns generated from current hidden state
        memory_keys:   (B, M, H, D_k) Stored episodic memory key patterns (M memory slots)
        memory_values: (B, M, H, D_v) Stored episodic memory value patterns
        beta:          Inverse temperature scaling factor
        mask:          Optional causal/episodic visibility mask (B, T, M)
        dropout_p:     Memory retrieval dropout
        training:      Whether model is in training mode
        
    Returns:
        retrieved:     (B, T, H, D_v) Associatively retrieved memory pattern
    """
    # query: (B, T, H, D_k) -> permute to (B, H, T, D_k)
    # memory_keys: (B, M, H, D_k) -> permute to (B, H, M, D_k)
    q = query.permute(0, 2, 1, 3)
    k = memory_keys.permute(0, 2, 1, 3)
    v = memory_values.permute(0, 2, 1, 3)
    
    D_k = q.shape[-1]
    scale = (beta / math.sqrt(D_k))
    
    # Hopfield energy affinity: (B, H, T, M)
    energy = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    if mask is not None:
        # mask is (B, T, M) -> unsqueeze to (B, 1, T, M)
        energy = energy.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
    attn_probs = F.softmax(energy, dim=-1)
    
    if training and dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        
    # Retrieved pattern: (B, H, T, D_v) -> permute back to (B, T, H, D_v)
    out = torch.matmul(attn_probs, v)
    out = out.permute(0, 2, 1, 3).contiguous()
    return out


def episodic_hopfield_pool(
    hidden_states: torch.Tensor,
    chunk_size: int = 16,
    max_memories: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunks a sequence of hidden states into episodic representations.
    
    Args:
        hidden_states: (B, T, D)
        chunk_size:    Length of each episodic chunk C
        max_memories:  Maximum episodic slots K_max
    Returns:
        memory_episodes: (B, M, D) where M = ceil(T / chunk_size) <= max_memories
        episodic_mask:   (B, T, M) Causal memory visibility mask ensuring query at t
                         can only retrieve from completed previous episodes (or current).
    """
    B, T, D = hidden_states.shape
    num_chunks = (T + chunk_size - 1) // chunk_size
    
    # Pad sequence if not evenly divisible
    pad_len = num_chunks * chunk_size - T
    if pad_len > 0:
        x_pad = F.pad(hidden_states, (0, 0, 0, pad_len), mode="constant", value=0.0)
    else:
        x_pad = hidden_states
        
    # Reshape to (B, num_chunks, chunk_size, D)
    x_chunks = x_pad.view(B, num_chunks, chunk_size, D)
    
    # Pool episodic representations using attentive / mean pooling
    # Mean pooling along chunk dimension: (B, num_chunks, D)
    episodes = x_chunks.mean(dim=2)
    
    if episodes.shape[1] > max_memories:
        # Keep most recent max_memories episodes
        episodes = episodes[:, -max_memories:, :]
        num_chunks = max_memories
        
    M = episodes.shape[1]
    
    # Construct Causal Episodic Mask:
    # A token at index t belongs to chunk_idx = t // chunk_size
    # It can attend to all past completed episodes (m < chunk_idx) and current episode (m == chunk_idx)
    token_indices = torch.arange(T, device=hidden_states.device).unsqueeze(1) # (T, 1)
    chunk_assignments = token_indices // chunk_size                             # (T, 1)
    memory_indices = torch.arange(M, device=hidden_states.device).unsqueeze(0) # (1, M)
    
    # (T, M) causal mask
    causal_mask = (memory_indices <= chunk_assignments).unsqueeze(0).expand(B, T, M) # (B, T, M)
    
    return episodes, causal_mask
