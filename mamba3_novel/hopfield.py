"""
Episodic Continuous Hopfield Memory (EHM) Module for Mamba-3 Hybrid Architecture
Implements a dual-system sub-quadratic associative memory layer:
- System 1 (Mamba-3 SSM): Fast linear-time state-tracking and local temporal dynamics.
- System 2 (Episodic Hopfield Memory): Exact associative pattern retrieval over episodic memory slots.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import Mamba3NovelConfig
from .ops import continuous_hopfield_retrieve, episodic_hopfield_pool


class EpisodicContinuousHopfieldMemory(nn.Module):
    """
    Episodic Continuous Hopfield Associative Memory Layer
    
    Stores pooled episodic representations into memory slots M and performs
    energy-based associative retrieval with sub-quadratic O(L * K) complexity.
    """
    def __init__(self, config: Mamba3NovelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.hopfield_heads
        self.head_dim = config.d_model // config.hopfield_heads
        self.chunk_size = config.hopfield_chunk_size
        self.max_memories = config.hopfield_max_memories
        self.beta = config.hopfield_beta
        self.dropout_p = config.hopfield_dropout
        
        # Projections for Query (from current sequence tokens)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        
        # Projections for Episodic Memory Key & Value
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        
        # Associative Read Gate (Gated Dual-System Fusion)
        self.gate_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        
        # Output Projection
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        
        # Layer Normalization for stability
        self.norm = nn.LayerNorm(self.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        external_memories: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:     Input tensor from Mamba-3 SSM layer (B, T, D)
            external_memories: Optional pre-existing episodic memory buffer (B, M_prev, D)
            
        Returns:
            out:               Memory-augmented representation (B, T, D)
            updated_memories:  New episodic memory bank for sequential streaming
        """
        B, T, D = hidden_states.shape
        x_norm = self.norm(hidden_states)
        
        # 1. Pool hidden states into Episodic Memories
        # episodes: (B, M, D), causal_mask: (B, T, M)
        current_episodes, causal_mask = episodic_hopfield_pool(
            x_norm,
            chunk_size=self.chunk_size,
            max_memories=self.max_memories,
        )
        
        if external_memories is not None:
            # Concatenate with external episodic memories from previous sequence chunks
            all_memories = torch.cat([external_memories, current_episodes], dim=1)
            if all_memories.shape[1] > self.max_memories:
                all_memories = all_memories[:, -self.max_memories:, :]
            M = all_memories.shape[1]
            # Update mask to allow attending to all external memories + causal current
            ext_len = all_memories.shape[1] - current_episodes.shape[1]
            ext_mask = torch.ones(B, T, ext_len, device=hidden_states.device, dtype=causal_mask.dtype)
            causal_mask = torch.cat([ext_mask, causal_mask], dim=-1)
        else:
            all_memories = current_episodes
            
        # 2. Project Queries, Keys, and Values
        # Q: (B, T, H, D_h)
        Q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim)
        # K, V: (B, M, H, D_h)
        M_count = all_memories.shape[1]
        K = self.k_proj(all_memories).view(B, M_count, self.num_heads, self.head_dim)
        V = self.v_proj(all_memories).view(B, M_count, self.num_heads, self.head_dim)
        
        # 3. Energy-Based Associative Hopfield Retrieval
        # retrieved: (B, T, H, D_h)
        retrieved = continuous_hopfield_retrieve(
            query=Q,
            memory_keys=K,
            memory_values=V,
            beta=self.beta,
            mask=causal_mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        
        # 4. Gated Dual-System Fusion
        retrieved_flat = retrieved.contiguous().view(B, T, D)
        gate = torch.sigmoid(self.gate_proj(x_norm))
        gated_memory = gate * retrieved_flat
        
        # 5. Output Projection + Residual
        out = self.out_proj(gated_memory)
        return out, all_memories
