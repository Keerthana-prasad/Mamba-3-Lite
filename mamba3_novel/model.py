"""
Full Mamba-3 Novel Language Model and Dual-System Hybrid Architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from .config import Mamba3NovelConfig
from .gmimo import RMSNorm
from .block import Mamba3NovelBlock


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network layer (LLaMA / Mamba-3 MLP backbone)"""
    def __init__(self, d_model: int, d_intermediate: int = 0, bias: bool = False):
        super().__init__()
        if d_intermediate == 0:
            d_intermediate = int(8 / 3 * d_model)
            # Make multiple of 64 for compute efficiency
            d_intermediate = ((d_intermediate + 63) // 64) * 64
            
        self.w1 = nn.Linear(d_model, d_intermediate, bias=bias)
        self.w2 = nn.Linear(d_intermediate, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, d_intermediate, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mamba3NovelLMHeadModel(nn.Module):
    """
    Mamba-3 Novel Extended Language Model Head Model
    Incorporates Grouped-MIMO (gMIMO), Constrained Learnable Neural Integration (LNI),
    and Episodic Continuous Hopfield Memory.
    """
    def __init__(self, config: Mamba3NovelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        
        # Token Embeddings
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)
        
        # Alternating Architecture: Mamba3NovelBlock layers and SwiGLU MLP layers
        self.layers = nn.ModuleList([
            Mamba3NovelBlock(config, layer_idx=i)
            for i in range(self.n_layers)
        ])
        
        self.mlp_norms = nn.ModuleList([
            RMSNorm(self.d_model, eps=config.rms_norm_eps)
            for _ in range(self.n_layers)
        ])
        
        self.mlps = nn.ModuleList([
            SwiGLU(self.d_model, config.d_intermediate, bias=config.bias)
            for _ in range(self.n_layers)
        ])
        
        # Final Layer Norm and LM Head
        self.final_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ssm_states: Optional[List[Optional[torch.Tensor]]] = None,
        memory_banks: Optional[List[Optional[torch.Tensor]]] = None,
        return_dict: bool = False,
    ) -> Any:
        """
        Args:
            input_ids:    (B, T) token indices
            labels:       (B, T) target token indices for loss computation
            ssm_states:   Optional list of previous SSM states per layer
            memory_banks: Optional list of previous Hopfield memory banks per layer
            return_dict:  Whether to return dict or (logits, loss) tuple
            
        Returns:
            Dict or Tuple containing logits, loss, ssm_states, etc.
        """
        B_sz, T = input_ids.shape
        x = self.embeddings(input_ids) # (B, T, D)
        
        next_ssm_states = []
        next_memory_banks = []
        
        for i, (layer, mlp_norm, mlp) in enumerate(zip(self.layers, self.mlp_norms, self.mlps)):
            curr_state = ssm_states[i] if ssm_states is not None else None
            curr_mem = memory_banks[i] if memory_banks is not None else None
            
            # 1. SSM + Hopfield Block
            x, nxt_state, nxt_mem = layer(x, ssm_state=curr_state, memory_bank=curr_mem)
            next_ssm_states.append(nxt_state)
            next_memory_banks.append(nxt_mem)
            
            # 2. SwiGLU MLP Block with Pre-RMSNorm
            residual = x
            x_norm = mlp_norm(x)
            x = residual + mlp(x_norm)
            
        # Final Norm & Head
        hidden_states = self.final_norm(x)
        logits = self.lm_head(hidden_states) # (B, T, vocab_size)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
        if not return_dict:
            return logits, loss
            
        return {
            "logits": logits,
            "loss": loss,
            "ssm_states": next_ssm_states,
            "memory_banks": next_memory_banks,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        """
        self.eval()
        curr_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Pass full or windowed sequence
            outputs = self.forward(curr_ids, return_dict=True)
            next_token_logits = outputs["logits"][:, -1, :] # (B, vocab_size)
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")
                    
                # Top-P filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float("Inf")
                    
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        return curr_ids
