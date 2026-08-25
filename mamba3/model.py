import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from .config import Mamba3Config
from .block import Mamba3Block, RMSNorm


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network following LLaMA / Mamba-3 architecture.
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        d_model = config.d_model
        if config.d_intermediate > 0:
            d_ff = config.d_intermediate
        else:
            # Standard 8/3 * d_model SwiGLU dimension rounded to multiple of 64
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 64 * ((d_ff + 63) // 64)
            
        self.w1 = nn.Linear(d_model, d_ff, bias=config.bias) # Gate proj
        self.w2 = nn.Linear(d_ff, d_model, bias=config.bias) # Down proj
        self.w3 = nn.Linear(d_model, d_ff, bias=config.bias) # Up proj
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Mamba3DecoderLayer(nn.Module):
    """
    Alternating Mamba-3 and SwiGLU layer with Pre-RMSNorm and residual connections.
    """
    def __init__(self, config: Mamba3Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.mixer_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mixer = Mamba3Block(config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # 1. Mixer sublayer: hidden_states + Mamba3(RMSNorm(hidden_states))
        residual = hidden_states
        hidden_states = self.mixer_norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states
        
        # 2. MLP sublayer: hidden_states + SwiGLU(RMSNorm(hidden_states))
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Mamba3LMHeadModel(nn.Module):
    """
    Complete Mamba-3 Causal Language Model for Pretraining and Generation.
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Mamba3DecoderLayer(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
            
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
        inference_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token indices
            labels: Optional (B, T) target token indices for cross-entropy loss
            inference_params: Optional dict for recurrent step caching
            
        Returns:
            logits: (B, T, vocab_size)
            loss: Cross entropy loss if labels provided, else None
        """
        hidden_states = self.embedding(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
            
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive text generation using recurrent state caching.
        """
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        
        if not use_cache or T == 0:
            # Full sequence recomputation fallback
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
            return generated

        # Recurrent state decoding
        inference_params = {"mamba_state": {}}
        
        # 1. Prefill prompt
        if T > 1:
            logits, _ = self.forward(input_ids[:, :-1], inference_params=None)
            # Process prompt step-by-step to populate recurrent cache
            for t in range(T - 1):
                _ = self.forward(input_ids[:, t:t+1], inference_params=inference_params)
                
        # 2. Decode step-by-step
        curr_token = input_ids[:, -1:]
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            logits, _ = self.forward(curr_token, inference_params=inference_params)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            curr_token = next_token
            
        return generated
