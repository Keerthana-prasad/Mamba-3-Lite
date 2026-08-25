from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Mamba3Config:
    d_model: int = 256              # Model dimension (D)
    d_intermediate: int = 0         # MLP hidden dimension (0 = auto 8/3 * d_model or parameter matching)
    n_layers: int = 8               # Number of alternating (Mamba-3 + SwiGLU) layers
    vocab_size: int = 50257         # Vocabulary size (GPT-2 tokenizer default)
    d_state: int = 64               # SSM state dimension (N). Must be even for complex RoPE pairs.
    headdim: int = 64               # Head dimension (P)
    expand: int = 2                 # Expansion factor for inner dimension E = expand * d_model
    ngroups: int = 1                # Number of groups for B, C projections (MVA / GQA style)
    mimo_rank: int = 1              # MIMO rank (R). 1 = SISO, >1 = MIMO (e.g. 4)
    
    # Mamba-3 specific innovations (toggleable for ablations)
    use_trapezoidal: bool = True    # Generalized Exponential-Trapezoidal discretization (vs Exponential-Euler)
    use_rope: bool = True           # Complex-valued state transitions via Data-Dependent RoPE
    use_bc_bias: bool = True        # Head-specific, channel-wise learnable B and C biases (initialized to 1.0)
    use_bc_norm: bool = True        # RMSNorm on B and C before biases / RoPE (BCNorm / QKNorm)
    
    # Layer configuration
    rms_norm_eps: float = 1e-5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False              # Linear projection bias
    tie_embeddings: bool = True     # Tie input and output embedding weights
    max_seq_len: int = 1024         # Context window length
    dropout: float = 0.0
