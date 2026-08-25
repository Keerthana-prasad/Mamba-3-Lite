from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Mamba3NovelConfig:
    """
    Configuration dataclass for Mamba-3 Novel Extended Architectures.
    Supports Grouped-MIMO (gMIMO), Constrained Learnable Neural Integration (LNI),
    and Episodic Continuous Hopfield Hybrid.
    """
    # Core Backbone Dimensions
    d_model: int = 256              # Model dimension (D)
    d_intermediate: int = 0         # MLP hidden dimension (0 = auto 8/3 * d_model)
    n_layers: int = 8               # Total number of layers in backbone
    vocab_size: int = 50257         # Vocabulary size (GPT-2 tokenizer default)
    d_state: int = 64               # SSM state dimension (N), must be even for complex RoPE
    headdim: int = 64               # SSM head dimension (P)
    expand: int = 2                 # Expansion factor for inner dimension E = expand * d_model
    ngroups: int = 1                # Groups for standard B, C projections (MVA style)
    
    # -------------------------------------------------------------
    # Novelty 1: Grouped-MIMO (gMIMO)
    # -------------------------------------------------------------
    mimo_rank: int = 4              # MIMO rank (R). 1 = SISO, >1 = MIMO
    use_gmimo: bool = True          # Enable Grouped-MIMO parameter sharing
    gmimo_groups: int = 2           # Number of parameter-sharing groups across SSM heads
                                    # (e.g. 8 heads split into 2 groups => 4 heads/group)
    gmimo_proj_rank: Optional[int] = None # Rank of shared subspace (defaults to mimo_rank)
    
    # -------------------------------------------------------------
    # Novelty 2: Constrained Learnable Neural Integration (LNI)
    # -------------------------------------------------------------
    use_lni: bool = True            # Enable Constrained Learnable Neural Integration
    lni_hidden_dim: int = 32        # Hidden projection dimension for integration neural net
    lni_temperature: float = 1.0    # Temperature for integration partition simplex softmax
    lni_contractive_margin: float = 1e-3 # Minimum contractive decay margin (1 - delta)
    lni_constraint_mode: str = "simplex_partition" # "simplex_partition", "bounded_softplus", "trapezoidal_general"
    
    # Base Mamba-3 features
    use_trapezoidal: bool = True    # Standard trapezoidal integration (overridden when use_lni=True)
    use_rope: bool = True           # Complex-valued state transitions via Data-Dependent RoPE
    use_bc_bias: bool = True        # Head-specific, channel-wise learnable B and C biases (init 1.0)
    use_bc_norm: bool = True        # RMSNorm on B and C before biases / RoPE (BCNorm)
    
    # -------------------------------------------------------------
    # Novelty 3: Episodic Continuous Hopfield Memory (EHM) Hybrid
    # -------------------------------------------------------------
    use_hopfield: bool = True       # Enable Episodic Continuous Hopfield Memory layers
    hopfield_interval: int = 2      # Interleave Hopfield memory every k layers (e.g. every 2 layers)
    hopfield_chunk_size: int = 16   # Episodic chunk size C for pooling into memory slots
    hopfield_max_memories: int = 64 # Max episodic memory slots (K_max) for strict sub-quadratic / linear scale
    hopfield_beta: float = 1.0      # Inverse temperature for continuous Hopfield energy retrieval
    hopfield_heads: int = 4         # Number of associative query heads
    hopfield_dropout: float = 0.0   # Dropout inside Hopfield associative retrieval
    
    # General Layer & Numerical Settings
    rms_norm_eps: float = 1e-5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    tie_embeddings: bool = True
    max_seq_len: int = 1024
    dropout: float = 0.0
