"""
Comprehensive Unit and Invariant Test Suite for Mamba-3 Novelties
Tests:
- Constrained Learnable Neural Integration (LNI) mathematical invariants
- Grouped-MIMO (gMIMO) parameter scaling and recurrence
- Episodic Continuous Hopfield Memory associative retrieval
- End-to-end forward/backward and generation
"""

import sys
import torch
import pytest

from mamba3_novel.config import Mamba3NovelConfig
from mamba3_novel.lni import ConstrainedLearnableNeuralIntegration
from mamba3_novel.gmimo import GroupedMIMOSSM
from mamba3_novel.hopfield import EpisodicContinuousHopfieldMemory
from mamba3_novel.block import Mamba3NovelBlock
from mamba3_novel.model import Mamba3NovelLMHeadModel


def test_lni_mathematical_invariants():
    print("Testing Constrained Learnable Neural Integration (LNI) invariants...")
    config = Mamba3NovelConfig(
        d_model=64,
        headdim=32,
        expand=2,
        use_lni=True,
        lni_constraint_mode="simplex_partition",
        lni_contractive_margin=1e-3,
    )
    num_heads = (config.expand * config.d_model) // config.headdim # 4
    lni = ConstrainedLearnableNeuralIntegration(config, num_heads)
    
    B, T, G, P = 2, 16, num_heads, config.headdim
    x = torch.randn(B, T, G, P)
    dt = torch.rand(B, T, G, 1) * 0.05 + 0.001
    A = -torch.rand(G, P, 1) * 2.0 - 0.1
    
    alpha, beta, gamma = lni(x, dt, A)
    
    # 1. Positivity invariant
    assert torch.all(beta > 0), "Violation: beta must be strictly positive"
    assert torch.all(gamma > 0), "Violation: gamma must be strictly positive"
    assert torch.all(alpha > 0), "Violation: alpha must be strictly positive"
    
    # 2. Contractive stability invariant: alpha <= 1 - margin
    assert torch.all(alpha <= (1.0 - config.lni_contractive_margin + 1e-5)), \
        "Violation: alpha exceeds contractive stability threshold"
        
    # 3. Simplex partition invariant: (beta / (dt * alpha)) + (gamma / dt) == 1.0
    dt_exp = dt.unsqueeze(-2).expand(-1, -1, -1, P, -1)
    pi_1 = beta / (dt_exp * alpha)
    pi_2 = gamma / dt_exp
    simplex_sum = pi_1 + pi_2
    assert torch.allclose(simplex_sum, torch.ones_like(simplex_sum), atol=1e-4), \
        f"Violation: Simplex partition sum does not equal 1.0 (mean={simplex_sum.mean().item()})"
        
    print("  [PASSED] LNI mathematical invariants verified successfully.")


def test_gmimo_parameter_efficiency_and_forward():
    print("Testing Grouped-MIMO (gMIMO) parameter scaling and execution...")
    # Standard MIMO (un-grouped: gmimo_groups = 4)
    cfg_standard = Mamba3NovelConfig(
        d_model=64,
        headdim=16,
        expand=2,
        mimo_rank=4,
        use_gmimo=False,
    )
    # Grouped MIMO (grouped: gmimo_groups = 2)
    cfg_grouped = Mamba3NovelConfig(
        d_model=64,
        headdim=16,
        expand=2,
        mimo_rank=4,
        use_gmimo=True,
        gmimo_groups=2,
    )
    
    ssm_std = GroupedMIMOSSM(cfg_standard)
    ssm_grp = GroupedMIMOSSM(cfg_grouped)
    
    params_std = sum(p.numel() for p in ssm_std.parameters())
    params_grp = sum(p.numel() for p in ssm_grp.parameters())
    
    print(f"  Standard MIMO Params: {params_std:,} | Grouped-MIMO Params: {params_grp:,}")
    assert params_grp < params_std, "Violation: Grouped-MIMO must have fewer parameters than standard MIMO"
    
    # Forward pass
    u = torch.randn(2, 16, cfg_grouped.expand * cfg_grouped.d_model)
    out, h_last = ssm_grp(u)
    assert out.shape == (2, 16, cfg_grouped.d_model)
    assert h_last.ndim == 5 # (B, G, N, P, R)
    print("  [PASSED] Grouped-MIMO forward execution and parameter efficiency verified.")


def test_episodic_hopfield_memory():
    print("Testing Episodic Continuous Hopfield Memory (EHM)...")
    config = Mamba3NovelConfig(
        d_model=64,
        hopfield_heads=4,
        hopfield_chunk_size=8,
        hopfield_max_memories=16,
        hopfield_beta=1.5,
    )
    hop = EpisodicContinuousHopfieldMemory(config)
    
    x = torch.randn(2, 32, 64)
    out, memories = hop(x)
    
    assert out.shape == (2, 32, 64), f"Unexpected output shape: {out.shape}"
    assert memories.shape[1] == (32 // 8), f"Unexpected episodic memory slot count: {memories.shape[1]}"
    
    # Backward pass verification
    loss = out.sum()
    loss.backward()
    assert hop.q_proj.weight.grad is not None, "Violation: Gradients did not propagate through Hopfield query projection"
    assert hop.k_proj.weight.grad is not None, "Violation: Gradients did not propagate through Hopfield key projection"
    print("  [PASSED] Episodic Hopfield Memory retrieval and gradient flow verified.")


def test_full_mamba3_novel_model():
    print("Testing Full Mamba-3 Novel Extended Model (Dual-System Hybrid)...")
    config = Mamba3NovelConfig(
        d_model=64,
        n_layers=2,
        d_state=32,
        headdim=16,
        expand=2,
        mimo_rank=4,
        use_gmimo=True,
        gmimo_groups=2,
        use_lni=True,
        use_hopfield=True,
        hopfield_interval=1, # Hopfield on every layer
        hopfield_chunk_size=8,
        vocab_size=1000,
    )
    model = Mamba3NovelLMHeadModel(config)
    
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()
    
    # Forward pass with loss
    res = model(input_ids, labels=labels)
    assert "logits" in res
    assert "loss" in res
    assert res["logits"].shape == (2, 16, 1000)
    assert not torch.isnan(res["loss"]).item(), "Loss contains NaN"
    
    # Backward pass
    res["loss"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            
    # Autoregressive generation
    prompt = torch.tensor([[10, 20, 30]])
    gen = model.generate(prompt, max_new_tokens=5, temperature=0.8)
    assert gen.shape == (1, 8), f"Unexpected generated shape: {gen.shape}"
    print("  [PASSED] Full Mamba-3 Novel Model end-to-end forward/backward and generation verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MAMBA-3 NOVEL ARCHITECTURAL INVARIANT TESTS")
    print("=" * 60)
    test_lni_mathematical_invariants()
    test_gmimo_parameter_efficiency_and_forward()
    test_episodic_hopfield_memory()
    test_full_mamba3_novel_model()
    print("=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
