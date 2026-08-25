import sys
sys.stdout.reconfigure(line_buffering=True)
import torch
from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel

def test():
    print("Testing Mamba-3 SISO architecture...", flush=True)
    cfg_siso = Mamba3Config(
        d_model=64,
        n_layers=2,
        d_state=16,
        headdim=16,
        mimo_rank=1,
        vocab_size=500,
        use_trapezoidal=True,
        use_rope=True,
        use_bc_bias=True,
        use_bc_norm=True,
    )
    model_siso = Mamba3LMHeadModel(cfg_siso)
    x = torch.randint(0, 500, (2, 16))
    logits, loss = model_siso(x, labels=x)
    print(f"  SISO Logits shape: {logits.shape}, Loss: {loss.item():.4f}", flush=True)

    print("Testing Mamba-3 MIMO (R=4) architecture...", flush=True)
    cfg_mimo = Mamba3Config(
        d_model=64,
        n_layers=2,
        d_state=16,
        headdim=16,
        mimo_rank=4,
        vocab_size=500,
        use_trapezoidal=True,
        use_rope=True,
        use_bc_bias=True,
        use_bc_norm=True,
    )
    model_mimo = Mamba3LMHeadModel(cfg_mimo)
    logits_m, loss_m = model_mimo(x, labels=x)
    print(f"  MIMO Logits shape: {logits_m.shape}, Loss: {loss_m.item():.4f}", flush=True)

    print("Testing Autoregressive Recurrent Generation (O(1) Memory Cache)...", flush=True)
    prompt = torch.randint(0, 500, (1, 4))
    gen_siso = model_siso.generate(prompt, max_new_tokens=6, use_cache=True)
    print(f"  Generated SISO Sequence shape: {gen_siso.shape}", flush=True)
    
    gen_mimo = model_mimo.generate(prompt, max_new_tokens=6, use_cache=True)
    print(f"  Generated MIMO Sequence shape: {gen_mimo.shape}", flush=True)

    print("\nAll Mamba-3 verification tests passed successfully!", flush=True)

if __name__ == "__main__":
    test()
