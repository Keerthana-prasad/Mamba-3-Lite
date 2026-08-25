"""
Master Comparative Benchmark Runner for Journal Publication
Compares:
1. Base Mamba-3 (Reference Paper Implementation)
2. Mamba-3 + Grouped-MIMO (gMIMO Parameter Sharing)
3. Mamba-3 + Constrained Learnable Neural Integration (LNI)
4. Mamba-3 Dual-System Complete Architecture (gMIMO + LNI + Episodic Hopfield Hybrid)

Evaluates on dataset/ (WikiText-103) across:
- Cross-Entropy Loss (Val & Test)
- Perplexity (Val & Test)
- Next-Token Accuracy (%)
- Active Parameters
- Prefill Latency (ms)
- Decode Latency (ms/token)
- Decoding Throughput (tokens/sec)
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel
from mamba3.dataset import get_dataloaders
from mamba3.metrics import evaluate_model, benchmark_inference_latency

from mamba3_novel.config import Mamba3NovelConfig
from mamba3_novel.model import Mamba3NovelLMHeadModel


def train_model_quick(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_steps: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    desc: str = "Model",
):
    """Fast training loop for fair comparative study"""
    print(f"\n--- Training {desc} ({max_steps} steps) ---", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine scheduler with linear warmup
    warmup_steps = 10
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    step = 0
    start_time = time.time()
    
    train_iter = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        
        if step % 25 == 0 or step == max_steps:
            ppl = math.exp(min(loss.item(), 20.0))
            print(f"  Step {step:3d}/{max_steps} | Loss: {loss.item():.4f} | PPL: {ppl:7.2f}", flush=True)
            
    elapsed = time.time() - start_time
    print(f"  Training finished in {elapsed:.2f}s", flush=True)


def run_comparative_study():
    print("=" * 80, flush=True)
    print("MAMBA-3 NOVEL ARCHITECTURES: SCIENTIFIC COMPARATIVE JOURNAL STUDY", flush=True)
    print("Dataset: WikiText-103 (dataset/) | Device: CPU / Hardware Baseline", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 128
    batch_size = 4
    max_steps = 40
    
    # 1. Load Data
    print("Loading WikiText dataset splits...", flush=True)
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        dataset_dir="dataset",
        batch_size=batch_size,
        seq_len=seq_len,
        max_train_samples=200,
        max_eval_samples=30,
    )
    
    # Common model dimensions
    d_model = 128
    n_layers = 4
    d_state = 32
    headdim = 32
    
    # Define the 4 model variants to evaluate
    variants = [
        {
            "name": "1. Base Mamba-3 (Paper)",
            "model_fn": lambda: Mamba3LMHeadModel(Mamba3Config(
                d_model=d_model, n_layers=n_layers, d_state=d_state, headdim=headdim,
                max_seq_len=seq_len, mimo_rank=1, use_trapezoidal=True, use_rope=True,
            )),
        },
        {
            "name": "2. Mamba-3 + Grouped-MIMO (gMIMO)",
            "model_fn": lambda: Mamba3NovelLMHeadModel(Mamba3NovelConfig(
                d_model=d_model, n_layers=n_layers, d_state=d_state, headdim=headdim,
                max_seq_len=seq_len, mimo_rank=4, use_gmimo=True, gmimo_groups=2,
                use_lni=False, use_hopfield=False,
            )),
        },
        {
            "name": "3. Mamba-3 + Constrained LNI",
            "model_fn": lambda: Mamba3NovelLMHeadModel(Mamba3NovelConfig(
                d_model=d_model, n_layers=n_layers, d_state=d_state, headdim=headdim,
                max_seq_len=seq_len, mimo_rank=1, use_gmimo=False,
                use_lni=True, lni_constraint_mode="simplex_partition", use_hopfield=False,
            )),
        },
        {
            "name": "4. Mamba-3 Complete Hybrid (gMIMO + LNI + Hopfield)",
            "model_fn": lambda: Mamba3NovelLMHeadModel(Mamba3NovelConfig(
                d_model=d_model, n_layers=n_layers, d_state=d_state, headdim=headdim,
                max_seq_len=seq_len, mimo_rank=4, use_gmimo=True, gmimo_groups=2,
                use_lni=True, lni_constraint_mode="simplex_partition",
                use_hopfield=True, hopfield_interval=2, hopfield_chunk_size=16,
            )),
        },
    ]
    
    results = []
    
    for v in variants:
        name = v["name"]
        print("\n" + "#" * 70, flush=True)
        print(f"EVALUATING VARIANT: {name}", flush=True)
        print("#" * 70, flush=True)
        
        model = v["model_fn"]().to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}", flush=True)
        
        # Train model
        train_model_quick(
            model,
            train_loader,
            val_loader,
            max_steps=max_steps,
            lr=1e-3,
            device=device,
            desc=name,
        )
        
        # Evaluate on Validation
        print("Evaluating on Validation split...", flush=True)
        val_metrics = evaluate_model(model, val_loader, device=device)
        
        # Evaluate on Test
        print("Evaluating on Test split...", flush=True)
        test_metrics = evaluate_model(model, test_loader, device=device)
        
        # Profile Inference Latency
        print("Profiling Inference Latency...", flush=True)
        bench = benchmark_inference_latency(model, device=device, seq_len=64, decode_steps=10)
        
        res_entry = {
            "name": name,
            "params": total_params,
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["perplexity"],
            "test_loss": test_metrics["loss"],
            "test_ppl": test_metrics["perplexity"],
            "test_acc": test_metrics["accuracy"],
            "prefill_ms": bench["prefill_latency_ms"],
            "decode_ms_tok": bench["decode_latency_per_token_ms"],
            "throughput_tok_s": bench["decode_throughput_tok_per_sec"],
        }
        results.append(res_entry)
        
    # Print Journal Master Table
    print("\n\n" + "=" * 105, flush=True)
    print("MASTER JOURNAL COMPARATIVE RESULTS TABLE", flush=True)
    print("=" * 105, flush=True)
    header = f"{'Model Architecture':<45} | {'Params':<9} | {'Val Loss':<8} | {'Val PPL':<8} | {'Test Loss':<9} | {'Test PPL':<8} | {'Test Acc':<8} | {'Dec Latency':<11}"
    print(header, flush=True)
    print("-" * 105, flush=True)
    for r in results:
        row = f"{r['name']:<45} | {r['params']:<9,d} | {r['val_loss']:<8.4f} | {r['val_ppl']:<8.2f} | {r['test_loss']:<9.4f} | {r['test_ppl']:<8.2f} | {r['test_acc']:<7.2f}% | {r['decode_ms_tok']:<8.2f} ms"
        print(row, flush=True)
    print("=" * 105, flush=True)
    
    # Save results dictionary
    torch.save(results, "checkpoints/novelty_comparative_results.pt")
    print("\nSaved comparative benchmark results to checkpoints/novelty_comparative_results.pt", flush=True)


if __name__ == "__main__":
    run_comparative_study()
