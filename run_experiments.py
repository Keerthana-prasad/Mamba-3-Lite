"""
Master Experiment Runner: Mamba-3 vs Baselines & Ablations
Trains and evaluates multiple configurations on WikiText and Synthetic Benchmarks,
producing comprehensive comparison tables matching the paper's experimental protocol.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import time
import torch
from typing import Dict, Any

from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel
from mamba3.dataset import get_dataloaders
from mamba3.metrics import evaluate_model, benchmark_inference_latency
from train import get_optimizer_and_scheduler


def train_and_eval_config(
    name: str,
    config: Mamba3Config,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    max_steps: int = 100,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    print("\n" + "=" * 70, flush=True)
    print(f"RUNNING EXPERIMENT: {name}", flush=True)
    print(f"Config: d_model={config.d_model}, layers={config.n_layers}, d_state={config.d_state}", flush=True)
    print(f"Features: Trapezoidal={config.use_trapezoidal}, RoPE={config.use_rope}, BC_Bias={config.use_bc_bias}, MIMO_Rank={config.mimo_rank}", flush=True)
    print("=" * 70, flush=True)
    
    model = Mamba3LMHeadModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=0.1,
        total_steps=max_steps,
        warmup_steps=15,
    )
    
    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    
    for step in range(1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits, loss = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 25 == 0 or step == max_steps:
            print(f"  [Step {step:3d}/{max_steps}] Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
            
    train_time = time.time() - t0
    
    # Evaluate
    print("  Evaluating on Validation set...", flush=True)
    val_res = evaluate_model(model, val_loader, device=device, max_batches=15)
    print("  Evaluating on Test set...", flush=True)
    test_res = evaluate_model(model, test_loader, device=device, max_batches=20)
    
    # Latency benchmark
    bench = benchmark_inference_latency(model, device=device, seq_len=64, decode_steps=16)
    
    result = {
        "model_name": name,
        "num_parameters": num_params,
        "train_time_sec": train_time,
        "val_loss": val_res["loss"],
        "val_perplexity": val_res["perplexity"],
        "val_accuracy": val_res["accuracy"],
        "test_loss": test_res["loss"],
        "test_perplexity": test_res["perplexity"],
        "test_accuracy": test_res["accuracy"],
        "prefill_latency_ms": bench["prefill_latency_ms"],
        "decode_latency_per_token_ms": bench["decode_latency_per_token_ms"],
        "decode_throughput_tok_per_sec": bench["decode_throughput_tok_per_sec"],
    }
    
    print(f"--> {name} Results: Val PPL = {val_res['perplexity']:.2f}, Test PPL = {test_res['perplexity']:.2f}, Decode Latency = {bench['decode_latency_per_token_ms']:.2f} ms/tok", flush=True)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Master Evaluation on Device: {device}", flush=True)
    
    # DataLoaders
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        dataset_dir="dataset",
        batch_size=4,
        seq_len=128,
        max_train_samples=500,
        max_eval_samples=50,
    )
    
    vocab_size = len(tokenizer)
    base_dim = 128
    layers = 4
    d_state = 32
    headdim = 32
    
    experiments = [
        (
            "Mamba-3 (SISO)",
            Mamba3Config(
                d_model=base_dim,
                n_layers=layers,
                vocab_size=vocab_size,
                d_state=d_state,
                headdim=headdim,
                expand=2,
                mimo_rank=1,
                use_trapezoidal=True,
                use_rope=True,
                use_bc_bias=True,
                use_bc_norm=True,
                max_seq_len=128,
            )
        ),
        (
            "Mamba-3 (MIMO R=4)",
            Mamba3Config(
                d_model=base_dim,
                n_layers=layers,
                vocab_size=vocab_size,
                d_state=d_state,
                headdim=headdim,
                expand=2,
                mimo_rank=4,
                use_trapezoidal=True,
                use_rope=True,
                use_bc_bias=True,
                use_bc_norm=True,
                max_seq_len=128,
            )
        ),
        (
            "Mamba-3 (w/o RoPE)",
            Mamba3Config(
                d_model=base_dim,
                n_layers=layers,
                vocab_size=vocab_size,
                d_state=d_state,
                headdim=headdim,
                expand=2,
                mimo_rank=1,
                use_trapezoidal=True,
                use_rope=False,
                use_bc_bias=True,
                use_bc_norm=True,
                max_seq_len=128,
            )
        ),
        (
            "Mamba-2 Baseline (Euler)",
            Mamba3Config(
                d_model=base_dim,
                n_layers=layers,
                vocab_size=vocab_size,
                d_state=d_state,
                headdim=headdim,
                expand=2,
                mimo_rank=1,
                use_trapezoidal=False,
                use_rope=False,
                use_bc_bias=False,
                use_bc_norm=False,
                max_seq_len=128,
            )
        ),
    ]
    
    all_results = []
    for name, cfg in experiments:
        res = train_and_eval_config(name, cfg, train_loader, val_loader, test_loader, device, max_steps=80)
        all_results.append(res)
        
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "=" * 90, flush=True)
    print("MAMBA-3 JOURNAL EXPERIMENT SUMMARY TABLE", flush=True)
    print("=" * 90, flush=True)
    headers = f"{'Model':<25} | {'Params':<10} | {'Val PPL':<9} | {'Test PPL':<9} | {'Test Acc':<9} | {'Decode (ms/tok)':<15}"
    print(headers, flush=True)
    print("-" * 90, flush=True)
    for r in all_results:
        row = f"{r['model_name']:<25} | {r['num_parameters']:<10,} | {r['val_perplexity']:<9.2f} | {r['test_perplexity']:<9.2f} | {r['test_accuracy']:<8.2f}% | {r['decode_latency_per_token_ms']:<15.2f}"
        print(row, flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
