"""
Training Pipeline for Mamba-3 Language Model on WikiText Dataset
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel
from mamba3.dataset import get_dataloaders
from mamba3.metrics import compute_metrics, evaluate_model


def get_optimizer_and_scheduler(model: nn.Module, lr: float, weight_decay: float, total_steps: int, warmup_steps: int):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, "_no_weight_decay", False) or "norm" in name or "bias" in name or "b_bias" in name or "c_bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train():
    parser = argparse.ArgumentParser(description="Train Mamba-3 on WikiText dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Directory containing train.txt, validation.txt, test.txt")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--d_model", type=int, default=128, help="Model embedding dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--d_state", type=int, default=32, help="SSM state dimension N")
    parser.add_argument("--headdim", type=int, default=32, help="Head dimension P")
    parser.add_argument("--expand", type=int, default=2, help="Inner expansion factor")
    parser.add_argument("--mimo_rank", type=int, default=1, help="MIMO rank R (1 for SISO, 4 for MIMO)")
    parser.add_argument("--no_trapezoidal", action="store_true", help="Disable Exponential-Trapezoidal discretization")
    parser.add_argument("--no_rope", action="store_true", help="Disable Complex-valued RoPE")
    parser.add_argument("--no_bias", action="store_true", help="Disable learnable B and C biases")
    parser.add_argument("--no_bc_norm", action="store_true", help="Disable BCNorm")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence context length")
    parser.add_argument("--max_steps", type=int, default=300, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=30, help="Linear warmup steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval steps")
    parser.add_argument("--max_train_samples", type=int, default=1000, help="Max train samples")
    parser.add_argument("--max_eval_samples", type=int, default=100, help="Max eval samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("MAMBA-3 PRETRAINING PIPELINE", flush=True)
    print(f"Device: {device} | Sequence Length: {args.seq_len} | Batch Size: {args.batch_size}", flush=True)
    print(f"Trapezoidal: {not args.no_trapezoidal} | Complex RoPE: {not args.no_rope} | MIMO Rank: {args.mimo_rank}", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load Data
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    
    # 2. Setup Model Config
    config = Mamba3Config(
        d_model=args.d_model,
        n_layers=args.n_layers,
        vocab_size=len(tokenizer),
        d_state=args.d_state,
        headdim=args.headdim,
        expand=args.expand,
        mimo_rank=args.mimo_rank,
        use_trapezoidal=not args.no_trapezoidal,
        use_rope=not args.no_rope,
        use_bc_bias=not args.no_bias,
        use_bc_norm=not args.no_bc_norm,
        max_seq_len=args.seq_len,
    )
    
    model = Mamba3LMHeadModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {num_params:,} trainable parameters.", flush=True)
    
    # 3. Optimizer & LR Scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
    )
    
    # 4. Training Loop
    model.train()
    step = 0
    best_val_ppl = float("inf")
    start_time = time.time()
    
    train_iter = iter(train_loader)
    
    print("\nStarting pretraining loop...", flush=True)
    while step < args.max_steps:
        step += 1
        t0 = time.time()
        
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        
        step_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        loss_val = loss.item()
        ppl = math.exp(min(loss_val, 100))
        
        if step % 20 == 0 or step == 1 or step == args.max_steps:
            tokens_per_sec = (args.batch_size * args.seq_len) / max(step_time, 1e-5)
            print(f"Step {step:4d}/{args.max_steps} | Loss: {loss_val:.4f} | PPL: {ppl:.2f} | LR: {current_lr:.2e} | Speed: {tokens_per_sec:.1f} tok/s", flush=True)
            
        # Periodic validation
        if step % args.eval_interval == 0 or step == args.max_steps:
            print(f"\n---> Running Validation at Step {step}...", flush=True)
            val_metrics = evaluate_model(model, val_loader, device=device, max_batches=15)
            print(f"Validation Loss: {val_metrics['loss']:.4f} | Validation PPL: {val_metrics['perplexity']:.2f} | Accuracy: {val_metrics['accuracy']:.2f}%", flush=True)
            
            if val_metrics["perplexity"] < best_val_ppl:
                best_val_ppl = val_metrics["perplexity"]
                save_path = os.path.join(args.save_dir, "best_mamba3.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                }, save_path)
                print(f"Saved new best checkpoint to {save_path}\n", flush=True)
            model.train()
            
    total_time = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"Pretraining completed in {total_time:.2f} seconds.", flush=True)
    print(f"Best Validation Perplexity: {best_val_ppl:.2f}", flush=True)
    print("=" * 70, flush=True)
    
    # 5. Final Evaluation on Test Set
    print("\nRunning Final Test Evaluation on dataset/test.txt...", flush=True)
    test_metrics = evaluate_model(model, test_loader, device=device, max_batches=30)
    print("=" * 70, flush=True)
    print(f"FINAL TEST METRICS:", flush=True)
    print(f"  - Test Loss:       {test_metrics['loss']:.4f}", flush=True)
    print(f"  - Test Perplexity: {test_metrics['perplexity']:.2f}", flush=True)
    print(f"  - Test Accuracy:   {test_metrics['accuracy']:.2f}%", flush=True)
    print(f"  - Throughput:      {test_metrics['throughput_tokens_per_sec']:.1f} tokens/sec", flush=True)
    print("=" * 70, flush=True)
    
    # Save final model
    final_save_path = os.path.join(args.save_dir, "final_mamba3.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "test_metrics": test_metrics,
    }, final_save_path)
    print(f"Saved final checkpoint to {final_save_path}", flush=True)


if __name__ == "__main__":
    train()
