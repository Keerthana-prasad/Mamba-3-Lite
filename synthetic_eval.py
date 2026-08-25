"""
Synthetic State-Tracking Benchmark (Section 4.2 & Table 5b)
Evaluates state-tracking capabilities on Chomsky formal language tasks:
1. Parity (Bit sequence parity: sum(x) mod 2)
2. Modular Arithmetic (Modular addition / subtraction)

Demonstrates that Mamba-3 with complex-valued Data-Dependent RoPE solves state tracking,
whereas models without RoPE fail.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel


class ParityDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 24):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.data = torch.randint(0, 2, (num_samples, seq_len))
        self.targets = torch.sum(self.data, dim=1) % 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        target = self.targets[idx]
        input_ids = torch.cat([seq, target.unsqueeze(0)])
        labels = torch.full_like(input_ids, -100)
        labels[-1] = target
        return {"input_ids": input_ids, "labels": labels}


class ModularArithmeticDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 24, mod: int = 5):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.mod = mod
        self.data = torch.randint(0, mod, (num_samples, seq_len))
        self.targets = torch.sum(self.data, dim=1) % mod

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        target = self.targets[idx]
        input_ids = torch.cat([seq, target.unsqueeze(0)])
        labels = torch.full_like(input_ids, -100)
        labels[-1] = target
        return {"input_ids": input_ids, "labels": labels}


def train_and_eval_task(task_name: str, use_rope: bool, use_trapezoidal: bool, device: torch.device, epochs: int = 4):
    print(f"\n=======================================================", flush=True)
    print(f"Task: {task_name} | Model: {'Mamba-3 (Complex RoPE)' if use_rope else 'Baseline (No RoPE)'}", flush=True)
    print(f"=======================================================", flush=True)
    
    if task_name == "Parity":
        vocab_size = 4
        train_ds = ParityDataset(num_samples=1000, seq_len=24)
        test_ds = ParityDataset(num_samples=250, seq_len=48) # Test length generalization
    else:
        vocab_size = 8
        train_ds = ModularArithmeticDataset(num_samples=1000, seq_len=24, mod=5)
        test_ds = ModularArithmeticDataset(num_samples=250, seq_len=48, mod=5)
        
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    config = Mamba3Config(
        d_model=64,
        n_layers=2,
        vocab_size=vocab_size,
        d_state=32,
        headdim=32,
        expand=2,
        use_trapezoidal=use_trapezoidal,
        use_rope=use_rope,
        use_bc_bias=True,
        use_bc_norm=True,
    )
    
    model = Mamba3LMHeadModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.01)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f}", flush=True)
            
    # Evaluate on held-out length (48 vs 24)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"][:, -1].to(device)
            
            logits, _ = model(input_ids)
            preds = torch.argmax(logits[:, -2, :], dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    acc = (correct / total) * 100.0
    print(f"--> Task: {task_name} | Eval Length: 48 (2x train length) | Accuracy: {acc:.2f}%", flush=True)
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Synthetic Benchmark on device: {device}", flush=True)
    
    # 1. Parity Task
    m3_parity_acc = train_and_eval_task("Parity", use_rope=True, use_trapezoidal=True, device=device)
    base_parity_acc = train_and_eval_task("Parity", use_rope=False, use_trapezoidal=False, device=device)
    
    # 2. Modular Arithmetic Task
    m3_arith_acc = train_and_eval_task("Modular Arithmetic", use_rope=True, use_trapezoidal=True, device=device)
    base_arith_acc = train_and_eval_task("Modular Arithmetic", use_rope=False, use_trapezoidal=False, device=device)
    
    print("\n" + "="*66, flush=True)
    print("SUMMARY: Synthetic State-Tracking Capabilities (Table 5b Reproduction)", flush=True)
    print("="*66, flush=True)
    print(f"{'Model Architecture':<30} | {'Parity Accuracy':<15} | {'Modular Arith.':<15}", flush=True)
    print("-" * 66, flush=True)
    print(f"{'Mamba-3 (Complex RoPE)':<30} | {m3_parity_acc:>13.2f}% | {m3_arith_acc:>13.2f}%", flush=True)
    print(f"{'Baseline (No RoPE / Euler)':<30} | {base_parity_acc:>13.2f}% | {base_arith_acc:>13.2f}%", flush=True)
    print("="*66, flush=True)


if __name__ == "__main__":
    main()
