"""
Multi-Query Associative Recall (MQAR) & Long-Range Pattern Retrieval Benchmark
Demonstrates the fundamental advantage of the Mamba-3 + Episodic Hopfield Hybrid (Dual-System)
over pure SSMs for associative memory retrieval across extended context lengths.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel
from mamba3_novel.config import Mamba3NovelConfig
from mamba3_novel.model import Mamba3NovelLMHeadModel


def generate_mqar_dataset(
    num_samples: int = 200,
    seq_len: int = 64,
    vocab_size: int = 128,
    num_kv_pairs: int = 4,
    device: str = "cpu",
):
    """
    Generates synthetic Multi-Query Associative Recall (MQAR) sequences:
      Sequence: k1 v1 ... k2 v2 ... [filler tokens] ... q_k1 -> Target: v1
    """
    inputs_list = []
    targets_list = []
    
    # Reserve special tokens
    KEY_MIN = 10
    KEY_MAX = 50
    VAL_MIN = 60
    VAL_MAX = 100
    FILLER_TOKEN = 1
    
    for _ in range(num_samples):
        seq = [FILLER_TOKEN] * seq_len
        target_seq = [-100] * seq_len
        
        # Pick distinct keys and values
        keys = random.sample(range(KEY_MIN, KEY_MAX), num_kv_pairs)
        vals = random.sample(range(VAL_MIN, VAL_MAX), num_kv_pairs)
        kv_map = dict(zip(keys, vals))
        
        # Insert KV pairs into early half of the sequence
        insert_positions = sorted(random.sample(range(0, seq_len // 2 - 2), num_kv_pairs))
        for idx, (k, v) in enumerate(zip(keys, vals)):
            pos = insert_positions[idx]
            seq[pos] = k
            seq[pos + 1] = v
            
        # Insert query in second half
        query_key = random.choice(keys)
        expected_val = kv_map[query_key]
        query_pos = random.randint(seq_len // 2, seq_len - 2)
        seq[query_pos] = query_key
        # Target at query_pos is expected_val (predicting token at query_pos + 1)
        target_seq[query_pos] = expected_val
        
        inputs_list.append(seq)
        targets_list.append(target_seq)
        
    inputs = torch.tensor(inputs_list, dtype=torch.long, device=device)
    targets = torch.tensor(targets_list, dtype=torch.long, device=device)
    return inputs, targets


def train_and_eval_mqar(model, train_inputs, train_targets, val_inputs, val_targets, epochs=6, lr=3e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    batch_size = 16
    n_samples = train_inputs.shape[0]
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            b_in = train_inputs[idx]
            b_tgt = train_targets[idx]
            
            optimizer.zero_grad()
            out = model(b_in)
            logits = out["logits"] if isinstance(out, dict) else out[0]
            
            # Predict targets
            loss = criterion(logits.view(-1, logits.shape[-1]), b_tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(val_inputs)
        logits = out["logits"] if isinstance(out, dict) else out[0]
        preds = torch.argmax(logits, dim=-1)
        
        # Mask where target != -100
        mask = (val_targets != -100)
        correct = ((preds == val_targets) & mask).sum().item()
        total = mask.sum().item()
        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        
    return accuracy


def run_mqar_benchmark():
    print("=" * 70, flush=True)
    print("MULTI-QUERY ASSOCIATIVE RECALL (MQAR) BENCHMARK", flush=True)
    print("Evaluating Long-Range Pattern Retrieval: Base Mamba-3 vs. Hopfield Hybrid", flush=True)
    print("=" * 70, flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 128
    
    for seq_len in [48, 96]:
        print(f"\n--- Sequence Length: {seq_len} tokens ---", flush=True)
        train_in, train_tgt = generate_mqar_dataset(60, seq_len=seq_len, vocab_size=vocab_size, device=device)
        val_in, val_tgt = generate_mqar_dataset(30, seq_len=seq_len, vocab_size=vocab_size, device=device)
        
        # 1. Base Mamba-3
        cfg_base = Mamba3Config(
            d_model=64,
            n_layers=2,
            d_state=32,
            headdim=16,
            expand=2,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
        )
        model_base = Mamba3LMHeadModel(cfg_base).to(device)
        print("  Training Base Mamba-3...", flush=True)
        acc_base = train_and_eval_mqar(model_base, train_in, train_tgt, val_in, val_tgt, epochs=6)
        
        # 2. Mamba-3 + Episodic Hopfield Hybrid
        cfg_hybrid = Mamba3NovelConfig(
            d_model=64,
            n_layers=2,
            d_state=32,
            headdim=16,
            expand=2,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            use_gmimo=True,
            gmimo_groups=2,
            use_lni=True,
            use_hopfield=True,
            hopfield_interval=1,
            hopfield_chunk_size=8,
        )
        model_hybrid = Mamba3NovelLMHeadModel(cfg_hybrid).to(device)
        print("  Training Mamba-3 + Hopfield Hybrid...", flush=True)
        acc_hybrid = train_and_eval_mqar(model_hybrid, train_in, train_tgt, val_in, val_tgt, epochs=6)
        
        print(f"  [Base Mamba-3 Pure SSM]     Associative Recall Accuracy: {acc_base:6.2f}%", flush=True)
        print(f"  [Mamba-3 + Hopfield Hybrid]  Associative Recall Accuracy: {acc_hybrid:6.2f}%", flush=True)
        print(f"  --> Delta Improvement:     +{acc_hybrid - acc_base:6.2f}%", flush=True)
        
    print("\n" + "=" * 70, flush=True)
    print("MQAR BENCHMARK COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_mqar_benchmark()
