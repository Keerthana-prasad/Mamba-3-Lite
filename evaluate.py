"""
Evaluation & Inference Demo Script for Mamba-3
Evaluates checkpoint on test and validation sets, benchmarks latency, and generates text completions.
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import torch
from transformers import AutoTokenizer

from mamba3.config import Mamba3Config
from mamba3.model import Mamba3LMHeadModel
from mamba3.dataset import get_dataloaders
from mamba3.metrics import evaluate_model, benchmark_inference_latency


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Mamba-3 Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_mamba3.pt", help="Path to checkpoint")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="The state space model architecture is", help="Prompt for generation")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = Mamba3LMHeadModel(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"No checkpoint found at {args.checkpoint}. Instantiating fresh Mamba-3 model for benchmark...", flush=True)
        config = Mamba3Config(d_model=128, n_layers=4, d_state=32, headdim=32, max_seq_len=args.seq_len)
        model = Mamba3LMHeadModel(config).to(device)
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 1. Dataset Evaluation
    _, val_loader, test_loader, _ = get_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_eval_samples=50,
    )
    
    print("\n" + "=" * 65, flush=True)
    print("MAMBA-3 MODEL EVALUATION", flush=True)
    print("=" * 65, flush=True)
    
    print("Evaluating on Validation Set (dataset/validation.txt)...", flush=True)
    val_res = evaluate_model(model, val_loader, device=device)
    print(f"  - Validation Loss:       {val_res['loss']:.4f}", flush=True)
    print(f"  - Validation Perplexity: {val_res['perplexity']:.2f}", flush=True)
    print(f"  - Validation Accuracy:   {val_res['accuracy']:.2f}%", flush=True)
    
    print("\nEvaluating on Test Set (dataset/test.txt)...", flush=True)
    test_res = evaluate_model(model, test_loader, device=device)
    print(f"  - Test Loss:             {test_res['loss']:.4f}", flush=True)
    print(f"  - Test Perplexity:       {test_res['perplexity']:.2f}", flush=True)
    print(f"  - Test Accuracy:         {test_res['accuracy']:.2f}%", flush=True)
    print(f"  - Throughput:            {test_res['throughput_tokens_per_sec']:.1f} tok/s", flush=True)
    
    # 2. Benchmark Inference Latency
    print("\n" + "=" * 65, flush=True)
    print("INFERENCE LATENCY PROFILING (Table 6 & Table 7 Reproduction)", flush=True)
    print("=" * 65, flush=True)
    bench = benchmark_inference_latency(model, device=device, seq_len=64, decode_steps=16)
    print(f"  - Prefill Latency (64 tokens):  {bench['prefill_latency_ms']:.2f} ms", flush=True)
    print(f"  - Decode Latency per token:     {bench['decode_latency_per_token_ms']:.2f} ms", flush=True)
    print(f"  - Decode Throughput:            {bench['decode_throughput_tok_per_sec']:.1f} tokens/sec", flush=True)
    
    # 3. Text Generation Demo
    print("\n" + "=" * 65, flush=True)
    print("AUTOREGRESSIVE GENERATION SAMPLE (Recurrent State Cache)", flush=True)
    print("=" * 65, flush=True)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    generated_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8, use_cache=True)
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    print(f"Prompt:     {args.prompt}", flush=True)
    print(f"Generated:  {generated_text}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    evaluate()
