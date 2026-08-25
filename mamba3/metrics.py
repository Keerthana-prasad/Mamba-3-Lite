import math
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Computes standard language modeling metrics:
      - CrossEntropy Loss
      - Perplexity (exp(Loss))
      - Next-Token Prediction Accuracy
      - Bits-Per-Byte / Bits-Per-Token
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    
    preds = torch.argmax(shift_logits, dim=-1)
    correct = (preds == shift_labels).float()
    accuracy = correct.mean().item()
    
    loss_val = loss.item()
    ppl = math.exp(min(loss_val, 100)) # prevent overflow
    bpc = loss_val / math.log(2.0)
    
    return {
        "loss": loss_val,
        "perplexity": ppl,
        "accuracy": accuracy,
        "bits_per_token": bpc,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates the model over an entire dataset / dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        logits, _ = model(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        
        preds = torch.argmax(shift_logits, dim=-1)
        correct = (preds == shift_labels).sum().item()
        num_tokens = shift_labels.numel()
        
        total_loss += loss.item()
        total_correct += correct
        total_tokens += num_tokens
        
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))
    acc = total_correct / max(total_tokens, 1)
    tokens_per_sec = total_tokens / max(elapsed, 1e-5)
    
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "accuracy": acc * 100.0, # percentage
        "total_tokens": total_tokens,
        "eval_time_sec": elapsed,
        "throughput_tokens_per_sec": tokens_per_sec,
    }


@torch.no_grad()
def benchmark_inference_latency(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 128,
    decode_steps: int = 64,
    warmup_reps: int = 2,
    test_reps: int = 5,
) -> Dict[str, float]:
    """
    Measures prefill and recurrent token decode latency.
    """
    model.eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)
    
    # Warmup
    for _ in range(warmup_reps):
        _ = model.generate(prompt, max_new_tokens=8, use_cache=True)
        
    # Benchmark Prefill
    prefill_times = []
    for _ in range(test_reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_times.append((time.time() - t0) * 1000.0) # ms
        
    avg_prefill_ms = sum(prefill_times) / len(prefill_times)
    
    # Benchmark Decode
    decode_times = []
    for _ in range(test_reps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model.generate(prompt, max_new_tokens=decode_steps, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_times.append((time.time() - t0) * 1000.0)
        
    avg_decode_total_ms = sum(decode_times) / len(decode_times)
    per_token_decode_ms = avg_decode_total_ms / decode_steps
    
    return {
        "prefill_latency_ms": avg_prefill_ms,
        "decode_latency_per_token_ms": per_token_decode_ms,
        "decode_throughput_tok_per_sec": 1000.0 / per_token_decode_ms if per_token_decode_ms > 0 else 0,
    }
