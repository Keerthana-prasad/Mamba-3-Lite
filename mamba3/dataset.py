import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """
    Fast streaming Dataset loader for WikiText language modeling benchmarks.
    Efficiently tokenizes line-by-line and chunks into fixed-size context windows.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        seq_len: int = 256,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(file_path))[0]
            suffix = f"_max{max_samples}" if max_samples else "_full"
            cache_file = os.path.join(cache_dir, f"{basename}_{seq_len}{suffix}_tokens.pt")
            
        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading cached tokens from {cache_file}...", flush=True)
            self.tokens = torch.load(cache_file)
        else:
            print(f"Tokenizing {os.path.basename(file_path)}...", flush=True)
            target_tokens = (max_samples * seq_len + seq_len) if max_samples is not None else None
            token_list = []
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                batch_text = []
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        batch_text.append(line_str)
                    if len(batch_text) >= 200:
                        joined = "\n".join(batch_text)
                        enc = self.tokenizer.encode(joined)
                        token_list.extend(enc)
                        batch_text = []
                        if target_tokens is not None and len(token_list) >= target_tokens:
                            break
                            
                if batch_text:
                    joined = "\n".join(batch_text)
                    enc = self.tokenizer.encode(joined)
                    token_list.extend(enc)
                    
            if target_tokens is not None and len(token_list) > target_tokens:
                token_list = token_list[:target_tokens]
                
            self.tokens = torch.tensor(token_list, dtype=torch.long)
            
            if cache_file is not None:
                torch.save(self.tokens, cache_file)
                print(f"Saved tokenized cache to {cache_file}", flush=True)
                
        self.num_chunks = len(self.tokens) // self.seq_len
        if max_samples is not None and max_samples > 0:
            self.num_chunks = min(self.num_chunks, max_samples)
            
        print(f"Loaded '{os.path.basename(file_path)}': {len(self.tokens):,} tokens, {self.num_chunks:,} samples (seq_len={seq_len})", flush=True)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.tokens[start:end]
        return {
            "input_ids": chunk,
            "labels": chunk.clone(),
        }


def get_dataloaders(
    dataset_dir: str = "dataset",
    batch_size: int = 4,
    seq_len: int = 128,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    cache_dir = os.path.join(dataset_dir, ".cache")
    
    train_path = os.path.join(dataset_dir, "train.txt")
    val_path = os.path.join(dataset_dir, "validation.txt")
    test_path = os.path.join(dataset_dir, "test.txt")
    
    train_dataset = WikiTextDataset(
        train_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_train_samples,
        cache_dir=cache_dir,
    )
    val_dataset = WikiTextDataset(
        val_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_eval_samples,
        cache_dir=cache_dir,
    )
    test_dataset = WikiTextDataset(
        test_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=max_eval_samples,
        cache_dir=cache_dir,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, test_loader, tokenizer
