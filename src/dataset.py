import os
from typing import Tuple

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    character-level dataset for for next-character prediction
    x = text[i : i + block_size]
    y = text[i+1 : i + block_size + 1]
    """
    def __init__(self, text:str, block_size: int):
        if not text:
            raise ValueError("Input text is empty!")
        if block_size <= 0:
            raise ValueError("Block size must be greater than 0")
        if len(text) <= block_size:
            raise ValueError(
                f"Text length must be greater than block size"
            )
        self.block_size = block_size
        self.text = text

        # build vocabulary
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.data = torch.tensor([self.stoi[c] for c in text])
        self.vocab_size = len(chars)

    def __len__(self) -> int:
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y
    
    def encode(self, s:str) -> torch.Tensor:
        """convert string to tensor of token ids"""
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)
    
    def decode(self, ids) -> str:
        """convert token ids back to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)
    
    def save_vocab(self, path: str) -> None:
        """save vocabulary to a text file"""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", encoding="utf-8") as f:
            for i in range(self.vocab_size):
                ch = self.itos[i]
                f.write(f"{i}\t{repr(ch)}\n")



