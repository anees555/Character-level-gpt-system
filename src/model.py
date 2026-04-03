import torch
import torch.nn as nn
import torch.nn.functional as F
# from dataclasses import dataclass

# @dataclass
# class GPTConfig:
#     vocab_size: int
#     block_size: int = 256
#     n_embed: int = 384
#     n_head: int = 6
#     n_layer: int = 6
#     dropout: float = 0.2

class Head(nn.Module):
    """Single attention head"""

    def __init__(self, head_size:int, block_size:int, dropout:float = 0.1):
        super().__init__()
        self.key = nn.Linear(head_size, head_size, bias=False)
        self.query = nn.Linear(head_size, head_size, bias=False)
        self.value = nn.Linear(head_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) * (C ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim = -1)
        attn = self.dropout(attn)
        out = attn @ v 
        return out