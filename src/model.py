import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_embed: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

