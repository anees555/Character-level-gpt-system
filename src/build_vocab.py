from pathlib import Path
import torch

from dataset import CharDataset

def build_and_save_vocab(
        text_path: str = "../dataset/data-nepali-cleaned",
        block_size: int = 128,
        vocab_text_path: str = "../vocabulary/vocab.txt",
        vocab_pt_path: str = "../vocabulary/vocab.pt"
):
    text = Path(text_path).read_text(encoding="utf-8")
    dataset = CharDataset(text=text, block_size=block_size)
    dataset.save_vocab(vocab_text_path)
    torch.save(
        {
            "stoi": dataset.stoi,
            "itos": dataset.itos,
            "vocab_size": dataset.vocab_size,
            "block_size": dataset.block_size
        },
        vocab_pt_path,
    )
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Saved text vocab: {vocab_text_path}")
    print(f"Saved binary vocab: {vocab_pt_path}")
    return dataset

if __name__ == "__main__":
    ds = build_and_save_vocab()
    

 