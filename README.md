## Mini LLM trained from scratch (character-level GPT) System

#### Simple Architecture:
**Characters** → **Embedding Layer** → **Transformer Blocks** → **Linear Layer** → **Next Character**

### Training Pipeline:
**Dataset** → **Tokenization** → **Batching** → **Model** → **Training** → **Loss** → **Backpropagation** → **Save Model** → **Text** **Generation**