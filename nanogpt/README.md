# nanogpt

GPT-2 (124M) reproduced from scratch, following Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt).

| File | Purpose |
|---|---|
| `model.py` | `GPT`, `GPTconfig`, attention/MLP blocks, `from_pretrained` (HF weights), optimizer groups |
| `train.py` | Tiny Shakespeare training: grad accumulation to 0.5M-token batches, warmup + cosine LR, bf16 autocast, `torch.compile`, DDP |
| `sample.py` | Top-k sampling through `model.py` with OpenAI's released weights |
| `data/input.txt` | Tiny Shakespeare (~1.1 MB) |
| `notebooks/` | The step-by-step build (`gpt2_walkthrough`) and scratch experiments (`snippets`) |

```bash
python sample.py --prompt "Hello, I'm a language model,"
python train.py
torchrun --standalone --nproc_per_node=8 train.py
```

Tests live in [`../tests/test_nanogpt.py`](../tests/test_nanogpt.py).
