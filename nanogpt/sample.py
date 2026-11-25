"""Sample text from GPT-2 using OpenAI's released weights (sanity check for model.py).

python sample.py --prompt "Hello, I'm a language model," --n 5 --max-length 30
"""

import argparse

import tiktoken
import torch
from torch.nn import functional as F

from model import GPT


@torch.no_grad()
def generate(model, idx, max_length, top_k=50):
    while idx.size(1) < max_length:
        logits, _ = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)  # (B, top_k)
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        idx = torch.cat((idx, torch.gather(topk_indices, -1, ix)), dim=1)
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Hello, I'm a language model,")
    p.add_argument("--n", type=int, default=5, help="number of samples")
    p.add_argument("--max-length", type=int, default=30)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = GPT.from_pretrained(args.model).to(device).eval()
    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(enc.encode(args.prompt), dtype=torch.long, device=device)
    idx = idx.unsqueeze(0).repeat(args.n, 1)

    out = generate(model, idx, args.max_length)
    for row in out:
        print(">", enc.decode(row.tolist()))


if __name__ == "__main__":
    main()
