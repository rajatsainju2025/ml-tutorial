"""Train GPT-2 (124M) on Tiny Shakespeare.

python train.py                                    # single device
torchrun --standalone --nproc_per_node=8 train.py  # 8-GPU DDP
"""

import math
import os
import time
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTconfig

DATA_PATH = Path(__file__).resolve().parent / "data" / "input.txt"

# --- hyperparameters --------------------------------------------------------
TOTAL_BATCH_SIZE = 524288  # 2**19, ~0.5M tokens per optimizer step
B = 16  # micro batch size
T = 1024  # sequence length
MAX_LR = 3e-4  # the schedule overrides the optimizer lr every step
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 50
WEIGHT_DECAY = 0.1
SEED = 1337


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, path=DATA_PATH):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        text = Path(path).read_text()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


def get_lr(it):
    # 1) linear warmup
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    # 2) past the decay horizon, hold at the minimum
    if it > MAX_STEPS:
        return MIN_LR
    # 3) cosine decay down to the minimum
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def setup_device():
    """Return (ddp, rank, local_rank, world_size, device, master_process)."""
    # torchrun sets RANK, LOCAL_RANK and WORLD_SIZE
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP currently requires CUDA"
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        return ddp, rank, local_rank, world_size, device, rank == 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return ddp, 0, 0, 1, device, True


def main():
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_device()
    device_type = "cuda" if device.startswith("cuda") else device

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # gradient accumulation to reach TOTAL_BATCH_SIZE tokens per step
    assert TOTAL_BATCH_SIZE % (B * T * ddp_world_size) == 0, (
        "TOTAL_BATCH_SIZE must be divisible by B * T * world_size"
    )
    grad_accum_steps = TOTAL_BATCH_SIZE // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {TOTAL_BATCH_SIZE}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls

    model = GPT(GPTconfig(vocab_size=50304))  # 50304 = nicer power-of-two multiple
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=WEIGHT_DECAY, learning_rate=MAX_LR, device=device_type
    )

    for step in range(MAX_STEPS):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # only all-reduce gradients on the last micro step
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps * ddp_world_size / dt
        if master_process:
            print(
                f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | "
                f"norm: {norm:.4f} | dt: {dt:.2f} s | tok/sec: {tokens_per_sec:.2f}"
            )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
