import math

import pytest

torch = pytest.importorskip("torch")

from model import GPT, GPTconfig  # noqa: E402

TINY = GPTconfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32)


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, TINY.block_size, TINY.vocab_size)
    # a freshly initialised model should be close to uniform: loss ~ ln(vocab)
    assert loss.item() == pytest.approx(math.log(TINY.vocab_size), rel=0.1)


def test_token_embedding_is_tied_to_lm_head():
    model = GPT(TINY)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_gpt2_small_parameter_count():
    # 124,439,808 is the figure quoted in the README chart
    with torch.device("meta"):
        model = GPT(GPTconfig())
    assert sum(p.numel() for p in model.parameters()) == 124_439_808


def test_rejects_sequences_longer_than_block_size():
    model = GPT(TINY)
    with pytest.raises(AssertionError):
        model(torch.zeros(1, TINY.block_size + 1, dtype=torch.long))


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(TINY)
    opt = model.configure_optimizers(weight_decay=0.0, learning_rate=3e-3, device="cpu")
    x = torch.randint(0, TINY.vocab_size, (2, TINY.block_size))
    for _ in range(100):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_lr_schedule():
    pytest.importorskip("tiktoken")
    import train

    assert train.get_lr(0) == pytest.approx(train.MAX_LR / train.WARMUP_STEPS)
    assert train.get_lr(train.WARMUP_STEPS) == pytest.approx(train.MAX_LR)
    assert train.get_lr(train.MAX_STEPS) == pytest.approx(train.MIN_LR)
    assert train.get_lr(train.MAX_STEPS + 100) == pytest.approx(train.MIN_LR)
