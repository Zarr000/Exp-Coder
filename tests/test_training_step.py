"""Loss correctness and a real trainable mini-step (forward/backward/step)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training import LanguageModelingLoss


def test_loss_matches_manual_shift():
    b, s, v = 2, 6, 128
    logits = torch.randn(b, s, v)
    labels = torch.randint(0, v, (b, s))

    criterion = LanguageModelingLoss()
    loss = criterion(logits, labels)

    shift_logits = logits[:, :-1, :].reshape(-1, v)
    shift_labels = labels[:, 1:].reshape(-1)
    manual = F.cross_entropy(shift_logits, shift_labels)
    assert torch.allclose(loss, manual, atol=1e-6)


def test_loss_ignores_last_position(tiny_model):
    ids = torch.randint(0, 64, (1, 5))
    logits = tiny_model(ids, use_cache=False, return_dict=True)["logits"]
    loss = LanguageModelingLoss()(logits, ids)
    # cross-entropy over 4 aligned positions (s-1 pairs)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_backward_and_optimizer_step(tiny_model):
    torch.manual_seed(1)
    model = tiny_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = LanguageModelingLoss()

    ids = torch.randint(0, 64, (2, 8))
    logits = model(ids, use_cache=False, return_dict=True)["logits"]
    loss = criterion(logits, ids)
    assert torch.isfinite(loss)

    loss.backward()

    named = dict(model.named_parameters())
    assert all(p.grad is not None for p in model.parameters())
    grads = torch.cat([p.grad.flatten() for p in model.parameters()])
    assert torch.isfinite(grads).all()

    before = next(model.parameters()).detach().clone()
    optimizer.step()
    optimizer.zero_grad()
    assert not torch.equal(next(model.parameters()).detach(), before)


def test_loss_decreases_on_repeated_step(tiny_model):
    torch.manual_seed(2)
    model = tiny_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = LanguageModelingLoss()
    ids = torch.randint(0, 64, (4, 16))

    losses = []
    for _ in range(6):
        logits = model(ids, use_cache=False, return_dict=True)["logits"]
        loss = criterion(logits, ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]