"""Checkpoint save/load/resume + train->save->load->inference."""

import torch

from src.model.architecture import ExperaModel
from src.training import CheckpointManager, LanguageModelingLoss, Trainer, TrainerConfig

MODEL_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=64,
    max_position_embeddings=128,
    activation="gelu",
    dropout=0.0,
    attention_dropout=0.0,
)


def build_model(seed=0, vocab_size=128):
    torch.manual_seed(seed)
    return ExperaModel(**{**MODEL_KWARGS, "vocab_size": vocab_size})


def _run_steps(model, steps, opt, crit, ids):
    for _ in range(steps):
        logits = model(ids, use_cache=False, return_dict=True)["logits"]
        loss = crit(logits, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_checkpoint_manager_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = build_model(seed=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(
        model=model, optimizer=optimizer, save_dir=str(tmp_path / "ckpt"), keep_last_n=2
    )
    path = manager.save(step=10, metrics={"loss": 0.5})

    fresh = torch.load(path, map_location="cpu")
    assert fresh["format"] == "exp-coder-v1"
    assert fresh["step"] == 10
    assert "rng_state" in fresh and "torch" in fresh["rng_state"]
    key = "embed_tokens.token_embedding.embedding.weight"
    assert torch.equal(fresh["model_state_dict"][key], model.state_dict()[key])
    assert manager.get_latest_step() == 10


def test_checkpoint_manager_resume(tmp_path):
    torch.manual_seed(4)
    ids = torch.randint(0, 128, (2, 8))
    crit = LanguageModelingLoss()

    model = build_model(seed=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _run_steps(model, 3, opt, crit, ids)

    manager = CheckpointManager(model=model, optimizer=opt, save_dir=str(tmp_path / "ckpt"))
    path = manager.save(step=3, metrics={})
    expected = {k: v.clone() for k, v in model.state_dict().items()}

    model2 = build_model(seed=99)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    manager2 = CheckpointManager(model=model2, optimizer=opt2, save_dir=str(tmp_path / "ckpt"))
    manager2.resume_from(manager2.load(path))

    for k in expected:
        assert torch.equal(expected[k], model2.state_dict()[k]), k
    # continue with a valid step
    _run_steps(model2, 1, opt2, crit, ids)


def test_trainer_save_load_continue(tmp_path):
    ids = torch.randint(0, 128, (2, 8))

    def make_trainer(model):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return Trainer(TrainerConfig(
            model=model,
            train_dataloader=[],
            optimizer=opt,
            max_steps=100,
            gradient_accumulation_steps=1,
            use_amp=False,
            device="cpu",
            use_ema=False,
            save_every=50,
            save_dir=str(tmp_path / "train"),
            log_every=1000,
        ))

    trainer1 = make_trainer(build_model(seed=5))
    trainer1.global_step = 4
    trainer1.save_checkpoint("step_4")

    trainer2 = make_trainer(build_model(seed=99))
    trainer2.load_checkpoint(str(tmp_path / "train/step_4.pt"))
    assert trainer2.global_step == 4

    batch = {"input_ids": ids, "labels": ids}
    metrics = trainer2.train_step(batch)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_train_save_load_inference(tiny_tokenizer, tmp_path):
    torch.manual_seed(6)
    vocab = len(tiny_tokenizer) + 4
    ids = torch.randint(0, 128, (1, 8))
    model = build_model(seed=6, vocab_size=vocab)
    crit = LanguageModelingLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _run_steps(model, 5, opt, crit, ids)

    path = tmp_path / "ckpt" / "model.pt"
    path.parent.mkdir(parents=True)
    torch.save({
        "format": "exp-coder-v1",
        "global_step": 5,
        "model_state_dict": model.state_dict(),
    }, path)

    from src.inference import GenerationConfig, Generator
    from src.inference.generator import DecodingStrategy

    loaded = build_model(seed=0, vocab_size=vocab)
    loaded.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
    loaded.eval()

    gen = Generator(loaded, tiny_tokenizer, GenerationConfig(
        strategy=DecodingStrategy.GREEDY, max_new_tokens=5, use_cache=False))
    text = gen.generate("def fibonacci")
    assert isinstance(text, str)