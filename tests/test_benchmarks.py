"""Unit tests for benchmark infrastructure (fast, no training involved)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import common  # noqa: E402

BENCH = ROOT / "benchmarks"
RESULTS = BENCH / "results"


@pytest.fixture(scope="module")
def scripts():
    return {
        "tokenizer": BENCH / "tokenizer_benchmark.py",
        "overfit": BENCH / "overfit.py",
        "train_benchmark": BENCH / "train_benchmark.py",
        "corpus": BENCH / "make_code_corpus.py",
    }


def _parseable(script: Path) -> bool:
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0


def test_result_serialization_roundtrip(tmp_path):
    out = tmp_path / "r.json"
    common.write_result(out, {"x": 1.5})
    data = common.load_result(out)
    assert data["x"] == 1.5
    assert data["meta"]["commit"]
    assert data["meta"]["timestamp"]


def test_cuda_info_is_dict():
    info = common.cuda_info()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "cuda_device_count" in info
    assert isinstance(info["cuda_available"], bool)


def test_memory_probe_is_none_or_float():
    mb = common.memory_mb()
    assert mb is None or isinstance(mb, float)


@pytest.mark.parametrize("name", ["tokenizer", "overfit", "train_benchmark", "corpus"])
def test_benchmark_scripts_parse_args(name, scripts):
    assert _parseable(scripts[name]), f"{name} --help failed"


def test_result_json_finite_metrics():
    """Committed benchmark results must be finite and non-degenerate."""
    files = [
        RESULTS / "tokenizer.json",
        RESULTS / "overfit.json",
        RESULTS / "overfit_lr_sweep.json",
        RESULTS / "cpu_120m.json",
    ]
    for path in files:
        assert path.exists(), f"missing committed result {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("meta", {}).get("commit")

    tok = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    o = tok["overall"]
    assert o["chars"] > 0 and o["tokens"] > 0
    assert o["chars_per_token"] > 0
    assert math.isfinite(o["tokens_per_1k_chars"])
    assert o["unk_rate"] == 0.0

    overfit = json.loads((RESULTS / "overfit.json").read_text(encoding="utf-8"))
    for run in overfit["runs"].values():
        assert math.isfinite(run["initial_loss"])
        assert math.isfinite(run["final_loss"])
        assert run["unstable"] == {
            "nan_step": None, "inf_grad": None, "inf_param": None
        }
        for point in run["curve"]:
            assert math.isfinite(point["loss"])
            assert math.isfinite(point["grad_norm"])
            assert math.isfinite(point["param_norm"])

    cpu = json.loads((RESULTS / "cpu_120m.json").read_text(encoding="utf-8"))
    assert cpu["params_measured"] > 100_000_000
    for cell in cpu["cells"]:
        assert cell["avg_step_sec"] > 0
        assert cell["tokens_per_sec"] > 0
        assert cell["finite"] is True