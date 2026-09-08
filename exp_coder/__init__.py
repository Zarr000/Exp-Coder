"""Exp-Coder — Exp Works, by Zarr.

Canonical public package. The implementation lives under ``src/`` (kept for
backward compatibility with legacy scripts); this package re-exports the core
subsystems so they can be imported canonically as ``exp_coder.model``,
``exp_coder.tokenizer``, ``exp_coder.training``, etc.
"""

import importlib
import sys as _sys

_CORE = (
    "model",
    "tokenizer",
    "training",
    "data",
    "inference",
    "utils",
    "config",
)

for _name in _CORE:
    _mod = importlib.import_module(f"src.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

__version__ = "0.1.0"
__author__ = "Zarr"
__studio__ = "Exp Works"