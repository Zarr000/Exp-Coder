"""
OpenAI-Compatible API Server for Expera AI.

Provides OpenAI-compatible endpoints:
- /v1/chat/completions
- /v1/completions
- /v1/models

Usage:
    python -m src.server.app --model checkpoints/final/
    uvicorn src.server.app:app --host 0.0.0.0 --port 8000
"""

from .app import create_app, app
from .routes import router

__all__ = [
    "create_app",
    "app",
    "router",
]