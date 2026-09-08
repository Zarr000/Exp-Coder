"""
FastAPI Application for OpenAI-Compatible API.

Provides:
- /v1/chat/completions (ChatGPT-compatible)
- /v1/completions (Legacy)
- /v1/models
- /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Optional, List, Dict, Any

from .routes import router
from .service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state
model_service: Optional[ModelService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    global model_service

    # Initialize on startup
    logger.info("Starting API server...")

    yield

    # Cleanup
    logger.info("Shutting down API server...")


def create_app(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        model_path: Model checkpoint path
        tokenizer_path: Tokenizer path
        device: Device to run on

    Returns:
        FastAPI app
    """
    global model_service

    # Create app
    app = FastAPI(
        title="Expera AI API",
        description="OpenAI-compatible API for Expera AI",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router)

    # Initialize model service
    if model_path:
        model_service = ModelService(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        logger.info(f"Model loaded: {model_path}")

    return app


# Default app
app = create_app()


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": model_service is not None,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Expera AI API",
        "version": "1.0.0",
    }


__all__ = ["create_app", "app"]