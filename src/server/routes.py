"""
API Routes for OpenAI-Compatible Endpoints.

Provides:
- /v1/chat/completions
- /v1/completions
- /v1/models
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()


# ===== Request Models =====


class ChatMessage(BaseModel):
    """Chat message."""
    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    """Legacy completion request."""
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


# ===== Response Models =====


class Usage(BaseModel):
    """Token usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatMessageResponse(BaseModel):
    """Chat message response."""
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """Completion choice."""
    index: int
    message: Optional[ChatMessageResponse] = None
    text: Optional[str] = None
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class CompletionResponse(BaseModel):
    """Legacy completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class Model(BaseModel):
    """Model info."""
    id: str
    object: str = "model"
    owned_by: str = "expera"
    permission: List[Dict[str, Any]] = Field(default_factory=list)  # type: ignore


class ModelList(BaseModel):
    """List of models."""
    object: str = "list"
    data: List[Model]


# ===== Routes =====


import time
import uuid


def get_model_service():
    """Get model service from app state."""
    from .app import model_service
    return model_service


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint (OpenAI-compatible).
    """
    service = get_model_service()

    if not service:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Extract messages
    messages = request.messages

    # Build prompt from messages
    prompt = _build_prompt(messages)

    # Generate
    if request.stream:
        return StreamingResponse(
            _stream_chat(prompt, service, request),
            media_type="text/event-stream",
        )
    else:
        result = await service.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    message=ChatMessageResponse(content=result["text"]),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
            ),
        )


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Legacy completions endpoint.
    """
    service = get_model_service()

    if not service:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Generate
    if request.stream:
        return StreamingResponse(
            _stream_complete(request, service),
            media_type="text/event-stream",
        )
    else:
        result = await service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
        )

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(text=result["text"], finish_reason="stop")],
            usage=Usage(
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
            ),
        )


@router.get("/v1/models")
async def list_models():
    """
    List available models.
    """
    return ModelList(
        data=[
            Model(id="expera-base", owned_by="expera"),
            Model(id="expera-small", owned_by="expera"),
            Model(id="expera-large", owned_by="expera"),
        ]
    )


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model info."""
    return Model(id=model_id, owned_by="expera")


# ===== Helpers =====


def _build_prompt(messages: List[ChatMessage]) -> str:
    """Build prompt from chat messages."""
    prompt = ""
    for msg in messages:
        role = msg.role
        content = msg.content
        prompt += f"{role.capitalize()}: {content}\n"
    prompt += "Assistant:"
    return prompt


async def _stream_chat(prompt: str, service, request: ChatCompletionRequest):
    """Stream chat response."""
    async for chunk in service.stream_generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7,
    ):
        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

    yield "data: [DONE]\n\n"


async def _stream_complete(prompt: str, service, request: CompletionRequest):
    """Stream completion response."""
    async for chunk in service.stream_generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7,
    ):
        yield f"data: {json.dumps({'choices': [{'text': chunk}]})}\n\n"

    yield "data: [DONE]\n\n"


__all__ = ["router"]