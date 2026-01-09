from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Prompt = Union[str, List[str]]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model identifier")
    prompt: Prompt = Field(..., description="Text prompt or list of prompts")
    stream_options: Optional[dict[str, Any]] = None
    suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    user: Optional[str] = None


class Choice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: Any = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage]
    stream_options: Optional[dict[str, Any]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None


class ChatMessageOut(BaseModel):
    role: str
    content: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessageOut
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "wrapper-api"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class HealthStatus(BaseModel):
    status: str
