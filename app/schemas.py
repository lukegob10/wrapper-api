from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Prompt = Union[str, List[str]]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="Model identifier")
    prompt: Prompt = Field(..., description="Text prompt or list of prompts")
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


class HealthStatus(BaseModel):
    status: str
