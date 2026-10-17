"""OpenAI-style request and response bodies for ``/v1/completions`` and ``/v1/models``."""

from pydantic import BaseModel, Field

from nanoengine.constants import LIST_OBJECT, MODEL_OBJECT, MODEL_OWNER, TEXT_COMPLETION_OBJECT


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False

    @property
    def should_include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """A full response, or one SSE chunk when streaming (then ``usage`` is set only at the end)."""

    id: str
    object: str = TEXT_COMPLETION_OBJECT
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None


class ModelCard(BaseModel):
    id: str
    object: str = MODEL_OBJECT
    owned_by: str = MODEL_OWNER
    max_model_len: int


class ModelList(BaseModel):
    object: str = LIST_OBJECT
    data: list[ModelCard]
