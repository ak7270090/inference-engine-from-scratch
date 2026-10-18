"""Qwen3 tokenizer wrapper: encode, decode, stop tokens, and streaming-safe incremental decoding."""

import json
from pathlib import Path

from tokenizers import Tokenizer

from nanoengine.constants import (
    CONFIG_FILE_NAME,
    EOS_TOKEN_ID_KEY,
    GENERATION_CONFIG_FILE_NAME,
    TOKENIZER_FILE_NAME,
    UTF8_REPLACEMENT_CHARACTER,
)


class TextTokenizer:
    def __init__(self, tokenizer: Tokenizer, stop_token_ids: frozenset[int]) -> None:
        self._tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)


class IncrementalDecoder:
    """Turns token IDs into text pieces one at a time without splitting a multi-byte character."""

    def __init__(self, tokenizer: TextTokenizer) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted_text = ""

    def add_token(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        text = self._tokenizer.decode(self._token_ids)
        if text.endswith(UTF8_REPLACEMENT_CHARACTER):
            return ""
        return self._take_new_text(text)

    def flush(self) -> str:
        return self._take_new_text(self._tokenizer.decode(self._token_ids))

    def _take_new_text(self, text: str) -> str:
        new_text = text[len(self._emitted_text) :]
        self._emitted_text = text
        return new_text


def read_stop_token_ids(model_dir: Path) -> frozenset[int]:
    """EOS IDs from ``generation_config.json`` and ``config.json``; either may hold a list."""
    stop_token_ids: set[int] = set()
    for file_name in (GENERATION_CONFIG_FILE_NAME, CONFIG_FILE_NAME):
        config_path = model_dir / file_name
        if not config_path.exists():
            continue
        eos_token_id = json.loads(config_path.read_text(encoding="utf-8")).get(EOS_TOKEN_ID_KEY)
        if isinstance(eos_token_id, int):
            stop_token_ids.add(eos_token_id)
        elif isinstance(eos_token_id, list):
            stop_token_ids.update(eos_token_id)
    return frozenset(stop_token_ids)


def load_text_tokenizer(model_dir: Path) -> TextTokenizer:
    tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE_NAME))
    return TextTokenizer(tokenizer, read_stop_token_ids(model_dir))
