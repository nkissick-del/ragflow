from typing import Any
from .base import Base


class OpenAI_APIChat(Base):
    _FACTORY_NAME = ("VLLM", "OpenAI-API-Compatible")

    def __init__(self, key: str, model_name: str, base_url: str, **kwargs: Any) -> None:
        if not base_url:
            raise ValueError("base_url cannot be empty or None")
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        # "___" is used as a delimiter to append metadata (e.g., version, backend, or config);
        # only the base model name (left of the delimiter) should be used for OpenAI calls.
        model_name = model_name.split("___")[0]
        super().__init__(key, model_name, base_url, **kwargs)
