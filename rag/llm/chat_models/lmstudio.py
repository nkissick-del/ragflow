from openai import OpenAI
from .base import Base


class LmStudioChat(Base):
    _FACTORY_NAME = "LM-Studio"

    def __init__(self, key, model_name, base_url, **kwargs):
        if not base_url:
            raise ValueError("Local llm url cannot be None")
        base_url = base_url.rstrip("/") + "/v1"
        super().__init__(key, model_name, base_url, **kwargs)
        if not key:
            key = "lm-studio"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name
