from urllib.parse import urljoin
from openai import OpenAI
from .base import Base


class LocalAIChat(Base):
    _FACTORY_NAME = "LocalAI"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        if not base_url:
            raise ValueError("Local llm url cannot be None")
        base_url = urljoin(base_url, "v1")
        self.client = OpenAI(api_key="empty", base_url=base_url)
        self.model_name = model_name.split("___")[0]
