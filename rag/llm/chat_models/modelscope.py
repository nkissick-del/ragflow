from .base import Base


class ModelScopeChat(Base):
    _FACTORY_NAME = "ModelScope"

    def __init__(self, key=None, model_name="", base_url="", **kwargs):
        if not base_url:
            raise ValueError("Local llm url cannot be empty or None")
        base_url = base_url.rstrip("/") + "/v1"
        super().__init__(key, model_name.split("___")[0], base_url, **kwargs)
