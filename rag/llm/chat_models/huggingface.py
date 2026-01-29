from .base import Base


class HuggingFaceChat(Base):
    _FACTORY_NAME = "HuggingFace"

    def __init__(self, key=None, model_name="", base_url="", **kwargs):
        if not base_url or not base_url.strip():
            raise ValueError("Local llm url cannot be empty or None")
        base_url = base_url.rstrip("/") + "/v1"
        # model_name may include a suffix after "___" (e.g., variant, adapter, or profile metadata) which is intentionally
        # stripped by model_name.split("___")[0] before passing to super().__init__
        super().__init__(key, model_name.split("___")[0], base_url, **kwargs)
