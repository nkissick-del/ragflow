from .base import Base


class XinferenceChat(Base):
    _FACTORY_NAME = "Xinference"

    def __init__(self, key=None, model_name="", base_url="", **kwargs):
        if not base_url:
            raise ValueError("base_url cannot be empty")
        base_url = base_url.rstrip("/") + "/v1"
        super().__init__(key, model_name, base_url, **kwargs)
