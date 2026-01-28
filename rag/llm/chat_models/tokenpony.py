from .base import Base


class TokenPonyChat(Base):
    _FACTORY_NAME = "TokenPony"

    DEFAULT_BASE_URL = "https://ragflow.vip-api.tokenpony.cn/v1"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(key, model_name, base_url, **kwargs)
