from .base import Base


class TokenPonyChat(Base):
    _FACTORY_NAME = "TokenPony"

    def __init__(self, key, model_name, base_url="https://ragflow.vip-api.tokenpony.cn/v1", **kwargs):
        if not base_url:
            base_url = "https://ragflow.vip-api.tokenpony.cn/v1"
        super().__init__(key, model_name, base_url, **kwargs)
