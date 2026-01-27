from .base import Base


class LeptonAIChat(Base):
    _FACTORY_NAME = "LeptonAI"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        if not base_url:
            base_url = f"https://{model_name}.lepton.run/api/v1"
        super().__init__(key, model_name, base_url, **kwargs)
