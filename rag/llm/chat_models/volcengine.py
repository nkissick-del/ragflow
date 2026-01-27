import json
from .base import Base


class VolcEngineChat(Base):
    _FACTORY_NAME = "VolcEngine"

    def __init__(self, key, model_name, base_url="https://ark.cn-beijing.volces.com/api/v3", **kwargs):
        """
        Since do not want to modify the original database fields, and the VolcEngine authentication method is quite special,
        Assemble ark_api_key, ep_id into api_key, store it as a dictionary type, and parse it for use
        model_name is for display only
        """
        try:
            key_dict = json.loads(key)
        except Exception as e:
            raise ValueError(f"Invalid key format for VolcEngine: {e}")

        ark_api_key = key_dict.get("ark_api_key", "")
        if model_name:
            final_model_name = model_name
        else:
            final_model_name = f"{key_dict.get('ep_id', '')}-{key_dict.get('endpoint_id', '')}"

        super().__init__(ark_api_key, final_model_name, base_url, **kwargs)
