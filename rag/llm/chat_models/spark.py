from .base import Base


class SparkChat(Base):
    _FACTORY_NAME = "XunFei Spark"

    MODEL_TO_VERSION = {
        "Spark-Max": "generalv3.5",
        "Spark-Lite": "general",
        "Spark-Pro": "generalv3",
        "Spark-Pro-128K": "pro-128k",
        "Spark-4.0-Ultra": "4.0Ultra",
    }
    VERSION_TO_MODEL = {v: k for k, v in MODEL_TO_VERSION.items()}

    def __init__(self, key, model_name, base_url="https://spark-api-open.xf-yun.com/v1", **kwargs):
        if not base_url:
            base_url = "https://spark-api-open.xf-yun.com/v1"

        if model_name not in self.MODEL_TO_VERSION and model_name not in self.VERSION_TO_MODEL:
            raise ValueError(f"The given model name is not supported yet. Support: {list(self.MODEL_TO_VERSION.keys())}")

        if model_name in self.MODEL_TO_VERSION:
            model_version = self.MODEL_TO_VERSION[model_name]
        else:
            model_version = model_name
        super().__init__(key, model_version, base_url, **kwargs)
