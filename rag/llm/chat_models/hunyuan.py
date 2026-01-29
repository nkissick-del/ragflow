import json
from .base import Base


class HunyuanChat(Base):
    _FACTORY_NAME = "Tencent Hunyuan"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        from tencentcloud.common import credential
        from tencentcloud.hunyuan.v20230901 import hunyuan_client

        try:
            key_json = json.loads(key)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse configuration key: invalid JSON")

        sid = key_json.get("hunyuan_sid", "")
        sk = key_json.get("hunyuan_sk", "")
        region = key_json.get("hunyuan_region", None)

        if not sid or not sk:
            raise ValueError("hunyuan_sid and hunyuan_sk are required")

        cred = credential.Credential(sid, sk)
        self.model_name = model_name
        self.client = hunyuan_client.HunyuanClient(cred, region)

    def _clean_conf(self, gen_conf):
        _gen_conf = {}
        if "temperature" in gen_conf:
            _gen_conf["Temperature"] = gen_conf["temperature"]
        if "top_p" in gen_conf:
            _gen_conf["TopP"] = gen_conf["top_p"]
        return _gen_conf

    def _chat(self, history, gen_conf=None, **kwargs):
        from tencentcloud.hunyuan.v20230901 import models

        gen_conf = gen_conf or {}
        cleaned_conf = self._clean_conf(gen_conf)

        hist = [{k.capitalize(): v for k, v in item.items()} for item in history]
        req = models.ChatCompletionsRequest()
        params = {"Model": self.model_name, "Messages": hist, **cleaned_conf}
        req.from_json_string(json.dumps(params))
        response = self.client.ChatCompletions(req)
        ans = response.Choices[0].Message.Content
        return ans, response.Usage.TotalTokens

    def chat_streamly(self, system, history, gen_conf=None, **kwargs):
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
            TencentCloudSDKException,
        )
        from tencentcloud.hunyuan.v20230901 import models

        _gen_conf = {}
        gen_conf = dict(gen_conf or {})
        _history = [{k.capitalize(): v for k, v in item.items()} for item in history]
        if system and history and history[0].get("role") != "system":
            _history.insert(0, {"Role": "system", "Content": system})

        gen_conf.pop("max_tokens", None)

        # Merge cleaned config
        _gen_conf.update(self._clean_conf(gen_conf))
        req = models.ChatCompletionsRequest()
        params = {
            "Model": self.model_name,
            "Messages": _history,
            "Stream": True,
            **_gen_conf,
        }
        req.from_json_string(json.dumps(params))

        total_tokens = 0
        try:
            response = self.client.ChatCompletions(req)
            for resp in response:
                resp = json.loads(resp["data"])

                # Extract usage if enabled/present in chunk
                if "Usage" in resp and "TotalTokens" in resp["Usage"]:
                    total_tokens = resp["Usage"]["TotalTokens"]

                if not resp["Choices"] or not resp["Choices"][0]["Delta"]["Content"]:
                    continue

                content = resp["Choices"][0]["Delta"]["Content"]
                yield {"type": "content", "text": content}

        except TencentCloudSDKException as e:
            yield {"type": "error", "text": str(e)}

        yield {"type": "usage", "total_tokens": total_tokens}
