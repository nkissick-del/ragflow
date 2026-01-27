import json
from common.token_utils import total_token_count_from_response
from .base import Base


class BaiduYiyanChat(Base):
    _FACTORY_NAME = "BaiduYiyan"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        import qianfan

        try:
            key = json.loads(key)
        except json.JSONDecodeError:
            raise ValueError("Invalid credentials: key must be a valid JSON string.")

        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        if not ak or not sk:
            raise ValueError("missing or invalid yiyan_ak/yiyan_sk")
        self.client = qianfan.ChatCompletion(ak=ak, sk=sk)
        self.model_name = model_name.lower()

    def _clean_conf(self, gen_conf):
        new_conf = gen_conf.copy()
        new_conf["penalty_score"] = ((new_conf.get("presence_penalty", 0) + new_conf.get("frequency_penalty", 0)) / 2) + 1
        if "max_tokens" in new_conf:
            del new_conf["max_tokens"]
        return new_conf

    def _chat(self, history, gen_conf):
        gen_conf = self._clean_conf(gen_conf)
        system = history[0]["content"] if history and history[0]["role"] == "system" else ""
        response = self.client.do(model=self.model_name, messages=[h for h in history if h["role"] != "system"], system=system, **gen_conf).body
        ans = response["result"]
        return ans, total_token_count_from_response(response)

    def chat_streamly(self, system, history, gen_conf=None, **kwargs):
        if gen_conf is None:
            gen_conf = {}
        gen_conf = self._clean_conf(gen_conf)
        ans = ""
        total_tokens = 0

        try:
            response = self.client.do(model=self.model_name, messages=history, system=system, stream=True, **gen_conf)
            for resp in response:
                resp = resp.body
                ans = resp["result"]
                total_tokens = total_token_count_from_response(resp)

                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e), 0
            return

        yield total_tokens
