import json
from common.token_utils import total_token_count_from_response
from .base import Base


class BaiduYiyanChat(Base):
    _FACTORY_NAME = "BaiduYiyan"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        import qianfan

        key = json.loads(key)
        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        self.client = qianfan.ChatCompletion(ak=ak, sk=sk)
        self.model_name = model_name.lower()

    def _clean_conf(self, gen_conf):
        gen_conf["penalty_score"] = ((gen_conf.get("presence_penalty", 0) + gen_conf.get("frequency_penalty", 0)) / 2) + 1
        if "max_tokens" in gen_conf:
            del gen_conf["max_tokens"]
        return gen_conf

    def _chat(self, history, gen_conf):
        system = history[0]["content"] if history and history[0]["role"] == "system" else ""
        response = self.client.do(model=self.model_name, messages=[h for h in history if h["role"] != "system"], system=system, **gen_conf).body
        ans = response["result"]
        return ans, total_token_count_from_response(response)

    def chat_streamly(self, system, history, gen_conf={}, **kwargs):
        gen_conf["penalty_score"] = ((gen_conf.get("presence_penalty", 0) + gen_conf.get("frequency_penalty", 0)) / 2) + 1
        if "max_tokens" in gen_conf:
            del gen_conf["max_tokens"]
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
            return ans + "\n**ERROR**: " + str(e), 0

        yield total_tokens
