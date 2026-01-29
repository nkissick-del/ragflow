from common.token_utils import num_tokens_from_string
from .base import Base


class ReplicateChat(Base):
    _FACTORY_NAME = "Replicate"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        from replicate.client import Client

        self.model_name = model_name
        self.client = Client(api_token=key)

    def _chat(self, history, gen_conf=None, **kwargs):
        if gen_conf is None:
            gen_conf = {}
        system = history[0]["content"] if history and history[0]["role"] == "system" else ""
        prompt = "\n".join([item["role"] + ":" + item["content"] for item in history[-5:] if item["role"] != "system"])
        response = self.client.run(
            self.model_name,
            input={"system_prompt": system, "prompt": prompt, **gen_conf},
        )
        ans = "".join(response)
        return ans, num_tokens_from_string(ans)

    def chat_streamly(self, system, history, gen_conf=None, **kwargs):
        if gen_conf is None:
            gen_conf = {}
        conf = dict(gen_conf)
        if "max_tokens" in conf:
            del conf["max_tokens"]
        prompt = "\n".join([item["role"] + ":" + item["content"] for item in history[-5:] if item["role"] != "system"])
        ans = ""
        try:
            response = self.client.run(
                self.model_name,
                input={"system_prompt": system, "prompt": prompt, **conf},
            )
            for resp in response:
                ans += resp
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield num_tokens_from_string(ans)
