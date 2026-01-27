import openai
from common.token_utils import total_token_count_from_response
from rag.nlp import is_chinese
from .base import Base, LENGTH_NOTIFICATION_CN, LENGTH_NOTIFICATION_EN


class MistralChat(Base):
    _FACTORY_NAME = "Mistral"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)

        from mistralai.client import MistralClient

        self.client = MistralClient(api_key=key)
        self.model_name = model_name

    def _clean_conf(self, gen_conf):
        for k in list(gen_conf.keys()):
            if k not in ["temperature", "top_p", "max_tokens"]:
                del gen_conf[k]
        return gen_conf

    def _chat(self, history, gen_conf={}, **kwargs):
        gen_conf = self._clean_conf(gen_conf)
        response = self.client.chat(model=self.model_name, messages=history, **gen_conf)
        ans = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            if is_chinese(ans):
                ans += LENGTH_NOTIFICATION_CN
            else:
                ans += LENGTH_NOTIFICATION_EN
        return ans, total_token_count_from_response(response)

    def chat_streamly(self, system, history, gen_conf={}, **kwargs):
        if system and history and history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": system})
        gen_conf = self._clean_conf(gen_conf)
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat_stream(model=self.model_name, messages=history, **gen_conf, **kwargs)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:
                    continue
                ans = resp.choices[0].delta.content
                total_tokens += 1
                if resp.choices[0].finish_reason == "length":
                    if is_chinese(ans):
                        ans += LENGTH_NOTIFICATION_CN
                    else:
                        ans += LENGTH_NOTIFICATION_EN
                yield ans

        except openai.APIError as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens
