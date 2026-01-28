from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

from common.token_utils import total_token_count_from_response, num_tokens_from_string
from rag.nlp import is_chinese
from .base import Base, LENGTH_NOTIFICATION_CN, LENGTH_NOTIFICATION_EN


class MistralChat(Base):
    _FACTORY_NAME = "Mistral"

    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)
        self.client = MistralClient(api_key=key, base_url=base_url)
        self.model_name = model_name

    def _clean_conf(self, gen_conf):
        return {k: v for k, v in gen_conf.items() if k in ["temperature", "top_p", "max_tokens"]}

    def _chat(self, history, gen_conf=None, **kwargs):
        gen_conf = gen_conf if gen_conf is not None else {}
        gen_conf = self._clean_conf(gen_conf)
        response = self.client.chat(model=self.model_name, messages=history, **gen_conf)
        ans = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            if is_chinese(ans):
                ans += LENGTH_NOTIFICATION_CN
            else:
                ans += LENGTH_NOTIFICATION_EN
        return ans, total_token_count_from_response(response)

    def chat_streamly(self, system, history, gen_conf=None, **kwargs):
        gen_conf = gen_conf if gen_conf is not None else {}
        if system and history and history[0].get("role") != "system":
            history = [{"role": "system", "content": system}] + history
        gen_conf = self._clean_conf(gen_conf)
        ans = ""
        full_ans = ""
        total_tokens = 0
        try:
            response = self.client.chat_stream(model=self.model_name, messages=history, **gen_conf, **kwargs)
            for resp in response:
                if not resp.choices or not resp.choices[0].delta.content:
                    continue
                ans = resp.choices[0].delta.content
                full_ans += ans
                if hasattr(resp, "usage") and resp.usage:
                    total_tokens = resp.usage.total_tokens
                else:
                    total_tokens += num_tokens_from_string(ans)
                if resp.choices[0].finish_reason == "length":
                    if is_chinese(full_ans):
                        ans += LENGTH_NOTIFICATION_CN
                        full_ans += LENGTH_NOTIFICATION_CN
                    else:
                        ans += LENGTH_NOTIFICATION_EN
                        full_ans += LENGTH_NOTIFICATION_EN
                yield ans

        except MistralAPIException as e:
            yield full_ans + "\n**ERROR**: " + str(e)
        except Exception as e:
            yield full_ans + "\n**ERROR**: " + str(e)

        yield total_tokens
