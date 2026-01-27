from common.token_utils import total_token_count_from_response, num_tokens_from_string
from rag.nlp import is_chinese
from .base import Base, LENGTH_NOTIFICATION_CN, LENGTH_NOTIFICATION_EN


class BaiChuanChat(Base):
    _FACTORY_NAME = "BaiChuan"

    def __init__(self, key, model_name="Baichuan3-Turbo", base_url="https://api.baichuan-ai.com/v1", **kwargs):
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        super().__init__(key, model_name, base_url, **kwargs)

    @staticmethod
    def _format_params(params):
        return {
            "temperature": params.get("temperature", 0.3),
            "top_p": params.get("top_p", 0.85),
        }

    def _clean_conf(self, gen_conf):
        return {
            "temperature": gen_conf.get("temperature", 0.3),
            "top_p": gen_conf.get("top_p", 0.85),
        }

    def _chat(self, history, gen_conf={}, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            extra_body={"tools": [{"type": "web_search", "web_search": {"enable": True, "search_mode": "performance_first"}}]},
            **gen_conf,
        )
        ans = response.choices[0].message.content.strip()
        if response.choices[0].finish_reason == "length":
            if is_chinese([ans]):
                ans += LENGTH_NOTIFICATION_CN
            else:
                ans += LENGTH_NOTIFICATION_EN
        return ans, total_token_count_from_response(response)

    def chat_streamly(self, system, history, gen_conf={}, **kwargs):
        if system and history and history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": system})
        if "max_tokens" in gen_conf:
            del gen_conf["max_tokens"]
        ans = ""
        total_tokens = 0
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=history,
                extra_body={"tools": [{"type": "web_search", "web_search": {"enable": True, "search_mode": "performance_first"}}]},
                stream=True,
                **self._format_params(gen_conf),
            )
            for resp in response:
                if not resp.choices:
                    continue
                if not resp.choices[0].delta.content:
                    resp.choices[0].delta.content = ""
                ans = resp.choices[0].delta.content
                tol = total_token_count_from_response(resp)
                if not tol:
                    total_tokens += num_tokens_from_string(resp.choices[0].delta.content)
                else:
                    total_tokens = tol
                if resp.choices[0].finish_reason == "length":
                    if is_chinese([ans]):
                        ans += LENGTH_NOTIFICATION_CN
                    else:
                        ans += LENGTH_NOTIFICATION_EN
                yield ans

        except Exception as e:
            yield ans + "\n**ERROR**: " + str(e)

        yield total_tokens
