import asyncio
from common.token_utils import num_tokens_from_string
from .base import Base


class LocalLLM(Base):
    def __init__(self, key, model_name, base_url=None, **kwargs):
        super().__init__(key, model_name, base_url=base_url, **kwargs)
        from jina import Client
        from urllib.parse import urlparse

        host = "localhost"
        port = 12345
        protocol = "grpc"

        if base_url:
            parsed = urlparse(base_url)
            if parsed.hostname:
                host = parsed.hostname
            if parsed.port:
                port = parsed.port
            if parsed.scheme:
                protocol = parsed.scheme

        self.client = Client(host=host, port=port, protocol=protocol, asyncio=True)

    def _prepare_prompt(self, system, history, gen_conf):
        from rag.svr.jina_server import Prompt

        new_history = list(history)
        if system and new_history and new_history[0].get("role") != "system":
            new_history.insert(0, {"role": "system", "content": system})
        return Prompt(message=new_history, gen_conf=gen_conf)

    def _stream_response(self, endpoint, prompt):
        from rag.svr.jina_server import Generation

        answer = ""
        try:
            res = self.client.stream_doc(on=endpoint, inputs=prompt, return_type=Generation)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                while True:
                    answer = loop.run_until_complete(res.__anext__()).text
                    yield answer
            except StopAsyncIteration:
                pass
            finally:
                loop.close()
        except Exception as e:
            yield answer + "\n**ERROR**: " + str(e)
        yield num_tokens_from_string(answer)

    def chat(self, system, history, gen_conf=None, **kwargs):
        local_conf = dict(gen_conf or {})
        if "max_tokens" in local_conf:
            del local_conf["max_tokens"]
        prompt = self._prepare_prompt(system, history, local_conf)
        chat_gen = self._stream_response("/chat", prompt)

        ans = ""
        total_tokens = 0
        for result in chat_gen:
            if isinstance(result, int):
                total_tokens = result
            else:
                ans = result

        return ans, total_tokens

    def chat_streamly(self, system, history, gen_conf=None, **kwargs):
        local_conf = dict(gen_conf or {})
        if "max_tokens" in local_conf:
            del local_conf["max_tokens"]
        prompt = self._prepare_prompt(system, history, local_conf)
        return self._stream_response("/stream", prompt)
