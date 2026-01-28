import json
import logging
import re
from quart import Response
from api.db.services.dialog_service import async_ask
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from common.constants import LLMType
from api.utils.api_utils import get_error_data_result, get_request_json, token_required, get_result


def register_ops_routes(manager):
    @manager.route("/sessions/ask", methods=["POST"])
    @token_required
    async def ask_about(tenant_id):
        req = await get_request_json()
        if not req.get("question"):
            return get_error_data_result("`question` is required.")
        if not req.get("dataset_ids"):
            return get_error_data_result("`dataset_ids` is required.")
        if not isinstance(req.get("dataset_ids"), list):
            return get_error_data_result("`dataset_ids` should be a list.")
        req["kb_ids"] = req.pop("dataset_ids")
        for kb_id in req["kb_ids"]:
            kbs = KnowledgebaseService.query(id=kb_id)
            if not kbs or not KnowledgebaseService.accessible(kb_id, tenant_id) or kbs[0].chunk_num == 0:
                return get_error_data_result("Dataset not found or inaccessible")

        uid = tenant_id

        async def stream():
            try:
                async for ans in async_ask(req["question"], req["kb_ids"], uid):
                    yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
                yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"
            except Exception as e:
                yield "data:" + json.dumps({"code": 500, "message": str(e), "data": {"answer": "**ERROR**: " + str(e), "reference": []}}, ensure_ascii=False) + "\n\n"

        resp = Response(stream(), mimetype="text/event-stream")
        resp.headers.add_header("Cache-control", "no-cache")
        resp.headers.add_header("Connection", "keep-alive")
        resp.headers.add_header("X-Accel-Buffering", "no")
        resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return resp

    @manager.route("/sessions/related_questions", methods=["POST"])
    @token_required
    async def related_questions(tenant_id):
        req = await get_request_json()
        if not req.get("question"):
            return get_error_data_result("`question` is required.")
        question = req["question"]
        industry = req.get("industry", "")
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
        prompt = """
Objective: To generate search terms related to the user's search keywords, helping users find more valuable information.
Instructions:
 - Based on the keywords provided by the user, generate 5-10 related search terms.
 - Each search term should be directly or indirectly related to the keyword, guiding the user to find more valuable information.
 - Use common, general terms as much as possible, avoiding obscure words or technical jargon.
 - Keep the term length between 2-4 words, concise and clear.
 - DO NOT translate, use the language of the original keywords.
"""
        if industry:
            prompt += f" - Ensure all search terms are relevant to the industry: {industry}.\n"
        prompt += """
### Example:
Keywords: Chinese football
Related search terms:
1. Current status of Chinese football
2. Reform of Chinese football
3. Youth training of Chinese football
4. Chinese football in the Asian Cup
5. Chinese football in the World Cup

Reason:
 - When searching, users often only use one or two keywords, making it difficult to fully express their information needs.
 - Generating related search terms can help users dig deeper into relevant information and improve search efficiency.
 - At the same time, related terms can also help search engines better understand user needs and return more accurate search results.

"""
        try:
            ans = await chat_mdl.async_chat(
                prompt,
                [
                    {
                        "role": "user",
                        "content": f"""
Keywords: {question}
Related search terms:
    """,
                    }
                ],
                {"temperature": 0.9},
            )
        except Exception as e:
            logging.exception("Error in related_questions")
            return get_error_data_result(str(e))
        return get_result(data=[re.sub(r"^[0-9]+\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]+\. ", a)])
