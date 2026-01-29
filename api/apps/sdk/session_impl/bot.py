import json
import logging
import re
from functools import wraps
from quart import Response, request
from api.db.db_models import APIToken
from api.db.services.conversation_service import async_iframe_completion as iframe_completion
from api.db.services.canvas_service import UserCanvasService
from api.db.services.canvas_service import completion as agent_completion
from api.db.services.dialog_service import DialogService, gen_mindmap, async_ask
from api.db.services.search_service import SearchService
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.db.services.user_service import UserTenantService
from common.metadata_utils import apply_meta_data_filter
from common.constants import RetCode, LLMType
from common import settings
from api.utils.api_utils import get_error_data_result, get_json_result, get_request_json, server_error_response, validate_request, get_result
from rag.app.tag import label_question
from rag.prompts.generator import cross_languages, keyword_extraction
from rag.prompts.template import load_prompt
from agent.canvas import Canvas
from elasticsearch.exceptions import NotFoundError

MAX_TOP_K = 10_000


def require_api_token(func):
    @wraps(func)
    async def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return get_error_data_result(message="Authorization header is missing")

        token_parts = auth_header.split()
        if len(token_parts) != 2:
            return get_error_data_result(message="Authorization is not valid")
        if token_parts[0].lower() != "bearer":
            return get_error_data_result(message="Authorization is not valid")
        token = token_parts[1]
        objs = APIToken.query(beta=token)
        if not objs:
            return get_error_data_result(message="Authentication error: API key is invalid")

        request.api_token = objs[0]
        return await func(*args, **kwargs)

    return decorated_function


def register_bot_routes(manager):
    @manager.route("/chatbots/<dialog_id>/completions", methods=["POST"])
    @require_api_token
    async def chatbot_completions(dialog_id):
        req = await get_request_json()

        if "quote" not in req:
            req["quote"] = False

        if req.get("stream", True):
            resp = Response(iframe_completion(dialog_id, **req), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        async for answer in iframe_completion(dialog_id, **req):
            return get_result(data=answer)

        return get_result(data=None)

    @manager.route("/chatbots/<dialog_id>/info", methods=["GET"])
    @require_api_token
    async def chatbots_inputs(dialog_id):
        e, dialog = DialogService.get_by_id(dialog_id)
        if not e:
            return get_error_data_result(message=f"Can't find dialog by ID: {dialog_id}")

        return get_result(
            data={
                "title": dialog.name,
                "avatar": dialog.icon,
                "prologue": dialog.prompt_config.get("prologue", ""),
            }
        )

    @manager.route("/agentbots/<agent_id>/completions", methods=["POST"])
    @require_api_token
    async def agent_bot_completions(agent_id):
        req = await get_request_json()

        if req.get("stream", True):
            resp = Response(agent_completion(request.api_token.tenant_id, agent_id, **req), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        async for answer in agent_completion(request.api_token.tenant_id, agent_id, **req):
            return get_result(data=answer)

        return get_result(data=None)

    @manager.route("/agentbots/<agent_id>/inputs", methods=["GET"])
    @require_api_token
    async def begin_inputs(agent_id):
        e, cvs = UserCanvasService.get_by_id(agent_id)
        if not e:
            return get_error_data_result(message=f"Can't find agent by ID: {agent_id}")

        canvas = Canvas(json.dumps(cvs.dsl), request.api_token.tenant_id, canvas_id=cvs.id)
        return get_result(data={"title": cvs.title, "avatar": cvs.avatar, "inputs": canvas.get_component_input_form("begin"), "prologue": canvas.get_prologue(), "mode": canvas.get_mode()})

    @manager.route("/searchbots/ask", methods=["POST"])
    @require_api_token
    @validate_request("question", "kb_ids")
    async def ask_about_embedded():
        req = await get_request_json()
        uid = request.api_token.tenant_id

        search_id = req.get("search_id", "")
        search_config = {}
        if search_id:
            if search_app := SearchService.get_detail(search_id):
                search_config = search_app.get("search_config", {})

        async def stream():
            nonlocal req, uid
            try:
                async for ans in async_ask(req["question"], req["kb_ids"], uid, search_config=search_config):
                    yield "data:" + json.dumps({"code": 0, "message": "", "data": ans}, ensure_ascii=False) + "\n\n"
            except Exception as e:
                logging.exception(e)
                yield "data:" + json.dumps({"code": 500, "message": "Internal server error", "data": {"answer": "**ERROR**: Internal server error", "reference": []}}, ensure_ascii=False) + "\n\n"
            yield "data:" + json.dumps({"code": 0, "message": "", "data": True}, ensure_ascii=False) + "\n\n"

        resp = Response(stream(), mimetype="text/event-stream")
        resp.headers.add_header("Cache-control", "no-cache")
        resp.headers.add_header("Connection", "keep-alive")
        resp.headers.add_header("X-Accel-Buffering", "no")
        resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return resp

    @manager.route("/searchbots/retrieval_test", methods=["POST"])
    @require_api_token
    @validate_request("kb_id", "question")
    async def retrieval_test_embedded():
        req = await get_request_json()
        try:
            page = int(req.get("page", 1))
            size = int(req.get("size", 30))
            similarity_threshold = float(req.get("similarity_threshold", 0.0))
            vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
            top = int(req.get("top_k", 1024))
            if top < 1 or top > MAX_TOP_K:
                raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
            if page < 1 or size < 1:
                raise ValueError("Page and size must be greater than 0")
            if not (0 <= similarity_threshold <= 1):
                raise ValueError("Similarity threshold must be between 0 and 1")
            if not (0 <= vector_similarity_weight <= 1):
                raise ValueError("Vector similarity weight must be between 0 and 1")
        except ValueError as e:
            return get_json_result(data=False, message=str(e), code=RetCode.DATA_ERROR)

        question = req["question"]
        kb_ids = req["kb_id"]
        if isinstance(kb_ids, str):
            kb_ids = [kb_ids]
        if not kb_ids:
            return get_json_result(data=False, message="Please specify dataset firstly.", code=RetCode.DATA_ERROR)
        doc_ids = req.get("doc_ids", [])
        use_kg = req.get("use_kg", False)
        langs = req.get("cross_languages", [])
        tenant_id = request.api_token.tenant_id
        if not tenant_id:
            return get_error_data_result(message="permission denied.")

        async def _retrieval():
            local_doc_ids = list(doc_ids) if doc_ids else []
            tenant_ids = []
            _question = question

            meta_data_filter = {}
            chat_mdl = None
            if req.get("search_id", ""):
                if detail := SearchService.get_detail(req.get("search_id", "")):
                    search_config = detail.get("search_config", {})
                    meta_data_filter = search_config.get("meta_data_filter", {})
                    if meta_data_filter.get("method") in ["auto", "semi_auto"]:
                        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_name=search_config.get("chat_id", ""))
            else:
                meta_data_filter = req.get("meta_data_filter") or {}
                if meta_data_filter.get("method") in ["auto", "semi_auto"]:
                    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)

            if meta_data_filter:
                metas = DocumentService.get_meta_by_kbs(kb_ids)
                local_doc_ids = await apply_meta_data_filter(meta_data_filter, metas, _question, chat_mdl, local_doc_ids)

            tenants = UserTenantService.query(user_id=tenant_id)
            for kb_id in kb_ids:
                for tenant in tenants:
                    if KnowledgebaseService.query(tenant_id=tenant.tenant_id, id=kb_id):
                        tenant_ids.append(tenant.tenant_id)
                        break
                else:
                    return get_json_result(data=False, message="Only owner of dataset authorized for this operation.", code=RetCode.OPERATING_ERROR)

            e, kb = KnowledgebaseService.get_by_id(kb_ids[0])
            if not e:
                return get_error_data_result(message="Knowledgebase not found!")

            if langs:
                _question = await cross_languages(kb.tenant_id, None, _question, langs)

            embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)

            rerank_mdl = None
            if req.get("rerank_id"):
                rerank_mdl = LLMBundle(kb.tenant_id, LLMType.RERANK.value, llm_name=req["rerank_id"])

            if req.get("keyword", False):
                chat_mdl = LLMBundle(kb.tenant_id, LLMType.CHAT)
                _question += await keyword_extraction(chat_mdl, _question)

            labels = label_question(_question, [kb])
            ranks = await settings.retriever.retrieval(
                _question,
                embd_mdl,
                tenant_ids,
                kb_ids,
                page,
                size,
                similarity_threshold,
                vector_similarity_weight,
                top,
                local_doc_ids,
                rerank_mdl=rerank_mdl,
                highlight=req.get("highlight"),
                rank_feature=labels,
            )
            if use_kg:
                ck = await settings.kg_retriever.retrieval(_question, tenant_ids, kb_ids, embd_mdl, LLMBundle(kb.tenant_id, LLMType.CHAT))
                if ck["content_with_weight"]:
                    ranks["chunks"].insert(0, ck)

            for c in ranks["chunks"]:
                c.pop("vector", None)
            ranks["labels"] = labels

            return get_json_result(data=ranks)

        try:
            return await _retrieval()
        except NotFoundError:
            return get_json_result(data=False, message="No chunk found! Check the chunk status please!", code=RetCode.DATA_ERROR)
        except Exception as e:
            return server_error_response(e)

    @manager.route("/searchbots/related_questions", methods=["POST"])
    @require_api_token
    @validate_request("question")
    async def related_questions_embedded():
        req = await get_request_json()
        tenant_id = request.api_token.tenant_id
        if not tenant_id:
            return get_error_data_result(message="permission denied.")

        search_id = req.get("search_id", "")
        search_config = {}
        if search_id:
            if search_app := SearchService.get_detail(search_id):
                search_config = search_app.get("search_config", {})

        question = req["question"]

        chat_id = search_config.get("chat_id", "")
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, chat_id)

        gen_conf = search_config.get("llm_setting", {"temperature": 0.9})
        prompt = load_prompt("related_question")
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
            gen_conf,
        )
        return get_json_result(data=[re.sub(r"^[0-9]+\. ", "", a) for a in ans.split("\n") if re.match(r"^[0-9]+\. ", a)])

    @manager.route("/searchbots/detail", methods=["GET"])
    @require_api_token
    async def detail_share_embedded():
        search_id = request.args.get("search_id")
        if not search_id:
            return get_error_data_result(message="search_id is required")

        tenant_id = request.api_token.tenant_id
        if not tenant_id:
            return get_error_data_result(message="permission denied.")
        try:
            tenants = UserTenantService.query(user_id=tenant_id)
            for tenant in tenants:
                if SearchService.query(tenant_id=tenant.tenant_id, id=search_id):
                    break
            else:
                return get_json_result(data=False, message="Has no permission for this operation.", code=RetCode.OPERATING_ERROR)

            search = SearchService.get_detail(search_id)
            if not search:
                return get_error_data_result(message="Can't find this Search App!")
            return get_json_result(data=search)
        except Exception as e:
            return server_error_response(e)

    @manager.route("/searchbots/mindmap", methods=["POST"])
    @require_api_token
    @validate_request("question", "kb_ids")
    async def mindmap():
        tenant_id = request.api_token.tenant_id
        req = await get_request_json()

        search_id = req.get("search_id", "")
        search_app = SearchService.get_detail(search_id) if search_id else {}

        mind_map = await gen_mindmap(req["question"], req["kb_ids"], tenant_id, search_app.get("search_config", {}))
        if "error" in mind_map:
            return server_error_response(Exception(mind_map["error"]))
        return get_json_result(data=mind_map)
