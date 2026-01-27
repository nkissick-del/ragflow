import json
import time
from quart import Response, jsonify, request
from api.db.services.conversation_service import ConversationService
from api.db.services.conversation_service import async_completion as rag_completion
from api.db.services.dialog_service import DialogService, async_chat
from api.db.services.document_metadata_service import DocumentMetadataService
from common.metadata_utils import convert_conditions, meta_filter
from common.misc_utils import get_uuid
from api.utils.api_utils import check_duplicate_ids, get_error_data_result, get_result, get_request_json, token_required, validate_request
from rag.prompts.generator import chunks_format
from common.constants import StatusEnum


def register_chat_routes(manager):
    @manager.route("/chats/<chat_id>/sessions", methods=["POST"])
    @token_required
    async def create(tenant_id, chat_id):
        req = await get_request_json()
        req["dialog_id"] = chat_id
        dia = DialogService.query(tenant_id=tenant_id, id=req["dialog_id"], status=StatusEnum.VALID.value)
        if not dia:
            return get_error_data_result(message="You do not own the assistant.")
        conv = {
            "id": get_uuid(),
            "dialog_id": req["dialog_id"],
            "name": req.get("name", "New session"),
            "message": [{"role": "assistant", "content": dia[0].prompt_config.get("prologue")}],
            "user_id": req.get("user_id", ""),
            "reference": [],
        }
        if not conv.get("name"):
            return get_error_data_result(message="`name` can not be empty.")
        ConversationService.save(**conv)
        e, conv = ConversationService.get_by_id(conv["id"])
        if not e:
            return get_error_data_result(message="Fail to create a session!")
        conv = conv.to_dict()
        conv["messages"] = conv.pop("message")
        conv["chat_id"] = conv.pop("dialog_id")
        del conv["reference"]
        return get_result(data=conv)

    @manager.route("/chats/<chat_id>/sessions/<session_id>", methods=["PUT"])
    @token_required
    async def update(tenant_id, chat_id, session_id):
        req = await get_request_json()
        req["dialog_id"] = chat_id
        conv_id = session_id
        conv = ConversationService.query(id=conv_id, dialog_id=chat_id)
        if not conv:
            return get_error_data_result(message="Session does not exist")
        if not DialogService.query(id=chat_id, tenant_id=tenant_id, status=StatusEnum.VALID.value):
            return get_error_data_result(message="You do not own the session")
        if "message" in req or "messages" in req:
            return get_error_data_result(message="`message` can not be change")
        if "reference" in req:
            return get_error_data_result(message="`reference` can not be change")
        if "name" in req and not req.get("name"):
            return get_error_data_result(message="`name` can not be empty.")
        if not ConversationService.update_by_id(conv_id, req):
            return get_error_data_result(message="Session updates error")
        return get_result()

    @manager.route("/chats/<chat_id>/completions", methods=["POST"])
    @token_required
    async def chat_completion(tenant_id, chat_id):
        req = await get_request_json()
        if not req:
            req = {"question": ""}
        if not req.get("session_id"):
            req["question"] = ""
        dia = DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value)
        if not dia:
            return get_error_data_result(f"You don't own the chat {chat_id}")
        dia = dia[0]
        if req.get("session_id"):
            if not ConversationService.query(id=req["session_id"], dialog_id=chat_id):
                return get_error_data_result(f"You don't own the session {req['session_id']}")

        metadata_condition = req.get("metadata_condition") or {}
        if metadata_condition and not isinstance(metadata_condition, dict):
            return get_error_data_result(message="metadata_condition must be an object.")

        if metadata_condition and req.get("question"):
            metas = DocumentMetadataService.get_meta_by_kbs(dia.kb_ids or [])
            filtered_doc_ids = meta_filter(
                metas,
                convert_conditions(metadata_condition),
                metadata_condition.get("logic", "and"),
            )
            if metadata_condition.get("conditions") and not filtered_doc_ids:
                filtered_doc_ids = ["-999"]

            if filtered_doc_ids:
                req["doc_ids"] = ",".join(filtered_doc_ids)
            else:
                req.pop("doc_ids", None)

        if req.get("stream", True):
            resp = Response(rag_completion(tenant_id, chat_id, **req), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")

            return resp
        else:
            answer = None
            async for ans in rag_completion(tenant_id, chat_id, **req):
                answer = ans
                break
            return get_result(data=answer)

    @manager.route("/chats_openai/<chat_id>/chat/completions", methods=["POST"])
    @validate_request("model", "messages")
    @token_required
    async def chat_completion_openai_like(tenant_id, chat_id):
        req = await get_request_json()

        extra_body = req.get("extra_body") or {}
        if extra_body and not isinstance(extra_body, dict):
            return get_error_data_result("extra_body must be an object.")

        need_reference = bool(extra_body.get("reference", False))

        messages = req.get("messages", [])
        # To prevent empty [] input
        if len(messages) < 1:
            return get_error_data_result("You have to provide messages.")
        if messages[-1]["role"] != "user":
            return get_error_data_result("The last content of this conversation is not from user.")

        prompt = messages[-1]["content"]
        # Treat context tokens as reasoning tokens
        context_token_used = sum(len(message["content"]) for message in messages)

        dia = DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value)
        if not dia:
            return get_error_data_result(f"You don't own the chat {chat_id}")
        dia = dia[0]

        metadata_condition = extra_body.get("metadata_condition") or {}
        if metadata_condition and not isinstance(metadata_condition, dict):
            return get_error_data_result(message="metadata_condition must be an object.")

        doc_ids_str = None
        if metadata_condition:
            metas = DocumentMetadataService.get_meta_by_kbs(dia.kb_ids or [])
            filtered_doc_ids = meta_filter(
                metas,
                convert_conditions(metadata_condition),
                metadata_condition.get("logic", "and"),
            )
            if metadata_condition.get("conditions") and not filtered_doc_ids:
                filtered_doc_ids = ["-999"]
            doc_ids_str = ",".join(filtered_doc_ids) if filtered_doc_ids else None

        # Filter system and non-sense assistant messages
        msg = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "assistant" and not msg:
                continue
            msg.append(m)

        tools = None
        toolcall_session = None

        if req.get("stream", True):

            async def streamed_response_generator(chat_id, dia, msg):
                token_used = 0
                last_ans = {}
                full_content = ""
                full_reasoning = ""
                final_answer = None
                final_reference = None
                in_think = False
                response = {
                    "id": f"chatcmpl-{chat_id}",
                    "choices": [
                        {
                            "delta": {
                                "content": "",
                                "role": "assistant",
                                "function_call": None,
                                "tool_calls": None,
                                "reasoning_content": "",
                            },
                            "finish_reason": None,
                            "index": 0,
                            "logprobs": None,
                        }
                    ],
                    "created": int(time.time()),
                    "model": "model",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": "",
                    "usage": None,
                }

                try:
                    chat_kwargs = {"toolcall_session": toolcall_session, "tools": tools, "quote": need_reference}
                    if doc_ids_str:
                        chat_kwargs["doc_ids"] = doc_ids_str
                    async for ans in async_chat(dia, msg, True, **chat_kwargs):
                        last_ans = ans
                        if ans.get("final"):
                            if ans.get("answer"):
                                full_content = ans["answer"]
                            final_answer = ans.get("answer") or full_content
                            final_reference = ans.get("reference", {})
                            continue
                        if ans.get("start_to_think"):
                            in_think = True
                            continue
                        if ans.get("end_to_think"):
                            in_think = False
                            continue
                        delta = ans.get("answer") or ""
                        if not delta:
                            continue
                        token_used += len(delta)
                        if in_think:
                            full_reasoning += delta
                            response["choices"][0]["delta"]["reasoning_content"] = delta
                            response["choices"][0]["delta"]["content"] = None
                        else:
                            full_content += delta
                            response["choices"][0]["delta"]["content"] = delta
                            response["choices"][0]["delta"]["reasoning_content"] = None
                        yield f"data:{json.dumps(response, ensure_ascii=False)}\n\n"
                except Exception as e:
                    response["choices"][0]["delta"]["content"] = "**ERROR**: " + str(e)
                    yield f"data:{json.dumps(response, ensure_ascii=False)}\n\n"

                # The last chunk
                response["choices"][0]["delta"]["content"] = None
                response["choices"][0]["delta"]["reasoning_content"] = None
                response["choices"][0]["finish_reason"] = "stop"
                response["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": token_used, "total_tokens": len(prompt) + token_used}
                if need_reference:
                    reference_payload = final_reference if final_reference is not None else last_ans.get("reference", [])
                    response["choices"][0]["delta"]["reference"] = chunks_format(reference_payload)
                    response["choices"][0]["delta"]["final_content"] = final_answer if final_answer is not None else full_content
                yield f"data:{json.dumps(response, ensure_ascii=False)}\n\n"
                yield "data:[DONE]\n\n"

            resp = Response(streamed_response_generator(chat_id, dia, msg), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp
        else:
            answer = None
            chat_kwargs = {"toolcall_session": toolcall_session, "tools": tools, "quote": need_reference}
            if doc_ids_str:
                chat_kwargs["doc_ids"] = doc_ids_str
            async for ans in async_chat(dia, msg, False, **chat_kwargs):
                # focus answer content only
                answer = ans
                break
            content = answer["answer"]

            response = {
                "id": f"chatcmpl-{chat_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", ""),
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(content),
                    "total_tokens": len(prompt) + len(content),
                    "completion_tokens_details": {
                        "reasoning_tokens": context_token_used,
                        "accepted_prediction_tokens": len(content),
                        "rejected_prediction_tokens": 0,  # 0 for simplicity
                    },
                },
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            }
            if need_reference:
                response["choices"][0]["message"]["reference"] = chunks_format(answer.get("reference", {}))

            return jsonify(response)

    @manager.route("/chats/<chat_id>/sessions", methods=["GET"])
    @token_required
    async def list_session(tenant_id, chat_id):
        if not DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value):
            return get_error_data_result(message=f"You don't own the assistant {chat_id}.")
        id = request.args.get("id")
        name = request.args.get("name")
        page_number = int(request.args.get("page", 1))
        items_per_page = int(request.args.get("page_size", 30))
        orderby = request.args.get("orderby", "create_time")
        user_id = request.args.get("user_id")
        if request.args.get("desc") == "False" or request.args.get("desc") == "false":
            desc = False
        else:
            desc = True
        convs = ConversationService.get_list(chat_id, page_number, items_per_page, orderby, desc, id, name, user_id)
        if not convs:
            return get_result(data=[])
        for conv in convs:
            conv["messages"] = conv.pop("message")
            infos = conv["messages"]
            for info in infos:
                if "prompt" in info:
                    info.pop("prompt")
            conv["chat_id"] = conv.pop("dialog_id")
            ref_messages = conv["reference"]
            if ref_messages:
                messages = conv["messages"]
                message_num = 0
                ref_num = 0
                while message_num < len(messages) and ref_num < len(ref_messages):
                    if messages[message_num]["role"] != "user":
                        chunk_list = []
                        if "chunks" in ref_messages[ref_num]:
                            chunks = ref_messages[ref_num]["chunks"]
                            for chunk in chunks:
                                new_chunk = {
                                    "id": chunk.get("chunk_id", chunk.get("id")),
                                    "content": chunk.get("content_with_weight", chunk.get("content")),
                                    "document_id": chunk.get("doc_id", chunk.get("document_id")),
                                    "document_name": chunk.get("docnm_kwd", chunk.get("document_name")),
                                    "dataset_id": chunk.get("kb_id", chunk.get("dataset_id")),
                                    "image_id": chunk.get("image_id", chunk.get("img_id")),
                                    "positions": chunk.get("positions", chunk.get("position_int")),
                                }

                                chunk_list.append(new_chunk)
                        messages[message_num]["reference"] = chunk_list
                        ref_num += 1
                    message_num += 1
            del conv["reference"]
        return get_result(data=convs)

    @manager.route("/chats/<chat_id>/sessions", methods=["DELETE"])
    @token_required
    async def delete(tenant_id, chat_id):
        if not DialogService.query(id=chat_id, tenant_id=tenant_id, status=StatusEnum.VALID.value):
            return get_error_data_result(message="You don't own the chat")

        errors = []
        success_count = 0
        req = await get_request_json()
        convs = ConversationService.query(dialog_id=chat_id)
        if not req:
            ids = None
        else:
            ids = req.get("ids")

        if not ids:
            conv_list = []
            for conv in convs:
                conv_list.append(conv.id)
        else:
            conv_list = ids

        unique_conv_ids, duplicate_messages = check_duplicate_ids(conv_list, "session")
        conv_list = unique_conv_ids

        for id in conv_list:
            conv = ConversationService.query(id=id, dialog_id=chat_id)
            if not conv:
                errors.append(f"The chat doesn't own the session {id}")
                continue
            ConversationService.delete_by_id(id)
            success_count += 1

        if errors:
            if success_count > 0:
                return get_result(data={"success_count": success_count, "errors": errors}, message=f"Partially deleted {success_count} sessions with {len(errors)} errors")
            else:
                return get_error_data_result(message="; ".join(errors))

        if duplicate_messages:
            if success_count > 0:
                return get_result(message=f"Partially deleted {success_count} sessions with {len(duplicate_messages)} errors", data={"success_count": success_count, "errors": duplicate_messages})
            else:
                return get_error_data_result(message=";".join(duplicate_messages))

        return get_result()
