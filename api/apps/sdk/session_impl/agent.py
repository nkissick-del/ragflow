import json
import copy
import tiktoken
from quart import Response, jsonify, request
from agent.canvas import Canvas
from api.db.services.api_service import API4ConversationService
from api.db.services.canvas_service import UserCanvasService, completion_openai
from api.db.services.canvas_service import completion as agent_completion
from common.misc_utils import get_uuid

from api.utils.api_utils import check_duplicate_ids, get_error_data_result, get_result, get_request_json, token_required, validate_request, get_data_openai


def register_agent_routes(manager):
    @manager.route("/agents/<agent_id>/sessions", methods=["POST"])
    @token_required
    async def create_agent_session(tenant_id, agent_id):
        user_id = request.args.get("user_id", tenant_id)
        e, cvs = UserCanvasService.get_by_id(agent_id)
        if not e:
            return get_error_data_result("Agent not found.")
        if not UserCanvasService.query(user_id=tenant_id, id=agent_id):
            return get_error_data_result("You cannot access the agent.")
        if not isinstance(cvs.dsl, str):
            cvs.dsl = json.dumps(cvs.dsl, ensure_ascii=False)

        session_id = get_uuid()
        canvas = Canvas(cvs.dsl, tenant_id, agent_id, canvas_id=cvs.id)
        canvas.reset()

        cvs.dsl = json.loads(str(canvas))
        conv = {"id": session_id, "dialog_id": cvs.id, "user_id": user_id, "message": [{"role": "assistant", "content": canvas.get_prologue()}], "source": "agent", "dsl": cvs.dsl}
        try:
            API4ConversationService.save(**conv)
        except Exception as e:
            import logging

            logging.exception(e)
            return get_error_data_result("Failed to save conversation")

        conv["agent_id"] = conv.pop("dialog_id")
        return get_result(data=conv)

    @manager.route("/agents_openai/<agent_id>/chat/completions", methods=["POST"])
    @validate_request("model", "messages")
    @token_required
    async def agents_completion_openai_compatibility(tenant_id, agent_id):
        req = await get_request_json()
        tiktoken_encode = tiktoken.get_encoding("cl100k_base")
        messages = req.get("messages", [])
        if not messages:
            return get_error_data_result("You must provide at least one message.")
        if not UserCanvasService.query(user_id=tenant_id, id=agent_id):
            return get_error_data_result(f"You don't own the agent {agent_id}")

        filtered_messages = [m for m in messages if m["role"] in ["user", "assistant"]]
        prompt_tokens = 0
        for m in filtered_messages:
            content = m.get("content")
            if isinstance(content, str):
                prompt_tokens += len(tiktoken_encode.encode(content))
        if not filtered_messages:
            return jsonify(
                get_data_openai(
                    id=agent_id,
                    content="No valid messages found (user or assistant).",
                    finish_reason="stop",
                    model=req.get("model", ""),
                    completion_tokens=len(tiktoken_encode.encode("No valid messages found (user or assistant).")),
                    prompt_tokens=prompt_tokens,
                )
            )

        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        session_id = req.pop("session_id", None)
        if not session_id:
            session_id = req.get("id", "")
            if not session_id:
                session_id = req.get("metadata", {}).get("id", "")
        stream = req.pop("stream", False)
        if stream:
            resp = Response(
                completion_openai(
                    tenant_id,
                    agent_id,
                    question,
                    session_id=session_id,
                    stream=True,
                    **req,
                ),
                mimetype="text/event-stream",
            )
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp
        else:
            # For non-streaming, just return the response directly
            has_result = False
            async for response in completion_openai(
                tenant_id,
                agent_id,
                question,
                session_id=session_id,
                stream=False,
                **req,
            ):
                has_result = True
                return jsonify(response)

            if not has_result:
                return jsonify(get_error_data_result("No completion generated")), 500

    @manager.route("/agents/<agent_id>/completions", methods=["POST"])
    @token_required
    async def agent_completions(tenant_id, agent_id):
        req = await get_request_json()
        return_trace = bool(req.get("return_trace", False))

        if req.get("stream", True):

            async def generate():
                trace_items = []
                async for answer in agent_completion(tenant_id=tenant_id, agent_id=agent_id, **req):
                    ans = None
                    if isinstance(answer, str):
                        try:
                            ans = json.loads(answer[5:])  # remove "data:"
                            event = ans.get("event")
                        except Exception:
                            continue
                    else:
                        # If answer is not string, it might be dict or we skip
                        continue

                    if event == "node_finished":
                        if return_trace:
                            data = ans.get("data", {})
                            trace_items.append(
                                {
                                    "component_id": data.get("component_id"),
                                    "trace": [copy.deepcopy(data)],
                                }
                            )
                            ans.setdefault("data", {})["trace"] = trace_items
                            answer = "data:" + json.dumps(ans, ensure_ascii=False) + "\n\n"
                        yield answer

                    if event not in ["message", "message_end"]:
                        continue

                    yield answer

                yield "data:[DONE]\n\n"

            resp = Response(generate(), mimetype="text/event-stream")
            resp.headers.add_header("Cache-control", "no-cache")
            resp.headers.add_header("Connection", "keep-alive")
            resp.headers.add_header("X-Accel-Buffering", "no")
            resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
            return resp

        full_content = ""
        reference = {}
        final_ans = ""
        trace_items = []
        async for answer in agent_completion(tenant_id=tenant_id, agent_id=agent_id, **req):
            try:
                ans = json.loads(answer[5:])

                if ans["event"] == "message":
                    full_content += ans["data"]["content"]

                if ans.get("data", {}).get("reference", None):
                    reference.update(ans["data"]["reference"])

                if return_trace and ans.get("event") == "node_finished":
                    data = ans.get("data", {})
                    trace_items.append(
                        {
                            "component_id": data.get("component_id"),
                            "trace": [copy.deepcopy(data)],
                        }
                    )

                final_ans = ans
            except Exception as e:
                return get_result(data=f"**ERROR**: {str(e)}")

        if not final_ans or not isinstance(final_ans, dict):
            # Initialize a valid structure if none was obtained
            final_ans = {"data": {}}

        final_ans.setdefault("data", {})["content"] = full_content
        final_ans["data"]["reference"] = reference
        if return_trace:
            final_ans["data"]["trace"] = trace_items
        return get_result(data=final_ans)

    @manager.route("/agents/<agent_id>/sessions", methods=["GET"])
    @token_required
    async def list_agent_session(tenant_id, agent_id):
        if not UserCanvasService.query(user_id=tenant_id, id=agent_id):
            return get_error_data_result(message=f"You don't own the agent {agent_id}.")
        id = request.args.get("id")
        user_id = request.args.get("user_id")
        try:
            page_number = int(request.args.get("page", 1))
            if page_number < 1:
                page_number = 1
        except ValueError:
            page_number = 1

        try:
            items_per_page = int(request.args.get("page_size", 30))
            if items_per_page < 1:
                items_per_page = 30
            items_per_page = min(items_per_page, 200)
        except ValueError:
            items_per_page = 30
        orderby = request.args.get("orderby", "update_time")
        if request.args.get("desc") == "False" or request.args.get("desc") == "false":
            desc = False
        else:
            desc = True
        # dsl defaults to True in all cases except for False and false
        include_dsl = request.args.get("dsl") != "False" and request.args.get("dsl") != "false"
        total, convs = API4ConversationService.get_list(agent_id, tenant_id, page_number, items_per_page, orderby, desc, id, user_id, include_dsl)
        if not convs:
            return get_result(data=[])
        for conv in convs:
            conv["messages"] = conv.pop("message")
            infos = conv["messages"]
            for info in infos:
                if "prompt" in info:
                    info.pop("prompt")
            conv["agent_id"] = conv.pop("dialog_id")
            # Fix for session listing endpoint
            if conv["reference"]:
                messages = conv["messages"]
                message_num = 0
                chunk_num = 0
                # Ensure reference is a list type to prevent KeyError
                if not isinstance(conv["reference"], list):
                    conv["reference"] = []
                while message_num < len(messages):
                    if message_num != 0 and messages[message_num]["role"] != "user":
                        chunk_list = []
                        # Add boundary and type checks to prevent KeyError
                        is_valid_chunk = (
                            chunk_num < len(conv["reference"])
                            and conv["reference"][chunk_num] is not None
                            and isinstance(conv["reference"][chunk_num], dict)
                            and "chunks" in conv["reference"][chunk_num]
                        )

                        if is_valid_chunk:
                            chunks = conv["reference"][chunk_num]["chunks"]
                            for chunk in chunks:
                                # Ensure chunk is a dictionary before calling get method
                                if not isinstance(chunk, dict):
                                    continue
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
                        chunk_num += 1
                        messages[message_num]["reference"] = chunk_list
                    message_num += 1
            del conv["reference"]
        return get_result(data=convs)

    @manager.route("/agents/<agent_id>/sessions", methods=["DELETE"])
    @token_required
    async def delete_agent_session(tenant_id, agent_id):
        errors = []
        success_count = 0
        req = await get_request_json()
        cvs = UserCanvasService.query(user_id=tenant_id, id=agent_id)
        if not cvs:
            return get_error_data_result(f"You don't own the agent {agent_id}")

        convs = API4ConversationService.query(dialog_id=agent_id)
        if not convs:
            # If explicit IDs provided and convs is empty, it means none of the requested IDs exist (or all deleted)
            # We will handle it in the ID checking loop or return 0 success count.
            # But if no IDs provided (bulk delete all for agent) and no sessions exist, then error.
            if not req or not req.get("ids"):
                return get_error_data_result(f"Agent {agent_id} has no sessions")

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

        for session_id in conv_list:
            conv = API4ConversationService.query(id=session_id, dialog_id=agent_id)
            if not conv:
                errors.append(f"The agent doesn't own the session {session_id}")
                continue
            try:
                res = API4ConversationService.delete_by_id(session_id)
                if res:
                    success_count += 1
            except Exception as e:
                import logging

                logging.exception(e)
                errors.append(f"Failed to delete session {session_id}")

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
