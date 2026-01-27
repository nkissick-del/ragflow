#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import asyncio
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from io import BytesIO
import xxhash
from api.db.db_models import DB, Task
from api.db.db_utils import bulk_insert_into_db
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from common.time_utils import get_format_time
from common.constants import TaskStatus, ParserType
from rag.nlp import rag_tokenizer, search
from rag.utils.redis_conn import REDIS_CONN
from common import settings
from common.misc_utils import get_uuid


class IngestionService(CommonService):
    @classmethod
    @DB.connection_context()
    def doc_upload_and_parse(cls, conversation_id, file_objs, user_id):
        from api.db.services.api_service import API4ConversationService
        from api.db.services.conversation_service import ConversationService
        from api.db.services.dialog_service import DialogService
        from api.db.services.file_service import FileService
        from api.db.services.llm_service import LLMBundle
        from api.db.services.user_service import TenantService
        from rag.app import audio, email, naive, picture, presentation

        e, conv = ConversationService.get_by_id(conversation_id)
        if not e:
            e, conv = API4ConversationService.get_by_id(conversation_id)
        if not e:
            raise LookupError("Conversation not found!")

        e, dia = DialogService.get_by_id(conv.dialog_id)
        if not e:
            raise LookupError("Dialog not found!")
        if not dia.kb_ids:
            raise LookupError("No dataset associated with this conversation. Please add a dataset before uploading documents")
        kb_id = dia.kb_ids[0]
        e, kb = KnowledgebaseService.get_by_id(kb_id)
        if not e:
            raise LookupError("Can't find this dataset!")

        embd_mdl = LLMBundle(kb.tenant_id, "embedding", llm_name=kb.embd_id, lang=kb.language)

        err, files = FileService.upload_document(kb, file_objs, user_id)
        if err:
            raise RuntimeError("\n".join(err))

        def dummy(prog=None, msg=""):
            pass

        FACTORY = {ParserType.PRESENTATION.value: presentation, ParserType.PICTURE.value: picture, ParserType.AUDIO.value: audio, ParserType.EMAIL.value: email}
        parser_config = {"chunk_token_num": 4096, "delimiter": "\n!?;。；！？", "layout_recognize": "Plain Text", "table_context_size": 0, "image_context_size": 0}
        doc_nm = {}
        for d, blob in files:
            doc_nm[d["id"]] = d["name"]

        docs = []
        threads = []
        with ThreadPoolExecutor(max_workers=12) as exe:
            for d, blob in files:
                kwargs = {"callback": dummy, "parser_config": parser_config, "from_page": 0, "to_page": 100000, "tenant_id": kb.tenant_id, "lang": kb.language}
                threads.append(exe.submit(FACTORY.get(d["parser_id"], naive).chunk, d["name"], blob, **kwargs))

        for (docinfo, _), th in zip(files, threads):
            doc = {"doc_id": docinfo["id"], "kb_id": [kb.id]}
            for ck in th.result():
                d = deepcopy(doc)
                d.update(ck)
                d["id"] = xxhash.xxh64((ck["content_with_weight"] + str(d["doc_id"])).encode("utf-8")).hexdigest()
                now = datetime.now()
                d["create_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
                d["create_timestamp_flt"] = now.timestamp()
                if not d.get("image"):
                    docs.append(d)
                    continue

                output_buffer = BytesIO()
                if isinstance(d["image"], bytes):
                    output_buffer = BytesIO(d["image"])
                else:
                    d["image"].save(output_buffer, format="JPEG")

                settings.STORAGE_IMPL.put(kb.id, d["id"], output_buffer.getvalue())
                d["img_id"] = "{}-{}".format(kb.id, d["id"])
                d.pop("image", None)
                docs.append(d)

        parser_ids = {d["id"]: d["parser_id"] for d, _ in files}
        docids = [d["id"] for d, _ in files]
        chunk_counts = {id: 0 for id in docids}
        token_counts = {id: 0 for id in docids}
        es_bulk_size = 64

        def embedding(doc_id, cnts, batch_size=16):
            nonlocal embd_mdl, chunk_counts, token_counts
            vectors = []
            for i in range(0, len(cnts), batch_size):
                vts, c = embd_mdl.encode(cnts[i : i + batch_size])
                vectors.extend(vts.tolist())
                chunk_counts[doc_id] += len(cnts[i : i + batch_size])
                token_counts[doc_id] += c
            return vectors

        idxnm = search.index_name(kb.tenant_id)
        try_create_idx = True

        _, tenant = TenantService.get_by_id(kb.tenant_id)
        llm_bdl = LLMBundle(kb.tenant_id, "chat", tenant.llm_id)

        async def process_mindmap(content_list):
            try:
                res = await mindmap(content_list)
                return json.dumps(res.output, ensure_ascii=False, indent=2)
            except Exception:
                logging.exception("Mind map generation error")
                return ""

        for doc_id in docids:
            cks = [c for c in docs if c["doc_id"] == doc_id]

            if parser_ids[doc_id] != ParserType.PICTURE.value:
                from graphrag.general.mind_map_extractor import MindMapExtractor

                mindmap = MindMapExtractor(llm_bdl)

                mind_map_results = asyncio.run(process_mindmap([c["content_with_weight"] for c in cks]))
                if mind_map_results:
                    if len(mind_map_results) < 32:
                        logging.error("Few content: " + mind_map_results)
                    else:
                        cks.append(
                            {
                                "id": get_uuid(),
                                "doc_id": doc_id,
                                "kb_id": [kb.id],
                                "docnm_kwd": doc_nm[doc_id],
                                "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", doc_nm[doc_id])),
                                "content_ltks": rag_tokenizer.tokenize("summary summarize 总结 概况 file 文件 概括"),
                                "content_with_weight": mind_map_results,
                                "knowledge_graph_kwd": "mind_map",
                            }
                        )

            vectors = embedding(doc_id, [c["content_with_weight"] for c in cks])
            assert len(cks) == len(vectors)
            for i, d in enumerate(cks):
                v = vectors[i]
                d["q_%d_vec" % len(v)] = v
            for b in range(0, len(cks), es_bulk_size):
                if try_create_idx:
                    if vectors:
                        dim = len(vectors[0])
                        if not settings.docStoreConn.index_exist(idxnm, kb_id):
                            settings.docStoreConn.create_idx(idxnm, kb_id, dim)
                    try_create_idx = False
                settings.docStoreConn.insert(cks[b : b + es_bulk_size], idxnm, kb_id)

            DocumentService.increment_chunk_num(doc_id, kb.id, token_counts[doc_id], chunk_counts[doc_id], 0)

        return [d["id"] for d, _ in files]

    @classmethod
    @DB.connection_context()
    def queue_raptor_o_graphrag_tasks(cls, sample_doc, ty, priority, fake_doc_id="", doc_ids=None):
        """
        You can provide a fake_doc_id to bypass the restriction of tasks at the knowledgebase level.
        Optionally, specify a list of doc_ids to determine which documents participate in the task.
        """
        if doc_ids is None:
            doc_ids = []

        assert ty in ["graphrag", "raptor", "mindmap"], "type should be graphrag, raptor or mindmap"

        chunking_config = DocumentService.get_chunking_config(sample_doc["id"])
        hasher = xxhash.xxh64()
        for field in sorted(chunking_config.keys()):
            hasher.update(str(chunking_config[field]).encode("utf-8"))

        def new_task():
            nonlocal sample_doc
            return {
                "id": get_uuid(),
                "doc_id": sample_doc["id"],
                "from_page": 100000000,
                "to_page": 100000000,
                "task_type": ty,
                "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task " + ty,
                "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        task = new_task()
        for field in ["doc_id", "from_page", "to_page"]:
            hasher.update(str(task.get(field, "")).encode("utf-8"))
        hasher.update(ty.encode("utf-8"))
        task["digest"] = hasher.hexdigest()
        bulk_insert_into_db(Task, [task], True)

        task["doc_id"] = fake_doc_id
        task["doc_ids"] = doc_ids
        cls.begin2parse(sample_doc["id"], keep_progress=True)
        assert REDIS_CONN.queue_product(settings.get_svr_queue_name(priority), message=task), "Can't access Redis. Please check the Redis' status."
        return task["id"]

    @classmethod
    @DB.connection_context()
    def begin2parse(cls, doc_id, keep_progress=False):
        info = {
            "progress_msg": "Task is queued...",
            "process_begin_at": get_format_time(),
        }
        if not keep_progress:
            info["progress"] = random.random() / 100.0
            info["run"] = TaskStatus.RUNNING.value
            # keep the doc in DONE state when keep_progress=True for GraphRAG, RAPTOR and Mindmap tasks

        DocumentService.update_by_id(doc_id, info)

    @classmethod
    def run(cls, tenant_id: str, doc: dict, kb_table_num_map: dict):
        from api.db.services.task_service import queue_dataflow, queue_tasks
        from api.db.services.file2document_service import File2DocumentService
        from common.misc_utils import get_uuid

        doc["tenant_id"] = tenant_id
        doc_parser = doc.get("parser_id", ParserType.NAIVE)
        if doc_parser == ParserType.TABLE:
            kb_id = doc.get("kb_id")
            if not kb_id:
                return
            if kb_id not in kb_table_num_map:
                count = DocumentService.count_by_kb_id(kb_id=kb_id, keywords="", run_status=[TaskStatus.DONE], types=[])
                kb_table_num_map[kb_id] = count
                if kb_table_num_map[kb_id] <= 0:
                    KnowledgebaseService.delete_field_map(kb_id)
        if doc.get("pipeline_id", ""):
            queue_dataflow(tenant_id, flow_id=doc["pipeline_id"], task_id=get_uuid(), doc_id=doc["id"])
        else:
            bucket, name = File2DocumentService.get_storage_address(doc_id=doc["id"])
            queue_tasks(doc, bucket, name, 0)

    @classmethod
    @DB.connection_context()
    def handle_run(cls, doc_ids, run_status, delete_flag, apply_kb_flag, user_id):
        from api.db.services.task_service import cancel_all_task_of, TaskService

        kb_table_num_map = {}
        for doc_id in doc_ids:
            if not DocumentService.accessible(doc_id, user_id):
                raise PermissionError("No authorization.")
            if delete_flag and not DocumentService.accessible4deletion(doc_id, user_id):
                raise PermissionError("No authorization.")

            should_cancel = False
            should_delete = False
            should_run = False
            doc_dict = {}
            doc_kb_id = ""
            tenant_id = ""

            with DB.atomic():
                info = {"run": str(run_status), "progress": 0}
                if str(run_status) == TaskStatus.RUNNING.value and delete_flag:
                    info["progress_msg"] = ""
                    info["chunk_num"] = 0
                    info["token_num"] = 0

                tenant_id = DocumentService.get_tenant_id(doc_id)
                if not tenant_id:
                    raise ValueError("Tenant not found!")
                e, doc = DocumentService.get_by_id(doc_id)
                if not e:
                    raise ValueError("Document not found!")
                doc_kb_id = doc.kb_id

                if str(run_status) == TaskStatus.CANCEL.value:
                    if str(doc.run) == TaskStatus.RUNNING.value:
                        should_cancel = True
                    else:
                        raise ValueError("Cannot cancel a task that is not in RUNNING status")

                is_rerun_condition = all([delete_flag, str(run_status) == TaskStatus.RUNNING.value, str(doc.run) == TaskStatus.DONE.value])
                if is_rerun_condition:
                    DocumentService.clear_chunk_num_when_rerun(doc.id)

                if delete_flag:
                    should_delete = True

                if str(run_status) == TaskStatus.RUNNING.value:
                    if apply_kb_flag:
                        e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
                        if not e:
                            raise LookupError("Can't find this dataset!")

                        new_config = dict(doc.parser_config or {})
                        new_config.update(
                            {"llm_id": kb.parser_config.get("llm_id"), "enable_metadata": kb.parser_config.get("enable_metadata", False), "metadata": kb.parser_config.get("metadata", {})}
                        )
                        DocumentService.update_parser_config(doc.id, new_config)
                    doc_dict = doc.to_dict()
                    should_run = True

                DocumentService.update_by_id(doc_id, info)

            # External side effects moved out of atomic block
            if should_cancel:
                try:
                    cancel_all_task_of(doc_id)
                except Exception as e:
                    logging.exception(f"Failed to cancel tasks for doc_id {doc_id}: {e}")

            if should_delete:
                try:
                    for i in range(3):  # Retry logic
                        try:
                            if settings.docStoreConn.index_exist(search.index_name(tenant_id), doc_kb_id):
                                settings.docStoreConn.delete({"doc_id": doc_id}, search.index_name(tenant_id), doc_kb_id)
                            TaskService.filter_delete([Task.doc_id == doc_id])
                            break
                        except Exception as e:
                            logging.warning(f"Failed to delete from docStore (attempt {i + 1}): {e}")
                            if i == 2:
                                logging.error(f"Final failure deleting doc {doc_id} from docStore {doc_kb_id} tenant {tenant_id}: {e}")
                                raise
                            time.sleep(2**i)
                except Exception as e:
                    logging.exception(f"Deletion failed for doc_id {doc_id}: {e}")
                    raise

            if should_run:
                cls.run(tenant_id, doc_dict, kb_table_num_map)
        return True
