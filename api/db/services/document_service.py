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
import json
import logging
import pathlib
from copy import deepcopy
from datetime import datetime

from peewee import fn, Case, JOIN

from api.constants import IMG_BASE64_PREFIX, FILE_NAME_LEN_LIMIT
from api.db import PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES, FileType, UserTenantRole, CanvasCategory
from api.db.db_models import DB, Document, Knowledgebase, Task, Tenant, UserTenant, File2Document, File, UserCanvas, User
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from common.metadata_utils import dedupe_list
from common.time_utils import current_timestamp, get_format_time
from common.constants import StatusEnum, TaskStatus, SVR_CONSUMER_GROUP_NAME
from rag.nlp import rag_tokenizer, search
from rag.utils.redis_conn import REDIS_CONN
from common.doc_store.doc_store_base import OrderByExpr
from common import settings


class DocumentService(CommonService):
    model = Document

    @classmethod
    def get_cls_model_fields(cls):
        return [
            cls.model.id,
            cls.model.thumbnail,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.pipeline_id,
            cls.model.parser_config,
            cls.model.source_type,
            cls.model.type,
            cls.model.created_by,
            cls.model.name,
            cls.model.location,
            cls.model.size,
            cls.model.token_num,
            cls.model.chunk_num,
            cls.model.progress,
            cls.model.progress_msg,
            cls.model.process_begin_at,
            cls.model.process_duration,
            cls.model.meta_fields,
            cls.model.suffix,
            cls.model.run,
            cls.model.status,
            cls.model.create_time,
            cls.model.create_date,
            cls.model.update_time,
            cls.model.update_date,
        ]

    @classmethod
    @DB.connection_context()
    def get_list(cls, kb_id, page_number, items_per_page, orderby, desc, keywords, id, name, suffix=None, run=None, doc_ids=None):
        fields = cls.get_cls_model_fields()
        docs = (
            cls.model.select(*[*fields, UserCanvas.title])
            .join(File2Document, on=(File2Document.document_id == cls.model.id))
            .join(File, on=(File.id == File2Document.file_id))
            .join(UserCanvas, on=((cls.model.pipeline_id == UserCanvas.id) & (UserCanvas.canvas_category == CanvasCategory.DataFlow.value)), join_type=JOIN.LEFT_OUTER)
            .where(cls.model.kb_id == kb_id)
        )
        if id:
            docs = docs.where(cls.model.id == id)
        if name:
            docs = docs.where(cls.model.name == name)
        if keywords:
            docs = docs.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if doc_ids:
            docs = docs.where(cls.model.id.in_(doc_ids))
        if suffix:
            docs = docs.where(cls.model.suffix.in_(suffix))
        if run:
            docs = docs.where(cls.model.run.in_(run))
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        count = docs.count()
        docs = docs.paginate(page_number, items_per_page)
        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def check_doc_health(cls, tenant_id: str, filename):
        import os

        MAX_FILE_NUM_PER_USER = int(os.environ.get("MAX_FILE_NUM_PER_USER", 0))
        if 0 < MAX_FILE_NUM_PER_USER <= DocumentService.get_doc_count(tenant_id):
            raise RuntimeError("Exceed the maximum file number of a free user!")
        if len(filename.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
            raise RuntimeError("Exceed the maximum length of file name!")
        return True

    @classmethod
    @DB.connection_context()
    def get_by_kb_id(cls, kb_id, page_number, items_per_page, orderby, desc, keywords, run_status, types, suffix, doc_ids=None, return_empty_metadata=False):
        fields = cls.get_cls_model_fields()
        if keywords:
            docs = (
                cls.model.select(*[*fields, UserCanvas.title.alias("pipeline_name"), User.nickname])
                .join(File2Document, on=(File2Document.document_id == cls.model.id))
                .join(File, on=(File.id == File2Document.file_id))
                .join(UserCanvas, on=(cls.model.pipeline_id == UserCanvas.id), join_type=JOIN.LEFT_OUTER)
                .join(User, on=(cls.model.created_by == User.id), join_type=JOIN.LEFT_OUTER)
                .where((cls.model.kb_id == kb_id), (fn.LOWER(cls.model.name).contains(keywords.lower())))
            )
        else:
            docs = (
                cls.model.select(*[*fields, UserCanvas.title.alias("pipeline_name"), User.nickname])
                .join(File2Document, on=(File2Document.document_id == cls.model.id))
                .join(UserCanvas, on=(cls.model.pipeline_id == UserCanvas.id), join_type=JOIN.LEFT_OUTER)
                .join(File, on=(File.id == File2Document.file_id))
                .join(User, on=(cls.model.created_by == User.id), join_type=JOIN.LEFT_OUTER)
                .where(cls.model.kb_id == kb_id)
            )

        if doc_ids:
            docs = docs.where(cls.model.id.in_(doc_ids))
        if run_status:
            docs = docs.where(cls.model.run.in_(run_status))
        if types:
            docs = docs.where(cls.model.type.in_(types))
        if suffix:
            docs = docs.where(cls.model.suffix.in_(suffix))
        if return_empty_metadata:
            docs = docs.where(fn.COALESCE(fn.JSON_LENGTH(cls.model.meta_fields), 0) == 0)

        count = docs.count()
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        if page_number and items_per_page:
            docs = docs.paginate(page_number, items_per_page)

        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def get_filter_by_kb_id(cls, kb_id, keywords, run_status, types, suffix):
        """
        returns:
        {
            "suffix": {
                "ppt": 1,
                "doxc": 2
            },
            "run_status": {
             "1": 2,
             "2": 2
            }
            "metadata": {
                "key1": {
                 "key1_value1": 1,
                 "key1_value2": 2,
                },
                "key2": {
                 "key2_value1": 2,
                 "key2_value2": 1,
                },
            }
        }, total
        where "1" => RUNNING, "2" => CANCEL
        """
        fields = cls.get_cls_model_fields()
        if keywords:
            query = (
                cls.model.select(*fields)
                .join(File2Document, on=(File2Document.document_id == cls.model.id))
                .join(File, on=(File.id == File2Document.file_id))
                .where((cls.model.kb_id == kb_id), (fn.LOWER(cls.model.name).contains(keywords.lower())))
            )
        else:
            query = cls.model.select(*fields).join(File2Document, on=(File2Document.document_id == cls.model.id)).join(File, on=(File.id == File2Document.file_id)).where(cls.model.kb_id == kb_id)

        if run_status:
            query = query.where(cls.model.run.in_(run_status))
        if types:
            query = query.where(cls.model.type.in_(types))
        if suffix:
            query = query.where(cls.model.suffix.in_(suffix))

        total = query.count()

        suffix_counter = {}
        for row in query.select(cls.model.suffix, fn.COUNT(cls.model.id).alias("count")).group_by(cls.model.suffix).dicts():
            suffix_counter[row["suffix"]] = row["count"]

        run_status_counter = {}
        for row in query.select(cls.model.run, fn.COUNT(cls.model.id).alias("count")).group_by(cls.model.run).dicts():
            run_status_counter[str(row["run"])] = row["count"]

        metadata_counter = {}
        empty_metadata_count = 0

        # We select only meta_fields to minimize data transfer for the loop
        meta_rows = query.select(cls.model.meta_fields)

        for row in meta_rows:
            meta_fields = row.meta_fields or {}
            if not meta_fields:
                empty_metadata_count += 1
                continue
            has_valid_meta = False
            for key, value in meta_fields.items():
                values = value if isinstance(value, list) else [value]
                for vv in values:
                    if vv is None:
                        continue
                    if isinstance(vv, str) and not vv.strip():
                        continue
                    sv = str(vv)
                    if key not in metadata_counter:
                        metadata_counter[key] = {}
                    metadata_counter[key][sv] = metadata_counter[key].get(sv, 0) + 1
                    has_valid_meta = True
            if not has_valid_meta:
                empty_metadata_count += 1

        metadata_counter["empty_metadata"] = {"true": empty_metadata_count}
        return {
            "suffix": suffix_counter,
            "run_status": run_status_counter,
            "metadata": metadata_counter,
        }, total

    @classmethod
    @DB.connection_context()
    def count_by_kb_id(cls, kb_id, keywords, run_status, types):
        if keywords:
            docs = cls.model.select().where((cls.model.kb_id == kb_id), (fn.LOWER(cls.model.name).contains(keywords.lower())))
        else:
            docs = cls.model.select().where(cls.model.kb_id == kb_id)

        if run_status:
            docs = docs.where(cls.model.run.in_(run_status))
        if types:
            docs = docs.where(cls.model.type.in_(types))

        count = docs.count()

        return count

    @classmethod
    @DB.connection_context()
    def get_total_size_by_kb_id(cls, kb_id, keywords="", run_status=[], types=[]):
        query = cls.model.select(fn.COALESCE(fn.SUM(cls.model.size), 0)).where(cls.model.kb_id == kb_id)

        if keywords:
            query = query.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if run_status:
            query = query.where(cls.model.run.in_(run_status))
        if types:
            query = query.where(cls.model.type.in_(types))

        return int(query.scalar()) or 0

    @classmethod
    @DB.connection_context()
    def get_all_doc_ids_by_kb_ids(cls, kb_ids):
        fields = [cls.model.id]
        docs = cls.model.select(*fields).where(cls.model.kb_id.in_(kb_ids))
        docs = docs.order_by(cls.model.create_time.asc())
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def get_all_docs_by_creator_id(cls, creator_id):
        fields = [cls.model.id, cls.model.kb_id, cls.model.token_num, cls.model.chunk_num, Knowledgebase.tenant_id]
        docs = cls.model.select(*fields).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.created_by == creator_id)
        docs = docs.order_by(cls.model.create_time.asc())
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def insert(cls, doc):
        if not cls.save(**doc):
            raise RuntimeError("Database error (Document)!")
        if not KnowledgebaseService.atomic_increase_doc_num_by_id(doc["kb_id"]):
            raise RuntimeError("Database error (Knowledgebase)!")
        return Document(**doc)

    @classmethod
    @DB.connection_context()
    def remove_document(cls, doc, tenant_id):
        from api.db.services.task_service import TaskService

        cls.clear_chunk_num(doc.id)

        # Delete tasks first
        try:
            TaskService.filter_delete([Task.doc_id == doc.id])
        except Exception as e:
            logging.warning(f"Failed to delete tasks for document {doc.id}: {e}")

        # Delete chunk images (non-critical, log and continue)
        try:
            cls.delete_chunk_images(doc, tenant_id)
        except Exception as e:
            logging.warning(f"Failed to delete chunk images for document {doc.id}: {e}")

        # Delete thumbnail (non-critical, log and continue)
        try:
            if doc.thumbnail and not doc.thumbnail.startswith(IMG_BASE64_PREFIX):
                if settings.STORAGE_IMPL.obj_exist(doc.kb_id, doc.thumbnail):
                    settings.STORAGE_IMPL.rm(doc.kb_id, doc.thumbnail)
        except Exception as e:
            logging.warning(f"Failed to delete thumbnail for document {doc.id}: {e}")

        # Delete chunks from doc store - this is critical, log errors
        try:
            settings.docStoreConn.delete({"doc_id": doc.id}, search.index_name(tenant_id), doc.kb_id)
        except Exception as e:
            logging.error(f"Failed to delete chunks from doc store for document {doc.id}: {e}")

        # Cleanup knowledge graph references (non-critical, log and continue)
        try:
            graph_source = settings.docStoreConn.get_fields(
                settings.docStoreConn.search(["source_id"], [], {"kb_id": doc.kb_id, "knowledge_graph_kwd": ["graph"]}, [], OrderByExpr(), 0, 1, search.index_name(tenant_id), [doc.kb_id]),
                ["source_id"],
            )
            if len(graph_source) > 0 and doc.id in list(graph_source.values())[0]["source_id"]:
                settings.docStoreConn.update(
                    {"kb_id": doc.kb_id, "knowledge_graph_kwd": ["entity", "relation", "graph", "subgraph", "community_report"], "source_id": doc.id},
                    {"remove": {"source_id": doc.id}},
                    search.index_name(tenant_id),
                    doc.kb_id,
                )
                settings.docStoreConn.update({"kb_id": doc.kb_id, "knowledge_graph_kwd": ["graph"]}, {"removed_kwd": "Y"}, search.index_name(tenant_id), doc.kb_id)
                settings.docStoreConn.delete(
                    {"kb_id": doc.kb_id, "knowledge_graph_kwd": ["entity", "relation", "graph", "subgraph", "community_report"], "must_not": {"exists": "source_id"}},
                    search.index_name(tenant_id),
                    doc.kb_id,
                )
        except Exception as e:
            logging.warning(f"Failed to cleanup knowledge graph for document {doc.id}: {e}")

        return cls.delete_by_id(doc.id)

    @classmethod
    @DB.connection_context()
    def delete_chunk_images(cls, doc, tenant_id):
        page = 0
        page_size = 1000
        while True:
            chunks = settings.docStoreConn.search(["img_id"], [], {"doc_id": doc.id}, [], OrderByExpr(), page * page_size, page_size, search.index_name(tenant_id), [doc.kb_id])
            chunk_ids = settings.docStoreConn.get_doc_ids(chunks)
            if not chunk_ids:
                break
            for cid in chunk_ids:
                if settings.STORAGE_IMPL.obj_exist(doc.kb_id, cid):
                    settings.STORAGE_IMPL.rm(doc.kb_id, cid)
            page += 1

    @classmethod
    @DB.connection_context()
    def get_newly_uploaded(cls):
        fields = [
            cls.model.id,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.parser_config,
            cls.model.name,
            cls.model.type,
            cls.model.location,
            cls.model.size,
            Knowledgebase.tenant_id,
            Tenant.embd_id,
            Tenant.img2txt_id,
            Tenant.asr_id,
            cls.model.update_time,
        ]
        docs = (
            cls.model.select(*fields)
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))
            .where(
                cls.model.status == StatusEnum.VALID.value,
                ~(cls.model.type == FileType.VIRTUAL.value),
                cls.model.progress == 0,
                cls.model.update_time >= current_timestamp() - 1000 * 600,
                cls.model.run == TaskStatus.RUNNING.value,
            )
            .order_by(cls.model.update_time.asc())
        )
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def get_unfinished_docs(cls):
        fields = [cls.model.id, cls.model.process_begin_at, cls.model.parser_config, cls.model.progress_msg, cls.model.run, cls.model.parser_id]
        unfinished_task_query = Task.select(Task.doc_id).where((Task.progress >= 0) & (Task.progress < 1))

        docs = cls.model.select(*fields).where(
            cls.model.status == StatusEnum.VALID.value,
            ~(cls.model.type == FileType.VIRTUAL.value),
            ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL.value)),
            (((cls.model.progress < 1) & (cls.model.progress > 0)) | (cls.model.id.in_(unfinished_task_query))),
        )  # including unfinished tasks like GraphRAG, RAPTOR and Mindmap
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def increment_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duration):
        num = (
            cls.model.update(token_num=cls.model.token_num + token_num, chunk_num=cls.model.chunk_num + chunk_num, process_duration=cls.model.process_duration + duration)
            .where(cls.model.id == doc_id)
            .execute()
        )
        if num == 0:
            logging.warning("Document not found which is supposed to be there")
        num = Knowledgebase.update(token_num=Knowledgebase.token_num + token_num, chunk_num=Knowledgebase.chunk_num + chunk_num).where(Knowledgebase.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def decrement_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duration):
        num = (
            cls.model.update(token_num=cls.model.token_num - token_num, chunk_num=cls.model.chunk_num - chunk_num, process_duration=cls.model.process_duration + duration)
            .where(cls.model.id == doc_id)
            .execute()
        )
        if num == 0:
            raise LookupError("Document not found which is supposed to be there")
        num = Knowledgebase.update(token_num=Knowledgebase.token_num - token_num, chunk_num=Knowledgebase.chunk_num - chunk_num).where(Knowledgebase.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't fine document in database."

        num = (
            Knowledgebase.update(token_num=Knowledgebase.token_num - doc.token_num, chunk_num=Knowledgebase.chunk_num - doc.chunk_num, doc_num=Knowledgebase.doc_num - 1)
            .where(Knowledgebase.id == doc.kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num_when_rerun(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't fine document in database."

        num = (
            Knowledgebase.update(
                token_num=Knowledgebase.token_num - doc.token_num,
                chunk_num=Knowledgebase.chunk_num - doc.chunk_num,
            )
            .where(Knowledgebase.id == doc.kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def get_tenant_id(cls, doc_id):
        docs = cls.model.select(Knowledgebase.tenant_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def get_knowledgebase_id(cls, doc_id):
        docs = cls.model.select(cls.model.kb_id).where(cls.model.id == doc_id)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["kb_id"]

    @classmethod
    @DB.connection_context()
    def get_tenant_id_by_name(cls, name):
        docs = cls.model.select(Knowledgebase.tenant_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.name == name, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def accessible(cls, doc_id, user_id):
        docs = (
            cls.model.select(cls.model.id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .join(UserTenant, on=(UserTenant.tenant_id == Knowledgebase.tenant_id))
            .where(cls.model.id == doc_id, UserTenant.user_id == user_id)
            .paginate(0, 1)
        )
        docs = docs.dicts()
        if not docs:
            return False
        return True

    @classmethod
    @DB.connection_context()
    def accessible4deletion(cls, doc_id, user_id):
        docs = (
            cls.model.select(cls.model.id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .join(UserTenant, on=((UserTenant.tenant_id == Knowledgebase.created_by) & (UserTenant.user_id == user_id)))
            .where(cls.model.id == doc_id, UserTenant.status == StatusEnum.VALID.value, ((UserTenant.role == UserTenantRole.NORMAL) | (UserTenant.role == UserTenantRole.OWNER)))
            .paginate(0, 1)
        )
        docs = docs.dicts()
        if not docs:
            return False
        return True

    @classmethod
    @DB.connection_context()
    def get_embd_id(cls, doc_id):
        docs = cls.model.select(Knowledgebase.embd_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["embd_id"]

    @classmethod
    @DB.connection_context()
    def get_chunking_config(cls, doc_id):
        configs = (
            cls.model.select(
                cls.model.id,
                cls.model.kb_id,
                cls.model.parser_id,
                cls.model.parser_config,
                Knowledgebase.language,
                Knowledgebase.embd_id,
                Tenant.id.alias("tenant_id"),
                Tenant.img2txt_id,
                Tenant.asr_id,
                Tenant.llm_id,
            )
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))
            .where(cls.model.id == doc_id)
        )
        configs = configs.dicts()
        if not configs:
            return None
        return configs[0]

    @classmethod
    @DB.connection_context()
    def get_doc_id_by_doc_name(cls, doc_name):
        fields = [cls.model.id]
        doc_id = cls.model.select(*fields).where(cls.model.name == doc_name)
        doc_id = doc_id.dicts()
        if not doc_id:
            return None
        return doc_id[0]["id"]

    @classmethod
    @DB.connection_context()
    def get_doc_ids_by_doc_names(cls, doc_names):
        if not doc_names:
            return []

        query = cls.model.select(cls.model.id).where(cls.model.name.in_(doc_names))
        return list(query.scalars().iterator())

    @classmethod
    @DB.connection_context()
    def get_thumbnails(cls, docids):
        fields = [cls.model.id, cls.model.kb_id, cls.model.thumbnail]
        return list(cls.model.select(*fields).where(cls.model.id.in_(docids)).dicts())

    @classmethod
    @DB.connection_context()
    def update_parser_config(cls, id, config):
        if not config:
            return
        e, d = cls.get_by_id(id)
        if not e:
            raise LookupError(f"Document({id}) not found.")

        def dfs_update(old, new):
            for k, v in new.items():
                if k not in old:
                    old[k] = v
                    continue
                if isinstance(v, dict):
                    assert isinstance(old[k], dict)
                    dfs_update(old[k], v)
                else:
                    old[k] = v

        dfs_update(d.parser_config, config)
        if not config.get("raptor") and d.parser_config.get("raptor"):
            del d.parser_config["raptor"]
        cls.update_by_id(id, {"parser_config": d.parser_config})

    @classmethod
    @DB.connection_context()
    def get_doc_count(cls, tenant_id):
        docs = cls.model.select(cls.model.id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(Knowledgebase.tenant_id == tenant_id)
        return len(docs)

    @classmethod
    @DB.connection_context()
    def update_meta_fields(cls, doc_id, meta_fields):
        return cls.update_by_id(doc_id, {"meta_fields": meta_fields})

    @classmethod
    @DB.connection_context()
    def get_meta_by_kbs(cls, kb_ids):
        """
        Legacy metadata aggregator (backward-compatible).
        - Does NOT expand list values and a list is kept as one string key.
          Example: {"tags": ["foo","bar"]} -> meta["tags"]["['foo', 'bar']"] = [doc_id]
        - Expects meta_fields is a dict.
        Use when existing callers rely on the old list-as-string semantics.
        """
        fields = [
            cls.model.id,
            cls.model.meta_fields,
        ]
        meta = {}
        for r in cls.model.select(*fields).where(cls.model.kb_id.in_(kb_ids)):
            doc_id = r.id
            for k, v in r.meta_fields.items():
                if k not in meta:
                    meta[k] = {}
                if not isinstance(v, list):
                    v = [v]
                for vv in v:
                    if vv not in meta[k]:
                        if isinstance(vv, list) or isinstance(vv, dict):
                            continue
                        meta[k][vv] = []
                    meta[k][vv].append(doc_id)
        return meta

    @classmethod
    @DB.connection_context()
    def get_flatted_meta_by_kbs(cls, kb_ids):
        """
        - Parses stringified JSON meta_fields when possible and skips non-dict or unparsable values.
        - Expands list values into individual entries.
          Example: {"tags": ["foo","bar"], "author": "alice"} ->
            meta["tags"]["foo"] = [doc_id], meta["tags"]["bar"] = [doc_id], meta["author"]["alice"] = [doc_id]
        Prefer for metadata_condition filtering and scenarios that must respect list semantics.
        """
        fields = [
            cls.model.id,
            cls.model.meta_fields,
        ]
        meta = {}
        for r in cls.model.select(*fields).where(cls.model.kb_id.in_(kb_ids)):
            doc_id = r.id
            meta_fields = r.meta_fields or {}
            if isinstance(meta_fields, str):
                try:
                    meta_fields = json.loads(meta_fields)
                except Exception:
                    continue
            if not isinstance(meta_fields, dict):
                continue
            for k, v in meta_fields.items():
                if k not in meta:
                    meta[k] = {}
                values = v if isinstance(v, list) else [v]
                for vv in values:
                    if vv is None:
                        continue
                    sv = str(vv)
                    if sv not in meta[k]:
                        meta[k][sv] = []
                    meta[k][sv].append(doc_id)
        return meta

    @classmethod
    @DB.connection_context()
    def get_metadata_summary(cls, kb_id):
        fields = [cls.model.id, cls.model.meta_fields]
        summary = {}
        for r in cls.model.select(*fields).where(cls.model.kb_id == kb_id):
            meta_fields = r.meta_fields or {}
            if isinstance(meta_fields, str):
                try:
                    meta_fields = json.loads(meta_fields)
                except Exception:
                    continue
            if not isinstance(meta_fields, dict):
                continue
            for k, v in meta_fields.items():
                values = v if isinstance(v, list) else [v]
                for vv in values:
                    if not vv:
                        continue
                    sv = str(vv)
                    if k not in summary:
                        summary[k] = {}
                    summary[k][sv] = summary[k].get(sv, 0) + 1
        return {k: sorted([(val, cnt) for val, cnt in v.items()], key=lambda x: x[1], reverse=True) for k, v in summary.items()}

    @classmethod
    @DB.connection_context()
    def batch_update_metadata(cls, kb_id, doc_ids, updates=None, deletes=None):
        updates = updates or []
        deletes = deletes or []
        if not doc_ids:
            return 0

        def _normalize_meta(meta):
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    return {}
            if not isinstance(meta, dict):
                return {}
            return deepcopy(meta)

        def _str_equal(a, b):
            return str(a) == str(b)

        def _apply_updates(meta):
            changed = False
            for upd in updates:
                key = upd.get("key")
                if not key or key not in meta:
                    continue

                new_value = upd.get("value")
                match_provided = "match" in upd
                if isinstance(meta[key], list):
                    if not match_provided:
                        if isinstance(new_value, list):
                            meta[key] = dedupe_list(new_value)
                        else:
                            meta[key] = new_value
                        changed = True
                    else:
                        match_value = upd.get("match")
                        replaced = False
                        new_list = []
                        for item in meta[key]:
                            if _str_equal(item, match_value):
                                new_list.append(new_value)
                                replaced = True
                            else:
                                new_list.append(item)
                        if replaced:
                            meta[key] = dedupe_list(new_list)
                            changed = True
                else:
                    if not match_provided:
                        meta[key] = new_value
                        changed = True
                    else:
                        match_value = upd.get("match")
                        if _str_equal(meta[key], match_value):
                            meta[key] = new_value
                            changed = True
            return changed

        def _apply_deletes(meta):
            changed = False
            for d in deletes:
                key = d.get("key")
                if not key or key not in meta:
                    continue
                value = d.get("value", None)
                if isinstance(meta[key], list):
                    if value is None:
                        del meta[key]
                        changed = True
                        continue
                    new_list = [item for item in meta[key] if not _str_equal(item, value)]
                    if len(new_list) != len(meta[key]):
                        if new_list:
                            meta[key] = new_list
                        else:
                            del meta[key]
                        changed = True
                else:
                    if value is None or _str_equal(meta[key], value):
                        del meta[key]
                        changed = True
            return changed

        updated_docs = 0
        with DB.atomic():
            rows = cls.model.select(cls.model.id, cls.model.meta_fields).where((cls.model.id.in_(doc_ids)) & (cls.model.kb_id == kb_id))
            for r in rows:
                meta = _normalize_meta(r.meta_fields or {})
                original_meta = deepcopy(meta)
                changed = _apply_updates(meta)
                changed = _apply_deletes(meta) or changed
                if changed and meta != original_meta:
                    cls.model.update(meta_fields=meta, update_time=current_timestamp(), update_date=get_format_time()).where(cls.model.id == r.id).execute()
                    updated_docs += 1
        return updated_docs

    @classmethod
    @DB.connection_context()
    def update_progress(cls):
        docs = cls.get_unfinished_docs()

        cls._sync_progress(docs)

    @classmethod
    @DB.connection_context()
    def update_progress_immediately(cls, docs: list[dict]):
        if not docs:
            return

        cls._sync_progress(docs)

    @classmethod
    @DB.connection_context()
    def _sync_progress(cls, docs: list[dict]):
        from api.db.services.task_service import TaskService

        for d in docs:
            try:
                tsks = TaskService.query(doc_id=d["id"], order_by=Task.create_time)
                if not tsks:
                    continue
                msg = []
                prg = 0
                finished = True
                bad = 0
                e, doc = DocumentService.get_by_id(d["id"])
                status = doc.run  # TaskStatus.RUNNING.value
                if status == TaskStatus.CANCEL.value:
                    continue
                doc_progress = doc.progress if doc and doc.progress else 0.0
                special_task_running = False
                priority = 0
                for t in tsks:
                    task_type = (t.task_type or "").lower()
                    if task_type in PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES:
                        special_task_running = True
                    if 0 <= t.progress < 1:
                        finished = False
                    if t.progress == -1:
                        bad += 1
                    prg += t.progress if t.progress >= 0 else 0
                    if t.progress_msg.strip():
                        msg.append(t.progress_msg)
                    priority = max(priority, t.priority)
                prg /= len(tsks)
                if finished and bad:
                    prg = -1
                    status = TaskStatus.FAIL.value
                elif finished:
                    prg = 1
                    status = TaskStatus.DONE.value

                # only for special task and parsed docs and unfinished
                freeze_progress = special_task_running and doc_progress >= 1 and not finished
                msg = "\n".join(sorted(msg))
                begin_at = d.get("process_begin_at")
                if not begin_at:
                    begin_at = datetime.now()
                    # fallback
                    cls.update_by_id(d["id"], {"process_begin_at": begin_at})

                info = {"process_duration": max(datetime.timestamp(datetime.now()) - begin_at.timestamp(), 0), "run": status}
                if prg != 0 and not freeze_progress:
                    info["progress"] = prg
                if msg:
                    info["progress_msg"] = msg
                    if msg.endswith("created task graphrag") or msg.endswith("created task raptor") or msg.endswith("created task mindmap"):
                        info["progress_msg"] += "\n%d tasks are ahead in the queue..." % get_queue_length(priority)
                else:
                    info["progress_msg"] = "%d tasks are ahead in the queue..." % get_queue_length(priority)
                info["update_time"] = current_timestamp()
                info["update_date"] = get_format_time()
                (cls.model.update(info).where((cls.model.id == d["id"]) & ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL.value))).execute())
            except Exception as e:
                if str(e).find("'0'") < 0:
                    logging.exception("fetch task exception")

    @classmethod
    @DB.connection_context()
    def get_kb_doc_count(cls, kb_id):
        return cls.model.select().where(cls.model.kb_id == kb_id).count()

    @classmethod
    @DB.connection_context()
    def get_all_kb_doc_count(cls):
        result = {}
        rows = cls.model.select(cls.model.kb_id, fn.COUNT(cls.model.id).alias("count")).group_by(cls.model.kb_id)
        for row in rows:
            result[row.kb_id] = row.count
        return result

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, doc_id):
        try:
            _, doc = DocumentService.get_by_id(doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception:
            pass
        return False

    @classmethod
    @DB.connection_context()
    def knowledgebase_basic_info(cls, kb_id: str) -> dict[str, int]:
        # cancelled: run == "2"
        cancelled = cls.model.select(fn.COUNT(1)).where((cls.model.kb_id == kb_id) & (cls.model.run == TaskStatus.CANCEL)).scalar()
        downloaded = cls.model.select(fn.COUNT(1)).where(cls.model.kb_id == kb_id, cls.model.source_type != "local").scalar()

        row = (
            cls.model.select(
                # finished: progress == 1
                fn.COALESCE(fn.SUM(Case(None, [(cls.model.progress == 1, 1)], 0)), 0).alias("finished"),
                # failed: progress == -1
                fn.COALESCE(fn.SUM(Case(None, [(cls.model.progress == -1, 1)], 0)), 0).alias("failed"),
                # processing: 0 <= progress < 1
                fn.COALESCE(
                    fn.SUM(
                        Case(
                            None,
                            [
                                (((cls.model.progress == 0) | ((cls.model.progress > 0) & (cls.model.progress < 1))), 1),
                            ],
                            0,
                        )
                    ),
                    0,
                ).alias("processing"),
            )
            .where((cls.model.kb_id == kb_id) & ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL)))
            .dicts()
            .get()
        )

        return {"processing": int(row["processing"]), "finished": int(row["finished"]), "failed": int(row["failed"]), "cancelled": int(cancelled), "downloaded": int(downloaded)}

    @classmethod
    @DB.connection_context()
    def rename_document(cls, doc_id, new_name, user_id):
        from api.db.services.file_service import FileService
        from api.db.services.file2document_service import File2DocumentService

        if not cls.accessible(doc_id, user_id):
            raise PermissionError("No authorization.")

        e, doc = cls.get_by_id(doc_id)
        if not e:
            raise ValueError("Document not found!")
        if pathlib.Path(new_name.lower()).suffix != pathlib.Path(doc.name.lower()).suffix:
            raise ValueError("The extension of file can't be changed")

        for d in cls.query(name=new_name, kb_id=doc.kb_id):
            if d.name == new_name:
                raise ValueError("Duplicated document name in the same dataset.")

        if not cls.update_by_id(doc_id, {"name": new_name}):
            raise RuntimeError("Database error (Document rename)!")

        informs = File2DocumentService.get_by_document_id(doc_id)
        if informs:
            e, file = FileService.get_by_id(informs[0].file_id)
            if e and file:
                FileService.update_by_id(file.id, {"name": new_name})

        tenant_id = cls.get_tenant_id(doc_id)
        title_tks = rag_tokenizer.tokenize(new_name)
        es_body = {
            "docnm_kwd": new_name,
            "title_tks": title_tks,
            "title_sm_tks": rag_tokenizer.fine_grained_tokenize(title_tks),
        }
        if settings.docStoreConn.index_exist(search.index_name(tenant_id), doc.kb_id):
            settings.docStoreConn.update(
                {"doc_id": doc_id},
                es_body,
                search.index_name(tenant_id),
                doc.kb_id,
            )
        return True

    @classmethod
    @DB.connection_context()
    def change_document_status(cls, doc_id, status, user_id):
        if not cls.accessible(doc_id, user_id):
            raise ValueError("No authorization.")

        try:
            e, doc = cls.get_by_id(doc_id)
            if not e:
                raise ValueError("Document not found.")
            e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
            if not e:
                raise RuntimeError("Can't find this dataset!")
            if not cls.update_by_id(doc_id, {"status": str(status)}):
                raise RuntimeError("Database error (Document update)!")

            status_int = int(status)
            if not settings.docStoreConn.update({"doc_id": doc_id}, {"available_int": status_int}, search.index_name(kb.tenant_id), doc.kb_id):
                raise RuntimeError("Database error (docStore update)!")
            return {"status": status}
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Internal server error: {e}") from e


def get_queue_length(priority):
    group_info = REDIS_CONN.queue_info(settings.get_svr_queue_name(priority), SVR_CONSUMER_GROUP_NAME)
    if not group_info:
        return 0
    return int(group_info.get("lag", 0) or 0)
