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
import logging
from datetime import datetime
import os
from typing import Tuple, List

from anthropic import BaseModel
from peewee import SQL, fn

from api.db import InputType
from api.db.db_models import Connector, SyncLogs, Connector2Kb, Knowledgebase
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.doc_metadata_service import DocMetadataService
from common.misc_utils import get_uuid
from common.constants import TaskStatus
from common.time_utils import current_timestamp, timestamp_to_date
from common.exceptions import ConnectorError


class ConnectorService(CommonService):
    model = Connector

    @classmethod
    def resume(cls, connector_id, status):
        failures = []
        for c2k in Connector2KbService.query(connector_id=connector_id):
            task = SyncLogsService.get_latest_task(connector_id, c2k.kb_id)
            if not task:
                if status == TaskStatus.SCHEDULE:
                    res = SyncLogsService.schedule(connector_id, c2k.kb_id)
                    if res == "failed":
                        failures.append(f"Failed to schedule sync task for connector {connector_id} and knowledge base {c2k.kb_id}")
                    continue

            if task.status == TaskStatus.DONE:
                if status == TaskStatus.SCHEDULE:
                    res = SyncLogsService.schedule(connector_id, c2k.kb_id, task.poll_range_end, total_docs_indexed=task.total_docs_indexed)
                    if res == "failed":
                        failures.append(f"Failed to schedule sync task for connector {connector_id} and knowledge base {c2k.kb_id}")
                    continue

            task = task.to_dict()
            task["status"] = status
            SyncLogsService.update_by_id(task["id"], task)
        ConnectorService.update_by_id(connector_id, {"status": status})
        return "; ".join(failures) if failures else ""

    @classmethod
    def list(cls, tenant_id):
        fields = [cls.model.id, cls.model.name, cls.model.source, cls.model.status]
        return list(cls.model.select(*fields).where(cls.model.tenant_id == tenant_id).dicts())

    @classmethod
    def rebuild(cls, kb_id: str, connector_id: str, tenant_id: str):
        from api.db.services.file_service import FileService

        e, conn = cls.get_by_id(connector_id)
        if not e:
            return "connector_not_found"
        SyncLogsService.filter_delete([SyncLogs.connector_id == connector_id, SyncLogs.kb_id == kb_id])
        docs = DocumentService.query(source_type=f"{conn.source}/{conn.id}", kb_id=kb_id)
        err = FileService.delete_docs([d.id for d in docs], tenant_id)
        if not err:
            if not SyncLogsService.schedule(connector_id, kb_id, reindex=True):
                return f"Failed to schedule sync task for connector {connector_id} and knowledge base {kb_id}"
        return err


class SyncLogsService(CommonService):
    model = SyncLogs

    @classmethod
    def list_sync_tasks(cls, connector_id=None, page_number=None, items_per_page=15) -> Tuple[List[dict], int]:
        fields = [
            cls.model.id,
            cls.model.connector_id,
            cls.model.kb_id,
            cls.model.update_date,
            cls.model.poll_range_start,
            cls.model.poll_range_end,
            cls.model.new_docs_indexed,
            cls.model.total_docs_indexed,
            cls.model.error_msg,
            cls.model.full_exception_trace,
            cls.model.error_count,
            Connector.name,
            Connector.source,
            Connector.tenant_id,
            Connector.timeout_secs,
            Knowledgebase.name.alias("kb_name"),
            Knowledgebase.avatar.alias("kb_avatar"),
            Connector2Kb.auto_parse,
            cls.model.from_beginning.alias("reindex"),
            cls.model.status,
            cls.model.update_time,
        ]
        if not connector_id:
            fields.append(Connector.config)

        query = (
            cls.model.select(*fields)
            .join(Connector, on=(cls.model.connector_id == Connector.id))
            .join(Connector2Kb, on=(cls.model.kb_id == Connector2Kb.kb_id))
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
        )

        if connector_id:
            query = query.where(cls.model.connector_id == connector_id)
        else:
            database_type = os.getenv("DB_TYPE", "mysql")
            if "postgres" in database_type.lower():
                interval_expr = SQL("make_interval(mins => t2.refresh_freq)")
            else:
                interval_expr = SQL("INTERVAL `t2`.`refresh_freq` MINUTE")
            query = query.where(
                Connector.input_type == InputType.POLL, Connector.status == TaskStatus.SCHEDULE, cls.model.status == TaskStatus.SCHEDULE, cls.model.update_date < (fn.NOW() - interval_expr)
            )

        query = query.distinct().order_by(cls.model.update_time.desc())
        total = query.count()
        if page_number:
            query = query.paginate(page_number, items_per_page)

        return list(query.dicts()), total

    @classmethod
    def start(cls, id, connector_id):
        cls.update_by_id(id, {"status": TaskStatus.RUNNING, "time_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        ConnectorService.update_by_id(connector_id, {"status": TaskStatus.RUNNING})

    @classmethod
    def done(cls, id, connector_id):
        cls.update_by_id(id, {"status": TaskStatus.DONE})
        ConnectorService.update_by_id(connector_id, {"status": TaskStatus.DONE})

    @classmethod
    def schedule(cls, connector_id, kb_id, poll_range_start=None, reindex=False, total_docs_indexed=0):
        try:
            if cls.model.select().where(cls.model.kb_id == kb_id, cls.model.connector_id == connector_id).count() > 100:
                rm_ids = [m.id for m in cls.model.select(cls.model.id).where(cls.model.kb_id == kb_id, cls.model.connector_id == connector_id).order_by(cls.model.update_time.asc()).limit(70)]
                deleted = cls.model.delete().where(cls.model.id.in_(rm_ids)).execute()
                logging.info(f"[SyncLogService] Cleaned {deleted} old logs.")
        except Exception as e:
            logging.exception(e)

        try:
            e = cls.query(kb_id=kb_id, connector_id=connector_id, status=TaskStatus.SCHEDULE)
            if e:
                logging.warning(f"{kb_id}--{connector_id} has already had a scheduling sync task which is abnormal.")
                return "already_scheduled"
            reindex = "1" if reindex else "0"
            ConnectorService.update_by_id(connector_id, {"status": TaskStatus.SCHEDULE})
            cls.save(
                **{
                    "id": get_uuid(),
                    "kb_id": kb_id,
                    "status": TaskStatus.SCHEDULE,
                    "connector_id": connector_id,
                    "poll_range_start": poll_range_start,
                    "from_beginning": reindex,
                    "total_docs_indexed": total_docs_indexed,
                }
            )
            return "scheduled"
        except Exception as e:
            logging.exception(f"Failed to schedule sync task for connector {connector_id} and knowledge base {kb_id}")
            task = cls.get_latest_task(connector_id, kb_id)
            if task:
                cls.model.update(
                    status=TaskStatus.SCHEDULE, poll_range_start=poll_range_start, error_msg=cls.model.error_msg + str(e), full_exception_trace=cls.model.full_exception_trace + str(e)
                ).where(cls.model.id == task.id).execute()
                ConnectorService.update_by_id(connector_id, {"status": TaskStatus.SCHEDULE})
                return "scheduled"
            return "failed"

    @classmethod
    def increase_docs(cls, id, min_update, max_update, doc_num, err_msg="", error_count=0):
        cls.model.update(
            new_docs_indexed=cls.model.new_docs_indexed + doc_num,
            total_docs_indexed=cls.model.total_docs_indexed + doc_num,
            poll_range_start=fn.COALESCE(fn.LEAST(cls.model.poll_range_start, min_update), min_update),
            poll_range_end=fn.COALESCE(fn.GREATEST(cls.model.poll_range_end, max_update), max_update),
            error_msg=cls.model.error_msg + err_msg,
            error_count=cls.model.error_count + error_count,
            update_time=current_timestamp(),
            update_date=timestamp_to_date(current_timestamp()),
        ).where(cls.model.id == id).execute()

    @staticmethod
    def build_filename(doc):
        name = doc.get("semantic_identifier", "")
        ext = doc.get("extension") or ""
        if ext and not name.endswith(ext):
            return name + ext
        return name

    @classmethod
    def duplicate_and_parse(cls, kb, docs, tenant_id, src, auto_parse=True):
        from api.db.services.file_service import FileService

        if not docs:
            return None

        class FileObj(BaseModel):
            id: str
            filename: str
            blob: bytes

            def read(self) -> bytes:
                return self.blob

        files = [FileObj(id=d["id"], filename=cls.build_filename(d), blob=d["blob"]) for d in docs]
        doc_ids = []
        err, doc_blob_pairs = FileService.upload_document(kb, files, tenant_id, src)

        # Create a mapping from filename to metadata for later use
        metadata_map = {}
        for d in docs:
            if d.get("metadata"):
                filename = cls.build_filename(d)
                metadata_map[filename] = d["metadata"]

        kb_table_num_map = {}
        for doc, _ in doc_blob_pairs:
            doc_ids.append(doc["id"])

            # Set metadata if available for this document
            if doc["name"] in metadata_map:
                DocMetadataService.update_document_metadata(doc["id"], metadata_map[doc["name"]])

            if not auto_parse or auto_parse == "0":
                continue
            DocumentService.run(tenant_id, doc, kb_table_num_map)

        return err, doc_ids

    @classmethod
    def get_latest_task(cls, connector_id, kb_id):
        return cls.model.select().where(cls.model.connector_id == connector_id, cls.model.kb_id == kb_id).order_by(cls.model.update_time.desc()).first()


class Connector2KbService(CommonService):
    model = Connector2Kb

    @classmethod
    def link_connectors(cls, kb_id: str, connectors: list[dict], tenant_id: str):
        try:
            arr = cls.query(kb_id=kb_id)
            old_conn_ids = [a.connector_id for a in arr]
            connector_ids = []
            failures = []
            for conn in connectors:
                conn_id = conn["id"]
                connector_ids.append(conn_id)
                if conn_id in old_conn_ids:
                    cls.filter_update([cls.model.connector_id == conn_id, cls.model.kb_id == kb_id], {"auto_parse": conn.get("auto_parse", "1")})
                    continue

                # Ensure existing scheduling or running sync tasks for this connector and knowledge base are cancelled.
                # This ensures that only the latest sync configuration is accessible and to prevent accidental data loss if the connector is re-linked.
                SyncLogsService.filter_update(
                    [SyncLogs.connector_id == conn_id, SyncLogs.kb_id == kb_id, SyncLogs.status.in_([TaskStatus.SCHEDULE, TaskStatus.RUNNING])], {"status": TaskStatus.CANCEL}
                )
                cls.save(**{"id": get_uuid(), "connector_id": conn_id, "kb_id": kb_id, "auto_parse": conn.get("auto_parse", "1")})
                res = SyncLogsService.schedule(conn_id, kb_id, reindex=True)
                if res == "failed":
                    failures.append(f"Failed to schedule sync task for connector {conn_id} and knowledge base {kb_id}")

            if failures:
                return "; ".join(failures)
            return ""
        except (ValueError, KeyError) as e:
            logging.exception("Error while linking connectors")
            raise ConnectorError(str(e)) from e
        except Exception as e:
            logging.exception("Unexpected error while linking connectors")
            raise e

    @classmethod
    def list_connectors(cls, kb_id):
        fields = [Connector.id, Connector.source, Connector.name, cls.model.auto_parse, Connector.status]
        return list(cls.model.select(*fields).join(Connector, on=(cls.model.connector_id == Connector.id)).where(cls.model.kb_id == kb_id).dicts())
