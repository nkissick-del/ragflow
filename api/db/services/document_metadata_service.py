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
from copy import deepcopy

from peewee import fn

from api.db.db_models import DB, Document, File2Document, File
from api.db.services.common_service import CommonService
from common.metadata_utils import dedupe_list
from common.time_utils import current_timestamp, get_format_time


class DocumentMetadataService(CommonService):
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
        # Use iterator to avoid loading all objects into memory
        meta_rows = query.select(cls.model.meta_fields).iterator()

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
            for k, v in (r.meta_fields or {}).items():
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
                    else:
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
        docs_to_update = []
        with DB.atomic():
            rows = cls.model.select(cls.model.id, cls.model.meta_fields).where((cls.model.id.in_(doc_ids)) & (cls.model.kb_id == kb_id))
            for r in rows:
                meta = _normalize_meta(r.meta_fields or {})
                original_meta = deepcopy(meta)
                changed = _apply_updates(meta)
                changed = _apply_deletes(meta) or changed
                if changed and meta != original_meta:
                    r.meta_fields = meta
                    r.update_time = current_timestamp()
                    r.update_date = get_format_time()
                    docs_to_update.append(r)

            if docs_to_update:
                # Update in batches of 100 to avoid overly large SQL statements
                batch_size = 100
                for i in range(0, len(docs_to_update), batch_size):
                    batch = docs_to_update[i : i + batch_size]
                    cls.model.bulk_update(batch, fields=[cls.model.meta_fields, cls.model.update_time, cls.model.update_date])
                    updated_docs += len(batch)
        return updated_docs
