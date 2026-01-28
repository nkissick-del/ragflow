#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

from abc import ABC
from common.doc_store.doc_store_base import DocStoreConnection


class PGVectorConnectionBase(DocStoreConnection, ABC):
    def __init__(self):
        super().__init__()
        self.conn = None
        self.cursor = None

    def db_type(self) -> str:
        return "pgvector"

    def get_total(self, res):
        return res.get("total", 0)

    def get_doc_ids(self, res):
        return [doc_id for hit in res.get("hits", []) if (doc_id := hit.get("id"))]

    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        ret = {}
        for hit in res.get("hits", []):
            hit_id = hit.get("id")
            if hit_id:
                ret[hit_id] = {f: hit.get(f) for f in fields}
        return ret

    def get_highlight(self, res, keywords: list[str], field_name: str):
        # We'll use the universal highlighter in the driver or search service
        # but if PG has a native one (ts_headline), we can use it here.
        return {}

    def get_aggregation(self, res, field_name: str):
        return res.get("aggregations", {}).get(field_name, [])
