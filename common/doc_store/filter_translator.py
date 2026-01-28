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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from common.doc_store.doc_store_models import MetadataFilters


class BaseFilterTranslator(ABC):
    @abstractmethod
    def translate(self, filters: MetadataFilters) -> Any:
        pass


class SQLFilterTranslator(BaseFilterTranslator):
    def translate(self, filters: Optional[MetadataFilters]) -> str:
        if not filters or not filters.filters:
            return "1=1"

        cond_list = []
        for f in filters.filters:
            key, val, op = f.key, f.value, f.operator

            # Simple mapping of operators to SQL
            if op == "==":
                sql_op = "="
            elif op == "!=":
                sql_op = "!="
            elif op == "in":
                sql_op = "IN"
            elif op == ">":
                sql_op = ">"
            elif op == "<":
                sql_op = "<"
            else:
                sql_op = "="  # Default

            if isinstance(val, str):
                val = val.replace("'", "''")
                formatted_val = f"'{val}'"
            elif isinstance(val, list):
                if not val:
                    continue
                items = []
                for i in val:
                    if isinstance(i, str):
                        i = i.replace("'", "''")
                        items.append(f"'{i}'")
                    else:
                        items.append(str(i))
                formatted_val = f"({', '.join(items)})"
            else:
                formatted_val = str(val)

            cond_list.append(f"{key} {sql_op} {formatted_val}")

        joiner = f" {filters.condition} "
        return joiner.join(cond_list) if cond_list else "1=1"


class ESFilterTranslator(BaseFilterTranslator):
    def translate(self, filters: Optional[MetadataFilters]) -> List[Dict[str, Any]]:
        if not filters or not filters.filters:
            return []

        es_filters = []
        for f in filters.filters:
            key, val, op = f.key, f.value, f.operator

            if op == "==":
                es_filters.append({"term": {key: val}})
            elif op == "in":
                es_filters.append({"terms": {key: val if isinstance(val, list) else [val]}})
            elif op == "range":
                # Assuming val is a dict like {'gt': 1, 'lt': 10}
                es_filters.append({"range": {key: val}})
            else:
                # Fallback to term
                es_filters.append({"term": {key: val}})

        return es_filters
