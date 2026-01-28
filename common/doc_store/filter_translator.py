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
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from common.doc_store.doc_store_models import MetadataFilters, Operator

logger = logging.getLogger(__name__)


_RANGE_OP_MAP = {
    ">": "gt",
    Operator.GT: "gt",
    "<": "lt",
    Operator.LT: "lt",
    ">=": "gte",
    Operator.GTE: "gte",
    "<=": "lte",
    Operator.LTE: "lte",
}


def _normalize_condition(condition: Any) -> str:
    if hasattr(condition, "value"):
        condition = condition.value
    return (condition or "AND").strip().upper()


class BaseFilterTranslator(ABC):
    @abstractmethod
    def translate(self, filters: Optional[MetadataFilters]) -> Any:
        pass


class SQLFilterTranslator(BaseFilterTranslator):
    def translate(self, filters: Optional[MetadataFilters]) -> tuple[str, list]:
        if not filters or not filters.filters:
            return "1=1", []

        cond_list = []
        params = []
        for f in filters.filters:
            key, val, op = f.key, f.value, f.operator

            # SQL Injection Protection for keys: validate against strict allowlist (alphanumeric + underscore)
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid identifier (column name) for filter: {key}")

            # Simple mapping of operators to SQL

            if op == "==" or op == Operator.EQ:
                sql_op = "="
            elif op == "!=" or op == Operator.NE:
                sql_op = "!="
            elif op == "in" or op == Operator.IN:
                sql_op = "IN"
            elif op == ">" or op == Operator.GT:
                sql_op = ">"
            elif op == "<" or op == Operator.LT:
                sql_op = "<"
            elif op == ">=" or op == Operator.GTE:
                sql_op = ">="
            elif op == "<=" or op == Operator.LTE:
                sql_op = "<="
            else:
                raise ValueError(f"Unsupported operator: {op}")

            if op == "in" or op == Operator.IN:
                if isinstance(val, list):
                    if not val:
                        raise ValueError(f"Empty list provided for IN operator on key: {key}")
                    placeholders = ["%s"] * len(val)
                    formatted_val = f"({', '.join(placeholders)})"
                    params.extend(val)
                else:
                    # Treat single value as 1-element list for IN
                    formatted_val = "(%s)"
                    params.append(val)
            else:
                formatted_val = "%s"
                params.append(val)

            cond_list.append(f"{key} {sql_op} {formatted_val}")

        if not cond_list:
            return "1=1", []

        condition = _normalize_condition(filters.condition)
        joiner = f" {condition} "
        return joiner.join(cond_list), params


class ESFilterTranslator(BaseFilterTranslator):
    def translate(self, filters: Optional[MetadataFilters]) -> List[Dict[str, Any]]:
        if not filters or not filters.filters:
            return []

        must_filters = []
        must_not_filters = []
        for f in filters.filters:
            key, val, op = f.key, f.value, f.operator

            if op == "==" or op == Operator.EQ:
                must_filters.append({"term": {key: val}})
            elif op == "!=" or op == Operator.NE:
                must_not_filters.append({"term": {key: val}})
            elif op in _RANGE_OP_MAP:
                range_op = _RANGE_OP_MAP[op]
                must_filters.append({"range": {key: {range_op: val}}})
            elif op == "in" or op == Operator.IN:
                if isinstance(val, (list, tuple)):
                    if not val:
                        raise ValueError("IN operator requires a non-empty list")
                    formatted_val = val
                else:
                    formatted_val = [val]
                must_filters.append({"terms": {key: formatted_val}})
            elif op == "range" or op == Operator.RANGE:
                if not isinstance(val, dict):
                    raise TypeError(f"Value for 'range' operator must be a dict, got {type(val)}")
                valid_range_ops = {"gt", "gte", "lt", "lte"}
                for k, v in val.items():
                    if k not in valid_range_ops:
                        raise ValueError(f"Invalid range operator: {k}")
                    if not isinstance(v, (int, float, str)):
                        raise TypeError(f"Value for range operator '{k}' must be a scalar (int, float, str), got {type(v)}")
                must_filters.append({"range": {key: val}})
            else:
                raise ValueError(f"Unsupported ES operator: {op}")

        # Combine filters based on condition
        if not must_filters and not must_not_filters:
            return []

        condition = _normalize_condition(filters.condition)

        if condition == "OR":
            # For OR, combine into a list if no must_not, or handle accordingly.
            # ES boolean logic: OR is usually "should"
            clauses = []
            if must_filters:
                clauses.extend(must_filters)
            if must_not_filters:
                # This is tricky in OR: (A or B or not C)
                # Elasticsearch requires wrapping negations this way so the NOT is applied per-clause
                for f in must_not_filters:
                    clauses.append({"bool": {"must_not": [f]}})
            return [{"bool": {"should": clauses, "minimum_should_match": 1}}]
        else:
            # AND condition (default)
            res = []
            if must_filters:
                res.extend(must_filters)
            if must_not_filters:
                res.append({"bool": {"must_not": must_not_filters}})
            return res
