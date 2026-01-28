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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np

VEC = Union[List[float], np.ndarray]


class SearchMode(Enum):
    DEFAULT = "default"
    SEMANTIC = "semantic"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
    TAG = "tag"


class Operator(Enum):
    EQ = "=="
    NE = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    IN = "in"
    NIN = "nin"
    CONTAINS = "contains"
    RANGE = "range"


class Condition(Enum):
    AND = "AND"
    OR = "OR"


@dataclass
class MetadataFilter:
    key: str
    value: Any
    operator: Operator = Operator.EQ


@dataclass
class MetadataFilters:
    filters: List[MetadataFilter] = field(default_factory=list)
    condition: Condition = Condition.AND


@dataclass
class VectorStoreQuery:
    query_vector: Optional[VEC] = None
    query_text: Optional[str] = None
    top_k: int = 10
    filters: Optional[MetadataFilters] = None
    mode: SearchMode = SearchMode.DEFAULT
    alpha: float = 0.5  # Weight for hybrid search (semantic vs fulltext)
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")

        if self.mode != SearchMode.TAG:
            # Check if query_vector is provided and not empty (if it's a list/sequence)
            has_vector = self.query_vector is not None
            if has_vector and hasattr(self.query_vector, "__len__") and len(self.query_vector) == 0:
                has_vector = False

            if not has_vector and not self.query_text:
                raise ValueError("Either query_vector or query_text must be provided for the selected search mode")


@dataclass
class VectorStoreHit:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    vector: Optional[VEC] = None
    highlight: Optional[str] = None


@dataclass
class VectorStoreQueryResult:
    hits: List[VectorStoreHit]
    total: int
    aggregations: Optional[Dict[str, Any]] = None
