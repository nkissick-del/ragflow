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


@dataclass
class MetadataFilter:
    key: str
    value: Any
    operator: str = "=="  # ==, !=, >, <, >=, <=, in, nin, contains


@dataclass
class MetadataFilters:
    filters: List[MetadataFilter] = field(default_factory=list)
    condition: str = "AND"  # AND, OR


@dataclass
class VectorStoreQuery:
    query_vector: Optional[VEC] = None
    query_text: Optional[str] = None
    top_k: int = 10
    filters: Optional[MetadataFilters] = None
    mode: SearchMode = SearchMode.DEFAULT
    alpha: float = 0.5  # Weight for hybrid search (semantic vs fulltext)
    extra_options: Dict[str, Any] = field(default_factory=dict)


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
