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
"""
Standardized Intermediate Representation (IR) for document parsing.

This module defines the contract between parsers (Layer 1) and templates (Layer 3).
All parser adapters in the orchestration layer must produce StandardizedDocument
objects, and all templates must consume them.

Reference: architecture_propsal.md Section 5
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DocumentElement:
    """
    A semantic element extracted from the document.

    Examples: heading, paragraph, table, code_block, list, image
    """

    type: str  # "heading", "paragraph", "table", "code_block", "list", "image"
    content: str
    level: Optional[int] = None  # For headings: 1-6
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardizedDocument:
    """
    The contract between parsers and templates.

    Parsers (via their adapters) produce this format.
    Templates consume this format without knowing which parser was used.

    Attributes:
        content: Full structured Markdown with headers, tables, code blocks
        metadata: Document-level metadata (source parser, page count, etc.)
        elements: Pre-parsed structure (optional, for efficiency)
    """

    content: str  # Full structured Markdown
    metadata: Dict[str, Any] = field(default_factory=dict)
    elements: List[DocumentElement] = field(default_factory=list)
