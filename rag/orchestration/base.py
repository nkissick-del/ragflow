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

Reference: architecture_proposal.md Section 5
"""

from dataclasses import dataclass, field, InitVar
from typing import List, Optional, Dict, Any, Literal, Iterable


@dataclass
class DocumentElement:
    """
    A semantic element extracted from the document.

    Examples: heading, paragraph, table, code_block, list, image
    """

    type: Literal["heading", "paragraph", "table", "code_block", "list", "image"]
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
        content: Full structured Markdown with headers, tables, code blocks (via property)
        metadata: Document-level metadata (source parser, page count, etc.)
        elements: Pre-parsed structure (lazy-loaded, cached)

    Usage:
        - Templates should access `.content` and `.elements` (properties)
        - Adapters can set content directly or call `populate_elements()` for efficiency
        - Modifying `content` automatically invalidates the `elements` cache
    """

    # Public content parameter (not stored directly)
    content_input: InitVar[str] = ""

    # Backing field for content (use property for access)
    _content: str = field(default="", repr=False)

    # Document-level metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached elements (not in __init__, not serialized)
    _elements: Optional[List[DocumentElement]] = field(default=None, init=False, repr=False)

    def __post_init__(self, content_input: str):
        if content_input and not self._content:
            self._content = content_input

    @property
    def content(self) -> str:
        """Get the document content."""
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        """Set content and invalidate cached elements."""
        self._content = value
        self._elements = None  # Invalidate cache

    @property
    def elements(self) -> List[DocumentElement]:
        """
        Lazy derivation strategy:
        - If elements have been supplied (e.g. by adapter via populate_elements), return them.
        - If not, parse `self._content` on demand and cache the result.

        Note: StandardizedDocument._parse_elements is a stub; full parsing must be
        provided by templates or semantic._parse_with_headers.
        """
        if self._elements is None:
            self._elements = self._parse_elements(self._content)
        return self._elements

    def populate_elements(self, elements: Iterable[DocumentElement]) -> None:
        """
        Caches the provided elements without further parsing.

        Use this when the adapter has efficient access to structure and wants
        to avoid re-parsing content. The elements will be cached until content
        is modified.

        Args:
            elements: Iterable of DocumentElement objects to cache
        """
        if not isinstance(elements, Iterable):
            raise TypeError(f"elements must be an Iterable, got {type(elements)}")

        validated_elements = []
        for i, el in enumerate(elements):
            if not isinstance(el, DocumentElement):
                raise TypeError(f"Item at index {i} is not a DocumentElement: {type(el)}")
            validated_elements.append(el)

        self._elements = validated_elements

    def _parse_elements(self, content: str) -> List[DocumentElement]:
        """
        Parse content into DocumentElement list.

        This is a stub. Full parsing is handled by specific logic in templates
        (e.g., semantic.py's _parse_with_headers).

        Args:
            content: Markdown content to parse

        Raises:
            NotImplementedError: Templates must provide parsed elements or use semantic._parse_with_headers
        """
        raise NotImplementedError("Templates must provide parsed elements or use semantic._parse_with_headers")
