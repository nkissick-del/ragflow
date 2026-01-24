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
Unit tests for StandardizedDocument dataclass and DocumentElement.

Tests the contract between parsers and templates as defined in:
- docs/architecture_proposal.md Section 5
"""

import unittest
from rag.orchestration.base import StandardizedDocument, DocumentElement


class TestDocumentElement(unittest.TestCase):
    """Unit tests for DocumentElement dataclass."""

    def test_create_heading_element(self):
        """Test creating a heading element with level."""
        element = DocumentElement(type="heading", content="Introduction", level=1, metadata={"page": 1})
        self.assertEqual(element.type, "heading")
        self.assertEqual(element.content, "Introduction")
        self.assertEqual(element.level, 1)
        self.assertEqual(element.metadata["page"], 1)

    def test_create_paragraph_element(self):
        """Test creating a paragraph element (no level needed)."""
        element = DocumentElement(type="paragraph", content="This is some text.")
        self.assertEqual(element.type, "paragraph")
        self.assertIsNone(element.level)
        self.assertEqual(element.metadata, {})

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        element = DocumentElement(type="code_block", content="print('hello')")
        self.assertEqual(element.metadata, {})


class TestStandardizedDocument(unittest.TestCase):
    """Unit tests for StandardizedDocument dataclass."""

    def test_create_document_with_content(self):
        """Test creating a document with just content."""
        doc = StandardizedDocument(_content="# Heading\n\nParagraph text.")
        self.assertEqual(doc.content, "# Heading\n\nParagraph text.")
        self.assertEqual(doc.metadata, {})
        self.assertEqual(doc.elements, [])

    def test_create_document_with_metadata(self):
        """Test creating a document with metadata."""
        doc = StandardizedDocument(_content="Test content", metadata={"parser": "docling", "pages": 10})
        self.assertEqual(doc.metadata["parser"], "docling")
        self.assertEqual(doc.metadata["pages"], 10)

    def test_default_values(self):
        """Test that defaults work correctly."""
        doc = StandardizedDocument(_content="")
        self.assertEqual(doc.content, "")
        self.assertEqual(doc.metadata, {})
        self.assertEqual(doc.elements, [])

    def test_content_property_setter(self):
        """Test that setting content via property works."""
        doc = StandardizedDocument(_content="initial")
        self.assertEqual(doc.content, "initial")

        # Set via property
        doc.content = "modified"
        self.assertEqual(doc.content, "modified")

    def test_cache_invalidation_on_content_change(self):
        """Test that modifying content invalidates elements cache."""
        doc = StandardizedDocument(_content="# Test")

        # Access elements to populate cache
        _ = doc.elements

        # Modify content
        doc.content = "# New Content"

        # Elements should be re-derived (empty in our stub implementation)
        self.assertEqual(doc.elements, [])

    def test_populate_elements_method(self):
        """Test the populate_elements() method for adapters."""
        doc = StandardizedDocument(_content="# Title\n\nText here.")

        elements = [
            DocumentElement(type="heading", content="Title", level=1),
            DocumentElement(type="paragraph", content="Text here."),
        ]
        doc.populate_elements(elements)

        self.assertEqual(len(doc.elements), 2)
        self.assertEqual(doc.elements[0].type, "heading")
        self.assertEqual(doc.elements[1].type, "paragraph")

    def test_content_change_invalidates_populated_elements(self):
        """Test that changing content invalidates manually populated elements."""
        doc = StandardizedDocument(_content="# Original")

        # Populate elements
        elements = [DocumentElement(type="heading", content="Original", level=1)]
        doc.populate_elements(elements)
        self.assertEqual(len(doc.elements), 1)

        # Change content - should invalidate cache
        doc.content = "# New Content"

        # Elements should be re-derived (returns empty list in stub)
        self.assertEqual(doc.elements, [])


if __name__ == "__main__":
    unittest.main()
