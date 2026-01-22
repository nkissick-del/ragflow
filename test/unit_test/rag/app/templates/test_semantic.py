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
Unit tests for Semantic Template.

Tests the stack-based header tracking algorithm and chunk generation.
Reference: LlamaIndex MarkdownNodeParser
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/markdown.py
"""

import unittest
from unittest.mock import patch

# Import the semantic template components
from rag.app.templates.semantic import Semantic, SemanticChunk


class TestSemanticChunkDataclass(unittest.TestCase):
    """Unit tests for SemanticChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a semantic chunk."""
        chunk = SemanticChunk(text="Some text content", header_path="/Introduction/Background/", metadata={"tokens": 10})
        self.assertEqual(chunk.text, "Some text content")
        self.assertEqual(chunk.header_path, "/Introduction/Background/")
        self.assertEqual(chunk.metadata["tokens"], 10)


class TestSemanticParseWithHeaders(unittest.TestCase):
    """Unit tests for the _parse_with_headers algorithm."""

    def test_single_header(self):
        """Test parsing content with a single header."""
        content = """# Introduction

This is the introduction text.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].header_path, "/Introduction/")
        self.assertIn("# Introduction", chunks[0].text)
        self.assertIn("introduction text", chunks[0].text)

    def test_nested_headers(self):
        """Test parsing content with nested headers (H1 > H2 > H3)."""
        content = """# Chapter 1

Intro text.

## Section 1.1

Section text.

### Subsection 1.1.1

Detail text.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        # Should have 3 chunks
        self.assertEqual(len(chunks), 3)

        # Verify header paths build correctly
        paths = [c.header_path for c in chunks]
        self.assertIn("/Chapter 1/", paths)
        self.assertIn("/Chapter 1/Section 1.1/", paths)
        self.assertIn("/Chapter 1/Section 1.1/Subsection 1.1.1/", paths)

    def test_sibling_headers(self):
        """Test parsing content with sibling headers (H2 followed by H2)."""
        content = """# Main

## Section A

A content.

## Section B

B content.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        paths = [c.header_path for c in chunks]

        # Section A and Section B should both be under Main
        self.assertIn("/Main/Section A/", paths)
        self.assertIn("/Main/Section B/", paths)

        # Section B should NOT be under Section A
        self.assertNotIn("/Main/Section A/Section B/", paths)

    def test_going_up_hierarchy(self):
        """Test that going from H3 to H2 correctly pops the stack."""
        content = """# Main

## Section 1

### Detail 1

Detail content.

## Section 2

Section 2 content.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        paths = [c.header_path for c in chunks]

        # Detail 1 should be under Section 1
        self.assertIn("/Main/Section 1/Detail 1/", paths)

        # Section 2 should be under Main, NOT under Detail 1
        self.assertIn("/Main/Section 2/", paths)
        self.assertNotIn("/Main/Section 1/Detail 1/Section 2/", paths)

    def test_code_block_protection(self):
        """Test that headers inside code blocks are NOT parsed."""
        content = """# Real Header

```python
# This is a comment, not a header
def foo():
    pass
```

More text after code block.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        # Should only have one chunk with one header
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].header_path, "/Real Header/")

        # The "# This is a comment" should be in the text, not treated as header
        self.assertIn("# This is a comment", chunks[0].text)

    def test_multiple_code_blocks(self):
        """Test that multiple code blocks are handled correctly."""
        content = """# Header

```
Code block 1
# Not a header
```

Middle text.

```
Code block 2
## Also not a header
```

End text.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        # Should be one chunk with all content under "Header"
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].header_path, "/Header/")

    def test_empty_content(self):
        """Test parsing empty content."""
        chunks = Semantic._parse_with_headers("", chunk_token_num=500, overlap_percent=0)
        self.assertEqual(len(chunks), 0)

    def test_no_headers(self):
        """Test parsing content with no headers."""
        content = """Just some text without any headers.

This is a paragraph.

Another paragraph.
"""
        chunks = Semantic._parse_with_headers(content, chunk_token_num=500, overlap_percent=0)

        # Should have one chunk with root path
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].header_path, "/")

    def test_chunk_splitting(self):
        """Test that large content is split into multiple chunks."""
        # Create content larger than chunk limit
        long_para = "This is a test paragraph. " * 100  # ~500+ tokens
        content = f"""# Header

{long_para}

{long_para}
"""
        # Use small chunk size to force splitting
        chunks = Semantic._parse_with_headers(content, chunk_token_num=50, overlap_percent=0)

        # Should have multiple chunks, all with same header path
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.header_path, "/Header/")


class TestSemanticChunkMethod(unittest.TestCase):
    """Tests for the Semantic.chunk() public method."""

    @patch("rag.app.templates.semantic.rag_tokenizer")
    def test_chunk_produces_ragflow_format(self, mock_tokenizer):
        """Test that chunk() produces dicts with required RAGFlow fields."""
        # Mock tokenizer
        mock_tokenizer.tokenize.return_value = "tokenized content"
        mock_tokenizer.fine_grained_tokenize.return_value = "fine grained"

        from rag.app.standardized_document import StandardizedDocument

        doc = StandardizedDocument(content="# Test\n\nContent here.")
        parser_config = {"chunk_token_num": 500}
        doc_metadata = {"docnm_kwd": "test.pdf", "title_tks": "test"}

        chunks = Semantic.chunk(filename="test.pdf", standardized_doc=doc, parser_config=parser_config, doc=doc_metadata, is_english=True)

        # Should have at least one chunk
        self.assertGreater(len(chunks), 0)

        # Check required fields exist
        chunk = chunks[0]
        self.assertIn("content_with_weight", chunk)
        self.assertIn("content_ltks", chunk)
        self.assertIn("content_sm_ltks", chunk)
        self.assertIn("header_path", chunk)
        self.assertIn("docnm_kwd", chunk)

    @patch("rag.app.templates.semantic.rag_tokenizer")
    def test_chunk_preserves_header_path(self, mock_tokenizer):
        """Test that header_path is correctly preserved in output."""
        mock_tokenizer.tokenize.return_value = "tokenized"
        mock_tokenizer.fine_grained_tokenize.return_value = "fine"

        from rag.app.standardized_document import StandardizedDocument

        content = """# Introduction

Intro content.

## Background

Background content.
"""
        doc = StandardizedDocument(content=content)
        parser_config = {"chunk_token_num": 500}

        chunks = Semantic.chunk(filename="test.pdf", standardized_doc=doc, parser_config=parser_config, doc={}, is_english=True)

        # Find chunks by header path
        paths = [c.get("header_path") for c in chunks]
        self.assertIn("/Introduction/", paths)
        self.assertIn("/Introduction/Background/", paths)


if __name__ == "__main__":
    unittest.main()
