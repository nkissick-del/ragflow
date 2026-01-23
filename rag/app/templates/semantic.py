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
Semantic Template - Structure-Aware Chunking

This template implements structure-aware chunking for documents parsed by
high-fidelity parsers like Docling. It preserves semantic boundaries (headers,
paragraphs) and adds header hierarchy metadata to each chunk.

Key Features:
1. Stack-based header tracking (adapted from LlamaIndex MarkdownNodeParser)
2. Code block protection (don't parse headers inside ```)
3. Never split mid-paragraph
4. Configurable chunk size with overlap
5. header_path metadata on each chunk

Reference: LlamaIndex MarkdownNodeParser
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/markdown.py
"""

import re
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from rag.app.standardized_document import StandardizedDocument
from rag.nlp import rag_tokenizer

# Try to import token counting utility, fallback to simple estimation
try:
    from tiktoken import encoding_for_model

    _enc = encoding_for_model("gpt-3.5-turbo")

    def num_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    logging.warning("[Semantic] tiktoken not installed. Using fallback estimator for num_tokens.")

    def num_tokens(text: str) -> int:
        # Simple estimation: ~4 chars per token for English
        return len(text) // 4


@dataclass
class SemanticChunk:
    """A chunk with header hierarchy metadata."""

    text: str
    header_path: str  # e.g., "/Introduction/Background/"
    metadata: Dict[str, Any]


class Semantic:
    """
    Parser-agnostic semantic chunking template.

    Consumes StandardizedDocument (the contract from orchestration layer)
    and produces chunks with header_path metadata.
    """

    @staticmethod
    def chunk(filename: str, standardized_doc: StandardizedDocument, parser_config: dict, doc: dict, is_english: bool, callback=None, **kwargs) -> List[dict]:
        """
        Chunk document respecting semantic boundaries.

        Args:
            filename: Name of the source file
            standardized_doc: Normalized document from adapter
            parser_config: Parser configuration (chunk_token_num, overlapped_percent, etc.)
            doc: Document metadata dict (docnm_kwd, title_tks, etc.)
            is_english: Whether document is in English
            callback: Progress callback function
            **kwargs: Additional arguments

        Returns:
            List of chunk dicts ready for storage/embedding
        """
        chunk_token_num = int(parser_config.get("chunk_token_num", 512) or 512)
        overlap_percent = int(parser_config.get("overlapped_percent", 10) or 10)
        overlap_percent = max(0, min(100, overlap_percent))

        if callback:
            callback(0.5, "[Semantic] Parsing document structure...")

        # Parse into semantic chunks using header tracking
        semantic_chunks = Semantic._parse_with_headers(standardized_doc.content, chunk_token_num, overlap_percent)

        if callback:
            callback(0.8, f"[Semantic] Tokenizing {len(semantic_chunks)} chunks...")

        # Convert to RAGFlow chunk format
        results = []
        for chunk in semantic_chunks:
            tokens = rag_tokenizer.tokenize(chunk.text)
            ck = {
                "content_with_weight": tokens,
                "content_ltks": tokens,
                "content_sm_ltks": rag_tokenizer.fine_grained_tokenize(tokens),
            }
            # Add document metadata
            ck.update(doc)

            # Add header hierarchy metadata (the key new feature)
            ck["header_path"] = chunk.header_path

            # Merge any additional metadata
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    if key not in ck:
                        ck[key] = value

            results.append(ck)

        if callback:
            callback(1.0, f"[Semantic] Created {len(results)} chunks with header hierarchy")

        logging.info(f"[Semantic] Chunked {filename}: {len(results)} chunks")

        return results

    @staticmethod
    def _parse_with_headers(content: str, chunk_token_num: int, overlap_percent: int) -> List[SemanticChunk]:
        """
        Stack-based header tracking algorithm.

        Adapted from LlamaIndex MarkdownNodeParser.get_nodes_from_node

        Algorithm:
        1. Iterate through lines
        2. Track code blocks (don't parse headers inside them)
        3. When hitting a header, emit previous section and update stack
        4. Pop stack when going "up" the hierarchy (e.g., H2 after H3)
        5. Build header_path from current stack

        Args:
            content: Full Markdown content
            chunk_token_num: Maximum tokens per chunk
            overlap_percent: Percentage of previous chunk to overlap

        Returns:
            List of SemanticChunk objects
        """
        if not content:
            return []

        lines = content.split("\n")

        header_stack: List[Tuple[int, str]] = []  # (level, text)
        code_block = False
        current_section = ""
        chunks: List[SemanticChunk] = []

        def get_header_path() -> str:
            """Build header path from stack."""
            if not header_stack:
                return "/"
            return "/" + "/".join(h[1] for h in header_stack) + "/"

        def emit_chunk(text: str, header_path: str):
            """Create chunk, splitting if too large."""
            if not text.strip():
                return

            tokens = num_tokens(text)

            if tokens <= chunk_token_num:
                # Fits in single chunk
                final_text = text.strip()
                chunks.append(SemanticChunk(text=final_text, header_path=header_path, metadata={"tokens": num_tokens(final_text)}))
            else:
                # Split large sections at paragraph boundaries
                paragraphs = text.split("\n\n")
                current_chunk = ""
                current_tokens = 0
                separator_tokens = num_tokens("\n\n")

                def split_large_paragraph(para: str) -> List[str]:
                    """Split a large paragraph at sentence boundaries."""
                    sentences = []
                    try:
                        import nltk

                        try:
                            sentences = nltk.sent_tokenize(para)
                        except LookupError:
                            # Attempt to download punkt if missing
                            logging.warning("[Semantic] 'punkt' not found. Attempting to download...")
                            nltk.download("punkt")
                            sentences = nltk.sent_tokenize(para)
                    except (ImportError, LookupError) as e:
                        logging.warning(f"[Semantic] NLTK/punkt not available ({e}). Falling back to regex splitting.")
                        # Fallback to regex splitting
                        sentences = re.split(r"(?<=[.!?])\s+", para)

                    if not sentences:
                        logging.warning("[Semantic] No sentences found. Returning original paragraph.")
                        return [para]

                    # Group sentences to fit within chunk_token_num
                    result = []
                    current_sentence_group = ""
                    current_sentence_tokens = 0

                    for sentence in sentences:
                        sentence_tokens = num_tokens(sentence)

                        # Handle single sentence exceeding limit
                        if sentence_tokens > chunk_token_num:
                            # 1. Flush any existing group
                            if current_sentence_group:
                                result.append(current_sentence_group)
                                current_sentence_group = ""
                                current_sentence_tokens = 0

                            # 2. Split the oversized sentence iteratively
                            words = sentence.split(" ")
                            temp_chunk = ""
                            temp_tokens = 0
                            for word in words:
                                word_tokens = num_tokens(word)
                                space_tokens = num_tokens(" ") if temp_chunk else 0
                                if temp_tokens + word_tokens + space_tokens > chunk_token_num and temp_chunk:
                                    result.append(temp_chunk)
                                    temp_chunk = word
                                    temp_tokens = word_tokens
                                else:
                                    temp_chunk += (" " if temp_chunk else "") + word
                                    temp_tokens += word_tokens + space_tokens

                            if temp_chunk:
                                result.append(temp_chunk)
                            continue

                        space_tokens = num_tokens(" ") if current_sentence_group else 0

                        if current_sentence_tokens + sentence_tokens + space_tokens > chunk_token_num and current_sentence_group:
                            result.append(current_sentence_group)
                            current_sentence_group = sentence
                            current_sentence_tokens = sentence_tokens
                        else:
                            current_sentence_group += (" " if current_sentence_group else "") + sentence
                            current_sentence_tokens += sentence_tokens + space_tokens

                    if current_sentence_group:
                        result.append(current_sentence_group)

                    return result if result else [para]

                for para in paragraphs:
                    para_tokens = num_tokens(para)

                    # Handle oversized paragraphs
                    if para_tokens > chunk_token_num:
                        # First, emit any accumulated content
                        if current_chunk:
                            chunks.append(SemanticChunk(text=current_chunk.strip(), header_path=header_path, metadata={"tokens": current_tokens}))
                            current_chunk = ""
                            current_tokens = 0

                        # Split the large paragraph
                        para_pieces = split_large_paragraph(para)
                        for piece in para_pieces:
                            piece_tokens = num_tokens(piece)
                            if piece_tokens <= chunk_token_num:
                                chunks.append(SemanticChunk(text=piece.strip(), header_path=header_path, metadata={"tokens": piece_tokens}))
                            else:
                                # Still too large, emit with warning (already logged in split_large_paragraph)
                                chunks.append(SemanticChunk(text=piece.strip(), header_path=header_path, metadata={"tokens": piece_tokens, "oversized": True}))
                        continue

                    # Calculate total tokens including separator
                    sep_cost = separator_tokens if current_chunk else 0
                    if current_tokens + para_tokens + sep_cost > chunk_token_num and current_chunk:
                        # Emit current chunk
                        chunks.append(SemanticChunk(text=current_chunk.strip(), header_path=header_path, metadata={"tokens": current_tokens}))

                        # Handle overlap (token-based)
                        if overlap_percent > 0:
                            overlap_tokens = int(current_tokens * overlap_percent / 100)
                            # Take last portion of current_chunk by rebuilding from end
                            # Optimized O(n) approach: compute count once and subtract
                            overlap_text = current_chunk
                            overlap_count = num_tokens(overlap_text)

                            while overlap_count > overlap_tokens and len(overlap_text) > 0:
                                # Remove from start until we're within overlap_tokens
                                first_space = overlap_text.find(" ")
                                if first_space == -1:
                                    break

                                removed_piece = overlap_text[: first_space + 1]
                                overlap_count -= num_tokens(removed_piece)
                                overlap_text = overlap_text[first_space + 1 :]

                            current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                            # Update current_tokens efficiently
                            sep_cost = separator_tokens if overlap_text else 0
                            current_tokens = overlap_count + sep_cost + para_tokens
                        else:
                            current_chunk = para
                            current_tokens = para_tokens
                    else:
                        # Add to current chunk, accounting for separator
                        if current_chunk:
                            current_chunk += "\n\n" + para
                            current_tokens += para_tokens + separator_tokens
                        else:
                            current_chunk = para
                            current_tokens = para_tokens

                # Emit final chunk
                if current_chunk.strip():
                    final_text = current_chunk.strip()
                    chunks.append(SemanticChunk(text=final_text, header_path=header_path, metadata={"tokens": num_tokens(final_text)}))

        for line in lines:
            # Track code blocks (don't parse headers inside them)
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section += line + "\n"
                continue

            if not code_block:
                # Check for Markdown header
                header_match = re.match(r"^(#+)\s+(.*)", line)
                if header_match:
                    # Emit previous section before processing new header
                    emit_chunk(current_section, get_header_path())

                    level = len(header_match.group(1))
                    text = header_match.group(2).strip()

                    # Pop headers of equal or higher level
                    # This handles going "up" the hierarchy (e.g., H3 -> H2)
                    while header_stack and header_stack[-1][0] >= level:
                        header_stack.pop()

                    # Push new header onto stack
                    header_stack.append((level, text))

                    # Start new section with header line
                    current_section = f"{'#' * level} {text}\n"
                    continue

            # Regular content line
            current_section += line + "\n"

        # Emit final section
        emit_chunk(current_section, get_header_path())

        return chunks
