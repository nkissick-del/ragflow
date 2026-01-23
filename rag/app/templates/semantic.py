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
2. Code block protection (don't parse headers inside ``` or ~~~)
3. Never split mid-paragraph
4. Configurable chunk size with overlap
5. header_path metadata on each chunk
6. Configurable tokenizer model (via env var or set_tokenizer_model())

Token Counting:
- Uses tiktoken when available (configurable model, default: gpt-3.5-turbo)
- Falls back to heuristic estimation for CJK vs English text
- Configure via SEMANTIC_TOKENIZER_MODEL env var or set_tokenizer_model()

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
    import os
    import threading
    from tiktoken import encoding_for_model

    # Default model for token counting
    # Can be overridden via environment variable SEMANTIC_TOKENIZER_MODEL
    # or by calling set_tokenizer_model()
    _DEFAULT_TOKENIZER_MODEL = "gpt-3.5-turbo"
    _tokenizer_model = os.environ.get("SEMANTIC_TOKENIZER_MODEL", _DEFAULT_TOKENIZER_MODEL)
    _enc = None  # Lazy-initialized on first use
    _tokenizer_lock = threading.Lock()  # Protects _tokenizer_model and _enc

    def _validate_tokenizer_model(model_name: str) -> tuple[bool, str | None]:
        """
        Validate that the tokenizer model is supported by tiktoken.

        Args:
            model_name: Model name to validate

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        try:
            # Try to create encoder to validate the model
            _ = encoding_for_model(model_name)
            return (True, None)
        except KeyError:
            # Model not found in tiktoken
            error_msg = (
                f"Invalid tokenizer model: '{model_name}'. "
                f"This model is not supported by tiktoken. "
                f"Common models include: 'gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo', "
                f"'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'. "
                f"See tiktoken documentation for full list."
            )
            logging.error(f"[Semantic] {error_msg}")
            return (False, error_msg)
        except Exception as e:
            # Other errors (e.g., network issues if model list needs updating)
            error_msg = f"Error validating tokenizer model '{model_name}': {type(e).__name__}: {e}"
            logging.error(f"[Semantic] {error_msg}")
            return (False, error_msg)

    # Validate environment variable at module load
    is_valid, error_msg = _validate_tokenizer_model(_tokenizer_model)
    if not is_valid:
        logging.warning(
            f"[Semantic] Falling back to default model '{_DEFAULT_TOKENIZER_MODEL}' due to invalid "
            f"SEMANTIC_TOKENIZER_MODEL='{_tokenizer_model}'"
        )
        _tokenizer_model = _DEFAULT_TOKENIZER_MODEL
        # Re-validate the default (should always succeed)
        is_valid, error_msg = _validate_tokenizer_model(_tokenizer_model)
        if not is_valid:
            raise RuntimeError(f"[Semantic] Default tokenizer model '{_DEFAULT_TOKENIZER_MODEL}' is invalid. This should never happen.")

    def set_tokenizer_model(model_name: str):
        """
        Configure the tokenizer model for token counting.

        This function validates the model name immediately and raises ValueError if invalid.
        It is thread-safe and can be called at any time, but it's recommended to configure
        the tokenizer once at application startup before any concurrent document processing
        begins to avoid unnecessary re-initialization overhead.

        Args:
            model_name: Model name for tiktoken (e.g., "gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002")

        Raises:
            ValueError: If the model name is not supported by tiktoken

        Example:
            # At application startup
            try:
                set_tokenizer_model("gpt-4")
            except ValueError as e:
                print(f"Invalid model: {e}")
                # Fall back to default or handle error

        Note:
            Changing the model after processing has started will cause the encoder
            to be re-initialized on the next call to num_tokens(), which may have
            a small performance impact.
        """
        global _tokenizer_model, _enc

        # Validate before acquiring lock (validation may be slow)
        is_valid, error_msg = _validate_tokenizer_model(model_name)
        if not is_valid:
            raise ValueError(error_msg)

        with _tokenizer_lock:
            _tokenizer_model = model_name
            _enc = None  # Reset encoder to force re-initialization

    def _get_encoder():
        """
        Lazy-initialize the tiktoken encoder (thread-safe).

        Returns:
            Initialized tiktoken encoder for the configured model

        Raises:
            RuntimeError: If the encoder cannot be initialized (should not happen
                         if validation was successful)
        """
        global _enc
        with _tokenizer_lock:
            if _enc is None:
                try:
                    _enc = encoding_for_model(_tokenizer_model)
                    logging.info(f"[Semantic] Initialized tiktoken encoder for model: {_tokenizer_model}")
                except Exception as e:
                    error_msg = (
                        f"Failed to initialize tiktoken encoder for model '{_tokenizer_model}': {e}. "
                        f"This should not happen if validation succeeded. Please report this issue."
                    )
                    logging.error(f"[Semantic] {error_msg}")
                    raise RuntimeError(error_msg) from e
            return _enc

    def num_tokens(text: str, is_english: bool = None) -> int:
        """
        Count tokens using tiktoken encoder.

        The encoder model can be configured via:
        1. Environment variable SEMANTIC_TOKENIZER_MODEL
        2. Calling set_tokenizer_model(model_name)
        3. Defaults to "gpt-3.5-turbo" if not configured

        Args:
            text: Text to count tokens for
            is_english: Ignored when tiktoken is available (uses actual tokenizer)

        Returns:
            Actual token count from tiktoken encoder
        """
        if not text:
            return 0
        encoder = _get_encoder()
        return len(encoder.encode(text))
except ImportError:
    logging.warning("[Semantic] tiktoken not installed. Using fallback estimator for num_tokens.")

    def num_tokens(text: str, is_english: bool = None) -> int:
        """
        Fallback token estimator when tiktoken is unavailable.

        Heuristic-based estimation:
        - English/Latin scripts: ~4 chars per token (conservative estimate)
        - CJK languages: ~1.5-2 chars per token (denser tokenization)

        Limitations:
        - This is a rough approximation and may vary significantly by content
        - Actual token counts depend on tokenizer vocabulary and algorithm
        - Mixed-language content uses the dominant language's ratio

        Args:
            text: Text to count tokens for
            is_english: If True, use English ratio. If False, use CJK ratio.
                       If None, defaults to English ratio (conservative).

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Default to English estimation if not specified
        if is_english is None or is_english:
            # English: ~4 chars per token (conservative for safety)
            return len(text) // 4
        else:
            # CJK: ~1.5-2 chars per token (denser tokenization)
            # Using 1.8 as middle ground: len(text) / 1.8 ≈ len(text) * 5 // 9
            return round(len(text) * 5 / 9)


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
        semantic_chunks = Semantic._parse_with_headers(standardized_doc.content, chunk_token_num, overlap_percent, is_english)

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
    def _parse_with_headers(content: str, chunk_token_num: int, overlap_percent: int, is_english: bool = True) -> List[SemanticChunk]:
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
            is_english: Whether content is primarily English (affects token estimation)

        Returns:
            List of SemanticChunk objects
        """
        if not content:
            return []

        # Create language-aware token counter for consistent estimation throughout parsing
        count_tokens = lambda text: num_tokens(text, is_english=is_english)

        lines = content.split("\n")

        header_stack: List[Tuple[int, str]] = []  # (level, text)
        code_block_fence = None  # Tracks fence type: None, "```", or "~~~"
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

            tokens = count_tokens(text)

            if tokens <= chunk_token_num:
                # Fits in single chunk
                final_text = text.strip()
                chunks.append(SemanticChunk(text=final_text, header_path=header_path, metadata={"tokens": count_tokens(final_text)}))
            else:
                # Split large sections at paragraph boundaries
                paragraphs = text.split("\n\n")
                current_chunk = ""
                current_tokens = 0
                separator_tokens = count_tokens("\n\n")

                def split_large_paragraph(para: str) -> List[str]:
                    """Split a large paragraph at sentence boundaries."""
                    sentences = []
                    sentences = []
                    try:
                        import nltk

                        # Check availability without downloading
                        try:
                            nltk.data.find("tokenizers/punkt")
                            if nltk.__version__ >= "3.8":
                                nltk.data.find("tokenizers/punkt_tab")
                            sentences = nltk.sent_tokenize(para)
                        except LookupError:
                            logging.warning("[Semantic] NLTK 'punkt' or 'punkt_tab' resource not found. Falling back to regex splitting.")
                            raise
                    except (ImportError, LookupError):
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
                        sentence_tokens = count_tokens(sentence)

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
                            space_tokens_val = count_tokens(" ")
                            for word in words:
                                word_tokens = count_tokens(word)
                                space_tokens = space_tokens_val if temp_chunk else 0
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

                        space_tokens = count_tokens(" ") if current_sentence_group else 0

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
                    para_tokens = count_tokens(para)

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
                            piece_tokens = count_tokens(piece)
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
                            overlap_count = count_tokens(overlap_text)

                            while overlap_count > overlap_tokens and len(overlap_text) > 0:
                                # Remove from start until we're within overlap_tokens
                                first_space = overlap_text.find(" ")
                                if first_space == -1:
                                    break

                                removed_piece = overlap_text[: first_space + 1]
                                overlap_count -= count_tokens(removed_piece)
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
                    chunks.append(SemanticChunk(text=final_text, header_path=header_path, metadata={"tokens": count_tokens(final_text)}))

        for line in lines:
            # Track code blocks (don't parse headers inside them)
            # Support both backtick (```) and tilde (~~~) fenced code blocks per CommonMark spec
            # Only matching fence types can close a block (e.g., ``` closes ```, not ~~~)
            stripped = line.lstrip()

            # Check for backtick fence
            if stripped.startswith("```"):
                if code_block_fence is None:
                    # Opening a backtick fence
                    code_block_fence = "```"
                elif code_block_fence == "```":
                    # Closing the matching backtick fence
                    code_block_fence = None
                # If code_block_fence is "~~~", this is content inside a tilde fence, not a fence marker
                current_section += line + "\n"
                continue

            # Check for tilde fence
            if stripped.startswith("~~~"):
                if code_block_fence is None:
                    # Opening a tilde fence
                    code_block_fence = "~~~"
                elif code_block_fence == "~~~":
                    # Closing the matching tilde fence
                    code_block_fence = None
                # If code_block_fence is "```", this is content inside a backtick fence, not a fence marker
                current_section += line + "\n"
                continue

            if code_block_fence is None:
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
