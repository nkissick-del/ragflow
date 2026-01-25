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
from timeit import default_timer as timer
from rag.orchestration.router import PARSERS, by_deepdoc, by_mineru, by_docling, by_tcadp, by_paddleocr, by_plaintext, UniversalRouter
from rag.templates.general import General
from .base import StandardizedDocument
from rag.utils.file_utils import extract_embed_file, extract_html
from rag.nlp import rag_tokenizer

# Re-exporting classes/functions for backward compatibility if needed
# Docx, Pdf, Markdown are internal classes now in format_parsers, not strictly needed to be exported unless external usage exists.
# Based on check, specific classes like Docx/Pdf defined in naive.py were local.


__all__ = [
    "PARSERS",
    "by_deepdoc",
    "by_mineru",
    "by_docling",
    "by_tcadp",
    "by_paddleocr",
    "by_plaintext",
    "chunk",
    "adapt_docling_output",
    "StandardizedDocument",
    "ParsingError",
]


class ParsingError(Exception):
    """Raised when document parsing fails and fallbacks are exhausted."""

    def __init__(self, message, original_exception=None, retry_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
        self.retry_exception = retry_exception


def adapt_docling_output(sections, tables, parser_config) -> StandardizedDocument:
    """
    Convert Docling parser output to Standardized IR.

    This adapter normalizes Docling's output into the StandardizedDocument
    format that semantic templates consume.

    Args:
        sections: Either a string (new semantic mode) or List[str] (legacy mode)
        tables: List of extracted tables (typically empty for Docling, embedded in markdown)
        parser_config: Parser configuration dict

    Returns:
        StandardizedDocument: Normalized document ready for template processing
    """
    # If sections is already a string (new Docling format), use it directly
    content = sections if isinstance(sections, str) else ""

    return StandardizedDocument(
        content_input=content,
        metadata={
            "parser": "docling",
            "layout_recognizer": parser_config.get("layout_recognizer", "Docling"),
            "tables": tables,
        },
    )


def chunk(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
    """
    Document Processor / Coordinator (formerly 'naive').

    This function acts as the central entry point for file processing. It coordinates:
    1.  Recursion for embedded files (if is_root).
    2.  Routing to specific format parsers (via UniversalRouter).
    3.  Fallback logic (e.g., Docling -> DeepDOC).
    4.  Recursion for extracted URLs.
    5.  Final text chunking and merging (via General template).
    """
    st = timer()

    # 1. Setup
    parser_config = kwargs.get("parser_config", {"chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognizer": "DeepDOC", "analyze_hyperlink": True})
    is_english = lang.lower() == "english"

    doc = {"docnm_kwd": filename, "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))}
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    # 2. Embedding Recursion (Root Only)
    is_root = kwargs.get("is_root", True)
    embed_res = []
    if is_root:
        embeds = []
        if binary is not None:
            # Only extract embedded files at the root call
            # Note: binary might be bytes or BytesIO? extract_embed_file expects bytes usually.
            try:
                embeds = extract_embed_file(binary)
            except Exception as e:
                logging.warning(f"Failed to extract embeds: {e}")
        else:
            # No binary content available, skip embedding extraction
            pass

        for embed_filename, embed_bytes in embeds:
            try:
                sub_res = chunk(embed_filename, binary=embed_bytes, lang=lang, callback=callback, is_root=False, **kwargs) or []
                embed_res.extend(sub_res)
            except Exception as e:
                error_msg = f"Failed to chunk embed {embed_filename}: {e}"
                logging.error(error_msg)
                if callback:
                    callback(0.05, error_msg)
                continue

    # 3. Routing & Parsing
    # Pass configuration to Router
    try:
        parsed = UniversalRouter.route(filename, binary, from_page, to_page, lang, callback, **kwargs)
        sections, tables, section_images, pdf_parser, is_markdown, urls = (parsed.sections, parsed.tables, parsed.section_images, parsed.pdf_parser, parsed.is_markdown, parsed.urls)
    except Exception as e:
        # Fallback Strategy for Docling
        if parser_config.get("layout_recognizer") == "Docling":
            msg = f"Docling parsing failed for {filename}: {e}. Falling back to DeepDOC."
            logging.warning(msg)
            if callback:
                callback(0.1, msg)

            # Switch to DeepDOC and retry
            parser_config["layout_recognizer"] = "DeepDOC"
            # Update kwargs with the modified parser_config so the router receives the override
            kwargs["parser_config"] = parser_config
            try:
                parsed = UniversalRouter.route(filename, binary, from_page, to_page, lang, callback, **kwargs)
                sections, tables, section_images, pdf_parser, is_markdown, urls = parsed
            except Exception as retry_e:
                logging.error(f"Fallback parsing (DeepDOC) also failed: {retry_e}")
                raise ParsingError(f"Parsing and fallback failed for {filename}", original_exception=e, retry_exception=retry_e)
        else:
            logging.error(f"Routing/Parsing failed: {e}")
            raise ParsingError(f"Routing/Parsing failed for {filename}", original_exception=e)

    # 4. URL Recursion
    url_res = []
    if urls and parser_config.get("analyze_hyperlink", False) and is_root:
        for index, url in enumerate(urls):
            html_bytes, metadata = extract_html(url)
            if not html_bytes:
                continue
            try:
                sub_url_res = chunk(url, html_bytes, callback=callback, lang=lang, is_root=False, **kwargs)
            except Exception as e:
                logging.error(f"Failed to chunk url in registered file type {url}: {e}")
                try:
                    sub_url_res = chunk(f"{index}.html", html_bytes, callback=callback, lang=lang, is_root=False, **kwargs)
                except Exception as fallback_e:
                    logging.error(f"Failed to chunk url with fallback {index}.html: {fallback_e}")
                    sub_url_res = []
            url_res.extend(sub_url_res)

    # 5. Template Processing
    is_docling = parser_config.get("layout_recognizer") == "Docling"
    use_semantic = parser_config.get("use_semantic_chunking", False)

    # Remove parser_config from kwargs if present to avoid multiple values error for 'parser_config'
    kwargs.pop("parser_config", None)

    if is_docling and use_semantic and is_markdown:
        # NEW PATH: Docling with semantic chunking enabled
        # Route to semantic template for structure-aware processing
        from rag.templates.semantic import Semantic

        standardized = adapt_docling_output(sections, tables, parser_config)
        res = Semantic.chunk(filename, standardized, parser_config, doc, is_english, callback, **kwargs)
        logging.info(f"[Orchestrator] Used Semantic template for {filename}")
    else:
        # LEGACY PATH: DeepDOC, or Docling without semantic flag

        # Compatibility: DoclingParser now returns strict Markdown string.
        # If we are in legacy mode (Semantic Chunking disabled), we need to split lines
        # to satisfy General.chunk's expectations for non-semantic processing.
        if is_docling and isinstance(sections, str):
            sections = sections.splitlines()

        res = General.chunk(filename, sections, tables, section_images, pdf_parser, is_markdown, parser_config, doc, is_english, callback, is_docling=is_docling, **kwargs)

    logging.info("chunk({}): {}".format(filename, timer() - st))

    # 6. Merge Results
    if embed_res:
        res.extend(embed_res)
    if url_res:
        res.extend(url_res)

    return res


if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass

    chunk(sys.argv[1], from_page=0, to_page=10, callback=dummy)
