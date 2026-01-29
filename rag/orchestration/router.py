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

import re
import os
import logging
from io import BytesIO
from typing import List, Any
from dataclasses import dataclass, field

from rag.parsers.deepdoc_client import DeepDocParser

from common.constants import LLMType
from api.db.services.llm_service import LLMBundle

from rag.parsers.deepdoc.excel_parser import RAGFlowExcelParser as ExcelParser
from rag.parsers.deepdoc.html_parser import RAGFlowHtmlParser as HtmlParser
from rag.parsers.deepdoc.json_parser import RAGFlowJsonParser as JsonParser
from rag.parsers.deepdoc.txt_parser import RAGFlowTxtParser as TxtParser
from rag.parsers.deepdoc.pdf_parser import VisionParser
from rag.parsers.tcadp_client import TCADPParser
from rag.parsers import PlainParser

from rag.parsers.docling_client import DoclingParser
# by_paddleocr uses LLMBundle for OCR capabilities.

from common.parser_config_utils import normalize_layout_recognizer
from rag.utils.file_utils import extract_links_from_pdf, extract_links_from_docx
# Embedding extraction logic is handled in naive.py shim for now to avoid moving too much logic at once.


@dataclass
class ParseResult:
    """Explicit container for parser results."""

    sections: List[Any] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    section_images: List[Any] = field(default_factory=list)
    pdf_parser: Any = None
    is_markdown: bool = False
    urls: set = field(default_factory=set)

    def __post_init__(self):
        if self.urls is None:
            self.urls = set()


class UniversalRouter:
    @staticmethod
    def route(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
        parser_config = kwargs.get("parser_config", {})
        is_root = kwargs.get("is_root", True)
        urls = set()

        # Helper for unified layout recognizer lookup
        layout_recognizer_val = parser_config.get("layout_recognizer", parser_config.get("layout_recognize", "DeepDOC"))

        # 1. Docling Override (Universal for supported types)
        if layout_recognizer_val == "Docling":
            # Docling supports multiple formats. We route them all via by_docling.
            sections, tables, _ = by_docling(filename, binary, from_page=from_page, to_page=to_page, lang=lang, callback=callback, **kwargs)
            # Docling output is Markdown, so we set is_markdown=True
            return ParseResult(sections=sections, tables=tables, is_markdown=True, urls=urls)

        # 2. Extension-based Routing
        if re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
            chunk_token_num = int(parser_config.get("chunk_token_num", 128))
            sections = HtmlParser()(filename, binary, chunk_token_num)
            sections = [(_, "") for _ in sections if _]
            return ParseResult(sections=sections, urls=urls)

        elif re.search(r"\.(json|jsonl|ldjson)$", filename, re.IGNORECASE):
            chunk_token_num = int(parser_config.get("chunk_token_num", 128))
            sections = JsonParser(chunk_token_num)(binary)
            sections = [(_, "") for _ in sections if _]
            return ParseResult(sections=sections, urls=urls)

        elif re.search(r"\.docx$", filename, re.IGNORECASE):
            if parser_config.get("analyze_hyperlink", False) and is_root:
                urls = extract_links_from_docx(binary)

            sections, _ = DeepDocParser().parse_docx(filename, binary)
            return ParseResult(sections=sections, urls=urls)

        elif re.search(r"\.pdf$", filename, re.IGNORECASE):
            if parser_config.get("analyze_hyperlink", False) and is_root:
                urls = extract_links_from_pdf(binary)

            layout_recognizer, parser_model_name = normalize_layout_recognizer(layout_recognizer_val)
            if isinstance(layout_recognizer, bool):
                layout_recognizer = "DeepDOC" if layout_recognizer else "Plain Text"

            name = layout_recognizer.strip().lower()
            parser = PARSERS.get(name, by_plaintext)

            sections, tables, pdf_parser = parser(
                filename=filename,
                binary=binary,
                from_page=from_page,
                to_page=to_page,
                lang=lang,
                callback=callback,
                layout_recognizer=layout_recognizer,
                mineru_llm_name=parser_model_name,
                paddleocr_llm_name=parser_model_name,
                **kwargs,
            )
            return ParseResult(sections=sections, tables=tables, pdf_parser=pdf_parser, urls=urls)

        elif re.search(r"\.(csv|xlsx?)$", filename, re.IGNORECASE):
            layout_recognizer, _ = normalize_layout_recognizer(layout_recognizer_val)
            if isinstance(layout_recognizer, bool):
                layout_recognizer = "DeepDOC" if layout_recognizer else "Plain Text"
            layout_recognizer_normalized = layout_recognizer.strip().lower() if isinstance(layout_recognizer, str) else ""
            if layout_recognizer_normalized == "tcadp parser":
                table_result_type = parser_config.get("table_result_type", "1")
                markdown_image_response_type = parser_config.get("markdown_image_response_type", "1")
                tcadp_parser = TCADPParser(table_result_type=table_result_type, markdown_image_response_type=markdown_image_response_type)
                if not tcadp_parser.check_installation():
                    if callback:
                        callback(-1, "TCADP parser not available.")
                    return ParseResult(urls=urls)

                file_type = "XLSX" if re.search(r"\.xlsx?$", filename, re.IGNORECASE) else "CSV"
                sections, tables = tcadp_parser.parse_pdf(filepath=filename, binary=binary, callback=callback, output_dir=os.environ.get("TCADP_OUTPUT_DIR", ""), file_type=file_type)
                return ParseResult(sections=sections, tables=tables, urls=urls)
            else:
                excel_parser = ExcelParser()
                # logic for html4excel
                if parser_config.get("html4excel"):
                    sections = [(_, "") for _ in excel_parser.html(binary, 12) if _]
                else:
                    sections = [(_, "") for _ in excel_parser(binary) if _]
                return ParseResult(sections=sections, urls=urls)

        elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
            sections = TxtParser()(filename, binary, parser_config.get("chunk_token_num", 128), parser_config.get("delimiter", "\n!?;。；！？"))
            return ParseResult(sections=sections, urls=urls)

        elif re.search(r"\.(md|markdown|mdx)$", filename, re.IGNORECASE):
            sections, tables, section_images, hyperlink_urls = DeepDocParser().parse_markdown(
                filename, binary, parser_config=parser_config, analyze_hyperlink=parser_config.get("analyze_hyperlink", False) and is_root
            )
            urls.update(hyperlink_urls)

            return ParseResult(sections=sections, tables=tables, section_images=section_images, is_markdown=True, urls=urls)

        elif re.search(r"\.doc$", filename, re.IGNORECASE):
            try:
                from tika import parser as tika_parser

                binary_io = BytesIO(binary)
                doc_parsed = tika_parser.from_buffer(binary_io)
                if doc_parsed.get("content", None) is not None:
                    sections = [(_, "") for _ in doc_parsed["content"].split("\n") if _]
                    return ParseResult(sections=sections, urls=urls)
                else:
                    msg = f"tika.parser got empty content from {filename}."
                    if callback:
                        callback(0.8, msg)
                    logging.warning(msg)
                    return ParseResult(urls=urls)
            except Exception as e:
                msg = f"tika not available: {e}"
                if callback:
                    callback(0.8, msg)
                logging.warning(msg)
                return ParseResult(urls=urls)

        else:
            raise NotImplementedError(f"file type not supported yet: {filename}")


# Dispatch Functions


def by_deepdoc(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, pdf_cls=None, **kwargs):
    parser = DeepDocParser()
    sections, tables, pdf_parser = parser.parse_pdf(filepath=filename, binary=binary, from_page=from_page, to_page=to_page, callback=callback, **kwargs)
    return sections, tables, pdf_parser


def by_mineru(
    filename,
    binary=None,
    from_page=0,
    to_page=100000,
    lang="Chinese",
    callback=None,
    pdf_cls=None,
    parse_method: str = "raw",
    mineru_llm_name: str | None = None,
    tenant_id: str | None = None,
    **kwargs,
):
    mineru_parser = None
    if tenant_id:
        if not mineru_llm_name:
            try:
                from api.db.services.tenant_llm_service import TenantLLMService

                env_name = TenantLLMService.ensure_mineru_from_env(tenant_id)
                candidates = TenantLLMService.query(tenant_id=tenant_id, llm_factory="MinerU", model_type=LLMType.OCR)
                if candidates:
                    mineru_llm_name = candidates[0].llm_name
                elif env_name:
                    mineru_llm_name = env_name
            except Exception as e:  # best-effort fallback
                logging.warning(f"fallback to env mineru: {e}")

        if mineru_llm_name:
            try:
                ocr_model = LLMBundle(tenant_id=tenant_id, llm_type=LLMType.OCR, llm_name=mineru_llm_name, lang=lang)
                mineru_parser = ocr_model.mdl
                sections, tables = mineru_parser.parse_pdf(
                    filepath=filename,
                    binary=binary,
                    callback=callback,
                    parse_method=parse_method,
                    lang=lang,
                    from_page=from_page,
                    to_page=to_page,
                    **kwargs,
                )
                return sections, tables, mineru_parser
            except Exception as e:
                logging.error(f"Failed to parse pdf via LLMBundle MinerU ({mineru_llm_name}): {e}")
                if callback:
                    callback(-1, f"MinerU ({mineru_llm_name}) found but failed to parse: {e}")
                return [], [], mineru_parser

    if callback:
        callback(-1, "MinerU not found.")
    return [], [], mineru_parser


def by_docling(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, pdf_cls=None, **kwargs):
    pdf_parser = DoclingParser()
    parse_method = kwargs.get("parse_method", "raw")

    if not pdf_parser.check_installation():
        if callback and callable(callback):
            callback(-1, "Docling not found.")
        return [], [], pdf_parser

    sections, tables = pdf_parser.parse_pdf(
        filepath=filename,
        binary=binary,
        callback=callback,
        output_dir=os.environ.get("DOCLING_OUTPUT_DIR", ""),
        delete_output=bool(int(os.environ.get("DOCLING_DELETE_OUTPUT", 1))),
        parse_method=parse_method,
    )
    return sections, tables, pdf_parser


def by_tcadp(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, pdf_cls=None, **kwargs):
    tcadp_parser = TCADPParser()

    if not tcadp_parser.check_installation():
        if callback and callable(callback):
            callback(-1, "TCADP parser not available. Please check Tencent Cloud API configuration.")
        return None, None, tcadp_parser

    sections, tables = tcadp_parser.parse_pdf(
        filepath=filename,
        binary=binary,
        callback=callback,
        output_dir=os.environ.get("TCADP_OUTPUT_DIR", ""),
        file_type="PDF",
        file_start_page=from_page + 1,
        file_end_page=to_page,
    )
    return sections, tables, tcadp_parser


def by_paddleocr(
    filename,
    binary=None,
    from_page=0,
    to_page=100000,
    lang="Chinese",
    callback=None,
    pdf_cls=None,
    parse_method: str = "raw",
    paddleocr_llm_name: str | None = None,
    tenant_id: str | None = None,
    **kwargs,
):
    paddle_parser = None
    if tenant_id:
        if not paddleocr_llm_name:
            try:
                from api.db.services.tenant_llm_service import TenantLLMService

                env_name = TenantLLMService.ensure_paddleocr_from_env(tenant_id)
                candidates = TenantLLMService.query(tenant_id=tenant_id, llm_factory="PaddleOCR", model_type=LLMType.OCR)
                if candidates:
                    paddleocr_llm_name = candidates[0].llm_name
                elif env_name:
                    paddleocr_llm_name = env_name
            except Exception as e:  # best-effort fallback
                logging.warning(f"fallback to env paddleocr: {e}")

        if paddleocr_llm_name:
            try:
                ocr_model = LLMBundle(tenant_id=tenant_id, llm_type=LLMType.OCR, llm_name=paddleocr_llm_name, lang=lang)
                paddle_parser = ocr_model.mdl
                sections, tables = paddle_parser.parse_pdf(
                    filepath=filename,
                    binary=binary,
                    callback=callback,
                    parse_method=parse_method,
                    from_page=from_page,
                    to_page=to_page,
                    **kwargs,
                )
                return sections, tables, paddle_parser
            except Exception as e:
                logging.error(f"Failed to parse pdf via LLMBundle PaddleOCR ({paddleocr_llm_name}): {e}")
                if callback and callable(callback):
                    callback(-1, f"PaddleOCR parsing failed: {e}")
                return [], [], paddle_parser

    if callback and callable(callback):
        callback(-1, "PaddleOCR not found.")
    return [], [], paddle_parser


def by_plaintext(filename, binary=None, from_page=0, to_page=100000, callback=None, **kwargs):
    layout_recognizer = (kwargs.get("layout_recognizer") or "").strip()
    if (not layout_recognizer) or (layout_recognizer == "Plain Text"):
        pdf_parser = PlainParser()
    else:
        tenant_id = kwargs.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id is required when using vision layout recognizer")
        vision_model = LLMBundle(
            tenant_id,
            LLMType.IMAGE2TEXT,
            llm_name=layout_recognizer,
            lang=kwargs.get("lang", "Chinese"),
        )
        pdf_parser = VisionParser(vision_model=vision_model, **kwargs)

    sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback)
    return sections, tables, pdf_parser


PARSERS = {
    "deepdoc": by_deepdoc,
    "mineru": by_mineru,
    "docling": by_docling,
    "tcadp": by_tcadp,
    "paddleocr": by_paddleocr,
    "plaintext": by_plaintext,  # default
    "plain text": by_plaintext,
}
