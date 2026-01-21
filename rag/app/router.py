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

from deepdoc.parser import ExcelParser, HtmlParser, JsonParser, TxtParser
from deepdoc.parser.tcadp_parser import TCADPParser

from rag.app.format_parsers import Docx, Markdown, PARSERS, by_plaintext

from common.parser_config_utils import normalize_layout_recognizer
from rag.utils.file_utils import extract_links_from_pdf, extract_links_from_docx
# Embedding extraction logic is handled in naive.py shim for now to avoid moving too much logic at once.


class UniversalRouter:
    @staticmethod
    def route(filename, binary=None, from_page=0, to_page=100000, lang="Chinese", callback=None, **kwargs):
        parser_config = kwargs.get("parser_config", {})
        is_root = kwargs.get("is_root", True)
        urls = set()

        # 1. Docling Override (Universal for supported types)
        # TODO: Unlock other types for Docling
        if parser_config.get("layout_recognizer") == "Docling":
            from rag.app.format_parsers import by_docling

            # Docling supports multiple formats. We route them all via by_docling.
            sections, tables, _ = by_docling(filename, binary, from_page=from_page, to_page=to_page, lang=lang, callback=callback, **kwargs)
            # Docling output is Markdown, so we set is_markdown=True
            return sections, tables, None, None, True, urls

        # 2. Extension-based Routing
        if re.search(r"\.docx$", filename, re.IGNORECASE):
            if parser_config.get("analyze_hyperlink", False) and is_root:
                urls = extract_links_from_docx(binary)

            sections = Docx()(filename, binary)
            return sections, None, None, None, False, urls

        elif re.search(r"\.pdf$", filename, re.IGNORECASE):
            if parser_config.get("analyze_hyperlink", False) and is_root:
                urls = extract_links_from_pdf(binary)

            layout_recognizer, parser_model_name = normalize_layout_recognizer(parser_config.get("layout_recognize", "DeepDOC"))
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
            return sections, tables, None, pdf_parser, False, urls

        elif re.search(r"\.(csv|xlsx?)$", filename, re.IGNORECASE):
            layout_recognizer = parser_config.get("layout_recognize", "DeepDOC")
            if layout_recognizer == "TCADP Parser":
                table_result_type = parser_config.get("table_result_type", "1")
                markdown_image_response_type = parser_config.get("markdown_image_response_type", "1")
                tcadp_parser = TCADPParser(table_result_type=table_result_type, markdown_image_response_type=markdown_image_response_type)
                if not tcadp_parser.check_installation():
                    if callback:
                        callback(-1, "TCADP parser not available.")
                    return [], [], None, None, False, urls

                file_type = "XLSX" if re.search(r"\.xlsx?$", filename, re.IGNORECASE) else "CSV"
                sections, tables = tcadp_parser.parse_pdf(filepath=filename, binary=binary, callback=callback, output_dir=os.environ.get("TCADP_OUTPUT_DIR", ""), file_type=file_type)
                return sections, tables, None, None, False, urls
            else:
                excel_parser = ExcelParser()
                # logic for html4excel
                if parser_config.get("html4excel"):
                    sections = [(_, "") for _ in excel_parser.html(binary, 12) if _]
                else:
                    sections = [(_, "") for _ in excel_parser(binary) if _]
                return sections, None, None, None, False, urls

        elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
            sections = TxtParser()(filename, binary, parser_config.get("chunk_token_num", 128), parser_config.get("delimiter", "\n!?;。；！？"))
            return sections, None, None, None, False, urls

        elif re.search(r"\.(md|markdown|mdx)$", filename, re.IGNORECASE):
            markdown_parser = Markdown(int(parser_config.get("chunk_token_num", 128)))
            sections, tables, section_images = markdown_parser(
                filename,
                binary,
                separate_tables=False,
                delimiter=parser_config.get("delimiter", "\n!?;。；！？"),
                return_section_images=True,
            )

            if parser_config.get("hyperlink_urls", False) and is_root:
                for idx, (section_text, _) in enumerate(sections):
                    soup = markdown_parser.md_to_html(section_text)
                    hyperlink_urls = markdown_parser.get_hyperlink_urls(soup)
                    urls.update(hyperlink_urls)

            return sections, tables, section_images, None, True, urls

        elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
            chunk_token_num = int(parser_config.get("chunk_token_num", 128))
            sections = HtmlParser()(filename, binary, chunk_token_num)
            sections = [(_, "") for _ in sections if _]
            return sections, None, None, None, False, urls

        elif re.search(r"\.(json|jsonl|ldjson)$", filename, re.IGNORECASE):
            chunk_token_num = int(parser_config.get("chunk_token_num", 128))
            sections = JsonParser(chunk_token_num)(binary)
            sections = [(_, "") for _ in sections if _]
            return sections, None, None, None, False, urls

        elif re.search(r"\.doc$", filename, re.IGNORECASE):
            try:
                from tika import parser as tika_parser

                binary_io = BytesIO(binary)
                doc_parsed = tika_parser.from_buffer(binary_io)
                if doc_parsed.get("content", None) is not None:
                    sections = doc_parsed["content"].split("\n")
                    sections = [(_, "") for _ in sections if _]
                    return sections, None, None, None, False, urls
                else:
                    msg = f"tika.parser got empty content from {filename}."
                    if callback:
                        callback(0.8, msg)
                    logging.warning(msg)
                    return [], None, None, None, False, urls
            except Exception as e:
                msg = f"tika not available: {e}"
                if callback:
                    callback(0.8, msg)
                logging.warning(msg)
                return [], None, None, None, False, urls

        else:
            raise NotImplementedError(f"file type not supported yet: {filename}")
