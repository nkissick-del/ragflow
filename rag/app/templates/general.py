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
from common.token_utils import num_tokens_from_string
from deepdoc.parser.figure_parser import vision_figure_parser_docx_wrapper_naive
from rag.nlp import (
    concat_img,
    naive_merge,
    naive_merge_with_images,
    naive_merge_docx,
    tokenize_chunks,
    tokenize_chunks_with_images,
    doc_tokenize_chunks_with_images,
    tokenize_table,
)


def _get_chunk_token_num(parser_config: dict, default: int = 128) -> int:
    """Extract and sanitize chunk_token_num from parser_config.

    Args:
        parser_config: Configuration dictionary containing chunk_token_num.
        default: Default value if chunk_token_num is not set (default: 128).

    Returns:
        A non-negative integer chunk token number.
    """
    value = parser_config.get("chunk_token_num")
    if value is None:
        return max(0, default)
    return max(0, int(value))


class General:
    @staticmethod
    def chunk(filename, sections, tables, section_images, pdf_parser, is_markdown, parser_config, doc, is_english, callback, **kwargs):
        res = []
        try:
            child_deli = (parser_config.get("children_delimiter") or "").encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
        except Exception:
            child_deli = parser_config.get("children_delimiter") or ""

        # 1. Handle Tables (if any)
        if tables:
            res.extend(tokenize_table(tables, doc, is_english))

        # 2. Handle DOCX specifically (Legacy behavior)
        # Check if filename ends with docx AND we are using standard sections (not Docling override which would produce standard layout)
        # Ideally Router tells us if it's "Docx" legacy mode.
        # For now, we rely on filename check + check if sections is Docx structure?
        # Or we can just check filename. If we use Docling, we might want standard flow.
        # But if Router returns sections for Docling, how do we distinguish?
        # Let's assume if tables/images are None and is_markdown is False, and filename is .docx, it *might* be legacy.
        # But wait, Router returns `sections` for Docx legacy which is a list of objects.

        if re.search(r"\.docx$", filename, re.IGNORECASE) and not kwargs.get("is_docling", False):
            # This logic mimics naive.py lines 812-820
            table_context_size = max(0, int(parser_config.get("table_context_size", 0) or 0))
            image_context_size = max(0, int(parser_config.get("image_context_size", 0) or 0))

            chunk_token_num = _get_chunk_token_num(parser_config)
            chunks, images = naive_merge_docx(sections, chunk_token_num, parser_config.get("delimiter", "\n!?。；！？"), table_context_size, image_context_size)

            vision_figure_parser_docx_wrapper_naive(chunks=chunks, idx_lst=images, callback=callback, **kwargs)

            res.extend(doc_tokenize_chunks_with_images(chunks, doc, is_english, child_delimiters_pattern=child_deli))
            return res

        # 3. Standard Flow (PDF, Markdown, Txt, or Docling-DOCX)
        if not sections:
            return res

        if is_markdown:
            merged_chunks = []
            merged_images = []
            chunk_limit = _get_chunk_token_num(parser_config)
            overlapped_percent = int(parser_config.get("overlapped_percent", 0) or 0)
            overlapped_percent = max(0, min(100, overlapped_percent))

            current_text = ""
            current_tokens = 0
            current_image = None

            for idx, sec in enumerate(sections):
                text = sec[0] if isinstance(sec, tuple) else sec
                sec_tokens = num_tokens_from_string(text)
                sec_image = section_images[idx] if section_images and idx < len(section_images) else None

                if current_text and current_tokens + sec_tokens > chunk_limit:
                    merged_chunks.append(current_text)
                    merged_images.append(current_image)
                    overlap_part = ""
                    if overlapped_percent > 0:
                        overlap_len = int(len(current_text) * overlapped_percent / 100)
                        if overlap_len > 0:
                            overlap_part = current_text[-overlap_len:]
                    current_text = overlap_part
                    current_tokens = num_tokens_from_string(current_text)
                    current_image = current_image if overlap_part else None

                if current_text:
                    current_text += "\n" + text
                    current_tokens += sec_tokens + num_tokens_from_string("\n")
                else:
                    current_text = text
                    current_tokens += sec_tokens

                if sec_image:
                    current_image = concat_img(current_image, sec_image) if current_image else sec_image

            if current_text:
                merged_chunks.append(current_text)
                merged_images.append(current_image)

            chunks = merged_chunks
            has_images = merged_images and any(img is not None for img in merged_images)

            if has_images:
                res.extend(tokenize_chunks_with_images(chunks, doc, is_english, merged_images, child_delimiters_pattern=child_deli))
            else:
                res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser, child_delimiters_pattern=child_deli))
        else:
            if section_images:
                if all(image is None for image in section_images):
                    section_images = None

            if section_images:
                chunk_token_num = _get_chunk_token_num(parser_config)
                chunks, images = naive_merge_with_images(sections, section_images, chunk_token_num, parser_config.get("delimiter", "\n!?。；！？"))
                res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images, child_delimiters_pattern=child_deli))
            else:
                chunk_token_num = _get_chunk_token_num(parser_config)
                chunks = naive_merge(sections, chunk_token_num, parser_config.get("delimiter", "\n!?。；！？"))
                res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser, child_delimiters_pattern=child_deli))

        return res
