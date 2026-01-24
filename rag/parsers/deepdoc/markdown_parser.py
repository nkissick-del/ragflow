# -*- coding: utf-8 -*-
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
import logging
from io import BytesIO
from PIL import Image
from functools import reduce
from bs4 import BeautifulSoup
import requests
from pathlib import Path

from markdown import markdown
from rag.nlp import find_codec, concat_img


class RAGFlowMarkdownParser:
    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = int(chunk_token_num)

    def extract_tables_and_remainder(self, markdown_text, separate_tables=True):
        tables = []
        working_text = markdown_text

        def replace_tables_with_rendered_html(pattern, table_list, render=True):
            new_text = ""
            last_end = 0
            for match in pattern.finditer(working_text):
                raw_table = match.group()
                table_list.append(raw_table)
                if separate_tables:
                    # Skip this match (i.e., remove it)
                    new_text += working_text[last_end : match.start()] + "\n\n"
                else:
                    # Replace with rendered HTML
                    html_table = markdown(raw_table, extensions=["markdown.extensions.tables"]) if render else raw_table
                    new_text += working_text[last_end : match.start()] + html_table + "\n\n"
                last_end = match.end()
            new_text += working_text[last_end:]
            return new_text

        if "|" in markdown_text:  # for optimize performance
            # Standard Markdown table
            border_table_pattern = re.compile(
                r"""
                (?:\n|^)
                (?:\|.*?\|.*?\|.*?\n)
                (?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)
                (?:\|.*?\|.*?\|.*?\n)+
            """,
                re.VERBOSE,
            )
            working_text = replace_tables_with_rendered_html(border_table_pattern, tables)

            # Borderless Markdown table
            no_border_table_pattern = re.compile(
                r"""
                (?:\n|^)
                (?:\S.*?\|.*?\n)
                (?:(?:\s*[:-]+[-| :]*\s*).*?\n)
                (?:\S.*?\|.*?\n)+
                """,
                re.VERBOSE,
            )
            working_text = replace_tables_with_rendered_html(no_border_table_pattern, tables)

        # Replace any TAGS e.g. <table ...> to <table>
        TAGS = ["table", "td", "tr", "th", "tbody", "thead", "div"]
        table_with_attributes_pattern = re.compile(rf"<(?:{'|'.join(TAGS)})[^>]*>", re.IGNORECASE)

        def replace_tag(m):
            tag_name = re.match(r"<(\w+)", m.group()).group(1)
            return "<{}>".format(tag_name)

        working_text = re.sub(table_with_attributes_pattern, replace_tag, working_text)

        if "<table>" in working_text.lower():  # for optimize performance
            # HTML table extraction - handle possible html/body wrapper tags
            html_table_pattern = re.compile(
                r"""
            (?:\n|^)
            \s*
            (?:
                # case1: <html><body><table>...</table></body></html>
                (?:<html[^>]*>\s*<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>\s*</html>)
                |
                # case2: <body><table>...</table></body>
                (?:<body[^>]*>\s*<table[^>]*>.*?</table>\s*</body>)
                |
                # case3: only<table>...</table>
                (?:<table[^>]*>.*?</table>)
            )
            \s*
            (?=\n|$)
            """,
                re.VERBOSE | re.DOTALL | re.IGNORECASE,
            )

            def replace_html_tables():
                nonlocal working_text
                new_text = ""
                last_end = 0
                for match in html_table_pattern.finditer(working_text):
                    raw_table = match.group()
                    tables.append(raw_table)
                    if separate_tables:
                        new_text += working_text[last_end : match.start()] + "\n\n"
                    else:
                        new_text += working_text[last_end : match.start()] + raw_table + "\n\n"
                    last_end = match.end()
                new_text += working_text[last_end:]
                working_text = new_text

            replace_html_tables()

        return working_text, tables

    def md_to_html(self, sections):
        if not sections:
            return []
        if isinstance(sections, type("")):
            text = sections
        elif isinstance(sections[0], type("")):
            text = sections[0]
        else:
            return []

        html_content = markdown(text)
        soup = BeautifulSoup(html_content, "html.parser")
        return soup

    def get_hyperlink_urls(self, soup):
        if soup:
            return {a.get("href") for a in soup.find_all("a") if a.get("href")}
        return set()

    def extract_image_urls_with_lines(self, text):
        md_img_re = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
        html_img_re = re.compile(r'src=["\\\']([^"\\\'>\\s]+)', re.IGNORECASE)
        urls = []
        seen = set()
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            for url in md_img_re.findall(line):
                if (url, idx) not in seen:
                    urls.append({"url": url, "line": idx})
                    seen.add((url, idx))
            for url in html_img_re.findall(line):
                if (url, idx) not in seen:
                    urls.append({"url": url, "line": idx})
                    seen.add((url, idx))

        # cross-line
        try:
            soup = BeautifulSoup(text, "html.parser")
            newline_offsets = [m.start() for m in re.finditer(r"\n", text)] + [len(text)]
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src")
                if not src:
                    continue

                tag_str = str(img_tag)
                pos = text.find(tag_str)
                if pos == -1:
                    # fallback
                    pos = max(text.find(src), 0)
                line_no = 0
                for i, off in enumerate(newline_offsets):
                    if pos <= off:
                        line_no = i
                        break
                if (src, line_no) not in seen:
                    urls.append({"url": src, "line": line_no})
                    seen.add((src, line_no))
        except Exception as e:
            logging.error("Failed to extract image urls: {}".format(e))
            pass

        return urls

    def load_images_from_urls(self, urls, cache=None):
        cache = cache or {}
        images = []
        for url in urls:
            if url in cache:
                if cache[url]:
                    images.append(cache[url])
                continue
            img_obj = None
            try:
                if url.startswith(("http://", "https://")):
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image/"):
                        img_obj = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    local_path = Path(url)
                    if local_path.exists():
                        img_obj = Image.open(url).convert("RGB")
                    else:
                        logging.warning(f"Local image file not found: {url}")
            except Exception as e:
                logging.error(f"Failed to download/open image from {url}: {e}")
            cache[url] = img_obj
            if img_obj:
                images.append(img_obj)
        return images, cache

    def __call__(self, filename, binary=None, separate_tables=True, delimiter=None, return_section_images=False):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            # Determine encoding using find_codec by reading the file as binary first
            with open(filename, "rb") as f:
                raw_bytes = f.read()
            encoding = find_codec(raw_bytes) or "utf-8"
            txt = raw_bytes.decode(encoding, errors="ignore")

        remainder, tables = self.extract_tables_and_remainder(f"{txt}\n", separate_tables=separate_tables)
        # To eliminate duplicate tables in chunking result, use remainder instead of txt if needed
        # extractor = MarkdownElementExtractor(remainder)
        extractor = MarkdownElementExtractor(txt)
        image_refs = self.extract_image_urls_with_lines(txt)
        element_sections = extractor.extract_elements(delimiter, include_meta=True)

        sections = []
        section_images = []
        image_cache = {}
        for element in element_sections:
            content = element["content"]
            start_line = element["start_line"]
            end_line = element["end_line"]
            urls_in_section = [ref["url"] for ref in image_refs if start_line <= ref["line"] <= end_line]
            imgs = []
            if urls_in_section:
                imgs, image_cache = self.load_images_from_urls(urls_in_section, image_cache)
            combined_image = None
            if imgs:
                combined_image = reduce(concat_img, imgs) if len(imgs) > 1 else imgs[0]
            sections.append((content, ""))
            section_images.append(combined_image)

        tbls = []
        for table in tables:
            tbls.append(((None, markdown(table, extensions=["markdown.extensions.tables"])), ""))
        if return_section_images:
            return sections, tbls, section_images
        return sections, tbls


class MarkdownElementExtractor:
    def __init__(self, markdown_content):
        self.markdown_content = markdown_content
        self.lines = markdown_content.split("\n")

    def get_delimiters(self, delimiters):
        toks = re.findall(r"`([^`]+)`", delimiters)
        toks = sorted(set(toks), key=lambda x: -len(x))
        return "|".join(re.escape(t) for t in toks if t)

    def extract_elements(self, delimiter=None, include_meta=False):
        """Extract individual elements (headers, code blocks, lists, etc.)"""
        sections = []

        i = 0
        dels = ""
        if delimiter:
            dels = self.get_delimiters(delimiter)
        if len(dels) > 0:
            text = "\n".join(self.lines)
            if include_meta:
                pattern = re.compile(dels)
                last_end = 0
                for m in pattern.finditer(text):
                    part = text[last_end : m.start()]
                    if part and part.strip():
                        sections.append(
                            {
                                "content": part.strip(),
                                "start_line": text.count("\n", 0, last_end),
                                "end_line": text.count("\n", 0, m.start()),
                            }
                        )
                    last_end = m.end()

                part = text[last_end:]
                if part and part.strip():
                    sections.append(
                        {
                            "content": part.strip(),
                            "start_line": text.count("\n", 0, last_end),
                            "end_line": text.count("\n", 0, len(text)),
                        }
                    )
            else:
                parts = re.split(dels, text)
                sections = [p.strip() for p in parts if p and p.strip()]
            return sections
        while i < len(self.lines):
            line = self.lines[i]

            if re.match(r"^#{1,6}\s+.*$", line):
                # header
                element = self._extract_header(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith("```"):
                # code block
                element = self._extract_code_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line):
                # list block
                element = self._extract_list_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip().startswith(">"):
                # blockquote
                element = self._extract_blockquote(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip():
                # text block (paragraphs and inline elements until next block element)
                element = self._extract_text_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            else:
                i += 1

        if include_meta:
            sections = [section for section in sections if section["content"].strip()]
        else:
            sections = [section for section in sections if section.strip()]
        return sections

    def _extract_header(self, start_pos):
        return {
            "type": "header",
            "content": self.lines[start_pos],
            "start_line": start_pos,
            "end_line": start_pos,
        }

    def _extract_code_block(self, start_pos):
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]

        # Find the end of the code block
        for i in range(start_pos + 1, len(self.lines)):
            content_lines.append(self.lines[i])
            end_pos = i
            if self.lines[i].strip().startswith("```"):
                break

        return {
            "type": "code_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_list_block(self, start_pos):
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            # check if this line is a list item or continuation of a list
            if (
                re.match(r"^\s*[-*+]\s+.*$", line)
                or re.match(r"^\s*\d+\.\s+.*$", line)
                or (i > start_pos and not line.strip())
                or (i > start_pos and re.match(r"^\s{2,}[-*+]\s+.*$", line))
                or (i > start_pos and re.match(r"^\s{2,}\d+\.\s+.*$", line))
                or (i > start_pos and re.match(r"^\s+\w+.*$", line))
            ):
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "list_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_blockquote(self, start_pos):
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            if line.strip().startswith(">") or (i > start_pos and not line.strip()):
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "blockquote",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_text_block(self, start_pos):
        """Extract a text block (paragraphs, inline elements) until next block element"""
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]

        i = start_pos + 1
        while i < len(self.lines):
            line = self.lines[i]
            # stop if we encounter a block element
            if re.match(r"^#{1,6}\s+.*$", line) or line.strip().startswith("```") or re.match(r"^\s*[-*+]\s+.*$", line) or re.match(r"^\s*\d+\.\s+.*$", line) or line.strip().startswith(">"):
                break
            elif not line.strip():
                # check if the next line is a block element
                if i + 1 < len(self.lines) and (
                    re.match(r"^#{1,6}\s+.*$", self.lines[i + 1])
                    or self.lines[i + 1].strip().startswith("```")
                    or re.match(r"^\s*[-*+]\s+.*$", self.lines[i + 1])
                    or re.match(r"^\s*\d+\.\s+.*$", self.lines[i + 1])
                    or self.lines[i + 1].strip().startswith(">")
                ):
                    break
                else:
                    content_lines.append(line)
                    end_pos = i
                    i += 1
            else:
                content_lines.append(line)
                end_pos = i
                i += 1

        return {
            "type": "text_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }
