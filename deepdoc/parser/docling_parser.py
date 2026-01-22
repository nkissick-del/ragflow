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
from __future__ import annotations

import logging
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from io import BytesIO
from typing import Callable, Optional
from os import PathLike

try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except Exception:

    class RAGFlowPdfParser:
        pass


class DoclingParser(RAGFlowPdfParser):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = os.environ.get("DOCLING_BASE_URL", "http://localhost:5001")
        self.auth_token = os.environ.get("DOCLING_AUTH_TOKEN")
        self.session = self._create_retry_session()

    def _create_retry_session(self, retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=None,  # Allow retries on all methods including POST
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def check_installation(self) -> bool:
        """Checks if the Docling server is reachable."""
        if not self.base_url:
            self.logger.warning("[Docling] DOCLING_BASE_URL not set.")
            return False

        try:
            # Simple health check - try a lightweight HEAD request to the base URL
            response = self.session.head(self.base_url.rstrip("/"), timeout=2)
            # Accept 2xx, 3xx (redirects), 404 (endpoint not found but server up), 405 (method not allowed but server up)
            if response.status_code < 400 or response.status_code in (404, 405):
                return True
            else:
                self.logger.warning(f"[Docling] Service returned error status {response.status_code} at {self.base_url}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"[Docling] Service unreachable at {self.base_url}: {e}")
            return False

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        method: str = "auto",
        delete_output: bool = True,
        parse_method: str = "raw",
        **kwargs,
    ):
        if callback:
            callback(0.1, "[Docling] Starting API conversion...")

        # Prepare input
        filename = Path(filepath).name if filepath else "document.pdf"
        file_content = None
        if binary:
            if isinstance(binary, (bytes, bytearray)):
                file_content = binary
            elif hasattr(binary, "read"):
                file_content = binary.read()
        elif filepath:
            try:
                with open(filepath, "rb") as f:
                    file_content = f.read()
            except Exception as e:
                self.logger.error(f"[Docling] Failed to read file {filepath}: {e}")
                return [], []

        if not file_content:
            self.logger.error("[Docling] No content to parse.")
            return [], []

        # Prepare Request
        url = f"{self.base_url.rstrip('/')}/v1/convert/file"
        files = {"files": (filename, file_content)}
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # Options
        data = {
            "do_ocr": "true",  # docling-serve usually takes string bools or json
            "do_table_structure": "true",
            # "format": "md" # Removed in new API
        }

        try:
            if callback:
                callback(0.2, "[Docling] Sending request to API...")
            response = self.session.post(url, files=files, data=data, headers=headers, timeout=300)
            response.raise_for_status()

            if callback:
                callback(0.6, "[Docling] Processing response...")

            # Parse Response
            # Assume response is JSON with "markdown" field or 'content'
            # Or if text/markdown content type, raw text.
            content_type = response.headers.get("Content-Type", "")

            result_text = ""
            if "application/json" in content_type:
                resp_json = response.json()
                result_text = resp_json.get("markdown") or resp_json.get("content") or ""
            else:
                result_text = response.text

            sections = [result_text] if result_text else []
            tables = []  # Tables are embedded in markdown

            if callback:
                callback(1.0, "[Docling] Done.")
            return sections, tables

        except Exception as e:
            self.logger.error(f"[Docling] API request failed: {e}")
            if callback:
                callback(-1, f"Docling API failed: {e}")
            return [], []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = DoclingParser()
    # Test valid connection if you have a running server, else this might fail
    # sections, tables = parser.parse_pdf("test.pdf", binary=b"%PDF-1.4...")
    pass
