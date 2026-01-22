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

    def _create_retry_session(self, retries=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
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
            # Use the /health endpoint which is designed for health checks
            health_url = f"{self.base_url.rstrip('/')}/health"
            response = self.session.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
            else:
                self.logger.warning(f"[Docling] Service returned status {response.status_code} at {health_url}")
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
        """Parse PDF using Docling async API (submit -> poll -> fetch)."""
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

        # Use fresh session for each request to avoid connection pooling issues
        session = self._create_retry_session()

        try:
            # Step 1: Submit async job
            submit_url = f"{self.base_url.rstrip('/')}/v1/convert/file/async"
            files = {"files": (filename, file_content)}
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            data = {
                "do_ocr": "true",
                "do_table_structure": "true",
            }

            if callback:
                callback(0.15, "[Docling] Submitting job...")

            self.logger.info(f"[Docling] POST to {submit_url}, file size: {len(file_content)} bytes")

            submit_response = session.post(submit_url, files=files, data=data, headers=headers, timeout=60)
            submit_response.raise_for_status()

            submit_data = submit_response.json()
            task_id = submit_data.get("task_id")

            if not task_id:
                raise RuntimeError(f"[Docling] No task_id in response: {submit_data}")

            self.logger.info(f"[Docling] Job submitted, task_id: {task_id}")
            if callback:
                callback(0.2, f"[Docling] Job submitted (task: {task_id[:8]}...)")

            # Step 2: Poll for completion
            poll_url = f"{self.base_url.rstrip('/')}/v1/status/poll/{task_id}"
            max_polls = 360  # 30 minutes max (5s per poll)
            poll_count = 0

            import time

            status = "timeout"
            status_data = {}

            while poll_count < max_polls:
                try:
                    poll_response = session.get(poll_url, timeout=15, headers={"Connection": "close"})
                    poll_response.raise_for_status()

                    status_data = poll_response.json()

                    # Log full response for debugging (first few polls only)
                    if poll_count < 3:
                        self.logger.info(f"[Docling] Poll response: {status_data}")

                    # Try multiple possible status field names
                    status = status_data.get("status") or status_data.get("state") or status_data.get("job_status") or status_data.get("task_status") or "pending"

                    # Normalize status to lowercase
                    if isinstance(status, str):
                        status = status.lower()

                    poll_count += 1
                    # Progress from 0.2 to 0.8 during polling
                    progress = 0.2 + (0.6 * min(poll_count / 60, 1.0))

                    if callback:
                        callback(progress, f"[Docling] Processing... ({status})")

                    self.logger.debug(f"[Docling] Poll {poll_count}: status={status}")

                    if status in ("success", "completed", "done", "finished"):
                        break
                    elif status in ("failure", "error", "failed"):
                        error_msg = status_data.get("error") or status_data.get("message") or "Unknown error"
                        raise RuntimeError(f"[Docling] Job failed: {error_msg}")

                    # Add sleep as fallback in case wait parameter isn't honored
                    time.sleep(2)

                except requests.exceptions.Timeout:
                    # Timeout is expected with long polling, just continue
                    poll_count += 1
                    continue

            # Step 2b: Validate status after polling
            if status not in ("success", "completed", "done", "finished"):
                error_msg = status_data.get("error") or status_data.get("message") or f"Job not completed (status: {status})"
                self.logger.error(f"[Docling] {error_msg}")
                if callback:
                    callback(-1, f"Docling API failed: {error_msg}")
                raise RuntimeError(f"[Docling] Job failed or timed out: {error_msg}")

            # Step 3: Fetch result
            if callback:
                callback(0.85, "[Docling] Fetching result...")

            result_url = f"{self.base_url.rstrip('/')}/v1/result/{task_id}"
            result_response = session.get(result_url, timeout=60)
            result_response.raise_for_status()

            # Parse response
            content_type = result_response.headers.get("Content-Type", "")
            result_text = ""

            if "application/json" in content_type:
                resp_json = result_response.json()
                # Try various fields where content might be
                result_text = resp_json.get("markdown") or resp_json.get("content") or resp_json.get("document", {}).get("md_content", "") or ""
                # Handle nested document structure
                if not result_text and "document" in resp_json:
                    doc = resp_json["document"]
                    if isinstance(doc, dict):
                        result_text = doc.get("export_to_markdown", "") or doc.get("md_content", "")
            else:
                result_text = result_response.text

            sections = [result_text] if result_text else []
            tables = []  # Tables are embedded in markdown

            if callback:
                callback(1.0, "[Docling] Done.")

            self.logger.info(f"[Docling] Successfully parsed, result length: {len(result_text)} chars")
            return sections, tables

        except Exception as e:
            self.logger.error(f"[Docling] API request failed: {e}")
            if callback:
                callback(-1, f"Docling API failed: {e}")
            return [], []
        finally:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = DoclingParser()
    # Test valid connection if you have a running server, else this might fail
    # sections, tables = parser.parse_pdf("test.pdf", binary=b"%PDF-1.4...")
    pass
