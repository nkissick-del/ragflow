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
import re
import time

from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DoclingParser:
    """
    Docling parser that communicates with a remote Docling API server.

    This parser creates fresh HTTP sessions for each request to avoid connection
    pooling issues and ensure clean state for long-running operations.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = os.environ.get("DOCLING_BASE_URL", "http://localhost:5001")
        self.auth_token = os.environ.get("DOCLING_AUTH_TOKEN")

    def _create_retry_session(self, retries=5, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "HEAD", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def check_installation(self) -> bool:
        """
        Checks if the Docling server is reachable.

        Creates a fresh session for this health check to avoid stale connections.

        Returns:
            True if the server is reachable and healthy, False otherwise.
        """
        if not self.base_url:
            self.logger.warning("[Docling] DOCLING_BASE_URL not set.")
            return False

        # Create fresh session for health check
        session = self._create_retry_session()
        try:
            # Use the /health endpoint which is designed for health checks
            health_url = f"{self.base_url.rstrip('/')}/health"
            response = session.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
            else:
                self.logger.warning(f"[Docling] Service returned status {response.status_code} at {health_url}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"[Docling] Service unreachable at {self.base_url}: {e}")
            return False
        finally:
            session.close()

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        delete_output: bool = True,
        parse_method: str = "raw",
        **kwargs,
    ):
        """
        Parse PDF using Docling async API (submit -> poll -> fetch).

        Creates a fresh HTTP session for this request to avoid connection pooling
        issues during long-running operations. The session is closed in the finally
        block to ensure proper resource cleanup.

        Args:
            filepath: Path to the PDF file
            binary: Binary content (alternative to filepath)
            callback: Progress callback function
            output_dir: Reserved for interface compatibility with other parsers (unused)
            delete_output: Reserved for interface compatibility with other parsers (unused)
            parse_method: Reserved for interface compatibility with other parsers (unused)
            **kwargs: Additional arguments including use_semantic_chunking flag

        Returns:
            Tuple of (sections, tables) where sections is either list[str] or str
            depending on use_semantic_chunking flag.

        Note:
            The parameters output_dir, delete_output, and parse_method are accepted
            for API compatibility with MinerUParser and other PDF parsers but are not
            used by the Docling implementation, which handles all processing remotely
            via the Docling API server.
        """
        if callback:
            callback(0.1, "[Docling] Starting API conversion...")

        # Prepare input
        filename = Path(filepath).name if filepath else "document.pdf"
        file_content = None
        if binary:
            if isinstance(binary, (bytes, bytearray)):
                file_content = binary
            elif hasattr(binary, "read"):
                if hasattr(binary, "seek"):
                    binary.seek(0)
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

            # Step 2: Poll for completion using wall-clock timeout
            poll_url = f"{self.base_url.rstrip('/')}/v1/status/poll/{task_id}"
            total_timeout = 30 * 60  # 30 minutes max wall-clock time
            start_time = time.monotonic()
            deadline = start_time + total_timeout
            poll_interval = 5.0  # Expected time between polls in seconds
            # Per-request timeout should be >= poll_interval to allow server-side long-polling
            # but with buffer for network latency
            per_request_timeout = 15.0

            status = "timeout"
            status_data = {}

            while time.monotonic() < deadline:
                # Measure time spent in HTTP request
                poll_start = time.monotonic()
                elapsed_total = time.monotonic() - start_time

                try:
                    poll_response = session.get(poll_url, timeout=per_request_timeout, headers={"Connection": "close"})
                    if poll_response.status_code == 404:
                        self.logger.warning(f"[Docling] Polling returned 404 for task {task_id}. Attempting to fetch result directly.")
                        # A 404 on the polling endpoint likely means the task has completed and been moved to the result endpoint.
                        # We set status to "success" to break the loop and trigger the direct fetch attempt below.
                        status = "success"
                        break

                    poll_response.raise_for_status()

                    status_data = poll_response.json()

                    # Log full response for debugging (first few seconds only)
                    if elapsed_total < 15:
                        self.logger.info(f"[Docling] Poll response: {status_data}")

                    # Try multiple possible status field names
                    status = status_data.get("status") or status_data.get("state") or status_data.get("job_status") or status_data.get("task_status") or "pending"

                    # Normalize status to lowercase
                    if isinstance(status, str):
                        status = status.lower()

                    # Progress from 0.2 to 0.8 during polling (based on wall-clock time)
                    progress = 0.2 + (0.6 * min(elapsed_total / total_timeout, 1.0))

                    if callback:
                        callback(progress, f"[Docling] Processing... ({status})")

                    self.logger.debug(f"[Docling] Poll at {elapsed_total:.1f}s: status={status}")

                    if status in ("success", "completed", "done", "finished"):
                        break
                    elif status in ("failure", "error", "failed"):
                        error_msg = status_data.get("error") or status_data.get("message") or "Unknown error"
                        raise RuntimeError(f"[Docling] Job failed: {error_msg}")

                    # Calculate elapsed time for this poll and sleep only for the remainder
                    # This avoids adding delay when the server already honored long-polling
                    poll_elapsed = time.monotonic() - poll_start
                    sleep_time = max(0, poll_interval - poll_elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except requests.exceptions.Timeout:
                    # Timeout is expected with long polling, just continue
                    # Check if we've exceeded the overall deadline
                    if time.monotonic() >= deadline:
                        break
                    continue

            # Step 2b: Validate status after polling
            if status not in ("success", "completed", "done", "finished"):
                error_msg = status_data.get("error") or status_data.get("message") or f"Job not completed (status: {status})"
                self.logger.error(f"[Docling] {error_msg}")
                if callback:
                    callback(-1, f"Docling API failed: {error_msg}")
                raise RuntimeError(f"[Docling] Job failed or timed out: {error_msg}")

            self.logger.info(f"[Docling] Job matched success status. Final poll data: {status_data}")

            # Step 3: Fetch result
            if callback:
                callback(0.85, "[Docling] Fetching result...")

            # Use result_url provided by server if available, otherwise construct it
            result_url = status_data.get("result_url")
            if not result_url:
                result_url = f"{self.base_url.rstrip('/')}/v1/result/{task_id}"

            # Retry fetching result a few times to handle potential race conditions
            result_response = None
            fetch_errors = []

            for i in range(3):
                try:
                    self.logger.info(f"[Docling] Fetching result from {result_url} (Attempt {i + 1}/3)")
                    result_response = session.get(result_url, timeout=60)
                    if result_response.status_code == 200:
                        break
                    elif result_response.status_code == 404:
                        self.logger.warning(f"[Docling] Result not found (404) on attempt {i + 1}. Waiting...")
                        time.sleep(2)
                    else:
                        result_response.raise_for_status()
                except Exception as e:
                    fetch_errors.append(str(e))
                    self.logger.warning(f"[Docling] Result fetch failed attempt {i + 1}: {e}")
                    time.sleep(2)

            if not result_response or result_response.status_code != 200:
                raise RuntimeError(f"[Docling] Failed to fetch result after success status. URL: {result_url}. Status: {result_response.status_code if result_response else 'None'}")

            # Parse response
            content_type = result_response.headers.get("Content-Type", "")
            result_text = None

            if "application/json" in content_type:
                resp_json = result_response.json()

                # Try various fields where content might be, checking for None explicitly
                # to preserve empty strings as valid values
                result_text = resp_json.get("markdown")
                if result_text is None:
                    result_text = resp_json.get("content")
                if result_text is None:
                    result_text = resp_json.get("text")

                # Handle nested document structure
                if result_text is None and "document" in resp_json:
                    doc = resp_json.get("document")
                    if isinstance(doc, dict):
                        # Check standard Docling API fields
                        result_text = doc.get("markdown")
                        if result_text is None:
                            result_text = doc.get("md_content")
                        if result_text is None:
                            result_text = doc.get("content")

                # If still None, default to empty string
                if result_text is None:
                    result_text = ""
            else:
                result_text = result_response.text

            # Clean Base64 images from Markdown to prevent "garbage" chunks
            # Pattern matches: ![Alt Text](data:image/...)
            if result_text:
                result_text = re.sub(r"!\[.*?\]\(data:image\/[^)]*;base64,[^)]*\)", "", result_text)

                # Feature flag for semantic chunking (new structured output)
                # When enabled, return the full Markdown string for semantic template processing
                # When disabled (default), maintain backward compatibility with splitlines()
                # Return structured Markdown (or plain text) directly
                # The orchestration layer handles chunking/splitting as needed
                sections = result_text if result_text else ""
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
    logging.basicConfig(level=logging.INFO)
    parser = DoclingParser()
    try:
        if parser.check_installation():
            print("Docling server is reachable.")
        else:
            print("Docling server is NOT reachable.")
    except Exception as e:
        print(f"Error checking verification: {e}")
