import re
import os
import time
from selenium.common.exceptions import TimeoutException
from deepdoc.parser.html_parser import RAGFlowHtmlParser
from api.db.services.file_service import FileService
import logging

logger = logging.getLogger(__name__)


class _LocalFile:
    filename: str
    filepath: str

    def __init__(self, filename, filepath):
        self.filename = filename
        self.filepath = filepath

    def read(self):
        with open(self.filepath, "rb") as f:
            return f.read()


class SeleniumCrawler:
    @staticmethod
    def wait_for_download(download_path, timeout=120, expected_filename=None):
        start_time = time.time()
        initial_files = set(os.listdir(download_path)) if expected_filename is None else set()

        while time.time() - start_time < timeout:
            if expected_filename:
                file_path = os.path.join(download_path, expected_filename)
                if os.path.isfile(file_path):
                    try:
                        initial_size = os.path.getsize(file_path)
                        time.sleep(1)
                        if os.path.getsize(file_path) == initial_size:
                            return expected_filename
                    except (FileNotFoundError, OSError):
                        continue
            else:
                files = os.listdir(download_path)
                for fname in files:
                    if fname in initial_files or fname.endswith(".crdownload"):
                        continue
                    file_path = os.path.join(download_path, fname)
                    if os.path.isfile(file_path):
                        try:
                            # Check if file size is stable
                            initial_size = os.path.getsize(file_path)
                            time.sleep(1)
                            if os.path.getsize(file_path) == initial_size:
                                return fname
                        except (FileNotFoundError, OSError):
                            continue
            time.sleep(1)
        raise TimeoutError("Download timed out")

    @staticmethod
    def parse_url(url, download_path, user_id):
        from seleniumwire.webdriver import Chrome, ChromeOptions

        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option("prefs", {"download.default_directory": download_path, "download.prompt_for_download": False, "download.directory_upgrade": True, "safebrowsing.enabled": True})

        driver = Chrome(options=options)
        driver.set_page_load_timeout(120)  # Set page load timeout

        try:
            try:
                driver.get(url)
            except TimeoutException as e:
                logger.warning(f"Timeout loading {url}: {e}")
                pass

            res_headers = [r.response.headers for r in driver.requests if r and r.response]
            if not res_headers:
                raise ValueError("No response headers found")

            last_headers = res_headers[-1]
            content_type = last_headers.get("Content-Type", "")

            if "text/html" in content_type or "application/xhtml+xml" in content_type:
                sections = RAGFlowHtmlParser().parser_txt(driver.page_source)
                return "\n".join(sections)

            # Try to get filename from Content-Disposition
            filename = None
            content_disposition = last_headers.get("Content-Disposition")
            if content_disposition:
                r = re.search(r"filename=\"?([^\";]+)\"?", content_disposition)
                if r and r.group(1):
                    filename = r.group(1)
                    # Sanitize filename
                    filename = os.path.basename(filename)
                    filename = filename.lstrip(".").replace("\x00", "")
                    if not filename:
                        filename = f"downloaded_{int(time.time())}"

            if filename:
                # Fallback to waiting for a file in the download directory
                try:
                    filename = SeleniumCrawler.wait_for_download(download_path, expected_filename=filename)
                except TimeoutError:
                    raise ValueError("Cannot identify downloaded file")

            if not filename:
                # Fallback to waiting for a file in the download directory
                try:
                    filename = SeleniumCrawler.wait_for_download(download_path)
                except TimeoutError:
                    raise ValueError("Cannot identify downloaded file")

            f = _LocalFile(filename, os.path.join(download_path, filename))
            return FileService.parse_docs([f], user_id)
        finally:
            driver.quit()
