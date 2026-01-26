import re
import os
from deepdoc.parser.html_parser import RAGFlowHtmlParser
from api.db.services.file_service import FileService


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
    def parse_url(url, download_path, user_id):
        from seleniumwire.webdriver import Chrome, ChromeOptions

        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option("prefs", {"download.default_directory": download_path, "download.prompt_for_download": False, "download.directory_upgrade": True, "safebrowsing.enabled": True})
        driver = Chrome(options=options)
        try:
            driver.get(url)
            res_headers = [r.response.headers for r in driver.requests if r and r.response]
            if len(res_headers) > 1:
                sections = RAGFlowHtmlParser().parser_txt(driver.page_source)
                return "\n".join(sections)

            r = re.search(r"filename=\"([^\"]+)\"", str(res_headers))
            if not r or not r.group(1):
                raise ValueError("Can't not identify downloaded file")

            filename = r.group(1)
            f = _LocalFile(filename, os.path.join(download_path, filename))
            return FileService.parse_docs([f], user_id)
        finally:
            driver.quit()
