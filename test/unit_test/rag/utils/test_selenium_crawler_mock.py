import unittest
import sys
import os

import importlib
from unittest.mock import MagicMock, patch


class TestSeleniumCrawler(unittest.TestCase):
    def setUp(self):
        # Create mocks for dependencies
        self.mock_selenium = MagicMock()
        self.mock_selenium.common.exceptions.TimeoutException = TimeoutError

        self.mock_seleniumwire = MagicMock()
        self.mock_deepdoc = MagicMock()
        self.mock_api = MagicMock()

        # Setup the patcher for sys.modules
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "selenium": self.mock_selenium,
                "selenium.common": self.mock_selenium.common,
                "selenium.common.exceptions": self.mock_selenium.common.exceptions,
                "seleniumwire": self.mock_seleniumwire,
                "seleniumwire.webdriver": self.mock_seleniumwire.webdriver,
                "deepdoc": self.mock_deepdoc,
                "deepdoc.parser": self.mock_deepdoc.parser,
                "deepdoc.parser.html_parser": self.mock_deepdoc.parser.html_parser,
                "api": self.mock_api,
                "api.db": self.mock_api.db,
                "api.db.services": self.mock_api.db.services,
                "api.db.services.file_service": self.mock_api.db.services.file_service,
            },
        )
        self.modules_patcher.start()

        # Import (and reload) the module under test to apply mocks
        import rag.utils.selenium_crawler

        self.crawler_module = importlib.reload(rag.utils.selenium_crawler)
        self.SeleniumCrawler = self.crawler_module.SeleniumCrawler

        # Shortcuts for verification
        self.MockChrome = self.mock_seleniumwire.webdriver.Chrome
        self.MockOptions = self.mock_seleniumwire.webdriver.ChromeOptions
        self.MockFileService = self.mock_api.db.services.file_service.FileService
        self.MockHtmlParser = self.mock_deepdoc.parser.html_parser.RAGFlowHtmlParser

    def tearDown(self):
        self.modules_patcher.stop()

    def test_parse_url_html(self):
        # Setup mock driver and response
        mock_driver = self.MockChrome.return_value
        mock_request = MagicMock()
        mock_request.response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_driver.requests = [mock_request]
        mock_driver.page_source = "<html><body><p>Hello World</p></body></html>"

        # Mock HtmlParser result
        self.MockHtmlParser.return_value.parser_txt.return_value = ["Hello World"]

        result = self.SeleniumCrawler.parse_url("http://example.com", "/tmp/downloads", "user1")

        self.assertEqual(result, "Hello World")
        mock_driver.quit.assert_called_once()
        mock_driver.set_page_load_timeout.assert_called_with(120)

    def test_parse_url_file_content_disposition(self):
        # Setup mock driver and response
        mock_driver = self.MockChrome.return_value
        mock_request = MagicMock()
        mock_request.response.headers = {"Content-Type": "application/pdf", "Content-Disposition": 'attachment; filename="test.pdf"'}
        mock_driver.requests = [mock_request]

        # Determine where _LocalFile is in the reloaded module
        # Since we use patched sys.modules, imports inside selenium_crawler are using mocks.
        # But _LocalFile is defined in selenium_crawler.py, so we should patch it on the module instance.

        with (
            patch.object(self.crawler_module, "_LocalFile") as MockLocalFile,
            patch.object(self.crawler_module, "FileService") as MockFileService,
            patch.object(self.SeleniumCrawler, "wait_for_download") as MockWait,
        ):
            MockWait.return_value = "test.pdf"
            MockFileService.parse_docs.return_value = ["parsed_doc"]

            result = self.SeleniumCrawler.parse_url("http://example.com/file.pdf", "/tmp/downloads", "user1")

            # Verify wait_for_download called to verify existence
            MockWait.assert_called_with("/tmp/downloads", expected_filename="test.pdf")

            # Verify LocalFile created
            MockLocalFile.assert_called_with("test.pdf", os.path.join("/tmp/downloads", "test.pdf"))

            # Verify validation
            self.assertEqual(result, ["parsed_doc"])
            MockFileService.parse_docs.assert_called_once()

            mock_driver.quit.assert_called_once()

    def test_parse_url_file_wait_download(self):
        # Setup mock driver
        mock_driver = self.MockChrome.return_value
        mock_request = MagicMock()
        mock_request.response.headers = {"Content-Type": "application/pdf"}
        mock_driver.requests = [mock_request]

        with (
            patch.object(self.crawler_module, "_LocalFile") as MockLocalFile,
            patch.object(self.SeleniumCrawler, "wait_for_download") as MockWait,
            patch.object(self.crawler_module, "FileService") as MockFileService,
        ):
            MockWait.return_value = "downloaded_file.pdf"
            MockFileService.parse_docs.return_value = ["parsed_doc"]

            result = self.SeleniumCrawler.parse_url("http://example.com/file.pdf", "/tmp/downloads", "user1")

            MockWait.assert_called_with("/tmp/downloads")
            MockLocalFile.assert_called_with("downloaded_file.pdf", os.path.join("/tmp/downloads", "downloaded_file.pdf"))
            self.assertEqual(result, ["parsed_doc"])
            mock_driver.quit.assert_called_once()

    @patch("os.listdir")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_download(self, mock_time, mock_sleep, mock_getsize, mock_isfile, mock_listdir):
        # Use a callable side_effect for time.time to be deterministic and robust
        # Start at 0, increment by 1 on each call
        self.time_counter = 0

        def time_side_effect():
            self.time_counter += 1
            return self.time_counter

        mock_time.side_effect = time_side_effect

        # Scenario steps:
        # 1. Start (time=1)
        # 2. Check initial files.
        # 3. Loop: time check (time=2 < start+10)
        # 4. listdir -> finds nothing or crdownload
        # 5. sleep
        # 6. Loop: time check (time=3)
        # 7. listdir -> finds target
        # 8. isfile -> True
        # 9. getsize -> 100
        # 10. sleep
        # 11. getsize -> 100 (stable)
        # 12. return

        mock_listdir.side_effect = [
            ["file.crdownload"],  # Initial check
            ["file.crdownload"],  # Loop 1
            ["file.crdownload", "target.pdf"],  # Loop 2
            ["file.crdownload", "target.pdf"],  # Loop 3 (if needed)
        ]

        mock_isfile.return_value = True
        mock_getsize.side_effect = lambda x: 100  # Stable size regardless of calls

        # We need to ensure we don't timeout. timeout=10.
        # time values will be 1, 2, ...

        result = self.SeleniumCrawler.wait_for_download("/downloads", timeout=10)
        self.assertEqual(result, "target.pdf")


if __name__ == "__main__":
    unittest.main()
