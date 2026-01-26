import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Global Mocking of selenium and seleniumwire to handle missing dependencies in test environment
if "selenium" not in sys.modules:
    selenium_mock = MagicMock()
    selenium_mock.common.exceptions.TimeoutException = TimeoutError
    sys.modules["selenium"] = selenium_mock
    sys.modules["selenium.common"] = selenium_mock.common
    sys.modules["selenium.common.exceptions"] = selenium_mock.common.exceptions

if "seleniumwire" not in sys.modules:
    sys.modules["seleniumwire"] = MagicMock()
    sys.modules["seleniumwire.webdriver"] = MagicMock()

if "deepdoc" not in sys.modules:
    sys.modules["deepdoc"] = MagicMock()
    sys.modules["deepdoc.parser"] = MagicMock()
    sys.modules["deepdoc.parser.html_parser"] = MagicMock()

if "api" not in sys.modules:
    sys.modules["api"] = MagicMock()
    sys.modules["api.db"] = MagicMock()
    sys.modules["api.db.services"] = MagicMock()
    sys.modules["api.db.services.file_service"] = MagicMock()

# Now we can safely import the module under test (it will use the mocks)
# However, we need to ensure patches apply to the mocked modules or import happens here
# It is better to import inside tests or after mocks if we rely on global mocks
# But we already mocked sys.modules so standard import should work.

from rag.utils.selenium_crawler import SeleniumCrawler


class TestSeleniumCrawler(unittest.TestCase):
    @patch("seleniumwire.webdriver.Chrome")
    @patch("seleniumwire.webdriver.ChromeOptions")
    def test_parse_url_html(self, MockOptions, MockChrome):
        # Setup mock driver and response
        mock_driver = MockChrome.return_value
        mock_request = MagicMock()
        mock_request.response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_driver.requests = [mock_request]
        mock_driver.page_source = "<html><body><p>Hello World</p></body></html>"

        # Mock HtmlParser
        with patch("rag.utils.selenium_crawler.RAGFlowHtmlParser") as MockParser:
            MockParser.return_value.parser_txt.return_value = ["Hello World"]

            result = SeleniumCrawler.parse_url("http://example.com", "/tmp/downloads", "user1")

            self.assertEqual(result, "Hello World")
            mock_driver.quit.assert_called_once()
            # Verify timeout set
            mock_driver.set_page_load_timeout.assert_called_with(120)

    @patch("seleniumwire.webdriver.Chrome")
    @patch("seleniumwire.webdriver.ChromeOptions")
    @patch("rag.utils.selenium_crawler.FileService")
    @patch("rag.utils.selenium_crawler._LocalFile")
    def test_parse_url_file_content_disposition(self, MockLocalFile, MockFileService, MockOptions, MockChrome):
        # Setup mock driver and response
        mock_driver = MockChrome.return_value
        mock_request = MagicMock()
        mock_request.response.headers = {"Content-Type": "application/pdf", "Content-Disposition": 'attachment; filename="test.pdf"'}
        mock_driver.requests = [mock_request]

        SeleniumCrawler.parse_url("http://example.com/file.pdf", "/tmp/downloads", "user1")

        # Verify LocalFile created with correct filename
        MockLocalFile.assert_called_with("test.pdf", os.path.join("/tmp/downloads", "test.pdf"))
        # MockFileService.parse_docs.assert_called_once()
        mock_driver.quit.assert_called_once()

    @patch("seleniumwire.webdriver.Chrome")
    @patch("seleniumwire.webdriver.ChromeOptions")
    @patch("rag.utils.selenium_crawler.FileService")
    @patch("rag.utils.selenium_crawler._LocalFile")
    @patch("rag.utils.selenium_crawler.SeleniumCrawler.wait_for_download")
    def test_parse_url_file_wait_download(self, MockWait, MockLocalFile, MockFileService, MockOptions, MockChrome):
        # Setup mock driver and response
        mock_driver = MockChrome.return_value
        mock_request = MagicMock()
        # No Content-Disposition filename
        mock_request.response.headers = {"Content-Type": "application/pdf"}
        mock_driver.requests = [mock_request]

        MockWait.return_value = "downloaded_file.pdf"

        SeleniumCrawler.parse_url("http://example.com/file.pdf", "/tmp/downloads", "user1")

        MockWait.assert_called_with("/tmp/downloads")
        MockLocalFile.assert_called_with("downloaded_file.pdf", os.path.join("/tmp/downloads", "downloaded_file.pdf"))
        mock_driver.quit.assert_called_once()

    @patch("os.listdir")
    @patch("os.path.isfile")
    @patch("os.path.getsize")
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_download(self, mock_time, mock_sleep, mock_getsize, mock_isfile, mock_listdir):
        # Sequence of time.time() calls: start, check loop 1, check loop 2...
        # We need to simulate the file appearing and becoming stable.

        # Iteration 1: Only temp file
        # Iteration 2: Real file appears, initial size
        # Iteration 3: Real file stable size

        mock_time.side_effect = [0, 1, 2, 3, 4, 100]  # simulating time passing

        mock_listdir.side_effect = [["file.crdownload"], ["target.pdf"], ["target.pdf"]]

        mock_isfile.return_value = True
        mock_getsize.side_effect = [100, 100]  # Size stable

        result = SeleniumCrawler.wait_for_download("/downloads", timeout=10)
        self.assertEqual(result, "target.pdf")


if __name__ == "__main__":
    unittest.main()
