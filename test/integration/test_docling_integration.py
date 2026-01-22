import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure ragflow is in python path (go up two levels from test/integration/ to project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Only import DoclingParser
from deepdoc.parser.docling_parser import DoclingParser


class TestDoclingIntegration(unittest.TestCase):
    @patch("requests.post")
    def test_docling_parser_api(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"markdown": "# Test Docling\n\nResult text."}
        mock_post.return_value = mock_response

        sys.modules["rag.app.orchestrator"] = MagicMock()
        # sys.modules["rag.app.naive_parsers"] = MagicMock() # Don't mock the module we are testing if possible, or mock selective parts

        # Now we can import the module under test
        # We want to test DoclingHTTPParser, which is in docling_parser.py
        # But the test originally imported 'naive' to test routing.
        # We should import orchestrator (formerly naive) if we are testing routing.

        # Setup Parser
        os.environ["DOCLING_BASE_URL"] = "http://mock-docling"
        parser = DoclingParser()
        self.assertTrue(parser.check_installation())

        # Test parse_pdf
        sections, tables = parser.parse_pdf("test.pdf", binary=b"dummy content")

        # Verify call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["url"], "http://mock-docling/v1/convert/file")
        self.assertIn("files", kwargs["files"])

        # Verify result
        self.assertEqual(len(sections), 1)
        self.assertIn("# Test Docling", sections[0])


if __name__ == "__main__":
    unittest.main()
