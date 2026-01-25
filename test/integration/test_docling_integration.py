import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure ragflow is in python path (go up two levels from test/integration/ to project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Only import DoclingParser
from deepdoc.parser.docling_parser import DoclingParser


class TestDoclingIntegration(unittest.TestCase):
    @patch("requests.Session")
    def test_docling_parser_api_success(self, mock_session_cls):
        # Setup mock session
        mock_session = mock_session_cls.return_value

        # 1. Mock Check Installation (Health Check)
        # 2. Mock Submit Response
        # 3. Mock Poll Response (Success)
        # 4. Mock Result Response

        # We need to handle different calls based on URL, or just use side_effect
        mock_submit = MagicMock()
        mock_submit.status_code = 200
        mock_submit.json.return_value = {"task_id": "test_task_123"}

        mock_poll = MagicMock()
        mock_poll.status_code = 200
        mock_poll.json.return_value = {"status": "success"}

        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.headers = {"Content-Type": "application/json"}
        mock_result.json.return_value = {"markdown": "# Test Docling\n\nResult text."}

        # Health check mock
        mock_health = MagicMock()
        mock_health.status_code = 200

        def side_effect_get(url, **kwargs):
            if "health" in url:
                return mock_health
            if "poll" in url:
                return mock_poll
            if "result" in url:
                return mock_result
            return MagicMock(status_code=404)

        mock_session.post.return_value = mock_submit
        mock_session.get.side_effect = side_effect_get

        # Setup Parser
        with patch.dict(os.environ, {"DOCLING_BASE_URL": "http://mock-docling"}):
            parser = DoclingParser()
            # Test parse_pdf
            sections, tables = parser.parse_pdf("test.pdf", binary=b"dummy content")

        # Verify calls
        mock_session.post.assert_called()
        args, kwargs = mock_session.post.call_args
        self.assertIn("v1/convert/file/async", args[0])

        # Verify result
        self.assertEqual(len(sections), 1)
        self.assertIn("# Test Docling", sections[0])
        self.assertEqual(tables, [])

    @patch("requests.Session")
    def test_docling_parser_api_failure(self, mock_session_cls):
        # Setup mock session
        mock_session = mock_session_cls.return_value

        mock_submit = MagicMock()
        mock_submit.status_code = 200
        mock_submit.json.return_value = {"task_id": "test_task_fail"}

        # Mock FAILURE status
        mock_poll = MagicMock()
        mock_poll.status_code = 200
        mock_poll.json.return_value = {"status": "failed", "error": "Processing failed"}

        def side_effect_get(url, **kwargs):
            if "health" in url:
                return MagicMock(status_code=200)
            if "poll" in url:
                return mock_poll
            return MagicMock(status_code=404)

        mock_session.post.return_value = mock_submit
        mock_session.get.side_effect = side_effect_get

        mock_session.get.side_effect = side_effect_get

        # Setup Parser
        with patch.dict(os.environ, {"DOCLING_BASE_URL": "http://mock-docling"}):
            parser = DoclingParser()
            # Test parse_pdf should catch the error and return empty lists
            mock_callback = MagicMock()
            sections, tables = parser.parse_pdf("test.pdf", binary=b"dummy content", callback=mock_callback)

        self.assertEqual(sections, [])
        self.assertEqual(tables, [])

        # Verify callback was called with error
        # The error message should match what was logged and passed to callback
        # "Docling API failed: Job failed or timed out: Processing failed"
        args, _ = mock_callback.call_args
        self.assertEqual(args[0], -1)
        self.assertIn("Processing failed", args[1])

    @patch("requests.Session")
    def test_docling_parser_semantic_mode(self, mock_session_cls):
        """Test that semantic mode returns structured string instead of splitlines()."""
        mock_session = mock_session_cls.return_value

        mock_submit = MagicMock()
        mock_submit.status_code = 200
        mock_submit.json.return_value = {"task_id": "test_task_semantic"}

        mock_poll = MagicMock()
        mock_poll.status_code = 200
        mock_poll.json.return_value = {"status": "success"}

        mock_result = MagicMock()
        mock_result.status_code = 200
        mock_result.headers = {"Content-Type": "application/json"}
        mock_result.json.return_value = {"markdown": "# Heading\n\nParagraph one.\n\n## Subheading\n\nParagraph two."}

        mock_health = MagicMock()
        mock_health.status_code = 200

        def side_effect_get(url, **kwargs):
            if "health" in url:
                return mock_health
            if "poll" in url:
                return mock_poll
            if "result" in url:
                return mock_result
            return MagicMock(status_code=404)

        mock_session.post.return_value = mock_submit
        mock_session.get.side_effect = side_effect_get

        with patch.dict(os.environ, {"DOCLING_BASE_URL": "http://mock-docling"}):
            parser = DoclingParser()
            # Test SEMANTIC mode - should return string
            sections_semantic, _ = parser.parse_pdf("test.pdf", binary=b"dummy", use_semantic_chunking=True)
            self.assertIsInstance(sections_semantic, str)
            self.assertIn("# Heading", sections_semantic)
            self.assertIn("## Subheading", sections_semantic)

            # Reset mock to avoid call count confusion
            mock_session.reset_mock()
            mock_submit.reset_mock()
            mock_poll.reset_mock()
            mock_result.reset_mock()

            # Test LEGACY mode (default) - should return list
            sections_legacy, _ = parser.parse_pdf("test.pdf", binary=b"dummy", use_semantic_chunking=False)
            self.assertIsInstance(sections_legacy, list)
            # Relaxed assertion: check if "# Heading" is in any of the sections
            self.assertTrue(any("# Heading" in s for s in sections_legacy), "Heading should be present in legacy sections")


if __name__ == "__main__":
    unittest.main()
