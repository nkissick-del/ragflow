import unittest
from unittest.mock import patch, MagicMock
import sys


class TestSingleChunk(unittest.TestCase):
    def setUp(self):
        # Setup mocks
        self.mock_modules = {
            "rag.parsers": MagicMock(),
            "rag.parsers.deepdoc_client": MagicMock(),
            "rag.parsers.PdfParser": MagicMock(),
            "rag.parsers.ExcelParser": MagicMock(),
            "rag.parsers.HtmlParser": MagicMock(),
            "deepdoc": MagicMock(),
            "deepdoc.parser": MagicMock(),
            "common": MagicMock(),
            "common.token_utils": MagicMock(),
            "bs4": MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        from rag.templates import single_chunk

        self.single_chunk = single_chunk

    def tearDown(self):
        self.patcher.stop()

    def test_chunk_txt(self):
        # Test simple text chunking fallback
        filename = "test.txt"
        binary = b"Line 1\nLine 2\nLine 3"

        # Mock get_text to return string directly
        with patch.object(self.single_chunk, "get_text") as mock_get_text:
            mock_get_text.return_value = "Line 1\nLine 2\nLine 3"

            res = self.single_chunk.chunk(filename, binary, callback=lambda p, m: None)

            # Should return 1 document
            self.assertEqual(len(res), 1)
            # The content tokenization is complex to check without exact tokenizer,
            # but we verify structure.
            self.assertIn("docnm_kwd", res[0])
            self.assertEqual(res[0]["docnm_kwd"], filename)

    def test_chunk_docx(self):
        # Verify logic branches to Docx parser
        mock_docx_instance = MagicMock()

        with patch.object(self.single_chunk, "Docx") as mock_docx, patch.object(self.single_chunk, "vision_figure_parser_docx_wrapper_naive") as _:
            mock_docx.return_value = mock_docx_instance

            res = self.single_chunk.chunk("test.docx", b"binary", callback=lambda p, m: None)

            mock_docx.assert_called_once()
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["content_with_weight"], "")


if __name__ == "__main__":
    unittest.main()
