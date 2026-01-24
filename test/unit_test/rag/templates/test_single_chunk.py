import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from rag.templates import single_chunk


class TestSingleChunk(unittest.TestCase):
    def test_chunk_txt(self):
        # Test simple text chunking fallback
        filename = "test.txt"
        binary = b"Line 1\nLine 2\nLine 3"

        # Mock get_text to return string directly
        with patch("rag.templates.single_chunk.get_text") as mock_get_text:
            mock_get_text.return_value = "Line 1\nLine 2\nLine 3"

            res = single_chunk.chunk(filename, binary, callback=lambda p, m: None)  # Add callback

            # Should return 1 document
            self.assertEqual(len(res), 1)
            # The content tokenization is complex to check without exact tokenizer,
            # but we verify structure.
            self.assertIn("docnm_kwd", res[0])
            self.assertEqual(res[0]["docnm_kwd"], filename)

    @patch("rag.templates.single_chunk.Docx")
    @patch("rag.templates.single_chunk.vision_figure_parser_docx_wrapper_naive")
    def test_chunk_docx(self, mock_figure_parser, mock_docx):
        # Verify logic branches to Docx parser
        mock_docx_instance = MagicMock()
        mock_docx_instance.return_value = [("Section 1", None, None)]
        mock_docx.return_value = mock_docx_instance

        res = single_chunk.chunk("test.docx", b"binary", callback=lambda p, m: None)

        mock_docx.assert_called_once()
        self.assertEqual(len(res), 1)


if __name__ == "__main__":
    unittest.main()
