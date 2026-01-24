import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Import the module under test before mocking too much
# We need to make sure the parent directories are in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from rag.templates import paper


class TestPaperTemplate(unittest.TestCase):
    def setUp(self):
        self.pdf = paper.Pdf()
        # Mock base class methods
        self.pdf.__images__ = MagicMock()
        self.pdf._layouts_rec = MagicMock()
        self.pdf._table_transformer_job = MagicMock()
        self.pdf._text_merge = MagicMock()
        self.pdf._extract_table_figure = MagicMock(return_value=[])
        self.pdf._concat_downward = MagicMock()
        self.pdf._filter_forpages = MagicMock()
        self.pdf._line_tag = MagicMock(return_value="[tag]")
        self.pdf.total_page = 1
        self.pdf.boxes = []
        self.pdf.page_images = []

    def test_pdf_call_callback_none(self):
        """Verify Pdf.__call__ works when callback is None"""
        # Mock methods that might fail if called with real hardware/models
        self.pdf.boxes = [{"text": "dummy", "x0": 0, "x1": 10, "layoutno": "text"}]
        self.pdf.page_images = [MagicMock()]
        self.pdf.page_images[0].size = (100, 100)
        self.pdf.total_page = 1

        try:
            self.pdf("test.pdf", callback=None)
        except TypeError as e:
            self.fail(f"Pdf.__call__ raised TypeError with callback=None: {e}")

    def test_pdf_call_empty_page_images(self):
        """Verify Pdf.__call__ handles empty page_images gracefully"""
        self.pdf.page_images = []
        self.pdf.boxes = [{"text": "dummy", "x0": 0, "x1": 10, "layoutno": "text"}]
        # This should not raise IndexError at line 71
        self.pdf("test.pdf", callback=None)

    def test_chunk_callback_none(self):
        """Verify chunk() works when callback is None (Line 155 guard)"""
        with patch("rag.templates.paper.normalize_layout_recognizer", return_value=("DeepDOC", "model")):
            with patch("rag.templates.paper.Pdf") as mock_pdf_cls:
                mock_pdf = mock_pdf_cls.return_value
                mock_pdf.return_value = {"sections": [], "tables": []}
                # Mock vision_figure_parser_pdf_wrapper
                with patch("rag.templates.paper.vision_figure_parser_pdf_wrapper", return_value=[]):
                    try:
                        paper.chunk("test.pdf", callback=None)
                    except TypeError as e:
                        self.fail(f"chunk() raised TypeError with callback=None: {e}")

    def test_chunk_logging_defensive(self):
        """Verify chunk() logging handles both tuples and strings in sorted_sections"""
        with patch("rag.templates.paper.bullets_category", return_value=[]):
            with patch("rag.templates.paper.title_frequency", return_value=(0, [0, 0])):
                # Case 1: tuple
                sorted_sections = [("text1", "label1"), ("text2", "label2")]
                with patch("rag.templates.paper.normalize_layout_recognizer", return_value=("DeepDOC", "model")):
                    with patch("rag.templates.paper.Pdf") as mock_pdf_cls:
                        mock_pdf = mock_pdf_cls.return_value
                        mock_pdf.return_value = {"sections": sorted_sections, "tables": [], "authors": "", "title": "", "abstract": ""}
                        with patch("rag.templates.paper.rag_tokenizer") as mock_tok:
                            mock_tok.tokenize.return_value = []
                            mock_tok.fine_grained_tokenize.return_value = []
                            with patch("rag.templates.paper.vision_figure_parser_pdf_wrapper", return_value=[]):
                                paper.chunk("test.pdf")

                # Case 2: string
                sorted_sections = ["text1", "text2"]
                with patch("rag.templates.paper.normalize_layout_recognizer", return_value=("DeepDOC", "model")):
                    with patch("rag.templates.paper.Pdf") as mock_pdf_cls:
                        mock_pdf = mock_pdf_cls.return_value
                        mock_pdf.return_value = {"sections": sorted_sections, "tables": [], "authors": "", "title": "", "abstract": ""}
                        with patch("rag.templates.paper.rag_tokenizer") as mock_tok:
                            mock_tok.tokenize.return_value = []
                            mock_tok.fine_grained_tokenize.return_value = []
                            with patch("rag.templates.paper.vision_figure_parser_pdf_wrapper", return_value=[]):
                                paper.chunk("test.pdf")

    def test_chunk_mismatch_error(self):
        """Verify chunk() raises ValueError when sections and levels mismatch"""
        with patch("rag.templates.paper.bullets_category", return_value=[]):
            with patch("rag.templates.paper.title_frequency", return_value=(0, [0])):
                sorted_sections = ["text1", "text2"]
                with patch("rag.templates.paper.normalize_layout_recognizer", return_value=("DeepDOC", "model")):
                    with patch("rag.templates.paper.Pdf") as mock_pdf_cls:
                        mock_pdf = mock_pdf_cls.return_value
                        mock_pdf.return_value = {"sections": sorted_sections, "tables": [], "authors": "", "title": "", "abstract": ""}
                        with patch("rag.templates.paper.rag_tokenizer") as mock_tok:
                            mock_tok.tokenize.return_value = []
                            mock_tok.fine_grained_tokenize.return_value = []
                            with patch("rag.templates.paper.vision_figure_parser_pdf_wrapper", return_value=[]):
                                with self.assertRaisesRegex(ValueError, "Mismatch between number of sections"):
                                    paper.chunk("test.pdf")


if __name__ == "__main__":
    unittest.main()
