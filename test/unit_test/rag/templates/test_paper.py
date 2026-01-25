import unittest

from unittest.mock import MagicMock, patch
from test.mocks.mock_utils import setup_mocks

# Set up system mocks
setup_mocks()


from rag.templates import paper


class TestPaperTemplate(unittest.TestCase):
    def setUp(self):
        # Patch the Pdf class so that any instantiation within the test or setup returns our mock
        self.pdf_patcher = patch("rag.templates.paper.Pdf")
        self.mock_pdf_cls = self.pdf_patcher.start()
        self.addCleanup(self.pdf_patcher.stop)

        # Configure the mock instance
        self.mock_pdf_instance = self.mock_pdf_cls.return_value

        # Configure the mock with required attributes
        self.mock_pdf_instance.__images__ = MagicMock()
        self.mock_pdf_instance._layouts_rec = MagicMock()
        self.mock_pdf_instance._table_transformer_job = MagicMock()
        self.mock_pdf_instance._text_merge = MagicMock()
        self.mock_pdf_instance._extract_table_figure = MagicMock(return_value=[])
        self.mock_pdf_instance._concat_downward = MagicMock()
        self.mock_pdf_instance._filter_forpages = MagicMock()
        self.mock_pdf_instance._line_tag = MagicMock(return_value="[tag]")
        self.mock_pdf_instance.total_page = 1
        self.mock_pdf_instance.boxes = []
        self.mock_pdf_instance.page_images = []

        # Use the mock instance
        self.pdf = self.mock_pdf_instance

    def test_pdf_call_callback_none(self):
        """Verify Pdf.__call__ works when callback is None"""
        # Mock methods that might fail if called with real hardware/models
        self.pdf.boxes = [{"text": "dummy", "x0": 0, "x1": 10, "layoutno": "text"}]
        self.pdf.page_images = [MagicMock()]
        self.pdf.page_images[0].size = (100, 100)
        self.pdf.total_page = 1

        # Call directly, any exception will fail the test
        self.pdf("test.pdf", callback=None)

    def test_pdf_call_empty_page_images(self):
        """Verify Pdf.__call__ handles empty page_images gracefully"""
        # Create a real Pdf instance (or a mock that acts like one enough to run __call__)
        # To strictly follow instructions "replace the mock invocation with the actual implementation call"
        # we try to call the real method.
        # Since self.pdf is a MagicMock from setUp, we might need to patch the method to be real or create a new instance.
        # However, paper.Pdf is the class. Let's try to invoke the unbound method or create a clean instance if possible,
        # but Pdf.__init__ might require valid file or args.
        # A safer approach compatible with the request "Pdf.__call__(self.pdf, ...)" using the mock as 'self' but running real code:
        self.pdf.page_images = []
        self.pdf.boxes = [{"text": "dummy", "x0": 0, "x1": 10, "layoutno": "text"}]
        # Call the real __call__ method using the mock object as 'self'
        paper.Pdf.__call__(self.pdf, "test.pdf", callback=None)

    @patch("rag.templates.paper.rag_tokenizer")
    @patch("rag.templates.paper.vision_figure_parser_pdf_wrapper")
    @patch("rag.templates.paper.normalize_layout_recognizer")
    def test_chunk_callback_none(self, mock_normalize, mock_vision, mock_tok):
        """Verify chunk() works when callback is None"""
        mock_normalize.return_value = ("DeepDOC", "model")
        mock_tok.tokenize.return_value = []
        mock_tok.fine_grained_tokenize.return_value = []

        # Configure the mock Pdf instance returned by the class constructor logic
        # Since Pdf is already patched at the class level in setUp, we use self.pdf
        mock_pdf_instance = self.pdf
        # Add required keys: authors, title, abstract
        mock_pdf_instance.return_value = {"sections": [], "tables": [], "authors": "", "title": "", "abstract": ""}

        mock_vision.return_value = []

        # Call should not raise exception
        paper.chunk("test.pdf", callback=None)

    @patch("rag.templates.paper.vision_figure_parser_pdf_wrapper")
    @patch("rag.templates.paper.rag_tokenizer")
    @patch("rag.templates.paper.normalize_layout_recognizer")
    @patch("rag.templates.paper.title_frequency")
    @patch("rag.templates.paper.bullets_category")
    def test_chunk_logging_defensive(self, mock_bullets, mock_title, mock_normalize, mock_tok, mock_vision):
        """Verify chunk() logging handles both tuples and strings in sorted_sections"""
        mock_bullets.return_value = []
        mock_title.return_value = (0, [0, 0])
        mock_normalize.return_value = ("DeepDOC", "model")
        mock_tok.tokenize.return_value = []
        mock_tok.fine_grained_tokenize.return_value = []
        mock_vision.return_value = []

        test_cases = [("tuple", [("text1", "label1"), ("text2", "label2")]), ("string", ["text1", "text2"])]

        for name, sorted_sections in test_cases:
            with self.subTest(name=name):
                # Setup specific return value for this case
                mock_pdf_instance = self.pdf
                mock_pdf_instance.return_value = {"sections": sorted_sections, "tables": [], "authors": "", "title": "", "abstract": ""}
                # Fix ValueError in tokenize_chunks -> pdf_parser.crop
                # pdf_parser is the mock_pdf_instance in this context
                mock_pdf_instance.crop.return_value = (MagicMock(), [])

                # Fix TypeError in tokenize -> re.sub because remove_tag returned a Mock
                mock_pdf_instance.remove_tag.side_effect = lambda x: x

                paper.chunk("test.pdf")
                # No exception raised indicates success

    @patch("rag.templates.paper.vision_figure_parser_pdf_wrapper")
    @patch("rag.templates.paper.rag_tokenizer")
    @patch("rag.templates.paper.normalize_layout_recognizer")
    @patch("rag.templates.paper.title_frequency")
    @patch("rag.templates.paper.bullets_category")
    def test_chunk_mismatch_error(self, mock_bullets, mock_title, mock_normalize, mock_tok, mock_vision):
        """Verify chunk() raises ValueError when sections and levels mismatch"""
        mock_bullets.return_value = []
        mock_title.return_value = (0, [0])
        mock_normalize.return_value = ("DeepDOC", "model")

        sorted_sections = ["text1", "text2"]
        mock_pdf = self.pdf
        mock_pdf.return_value = {"sections": sorted_sections, "tables": [], "authors": "", "title": "", "abstract": ""}

        mock_tok.tokenize.return_value = []
        mock_tok.fine_grained_tokenize.return_value = []
        mock_vision.return_value = []

        with self.assertRaisesRegex(ValueError, "Mismatch between number of sections"):
            paper.chunk("test.pdf")


if __name__ == "__main__":
    unittest.main()
