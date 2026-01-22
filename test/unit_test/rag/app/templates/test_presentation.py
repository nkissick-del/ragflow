import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

# ----------------- MOCK SETUP START -----------------
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()
sys.modules["aspose"] = MagicMock()
sys.modules["aspose.slides"] = MagicMock()
sys.modules["aspose.pydrawing"] = MagicMock()


class MockPdfParser:
    def __init__(self):
        self.page_images = []
        self.boxes = []

    def __call__(self, *args, **kwargs):
        return ["text"]

    def __images__(self, *args, **kwargs):
        pass

    def _layouts_rec(self, *args, **kwargs):
        pass

    def _table_transformer_job(self, *args, **kwargs):
        pass

    def _text_merge(self, *args, **kwargs):
        pass

    def _extract_table_figure(self, *args, **kwargs):
        return []


class MockPptParser:
    def __call__(self, fnm, from_page, to_page, callback=None):
        return ["text"]


class MockPlainParser:
    pass


mock_deepdoc = MagicMock()
mock_deepdoc.parser.PdfParser = MockPdfParser
mock_deepdoc.parser.PptParser = MockPptParser
mock_deepdoc.parser.PlainParser = MockPlainParser
sys.modules["deepdoc"] = mock_deepdoc
sys.modules["deepdoc.parser"] = mock_deepdoc.parser

sys.modules["rag.nlp"] = MagicMock()
sys.modules["rag.nlp"].rag_tokenizer.tokenize = MagicMock(return_value=[])
sys.modules["rag.nlp"].rag_tokenizer.fine_grained_tokenize = MagicMock(return_value=[])
sys.modules["rag.nlp"].is_english = MagicMock(return_value=True)
sys.modules["rag.nlp"].tokenize = MagicMock()

mock_parser_dict = {}
mock_by_plaintext = MagicMock()
sys.modules["rag.app.format_parsers"] = MagicMock()
sys.modules["rag.app.format_parsers"].PARSERS = mock_parser_dict
sys.modules["rag.app.format_parsers"].by_plaintext = mock_by_plaintext

mock_common = MagicMock()
mock_common.parser_config_utils.normalize_layout_recognizer = MagicMock(return_value=("DeepDOC", "model"))
sys.modules["common"] = mock_common
sys.modules["common.parser_config_utils"] = mock_common.parser_config_utils
# ----------------- MOCK SETUP END -----------------

HAS_PRESENTATION = False
Pdf = None
Ppt = None
chunk = None

try:
    from rag.app.templates.presentation import Pdf, Ppt, chunk

    HAS_PRESENTATION = True
except ImportError:
    pass


@unittest.skipUnless(HAS_PRESENTATION, "presentation module not available")
class TestPresentationTemplate(unittest.TestCase):
    """Tests for the presentation template to verify callback safety and argument handling."""

    def setUp(self):
        patcher = patch("rag.app.templates.presentation.BytesIO")
        self.MockBytesIO = patcher.start()
        self.addCleanup(patcher.stop)

    def test_pdf_callback_none_safe(self):
        """Test Pdf parser with callback=None does NOT raise TypeError."""
        parser = Pdf()
        parser.__images__ = MagicMock()
        parser._layouts_rec = MagicMock()
        parser._table_transformer_job = MagicMock()
        parser._text_merge = MagicMock()
        parser._extract_table_figure = MagicMock(return_value=[])

        try:
            parser("dummy.pdf", callback=None)
        except TypeError as e:
            self.fail(f"Pdf raised TypeError with callback=None: {e}")
        except (FileNotFoundError, AttributeError, KeyError):
            # These exceptions are acceptable due to mocking limitations
            # (e.g., mock object missing attributes, file operations on dummy paths)
            pass

    def test_ppt_callback_none_safe(self):
        """Test Ppt parser with callback=None does NOT raise TypeError."""
        parser = Ppt()
        try:
            parser("dummy.pptx", 0, 10, callback=None)
        except TypeError as e:
            self.fail(f"Ppt raised TypeError with callback=None: {e}")
        except (FileNotFoundError, AttributeError, KeyError):
            # These exceptions are acceptable due to mocking limitations
            pass

    def test_chunk_callback_none_safe(self):
        """Test chunk function with callback=None."""
        mock_parser_impl = MagicMock(return_value=([], None, None))
        sys.modules["rag.app.format_parsers"].PARSERS["deepdoc"] = mock_parser_impl

        try:
            chunk("dummy.pdf", callback=None)
        except TypeError as e:
            self.fail(f"chunk(pdf) raised TypeError with callback=None: {e}")
        except (FileNotFoundError, AttributeError, KeyError):
            # These exceptions are acceptable due to mocking limitations
            pass

        try:
            chunk("dummy.pptx", callback=None)
        except TypeError as e:
            self.fail(f"chunk(ppt) raised TypeError with callback=None: {e}")
        except (FileNotFoundError, AttributeError, KeyError):
            # These exceptions are acceptable due to mocking limitations
            pass

    def test_chunk_passes_correct_to_page(self):
        """Test that chunk passes the correct to_page parameter to PptParser."""
        with patch("rag.app.templates.presentation.Ppt") as MockPptClass:
            mock_ppt_instance = MagicMock()
            mock_ppt_instance.return_value = []
            MockPptClass.return_value = mock_ppt_instance

            chunk("test.pptx", from_page=0, to_page=123)

            args, _ = mock_ppt_instance.call_args
            self.assertEqual(args[2], 123, f"Expected to_page=123, got {args[2]}")


if __name__ == "__main__":
    unittest.main()
