import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# 1. Mock external dependencies BEFORE importing the target module
# This is critical because the target module imports these at top level
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()
sys.modules["aspose"] = MagicMock()
sys.modules["aspose.slides"] = MagicMock()
sys.modules["aspose.pydrawing"] = MagicMock()

# Mock deepdoc structure
mock_deepdoc = MagicMock()


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


mock_deepdoc.parser.PdfParser = MockPdfParser
mock_deepdoc.parser.PptParser = MockPptParser
mock_deepdoc.parser.PlainParser = MockPlainParser
sys.modules["deepdoc"] = mock_deepdoc
sys.modules["deepdoc.parser"] = mock_deepdoc.parser


# Mock rag.* structure
# We DO NOT mock 'rag' or 'rag.app' top level modules because we need to import
# the real 'rag.app.templates.presentation'.
# We only mock the specific leaf modules that presentation.py imports.

sys.modules["rag.nlp"] = MagicMock()
sys.modules["rag.app.format_parsers"] = MagicMock()

# Setup specific attributes for rag.nlp
sys.modules["rag.nlp"].rag_tokenizer.tokenize = MagicMock(return_value=[])
sys.modules["rag.nlp"].rag_tokenizer.fine_grained_tokenize = MagicMock(return_value=[])
sys.modules["rag.nlp"].is_english = MagicMock(return_value=True)
sys.modules["rag.nlp"].tokenize = MagicMock()

# format_parsers needs PARSERS dict and by_plaintext function
mock_parser_dict = {}
mock_by_plaintext = MagicMock()
sys.modules["rag.app.format_parsers"].PARSERS = mock_parser_dict
sys.modules["rag.app.format_parsers"].by_plaintext = mock_by_plaintext

# common.parser_config_utils
mock_common = MagicMock()
mock_common.parser_config_utils.normalize_layout_recognizer = MagicMock(return_value=("DeepDOC", "model"))
sys.modules["common"] = mock_common
sys.modules["common.parser_config_utils"] = mock_common.parser_config_utils


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import the target. It should use the mocked modules.
try:
    from rag.app.templates.presentation import Pdf, Ppt, chunk
except ImportError as e:
    print(f"Failed to import presentation: {e}")
    # Inspect what failed
    import traceback

    traceback.print_exc()
    sys.exit(1)


class TestPresentationTemplate(unittest.TestCase):
    def setUp(self):
        # Patch BytesIO in presentation module to handle string inputs without crashing,
        # because existing code (unrelated to our fix) passes string to BytesIO.
        patcher = patch("rag.app.templates.presentation.BytesIO")
        self.MockBytesIO = patcher.start()
        self.addCleanup(patcher.stop)

    def test_pdf_callback_none_safe(self):
        """Test Pdf parser with callback=None does NOT raise TypeError"""
        parser = Pdf()
        # Mock internal methods
        parser.__images__ = MagicMock()
        parser._layouts_rec = MagicMock()
        parser._table_transformer_job = MagicMock()
        parser._text_merge = MagicMock()
        parser._extract_table_figure = MagicMock(return_value=[])

        try:
            # invocations inside __call__ are now guarded
            parser("dummy.pdf", callback=None)
        except TypeError as e:
            self.fail(f"Pdf raised TypeError with callback=None: {e}")
        except Exception:
            # We ignore other errors as we are only verifying callback safety
            pass

    def test_ppt_callback_none_safe(self):
        """Test Ppt parser with callback=None does NOT raise TypeError"""
        parser = Ppt()
        try:
            parser("dummy.pptx", 0, 10, callback=None)
        except TypeError as e:
            self.fail(f"Ppt raised TypeError with callback=None: {e}")
        except Exception:
            pass

    def test_chunk_callback_none_safe(self):
        """Test chunk function with callback=None"""
        # pdf case
        # Ensure PARSERS.get returns a callable mock
        mock_parser_impl = MagicMock(return_value=([], None, None))
        sys.modules["rag.app.format_parsers"].PARSERS["deepdoc"] = mock_parser_impl

        try:
            chunk("dummy.pdf", callback=None)
        except TypeError as e:
            self.fail(f"chunk(pdf) raised TypeError with callback=None: {e}")
        except Exception:
            pass

        # ppt case
        try:
            chunk("dummy.pptx", callback=None)
        except TypeError as e:
            self.fail(f"chunk(ppt) raised TypeError with callback=None: {e}")
        except Exception:
            pass

    def test_chunk_ppt_argument(self):
        """Test that chunk passes the correct to_page parameter to PptParser"""
        # We need to verify that PptParser is called with the 'to_page' value from chunk,
        # not the hardcoded 1000000.

        # Ppt is instantiated inside chunk. Ppt calls super calls etc.
        # But we mocked Ppt class in module imports? No, we mocked PptParser base class.
        # presentation.Ppt inherits from PptParser.

        # We can patch 'rag.app.templates.presentation.Ppt' but that is the class under test.
        # We want to check Ppt(...)(filename, from_page, to_page, callback)

        with patch("rag.app.templates.presentation.Ppt") as MockPptClass:
            mock_ppt_instance = MagicMock()
            # The iterator
            mock_ppt_instance.return_value = []
            MockPptClass.return_value = mock_ppt_instance

            chunk("test.pptx", from_page=0, to_page=123)

            # Check call args of instance call: (filename, from_page, to_page, callback)
            args, _ = mock_ppt_instance.call_args
            # args[2] should be to_page (123)
            self.assertEqual(args[2], 123, f"Expected to_page=123, got {args[2]}")


if __name__ == "__main__":
    unittest.main()
