import sys
import os
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

# Configure sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


# ----------------- MOCK SETUP START -----------------
# Define Dummy classes to serve as base classes instead of MagicMock
class DummyPdfParser:
    def __init__(self):
        self.boxes = [{"text": "dummy", "x0": 0}]  # minimal structure

    def __images__(self, *args, **kwargs):
        pass

    def _layouts_rec(self, *args, **kwargs):
        pass

    def _naive_vertical_merge(self, *args, **kwargs):
        pass

    def _line_tag(self, *args, **kwargs):
        return "tag"


class DummyDocxParser:
    def __init__(self):
        pass


class DummyHtmlParser:
    def __init__(self):
        pass


# Mock modules
sys.modules["docx"] = MagicMock()
sys.modules["docx.Document"] = MagicMock()  # The class constructor
sys.modules["docx.opc.exceptions"] = MagicMock()
sys.modules["docx.oxml"] = MagicMock()

deepdoc = MagicMock()
sys.modules["deepdoc"] = deepdoc
sys.modules["deepdoc.parser"] = MagicMock()
# Assign our dummy classes as the attributes of the mocked module
sys.modules["deepdoc.parser"].PdfParser = DummyPdfParser
sys.modules["deepdoc.parser"].DocxParser = DummyDocxParser
sys.modules["deepdoc.parser"].HtmlParser = DummyHtmlParser

sys.modules["deepdoc.parser.utils"] = MagicMock()

sys.modules["common"] = MagicMock()
# Constants need to be resolvable
mock_constants = MagicMock()
# Create a fake enum-like behavior for ParserType.LAWS.value
# We can just say ParserType.LAWS.value = "laws" by making ParserType.LAWS a mock with value attribute
mock_parser_type = MagicMock()
mock_parser_type.LAWS.value = "laws"
mock_constants.ParserType = mock_parser_type
sys.modules["common.constants"] = mock_constants

sys.modules["common.parser_config_utils"] = MagicMock()

sys.modules["rag.nlp"] = MagicMock()
sys.modules["rag.nlp"].bullets_category = MagicMock(return_value=[])
sys.modules["rag.nlp"].docx_question_level = MagicMock(return_value=(1, "text"))

sys.modules["rag.app.format_parsers"] = MagicMock()
# ----------------- MOCK SETUP END -----------------

# Import the module under test
from rag.app.templates import laws


class TestLawsTemplate(unittest.TestCase):
    def test_docx_str_removed(self):
        """Verify Docx.__str__ is removed or safe."""
        docx = laws.Docx()
        # Verify str(docx) works and doesn't crash
        s = str(docx)
        self.assertIsInstance(s, str)
        # It should use the default object repr, e.g. <rag.app.templates.laws.Docx object at ...>
        # Check it does NOT contain the old broken format
        self.assertNotIn("question:", s)
        self.assertNotIn("childs:", s)
        print("Verified Docx.__str__ is safe.")

    def test_pdf_call_callback_none_fixed(self):
        """Verify Pdf.__call__ works with callback=None."""
        pdf = laws.Pdf()
        # Should not raise TypeError
        try:
            pdf("dummy.pdf", binary=b"dummy", callback=None)
        except Exception as e:
            self.fail(f"Pdf.__call__ raised exception with callback=None: {e}")
        print("Verified Pdf.__call__ handles None callback.")

    def test_doc_binary_handling_fixed(self):
        """Verify .doc parsing handles None binary correctly."""
        with patch("rag.app.templates.laws.re.search") as mock_re_search:
            # Force .doc match
            mock_re_search.side_effect = lambda pat, f, flags=0: pat == r"\.doc$"

            # Mock tika
            with patch.dict(sys.modules, {"tika": MagicMock(), "tika.parser": MagicMock()}):
                tika_mk = sys.modules["tika"].parser
                tika_mk.from_file.return_value = {"content": "parsed content"}

                # Test with binary=None
                # Pass a callback to ensure checking behavior is isolated
                def cb(*args, **kwargs):
                    pass

                _res = laws.chunk("test.doc", binary=None, callback=cb)

                # Verify tika_parser.from_file was called (since binary is None)
                tika_mk.from_file.assert_called_with("test.doc")
                tika_mk.from_buffer.assert_not_called()
                print("Verified chunk .doc handling for None binary uses from_file.")

    def test_doc_binary_bytes_handling(self):
        """Verify .doc parsing handles bytes binary correctly."""
        with patch("rag.app.templates.laws.re.search") as mock_re_search:
            mock_re_search.side_effect = lambda pat, f, flags=0: pat == r"\.doc$"
            with patch.dict(sys.modules, {"tika": MagicMock(), "tika.parser": MagicMock()}):
                tika_mk = sys.modules["tika"].parser
                tika_mk.from_buffer.return_value = {"content": "parsed content"}

                def cb(*args, **kwargs):
                    pass

                laws.chunk("test.doc", binary=b"some bytes", callback=cb)

                tika_mk.from_buffer.assert_called()
                args, _ = tika_mk.from_buffer.call_args
                # args[0] should be BytesIO
                self.assertIsInstance(args[0], BytesIO)
                print("Verified chunk .doc handling for bytes binary uses from_buffer.")

    def test_not_implemented_error_message_fixed(self):
        """Verify NotImplementedError message lists all supported formats."""
        try:
            laws.chunk("unsupported.xyz", callback=lambda *args, **kwargs: None)
        except NotImplementedError as e:
            msg = str(e)
            print(f"Verified NotImplementedError message: {msg}")
            expected = ["doc", "docx", "pdf", "txt", "md", "markdown", "mdx", "htm", "html"]
            for ext in expected:
                self.assertIn(ext, msg, f"Message missing extension: {ext}")
        else:
            self.fail("Did not raise NotImplementedError")


if __name__ == "__main__":
    unittest.main()
