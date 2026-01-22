import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from io import BytesIO

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))


# ----------------- MOCK SETUP START -----------------
class DummyPdfParser:
    def __init__(self):
        self.boxes = [{"text": "dummy", "x0": 0}]

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


sys.modules["docx"] = MagicMock()
sys.modules["docx.Document"] = MagicMock()
sys.modules["docx.opc.exceptions"] = MagicMock()
sys.modules["docx.oxml"] = MagicMock()

deepdoc = MagicMock()
sys.modules["deepdoc"] = deepdoc
sys.modules["deepdoc.parser"] = MagicMock()
sys.modules["deepdoc.parser"].PdfParser = DummyPdfParser
sys.modules["deepdoc.parser"].DocxParser = DummyDocxParser
sys.modules["deepdoc.parser"].HtmlParser = DummyHtmlParser
sys.modules["deepdoc.parser.utils"] = MagicMock()

sys.modules["common"] = MagicMock()
mock_constants = MagicMock()
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

from rag.app.templates import laws


class TestLawsTemplate(unittest.TestCase):
    """Tests for the laws template to verify callback safety and error handling."""

    def test_docx_str_safe(self):
        """Verify Docx.__str__ is removed or safe."""
        docx = laws.Docx()
        s = str(docx)
        self.assertIsInstance(s, str)
        # Should NOT contain old broken format
        self.assertNotIn("question:", s)
        self.assertNotIn("childs:", s)

    def test_pdf_callback_none_safe(self):
        """Verify Pdf.__call__ works with callback=None."""
        pdf = laws.Pdf()
        try:
            pdf("dummy.pdf", binary=b"dummy", callback=None)
        except TypeError as e:
            self.fail(f"Pdf.__call__ raised TypeError with callback=None: {e}")
        except (FileNotFoundError, ValueError, AttributeError, KeyError):
            # These exceptions are acceptable due to mocking limitations
            # (e.g., missing file, mock object attribute access)
            pass

    def test_doc_binary_none_uses_from_file(self):
        """Verify .doc parsing handles None binary correctly by using from_file."""
        with patch("rag.app.templates.laws.re.search") as mock_re_search:
            # Return a Match-like object when pattern matches, None otherwise
            def search_side_effect(pat, f, flags=0):
                if pat == r"\.doc$":
                    match_mock = MagicMock()
                    match_mock.group.return_value = ".doc"
                    return match_mock
                return None

            mock_re_search.side_effect = search_side_effect

            # Create a consistent tika parser mock
            tika_parser_mock = MagicMock()
            tika_parser_mock.from_file.return_value = {"content": "parsed content"}
            tika_mock = MagicMock()
            tika_mock.parser = tika_parser_mock

            with patch.dict(sys.modules, {"tika": tika_mock, "tika.parser": tika_parser_mock}):
                laws.chunk("test.doc", binary=None, callback=lambda *args, **kwargs: None)

                tika_parser_mock.from_file.assert_called_with("test.doc")
                tika_parser_mock.from_buffer.assert_not_called()

    def test_doc_binary_bytes_uses_from_buffer(self):
        """Verify .doc parsing handles bytes binary correctly by using from_buffer."""
        with patch("rag.app.templates.laws.re.search") as mock_re_search:
            # Return a Match-like object when pattern matches, None otherwise
            def search_side_effect(pat, f, flags=0):
                if pat == r"\.doc$":
                    match_mock = MagicMock()
                    match_mock.group.return_value = ".doc"
                    return match_mock
                return None

            mock_re_search.side_effect = search_side_effect

            # Create a consistent tika parser mock
            tika_parser_mock = MagicMock()
            tika_parser_mock.from_buffer.return_value = {"content": "parsed content"}
            tika_mock = MagicMock()
            tika_mock.parser = tika_parser_mock

            with patch.dict(sys.modules, {"tika": tika_mock, "tika.parser": tika_parser_mock}):
                laws.chunk("test.doc", binary=b"some bytes", callback=lambda *args, **kwargs: None)

                tika_parser_mock.from_buffer.assert_called()
                args, _ = tika_parser_mock.from_buffer.call_args
                self.assertIsInstance(args[0], BytesIO)

    def test_not_implemented_error_lists_all_formats(self):
        """Verify NotImplementedError message lists all supported formats."""
        try:
            laws.chunk("unsupported.xyz", callback=lambda *args, **kwargs: None)
        except NotImplementedError as e:
            msg = str(e)
            expected = ["doc", "docx", "pdf", "txt", "md", "markdown", "mdx", "htm", "html"]
            for ext in expected:
                self.assertIn(ext, msg, f"Message missing extension: {ext}")
        else:
            self.fail("Did not raise NotImplementedError")


if __name__ == "__main__":
    unittest.main()
