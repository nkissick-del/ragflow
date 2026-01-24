import unittest
from unittest.mock import MagicMock, patch
import sys
from io import BytesIO

# Project root is automatically added to sys.path by test/unit_test/conftest.py


# ----------------- MOCK SETUP START -----------------
# Save original sys.modules state for cleanup
_MOCKED_MODULES = [
    "docx",
    "docx.Document",
    "docx.opc.exceptions",
    "docx.oxml",
    "deepdoc",
    "deepdoc.parser",
    "deepdoc.parser.utils",
    "common",
    "common.constants",
    "common.parser_config_utils",
    "rag.nlp",
    "rag.orchestration.router",
]

# Track intermediate package names that may be created by imports
_INTERMEDIATE_PACKAGES = [
    "rag",
    "rag.app",
    "rag.app.templates",
]

_original_modules = {mod: sys.modules.get(mod) for mod in _MOCKED_MODULES}
_original_intermediate_packages = {pkg: sys.modules.get(pkg) for pkg in _INTERMEDIATE_PACKAGES}


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
mock_settings = MagicMock()
mock_settings.PARALLEL_DEVICES = 0
sys.modules["common.settings"] = mock_settings
sys.modules["common"].settings = mock_settings
sys.modules["common.parser_config_utils"] = MagicMock()

sys.modules["rag.nlp"] = MagicMock()
sys.modules["rag.nlp"].bullets_category = MagicMock(return_value=[])
sys.modules["rag.nlp"].docx_question_level = MagicMock(return_value=(1, "text"))
sys.modules["rag.nlp"].docx_question_level = MagicMock(return_value=(1, "text"))
sys.modules["rag.orchestration.router"] = MagicMock()
# ----------------- MOCK SETUP END -----------------

from rag.templates import laws


class TestLawsTemplate(unittest.TestCase):
    """Tests for the laws template to verify callback safety and error handling."""

    @staticmethod
    def _doc_search_side_effect(pat, f, flags=0):
        """Shared helper for mocking re.search to match .doc files."""
        if pat == r"\.doc$":
            match_mock = MagicMock()
            match_mock.group.return_value = ".doc"
            return match_mock
        return None

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
        # Force set the setting to avoid MagicMock comparison error
        if "common" in sys.modules and hasattr(sys.modules["common"], "settings"):
            sys.modules["common"].settings.PARALLEL_DEVICES = 0
        if "common.settings" in sys.modules:
            sys.modules["common.settings"].PARALLEL_DEVICES = 0

        pdf = laws.Pdf()
        # Mock internal methods that might rely on complex dependencies (like DeepDOC layouter)
        pdf.__images__ = MagicMock()
        pdf._layouts_rec = MagicMock()
        pdf._table_transformer_job = MagicMock()
        pdf._text_merge = MagicMock()
        pdf._extract_table_figure = MagicMock(return_value=[])
        pdf._naive_vertical_merge = MagicMock()
        pdf.boxes = [{"text": "dummy", "x0": 0, "x1": 10, "top": 0, "bottom": 10, "layoutno": "text", "page_number": 1}]
        pdf.page_cum_height = [0]
        pdf.page_images = [MagicMock()]
        pdf.page_images[0].size = (100, 100)

        try:
            pdf("dummy.pdf", binary=b"dummy", callback=None)
        except TypeError as e:
            self.fail(f"Pdf.__call__ raised TypeError with callback=None: {e}")
        except (FileNotFoundError, ValueError, AttributeError, KeyError) as e:
            # Report the exception so the test provides visibility into what went wrong
            self.fail(f"Pdf.__call__ raised {type(e).__name__} with callback=None: {e}. This may indicate incomplete mocking or a real issue in the callback=None path.")

    def test_doc_binary_none_uses_from_file(self):
        """Verify .doc parsing handles None binary correctly by using from_file."""
        with patch("rag.templates.laws.re.search") as mock_re_search:
            mock_re_search.side_effect = self._doc_search_side_effect

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
        with patch("rag.templates.laws.re.search") as mock_re_search:
            mock_re_search.side_effect = self._doc_search_side_effect

            # Create a consistent tika parser mock
            tika_parser_mock = MagicMock()
            tika_parser_mock.from_buffer.return_value = {"content": "parsed content"}
            tika_mock = MagicMock()
            tika_mock.parser = tika_parser_mock

            with patch.dict(sys.modules, {"tika": tika_mock, "tika.parser": tika_parser_mock}):
                laws.chunk("test.doc", binary=b"some bytes", callback=lambda *args, **kwargs: None)

                tika_parser_mock.from_buffer.assert_called_once()
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


def teardown_module():
    """Restore sys.modules to original state after tests complete."""
    # Remove the imported laws module to allow fresh imports in other tests
    if "rag.app.templates.laws" in sys.modules:
        del sys.modules["rag.app.templates.laws"]

    # Restore original sys.modules entries
    for mod, original_value in _original_modules.items():
        if original_value is None:
            # Module didn't exist originally, remove it
            sys.modules.pop(mod, None)
        else:
            # Restore original module
            sys.modules[mod] = original_value

    # Restore intermediate package names to ensure complete isolation
    # Process in reverse order (deepest to shallowest) to avoid parent/child issues
    for pkg in reversed(_INTERMEDIATE_PACKAGES):
        original_value = _original_intermediate_packages.get(pkg)
        if original_value is None:
            # Package didn't exist originally, remove it
            sys.modules.pop(pkg, None)
        else:
            # Restore original package
            sys.modules[pkg] = original_value


if __name__ == "__main__":
    unittest.main()
