import unittest
from unittest.mock import MagicMock, patch
import sys


# Project root is automatically added to sys.path by test/unit_test/conftest.py


# ----------------- MOCK SETUP START -----------------
# Save original sys.modules state for cleanup


class TestLawsTemplate(unittest.TestCase):
    """Tests for the laws template to verify callback safety and error handling."""

    def setUp(self):
        self.mock_modules = {
            "docx": MagicMock(),
            "docx.Document": MagicMock(),
            "docx.opc.exceptions": MagicMock(),
            "docx.oxml": MagicMock(),
            "docx.image": MagicMock(),
            "docx.image.exceptions": MagicMock(),
            "docx.opc": MagicMock(),
            "docx.opc.pkgreader": MagicMock(),
            "docx.opc.oxml": MagicMock(),
            "docx.table": MagicMock(),
            "docx.text": MagicMock(),
            "docx.text.paragraph": MagicMock(),
            "bs4": MagicMock(),
            "bs4.BeautifulSoup": MagicMock(),
            "bs4.NavigableString": MagicMock(),
            "bs4.Tag": MagicMock(),
            "bs4.Comment": MagicMock(),
            "huggingface_hub": MagicMock(),
            "pdfplumber": MagicMock(),
            "xgboost": MagicMock(),
            "sklearn": MagicMock(),
            "sklearn.cluster": MagicMock(),
            "sklearn.metrics": MagicMock(),
            "pypdf": MagicMock(),
            "pptx": MagicMock(),  # Mock pptx manually here if needed or rely on conftest
            "deepdoc": MagicMock(),
            "deepdoc.parser": MagicMock(),
            "deepdoc.vision": MagicMock(),
            "common": MagicMock(),
            "common.constants": MagicMock(),
            "common.parser_config_utils": MagicMock(),
            "common.token_utils": MagicMock(),  # Ensure this is mocked!
            "rag.nlp": MagicMock(),
            "rag.orchestration.router": MagicMock(),
        }

        # Setup specific mocks
        self.mock_modules["rag.nlp"].bullets_category.return_value = []
        self.mock_modules["rag.nlp"].docx_question_level.return_value = (1, "text")

        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import after patching
        from rag.templates import laws

        self.laws = laws

    def tearDown(self):
        self.patcher.stop()

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
        docx = self.laws.Docx()
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

        pdf = self.laws.Pdf()
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

        # The test ensures no exception is raised when callback is None
        pdf("dummy.pdf", binary=b"dummy", callback=None)

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
                self.laws.chunk("test.doc", binary=None, callback=lambda *args, **kwargs: None)

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
                self.laws.chunk("test.doc", binary=b"some bytes", callback=lambda *args, **kwargs: None)

                tika_parser_mock.from_buffer.assert_called_once()
                args, _ = tika_parser_mock.from_buffer.call_args
                self.assertIsInstance(args[0], bytes)

    def test_not_implemented_error_lists_all_formats(self):
        """Verify NotImplementedError message lists all supported formats."""
        with self.assertRaises(NotImplementedError) as cm:
            self.laws.chunk("unsupported.xyz", callback=lambda *args, **kwargs: None)

        msg = str(cm.exception)
        expected = ["doc", "docx", "pdf", "txt", "md", "markdown", "mdx", "htm", "html"]
        for ext in expected:
            self.assertIn(ext, msg, f"Message missing extension: {ext}")


# Removed teardown_module as it is no longer needed with setUp/patcher


if __name__ == "__main__":
    unittest.main()
