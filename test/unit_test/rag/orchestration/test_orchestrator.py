import unittest
from unittest.mock import patch
import sys


# Mock dependencies to prevent heavy imports (DB, ML models, etc.)
# This allows running unit tests without installing the full environment.
from unittest.mock import MagicMock

_mock_modules = [
    "rag.app.format_parsers",
    "rag.orchestration.router",
    "rag.templates.general",
    "rag.templates.semantic",
    "rag.nlp",
    "rag.utils.file_utils",
    "common",
    "common.settings",
    "api.db.services.llm_service",
]


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Setup mocks
        self.mock_modules = {
            "rag.app.format_parsers": MagicMock(),
            "rag.orchestration.router": MagicMock(),
            "rag.templates.general": MagicMock(),
            "rag.templates.semantic": MagicMock(),
            "rag.nlp": MagicMock(),
            "rag.utils.file_utils": MagicMock(),
            "common": MagicMock(),
            "common.settings": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
        }

        # Setup specific mock attributes needed by orchestrator import
        self.mock_modules["rag.app.format_parsers"].PARSERS = {}
        # rag.nlp.rag_tokenizer is imported as `from rag.nlp import rag_tokenizer`
        self.mock_modules["rag.nlp"].rag_tokenizer = MagicMock()

        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import after patching
        from rag.orchestration import orchestrator
        from rag.orchestration.orchestrator import ParsingError
        from rag.orchestration.router import ParseResult
        from rag.orchestration.base import StandardizedDocument, DocumentElement

        self.orchestrator = orchestrator
        self.ParsingError = ParsingError
        self.ParseResult = ParseResult
        self.StandardizedDocument = StandardizedDocument
        self.DocumentElement = DocumentElement

    def tearDown(self):
        self.patcher.stop()

    @patch("rag.orchestration.orchestrator.UniversalRouter")
    @patch("rag.orchestration.orchestrator.General")
    def test_chunk(self, mock_general, mock_router):
        # Setup mocks
        mock_router.return_value.route.return_value = self.ParseResult(sections=["section1"])
        mock_general.return_value.chunk.return_value = [{"content": "result"}]

        # Test call
        filename = "test.docx"
        binary = b"content"
        res = self.orchestrator.chunk(filename, binary)

        # Verify calls
        mock_router.return_value.route.assert_called_once()
        mock_general.return_value.chunk.assert_called_once()

        # Verify result
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "result")

    @patch("rag.orchestration.orchestrator.extract_embed_file")
    def test_chunk_with_embeds(self, mock_extract):
        # Simulate an embedded file
        mock_extract.return_value = [("embed.pdf", b"embed_content")]

        # Mock dependencies to avoid actual parsing
        with patch("rag.orchestration.orchestrator.UniversalRouter") as mock_router, patch("rag.orchestration.orchestrator.General") as mock_general:
            mock_router.return_value.route.return_value = self.ParseResult()
            mock_general.return_value.chunk.return_value = []

            self.orchestrator.chunk("root.docx", b"root_content")

            # Verify extract called
            # Verify recursive call happens (implied by extract_embed_file being called and loop running)
            # Since we mocked chunk's internals, we expect at least the root processing attempts.
            # We expect 2 calls to route: one for the root document, and one for the embedded file.
            self.assertEqual(mock_router.return_value.route.call_count, 2)


class TestAdaptDoclingOutput(unittest.TestCase):
    """Tests for the adapt_docling_output() adapter function."""

    def setUp(self):
        mock_modules = {
            "rag.app.format_parsers": MagicMock(),
            "rag.orchestration.router": MagicMock(),
            "rag.templates.general": MagicMock(),
            "rag.templates.semantic": MagicMock(),
            "rag.nlp": MagicMock(),
            "rag.utils.file_utils": MagicMock(),
            "common": MagicMock(),
            "common.settings": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, mock_modules)
        self.patcher.start()

        from rag.orchestration import orchestrator
        from rag.orchestration.base import StandardizedDocument

        self.orchestrator = orchestrator
        self.StandardizedDocument = StandardizedDocument

    def tearDown(self):
        self.patcher.stop()

    def test_adapt_string_input(self):
        """Test adapter with new string format (semantic mode)."""
        sections = "# Heading\n\nParagraph text here."
        tables = []
        parser_config = {"layout_recognizer": "Docling"}

        # Need to re-import or use class imports if possible, but this is a separate class.
        # So we need to patch here too.
        # Actually, AdaptDoclingOutput tests standard library function, it imports orchestrator.
        # We should use a base class or setUp here too.
        # For now, let's just patch sys.modules in setUp of this class as well, or at least import inside tests.
        # But we need mocks for dependencies too?
        # adapt_docling_output relies on StandardizedDocument and DocumentElement from base.py
        # orchestrator imports them.
        # adapt_docling_output doesn't seem to rely on heavy stuff.
        # But we import orchestrator which triggers heavy imports.
        # So we MUST patch/mock modules before import.

        # To avoid duplicating setUp, I should probably merge classes or inherit from a base.
        # I'll patch sys.modules here too.
        mock_modules = {
            "rag.app.format_parsers": MagicMock(),
            "rag.orchestration.router": MagicMock(),
            "rag.templates.general": MagicMock(),
            "rag.templates.semantic": MagicMock(),
            "rag.nlp": MagicMock(),
            "rag.utils.file_utils": MagicMock(),
            "common": MagicMock(),
            "common.settings": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
        }
        with patch.dict(sys.modules, mock_modules):
            from rag.orchestration import orchestrator
            from rag.orchestration.base import StandardizedDocument

            result = orchestrator.adapt_docling_output(sections, tables, parser_config)

            self.assertIsInstance(result, StandardizedDocument)
            self.assertEqual(result.content, sections)
            self.assertEqual(result.metadata["parser"], "docling")
            self.assertEqual(result.metadata["layout_recognizer"], "Docling")

    def test_adapt_with_tables(self):
        """Test adapter handling of tables."""
        sections = "# Header\nText"
        tables = [{"type": "table", "content": "| A | B |\n|---|---|\n| 1 | 2 |"}]
        parser_config = {"layout_recognizer": "Docling"}

        mock_modules = {
            "rag.app.format_parsers": MagicMock(),
            "rag.orchestration.router": MagicMock(),
            "rag.templates.general": MagicMock(),
            "rag.templates.semantic": MagicMock(),
            "rag.nlp": MagicMock(),
            "rag.utils.file_utils": MagicMock(),
            "common": MagicMock(),
            "common.settings": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
        }
        with patch.dict(sys.modules, mock_modules):
            from rag.orchestration import orchestrator
            from rag.orchestration.base import StandardizedDocument

            result = orchestrator.adapt_docling_output(sections, tables, parser_config)

            self.assertIsInstance(result, StandardizedDocument)
            self.assertEqual(result.content, sections)
            self.assertEqual(result.metadata["tables"], tables)
            self.assertEqual(result.metadata["parser"], "docling")
            self.assertEqual(result.metadata["layout_recognizer"], "Docling")

    def test_adapt_list_input_legacy(self):
        """Test adapter with legacy list format."""
        sections = ["# Heading", "Paragraph text here."]
        tables = []
        parser_config = {"layout_recognizer": "Docling"}

        result = self.orchestrator.adapt_docling_output(sections, tables, parser_config)

        self.assertIsInstance(result, self.StandardizedDocument)
        # List should be joined with newlines
        self.assertEqual(result.content, "# Heading\nParagraph text here.")

    def test_adapt_empty_string(self):
        """Test adapter with empty string."""
        result = self.orchestrator.adapt_docling_output("", [], {})
        self.assertEqual(result.content, "")

    def test_adapt_empty_list(self):
        """Test adapter with empty list."""
        result = self.orchestrator.adapt_docling_output([], [], {})
        self.assertEqual(result.content, "")

    def test_adapter_preserves_elements_empty(self):
        """Test that elements list is empty initially and then raises if accessed without implementation."""
        result = self.orchestrator.adapt_docling_output("test", [], {})
        # In current design, _elements=None leads to _parse_elements which raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = result.elements


class TestSemanticRouting(unittest.TestCase):
    """Tests for semantic template routing in orchestrator.chunk()."""

    def setUp(self):
        mock_modules = {
            "rag.app.format_parsers": MagicMock(),
            "rag.orchestration.router": MagicMock(),
            "rag.templates.general": MagicMock(),
            "rag.templates.semantic": MagicMock(),
            "rag.nlp": MagicMock(),
            "rag.utils.file_utils": MagicMock(),
            "common": MagicMock(),
            "common.settings": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, mock_modules)
        self.patcher.start()

        from rag.orchestration import orchestrator
        from rag.orchestration.router import ParseResult

        self.orchestrator = orchestrator
        self.ParseResult = ParseResult

    def tearDown(self):
        self.patcher.stop()

    @patch("rag.orchestration.orchestrator.UniversalRouter")
    @patch("rag.orchestration.orchestrator.General")
    def test_legacy_path_with_list_sections(self, mock_general, mock_router):
        """Test that list sections still route to General template."""
        # Sections as list = legacy path
        mock_router.return_value.route.return_value = self.ParseResult(sections=["section1", "section2"], is_markdown=False)
        mock_general.return_value.chunk.return_value = [{"content": "general result"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": False}

        # os.environ patch removed as requested
        self.orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because sections is a list, not string
        mock_general.return_value.chunk.assert_called_once()

    @patch("rag.orchestration.orchestrator.UniversalRouter")
    @patch("rag.orchestration.orchestrator.Semantic")
    def test_semantic_path_with_string_sections(self, mock_semantic, mock_router):
        """Test that string sections with use_semantic_chunking routes to Semantic template."""
        # Sections as string = new semantic path
        mock_router.return_value.route.return_value = self.ParseResult(sections="# Heading\n\nContent", is_markdown=True)
        mock_semantic.return_value.chunk.return_value = [{"content": "semantic result", "header_path": "/Heading/"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": True}

        # The decorator patch on rag.app.orchestrator.Semantic is sufficient
        # No need for redundant inner patch of rag.app.templates.semantic.Semantic
        res = self.orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use Semantic template
        mock_semantic.return_value.chunk.assert_called_once()

        # Verify the result contains the expected semantic chunk
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "semantic result")
        self.assertEqual(res[0]["header_path"], "/Heading/")

    @patch("rag.orchestration.orchestrator.UniversalRouter")
    @patch("rag.orchestration.orchestrator.General")
    def test_legacy_path_without_semantic_flag(self, mock_general, mock_router):
        """Test that without use_semantic_chunking flag, General is used."""
        mock_router.return_value.route.return_value = self.ParseResult(sections="# Heading\n\nContent")
        mock_general.return_value.chunk.return_value = [{"content": "general result"}]

        # No use_semantic_chunking flag
        parser_config = {"layout_recognizer": "Docling"}

        res = self.orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because flag is not set
        mock_general.return_value.chunk.assert_called_once()
        self.assertEqual(res, [{"content": "general result"}])


if __name__ == "__main__":
    unittest.main()
