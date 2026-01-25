import unittest
from unittest.mock import patch, MagicMock
import sys


class OrchestratorTestBase(unittest.TestCase):
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


class TestOrchestrator(OrchestratorTestBase):
    def test_chunk(self):
        # Setup mocks via existing mocked modules from setUp
        # orchestrator.UniversalRouter is imported from rag.orchestration.router
        mock_router = self.mock_modules["rag.orchestration.router"].UniversalRouter
        mock_general = self.mock_modules["rag.templates.general"].General

        mock_router.route.return_value = self.ParseResult(sections=["section1"])
        mock_general.chunk.return_value = [{"content": "result"}]

        # Reset counts for this test
        mock_router.route.reset_mock()
        mock_general.chunk.reset_mock()

        # Test call
        filename = "test.docx"
        binary = b"content"
        res = self.orchestrator.chunk(filename, binary)

        # Verify calls
        mock_router.route.assert_called_once()
        mock_general.chunk.assert_called_once()

        # Verify result
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "result")

    @patch("rag.orchestration.orchestrator.extract_embed_file")
    def test_chunk_with_embeds(self, mock_extract):
        # Simulate an embedded file
        mock_extract.return_value = [("embed.pdf", b"embed_content")]

        # Mock dependencies
        mock_router = self.mock_modules["rag.orchestration.router"].UniversalRouter
        mock_general = self.mock_modules["rag.templates.general"].General

        mock_router.route.return_value = self.ParseResult()
        mock_general.chunk.return_value = []

        # Reset counts for this test
        mock_router.route.reset_mock()

        self.orchestrator.chunk("root.docx", b"root_content")

        # Verify extract called
        # Verify 2 calls to route: one for the root document, and one for the embedded file.
        self.assertEqual(mock_router.route.call_count, 2)


class TestAdaptDoclingOutput(OrchestratorTestBase):
    """Tests for the adapt_docling_output() adapter function."""

    def test_adapt_string_input(self):
        """Test adapter with new string format (semantic mode)."""
        sections = "# Heading\n\nParagraph text here."
        tables = []
        parser_config = {"layout_recognizer": "Docling"}

        result = self.orchestrator.adapt_docling_output(sections, tables, parser_config)

        self.assertIsInstance(result, self.StandardizedDocument)
        self.assertEqual(result.content, sections)
        self.assertEqual(result.metadata["parser"], "docling")
        self.assertEqual(result.metadata["layout_recognizer"], "Docling")

    def test_adapt_with_tables(self):
        """Test adapter handling of tables."""
        sections = "# Header\nText"
        tables = [{"type": "table", "content": "| A | B |\n|---|---|\n| 1 | 2 |"}]
        parser_config = {"layout_recognizer": "Docling"}

        result = self.orchestrator.adapt_docling_output(sections, tables, parser_config)

        self.assertIsInstance(result, self.StandardizedDocument)
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


class TestSemanticRouting(OrchestratorTestBase):
    """Tests for semantic template routing in orchestrator.chunk()."""

    def setUp(self):
        super().setUp()
        # Helper to access mocks
        self.mock_router = self.mock_modules["rag.orchestration.router"].UniversalRouter
        self.mock_general = self.mock_modules["rag.templates.general"].General
        self.mock_semantic = self.mock_modules["rag.templates.semantic"].Semantic

    def test_legacy_path_with_list_sections(self):
        """Test that list sections still route to General template."""
        # Sections as list = legacy path
        self.mock_router.route.return_value = self.ParseResult(sections=["section1", "section2"], is_markdown=False)
        self.mock_general.chunk.return_value = [{"content": "general result"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": False}

        self.mock_general.chunk.reset_mock()

        # os.environ patch removed as requested
        self.orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because sections is a list, not string
        self.mock_general.chunk.assert_called_once()

    def test_semantic_path_with_string_sections(self):
        """Test that string sections with use_semantic_chunking routes to Semantic template."""
        # Sections as string = new semantic path
        self.mock_router.route.return_value = self.ParseResult(sections="# Heading\n\nContent", is_markdown=True)
        self.mock_semantic.chunk.return_value = [{"content": "semantic result", "header_path": "/Heading/"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": True}

        self.mock_semantic.chunk.reset_mock()

        # The decorator patch on rag.app.orchestrator.Semantic is sufficient
        # No need for redundant inner patch of rag.app.templates.semantic.Semantic
        res = self.orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use Semantic template
        self.mock_semantic.chunk.assert_called_once()

        # Verify the result contains the expected semantic chunk
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "semantic result")
        self.assertEqual(res[0]["header_path"], "/Heading/")

    def test_legacy_path_without_semantic_flag(self):
        """Test that without use_semantic_chunking flag, General is used."""
        self.mock_router.route.return_value = self.ParseResult(sections="# Heading\n\nContent")
        self.mock_general.chunk.return_value = [{"content": "general result"}]

        # No use_semantic_chunking flag
        parser_config = {"layout_recognizer": "Docling"}

        self.mock_general.chunk.reset_mock()

        res = self.orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because flag is not set
        self.mock_general.chunk.assert_called_once()
        self.assertEqual(res, [{"content": "general result"}])


if __name__ == "__main__":
    unittest.main()
