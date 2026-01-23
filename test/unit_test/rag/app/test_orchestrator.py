import unittest
from unittest.mock import patch
import sys
import os

# Ensure ragflow is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Mock dependencies to prevent heavy imports (DB, ML models, etc.)
# This allows running unit tests without installing the full environment.
from unittest.mock import MagicMock

_mock_modules = [
    "rag.app.format_parsers",
    "rag.app.router",
    "rag.app.templates.general",
    "rag.app.templates.semantic",
    "rag.nlp",
    "rag.utils.file_utils",
    "common",
    "common.settings",
    "api.db.services.llm_service",
]

module_mocks = {}
for mod_name in _mock_modules:
    module_mocks[mod_name] = MagicMock()
    sys.modules[mod_name] = module_mocks[mod_name]

# Setup specific mock attributes needed by orchestrator import
module_mocks["rag.app.format_parsers"].PARSERS = {}
# rag.nlp.rag_tokenizer is imported as `from rag.nlp import rag_tokenizer`
# We ensure rag.nlp has the attribute
module_mocks["rag.nlp"].rag_tokenizer = MagicMock()

from rag.app import orchestrator
from rag.app.standardized_document import StandardizedDocument


class TestOrchestrator(unittest.TestCase):
    @patch("rag.app.orchestrator.UniversalRouter")
    @patch("rag.app.orchestrator.General")
    def test_chunk(self, mock_general, mock_router):
        # Setup mocks
        mock_router.route.return_value = (["section1"], [], [], None, False, [])
        mock_general.chunk.return_value = [{"content": "result"}]

        # Test call
        filename = "test.docx"
        binary = b"content"
        res = orchestrator.chunk(filename, binary)

        # Verify calls
        mock_router.route.assert_called_once()
        mock_general.chunk.assert_called_once()

        # Verify result
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "result")

    @patch("rag.app.orchestrator.extract_embed_file")
    def test_chunk_with_embeds(self, mock_extract):
        # Simulate an embedded file
        mock_extract.return_value = [("embed.pdf", b"embed_content")]

        # Mock dependencies to avoid actual parsing
        with patch("rag.app.orchestrator.UniversalRouter") as mock_router, patch("rag.app.orchestrator.General") as mock_general:
            mock_router.route.return_value = ([], [], [], None, False, [])
            mock_general.chunk.return_value = []

            orchestrator.chunk("root.docx", b"root_content")

            # Verify extract called
            mock_extract.assert_called_once()
            # Verify recursive call happens (implied by extract_embed_file being called and loop running)
            # Since we mocked chunk's internals, we expect at least the root processing attempts.


class TestAdaptDoclingOutput(unittest.TestCase):
    """Tests for the adapt_docling_output() adapter function."""

    def test_adapt_string_input(self):
        """Test adapter with new string format (semantic mode)."""
        sections = "# Heading\n\nParagraph text here."
        tables = []
        parser_config = {"layout_recognizer": "Docling"}

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

        result = orchestrator.adapt_docling_output(sections, tables, parser_config)

        self.assertIsInstance(result, StandardizedDocument)
        # List should be joined with newlines
        self.assertEqual(result.content, "# Heading\nParagraph text here.")

    def test_adapt_empty_string(self):
        """Test adapter with empty string."""
        result = orchestrator.adapt_docling_output("", [], {})
        self.assertEqual(result.content, "")

    def test_adapt_empty_list(self):
        """Test adapter with empty list."""
        result = orchestrator.adapt_docling_output([], [], {})
        self.assertEqual(result.content, "")

    def test_adapter_preserves_elements_empty(self):
        """Test that elements list is empty (populated by semantic template)."""
        result = orchestrator.adapt_docling_output("test", [], {})
        self.assertEqual(result.elements, [])


class TestSemanticRouting(unittest.TestCase):
    """Tests for semantic template routing in orchestrator.chunk()."""

    @patch("rag.app.orchestrator.UniversalRouter")
    @patch("rag.app.orchestrator.General")
    def test_legacy_path_with_list_sections(self, mock_general, mock_router):
        """Test that list sections still route to General template."""
        # Sections as list = legacy path
        mock_router.route.return_value = (["section1", "section2"], [], [], None, False, [])
        mock_general.chunk.return_value = [{"content": "general result"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": True}

        # os.environ patch removed as requested
        orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because sections is a list, not string
        mock_general.chunk.assert_called_once()

    @patch("rag.app.orchestrator.UniversalRouter")
    @patch("rag.app.templates.semantic.Semantic")
    def test_semantic_path_with_string_sections(self, mock_semantic, mock_router):
        """Test that string sections with use_semantic_chunking routes to Semantic template."""
        # Sections as string = new semantic path
        mock_router.route.return_value = ("# Heading\n\nContent", [], [], None, False, [])
        mock_semantic.chunk.return_value = [{"content": "semantic result", "header_path": "/Heading/"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": True}

        # The decorator patch on rag.app.orchestrator.Semantic is sufficient
        # No need for redundant inner patch of rag.app.templates.semantic.Semantic
        res = orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use Semantic template
        mock_semantic.chunk.assert_called_once()

        # Verify the result contains the expected semantic chunk
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["content"], "semantic result")
        self.assertEqual(res[0]["header_path"], "/Heading/")

    @patch("rag.app.orchestrator.UniversalRouter")
    @patch("rag.app.orchestrator.General")
    def test_legacy_path_without_semantic_flag(self, mock_general, mock_router):
        """Test that without use_semantic_chunking flag, General is used."""
        mock_router.route.return_value = ("# Heading\n\nContent", [], [], None, False, [])
        mock_general.chunk.return_value = [{"content": "general result"}]

        # No use_semantic_chunking flag
        parser_config = {"layout_recognizer": "Docling"}

        res = orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because flag is not set
        mock_general.chunk.assert_called_once()
        self.assertEqual(res, [{"content": "general result"}])


if __name__ == "__main__":
    unittest.main()
