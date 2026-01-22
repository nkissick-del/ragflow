import unittest
from unittest.mock import patch
import sys
import os

# Ensure ragflow is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

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

            res = orchestrator.chunk("root.docx", b"root_content")

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

        with patch.dict("os.environ", {}):
            res = orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use General because sections is a list, not string
        mock_general.chunk.assert_called_once()

    @patch("rag.app.orchestrator.UniversalRouter")
    @patch("rag.app.orchestrator.Semantic")
    def test_semantic_path_with_string_sections(self, mock_semantic, mock_router):
        """Test that string sections with use_semantic_chunking routes to Semantic template."""
        # Sections as string = new semantic path
        mock_router.route.return_value = ("# Heading\n\nContent", [], [], None, False, [])
        mock_semantic.chunk.return_value = [{"content": "semantic result", "header_path": "/Heading/"}]

        parser_config = {"layout_recognizer": "Docling", "use_semantic_chunking": True}

        with patch("rag.app.templates.semantic.Semantic", mock_semantic):
            res = orchestrator.chunk("test.pdf", b"content", parser_config=parser_config)

        # Should use Semantic template
        mock_semantic.chunk.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
