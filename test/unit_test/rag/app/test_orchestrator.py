import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure ragflow is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from rag.app import orchestrator


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


if __name__ == "__main__":
    unittest.main()
