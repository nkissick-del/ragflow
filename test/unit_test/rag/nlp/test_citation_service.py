import sys
import unittest
from unittest.mock import MagicMock


class TestCitationService(unittest.TestCase):
    MODULES_TO_MOCK = ["tiktoken", "rag.nlp.rag_tokenizer", "valkey", "valkey.lock", "common.settings"]

    @classmethod
    def setUpClass(cls):
        cls.original_modules = {}
        for mod in cls.MODULES_TO_MOCK:
            if mod in sys.modules:
                cls.original_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()

        # Import after mocking
        import rag.nlp.citation_service

        cls.CitationService = rag.nlp.citation_service.CitationService

    @classmethod
    def tearDownClass(cls):
        for mod, original in cls.original_modules.items():
            sys.modules[mod] = original

        # Remove mocks that weren't there before
        for mod in cls.MODULES_TO_MOCK:
            if mod not in cls.original_modules and mod in sys.modules:
                del sys.modules[mod]

    def test_insert_citations_empty(self):
        service = self.CitationService(MagicMock())
        ans, cites = service.insert_citations("answer", [], [], MagicMock())
        self.assertEqual(ans, "answer")
        self.assertEqual(cites, set())

    def test_insert_citations_basic(self):
        mock_qryr = MagicMock()
        mock_qryr.rmWWW.return_value = "chunk text"
        # Mock hybrid_similarity to return high similarity
        mock_qryr.hybrid_similarity.return_value = ([0.9], [0.9], [0.9])

        service = self.CitationService(mock_qryr)

        chunks = ["chunk1"]
        chunk_v = [[0.1, 0.2]]
        embd_mdl = MagicMock()
        # Mock encode to return matching dimension
        embd_mdl.encode.return_value = ([[0.1, 0.2]], None)

        # Mock tokenizer tokenize result
        sys.modules["rag.nlp.rag_tokenizer"].tokenize.return_value = "tokenized"

        ans, cites = service.insert_citations("This is a statement.", chunks, chunk_v, embd_mdl)

        # Expect citation [ID:0]
        self.assertIn("[ID:0]", ans)
        # Expect integer 0, not string "0"
        self.assertIn(0, cites)


if __name__ == "__main__":
    unittest.main()
