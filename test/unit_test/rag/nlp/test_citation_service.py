import sys
from unittest.mock import MagicMock

sys.modules["tiktoken"] = MagicMock()
sys.modules["rag.nlp.rag_tokenizer"] = MagicMock()
sys.modules["valkey"] = MagicMock()
sys.modules["valkey.lock"] = MagicMock()
sys.modules["common.settings"] = MagicMock()

import unittest
from rag.nlp.citation_service import CitationService


class TestCitationService(unittest.TestCase):
    def test_insert_citations_empty(self):
        service = CitationService(MagicMock())
        ans, cites = service.insert_citations("answer", [], [], MagicMock())
        self.assertEqual(ans, "answer")
        self.assertEqual(cites, set())

    def test_insert_citations_basic(self):
        mock_qryr = MagicMock()
        mock_qryr.rmWWW.return_value = "chunk text"
        # Mock hybrid_similarity to return high similarity
        mock_qryr.hybrid_similarity.return_value = ([0.9], [0.9], [0.9])

        service = CitationService(mock_qryr)

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
        self.assertIn("0", cites)


if __name__ == "__main__":
    unittest.main()
