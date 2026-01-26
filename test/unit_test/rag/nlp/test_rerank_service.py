import sys
from unittest.mock import MagicMock

sys.modules["tiktoken"] = MagicMock()
sys.modules["rag.nlp.rag_tokenizer"] = MagicMock()
sys.modules["rag.prompts.generator"] = MagicMock()
sys.modules["valkey"] = MagicMock()
sys.modules["valkey.lock"] = MagicMock()
sys.modules["common.settings"] = MagicMock()

import unittest
from rag.nlp.rerank_service import RerankService
from rag.nlp.search import SearchResult


class TestRerankService(unittest.TestCase):
    def test_rerank(self):
        mock_qryr = MagicMock()
        # Mock hybrid_similarity returning (sim, tksim, vtsim)
        mock_qryr.hybrid_similarity.return_value = ([0.8], [0.7], [0.9])
        mock_qryr.question.return_value = ("match", ["keyword"])

        service = RerankService(mock_qryr)

        sres = SearchResult(total=1, ids=["doc1"], query_vector=[0.1, 0.2], field={"doc1": {"content_ltks": "content", "title_tks": "title", "q_2_vec": [0.1, 0.2]}})

        sim, tksim, vtsim = service.rerank(sres, "query")

        mock_qryr.hybrid_similarity.assert_called_once()
        self.assertEqual(len(sim), 1)


if __name__ == "__main__":
    unittest.main()
