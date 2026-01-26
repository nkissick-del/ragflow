import sys
import unittest
from unittest.mock import MagicMock, patch


class TestRerankService(unittest.TestCase):
    def setUp(self):
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "tiktoken": MagicMock(),
                "rag.nlp.rag_tokenizer": MagicMock(),
                "rag.prompts.generator": MagicMock(),
                "valkey": MagicMock(),
                "valkey.lock": MagicMock(),
                "common.settings": MagicMock(),
            },
        )
        self.modules_patcher.start()
        # Re-import to ensure mocks are used
        global RerankService, SearchResult
        from rag.nlp.rerank_service import RerankService
        from rag.nlp.search import SearchResult

    def tearDown(self):
        self.modules_patcher.stop()

    def test_rerank(self):
        mock_qryr = MagicMock()
        # Mock hybrid_similarity returning (sim, tksim, vtsim)
        mock_qryr.hybrid_similarity.return_value = ([0.8], [0.7], [0.9])
        mock_qryr.question.return_value = ("match", ["keyword"])

        service = RerankService(mock_qryr)

        sres = SearchResult(total=1, ids=["doc1"], query_vector=[0.1, 0.2], field={"doc1": {"content_ltks": "content", "title_tks": "title", "q_2_vec": [0.1, 0.2]}})

        sim, tksim, vtsim = service.rerank(sres, "query")

        # Verify question call
        mock_qryr.question.assert_called_once_with("query")

        # Verify hybrid_similarity call arguments
        # args: query_vector, ins_embd, keywords, ins_tw, tkweight, vtweight
        # ins_embd should be [[0.1, 0.2]]
        # ins_tw construction depends on fields.
        # Content "content" -> ["content"]
        # Title "title" -> ["title"] (x2)
        # Important kwd [] -> [] (x5)
        # Question tks [] -> [] (x6)
        # Total tks list: ["content", "title", "title"]

        args, _ = mock_qryr.hybrid_similarity.call_args
        self.assertEqual(len(args), 6)
        self.assertEqual(args[0], [0.1, 0.2])  # query_vector
        self.assertEqual(args[1], [[0.1, 0.2]])  # ins_embd
        self.assertEqual(args[2], ["keyword"])  # keywords
        self.assertEqual(args[3], [["content", "title", "title"]])  # ins_tw

        # Verify output
        # sim = hybrid_sim + rank_fea
        # rank_fea default when no tag_fea is 0 + pagerank(0)
        # so sim should be 0.8
        self.assertEqual(len(sim), 1)
        self.assertEqual(sim[0], 0.8)
        self.assertEqual(tksim, [0.7])
        self.assertEqual(vtsim, [0.9])


if __name__ == "__main__":
    unittest.main()
