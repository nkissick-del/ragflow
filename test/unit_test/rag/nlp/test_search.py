import sys
import unittest
from unittest.mock import MagicMock, patch

_original_modules = {}


MODULES_TO_MOCK = ["tiktoken", "rag.nlp.rag_tokenizer", "rag.prompts.generator", "valkey", "valkey.lock", "common.settings"]


def setUpModule():
    for mod in MODULES_TO_MOCK:
        if mod in sys.modules:
            _original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = MagicMock()


def tearDownModule():
    for mod, original in _original_modules.items():
        sys.modules[mod] = original

    # Remove mocks for modules that weren't originally present
    for mod in MODULES_TO_MOCK:
        if mod not in _original_modules and mod in sys.modules:
            del sys.modules[mod]


class TestChunkList(unittest.TestCase):
    @patch("rag.nlp.search.query.FulltextQueryer", autospec=True)
    def test_chunk_list_termination(self, mock_queryer):
        from rag.nlp.search import Dealer

        # Mock dataStore
        mock_store = MagicMock()

        # Setup Dealer
        dealer = Dealer(mock_store)

        # Scenario:
        # Page 0: search returns hits, but get_fields returns empty (simulating filtering)
        # Page 1: search returns hits, get_fields returns chunks
        # Page 2: search returns empty (end of results)

        # Mock search results
        res_page_0 = {"hits": {"hits": [{"_id": "1"}]}}
        res_page_1 = {"hits": {"hits": [{"_id": "2"}]}}
        res_page_2 = {"hits": {"hits": []}}

        mock_store.search.side_effect = [res_page_0, res_page_1, res_page_2]

        # Mock get_fields
        mock_store.get_fields.side_effect = [{}, {"2": {"content_with_weight": "some content"}}, {}]

        def get_doc_ids_side_effect(res):
            return [d["_id"] for d in res["hits"]["hits"]]

        mock_store.get_doc_ids.side_effect = get_doc_ids_side_effect

        # Run chunk_list
        chunks = list(dealer.chunk_list("doc_id", "tenant_id", ["kb_id"], max_count=500))

        self.assertEqual(len(chunks), 1, "Should have retrieved chunks from the second page")

    def test_delegation_to_services(self):
        """Verify that Dealer delegates to the new services"""
        from rag.nlp.search import Dealer

        mock_store = MagicMock()

        dealer = Dealer(mock_store)

        # Mock services
        dealer.citation_service = MagicMock()
        dealer.rerank_service = MagicMock()
        dealer.tag_service = MagicMock()

        # Test insert_citations
        # Verify default parameters:
        # - insert_citations: tkweight=0.1, vtweight=0.9
        # - rerank: weights=0.3/0.7, chunk_id_name="content_ltks", match_ids=None
        # - rerank_by_model: weights=0.3/0.7, chunk_id_name="content_ltks", match_ids=None
        # - all_tags/all_tags_in_portion: num=1000
        # - tag_content: top_n=3, top_k=30, num=1000
        # - tag_query: top_n=3, num=1000
        ans = "ans"
        docs = []
        meta = []
        caller = MagicMock()
        dealer.insert_citations(ans, docs, meta, caller)
        dealer.citation_service.insert_citations.assert_called_once_with(ans, docs, meta, caller, 0.1, 0.9)

        # Test rerank
        query = "query"
        sres = MagicMock()
        dealer.rerank(sres, query)
        dealer.rerank_service.rerank.assert_called_once_with(sres, query, 0.3, 0.7, "content_ltks", None)

        # Test rerank_by_model
        model = MagicMock()
        dealer.rerank_by_model(model, sres, query)
        dealer.rerank_service.rerank_by_model.assert_called_once_with(model, sres, query, 0.3, 0.7, "content_ltks", None)

        # Test all_tags
        tid = "tid"
        kbs = ["kb"]
        dealer.all_tags(tid, kbs)
        dealer.tag_service.all_tags.assert_called_once_with(tid, kbs, 1000)

        # Test all_tags_in_portion
        dealer.all_tags_in_portion(tid, kbs)
        dealer.tag_service.all_tags_in_portion.assert_called_once_with(tid, kbs, 1000)

        # Test tag_content
        doc = {}
        all_tags = {}
        dealer.tag_content(tid, kbs, doc, all_tags)
        dealer.tag_service.tag_content.assert_called_once_with(tid, kbs, doc, all_tags, 3, 30, 1000)

        # Test tag_query
        dealer.tag_query(query, tid, kbs, all_tags)
        dealer.tag_service.tag_query.assert_called_once_with(query, tid, kbs, all_tags, 3, 1000)


if __name__ == "__main__":
    unittest.main()
