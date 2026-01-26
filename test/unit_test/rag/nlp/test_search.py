import sys
from unittest.mock import MagicMock

# Mock rag_tokenizer module to avoid tiktoken PermissionError
# We enter it into sys.modules so it's returning a mock when imported.
sys.modules["tiktoken"] = MagicMock()
sys.modules["rag.nlp.rag_tokenizer"] = MagicMock()
sys.modules["rag.prompts.generator"] = MagicMock()
sys.modules["valkey"] = MagicMock()
sys.modules["valkey.lock"] = MagicMock()
sys.modules["common.settings"] = MagicMock()

import unittest
from unittest.mock import MagicMock, patch
from rag.nlp.search import Dealer


class TestChunkList(unittest.TestCase):
    @patch("rag.nlp.search.query.FulltextQueryer")
    def test_chunk_list_termination(self, mock_queryer):
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
        mock_store = MagicMock()
        dealer = Dealer(mock_store)

        # Mock services
        dealer.citation_service = MagicMock()
        dealer.rerank_service = MagicMock()
        dealer.tag_service = MagicMock()

        # Test insert_citations
        dealer.insert_citations("ans", [], [], MagicMock())
        dealer.citation_service.insert_citations.assert_called_once()

        # Test rerank
        dealer.rerank(MagicMock(), "query")
        dealer.rerank_service.rerank.assert_called_once()

        # Test rerank_by_model
        dealer.rerank_by_model(MagicMock(), MagicMock(), "query")
        dealer.rerank_service.rerank_by_model.assert_called_once()

        # Test all_tags
        dealer.all_tags("tid", ["kb"])
        dealer.tag_service.all_tags.assert_called_once()

        # Test all_tags_in_portion
        dealer.all_tags_in_portion("tid", ["kb"])
        dealer.tag_service.all_tags_in_portion.assert_called_once()

        # Test tag_content
        dealer.tag_content("tid", ["kb"], {}, {})
        dealer.tag_service.tag_content.assert_called_once()

        # Test tag_query
        dealer.tag_query("q", "tid", ["kb"], {})
        dealer.tag_service.tag_query.assert_called_once()


if __name__ == "__main__":
    unittest.main()
