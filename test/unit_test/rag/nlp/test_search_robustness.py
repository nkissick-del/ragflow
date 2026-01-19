
import unittest
from unittest.mock import MagicMock, patch
from rag.nlp.search import Dealer

class TestChunkListRobustness(unittest.TestCase):
    @patch('rag.nlp.search.query.FulltextQueryer')
    def test_empty_search(self, mock_queryer):
        """Test that the loop terminates immediately if search returns no hits."""
        mock_store = MagicMock()
        dealer = Dealer(mock_store)

        # Search returns empty hits immediately
        mock_store.search.return_value = {"hits": {"hits": []}}
        mock_store.get_fields.return_value = {}
        mock_store.get_doc_ids.return_value = []

        chunks = dealer.chunk_list("doc_id", "tenant_id", ["kb_id"], max_count=1024)

        self.assertEqual(len(chunks), 0)
        # Should have called search once
        mock_store.search.assert_called_once()

    @patch('rag.nlp.search.query.FulltextQueryer')
    def test_filtered_hits_continue_loop(self, mock_queryer):
        """Test that the loop continues if hits exist but get_fields filters them out."""
        mock_store = MagicMock()
        dealer = Dealer(mock_store)

        # Page 1: 2 hits, but filtered out (get_fields returns empty)
        res_page_1 = {"hits": {"hits": [{"_id": "1"}, {"_id": "2"}]}}
        # Page 2: 1 hit, kept
        res_page_2 = {"hits": {"hits": [{"_id": "3"}]}}
        # Page 3: 0 hits (end)
        res_page_3 = {"hits": {"hits": []}}

        mock_store.search.side_effect = [res_page_1, res_page_2, res_page_3]

        # get_fields side effect
        mock_store.get_fields.side_effect = [
            {}, # Page 1 filtered out
            {"3": {"content_with_weight": "content3"}}, # Page 2 kept
            {}
        ]

        # get_doc_ids side effect
        mock_store.get_doc_ids.side_effect = [
            ["1", "2"],
            ["3"],
            []
        ]

        chunks = dealer.chunk_list("doc_id", "tenant_id", ["kb_id"], max_count=1024)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["id"], "3")
        # Should have called search 3 times
        self.assertEqual(mock_store.search.call_count, 3)

    @patch('rag.nlp.search.query.FulltextQueryer')
    def test_pagination_end(self, mock_queryer):
        """Test standard pagination termination."""
        mock_store = MagicMock()
        dealer = Dealer(mock_store)

        # BS is 128. Let's simulate getting exactly 1 hit per page for 2 pages, then 0.
        res_page_1 = {"hits": {"hits": [{"_id": "1"}]}}
        res_page_2 = {"hits": {"hits": [{"_id": "2"}]}}
        res_page_3 = {"hits": {"hits": []}}

        mock_store.search.side_effect = [res_page_1, res_page_2, res_page_3]

        mock_store.get_fields.side_effect = [
            {"1": {"content": "c1"}},
            {"2": {"content": "c2"}},
            {}
        ]

        mock_store.get_doc_ids.side_effect = [
            ["1"],
            ["2"],
            []
        ]

        chunks = dealer.chunk_list("doc_id", "tenant_id", ["kb_id"], max_count=1024)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(mock_store.search.call_count, 3)

    @patch('rag.nlp.search.query.FulltextQueryer')
    def test_max_count_reached(self, mock_queryer):
        """Test that loop stops when max_count is reached (by range loop)."""
        mock_store = MagicMock()
        dealer = Dealer(mock_store)

        # Max count = 200. BS = 128.
        # Range will generate: 0, 128. (Stop at 256 which is > 200? No, range(0, 200, 128) -> [0, 128])
        # So it should call search twice.

        res_page = {"hits": {"hits": [{"_id": "x"}]}} # Always returns hits
        mock_store.search.return_value = res_page
        mock_store.get_fields.return_value = {"x": {"content": "x"}}
        mock_store.get_doc_ids.return_value = ["x"]

        chunks = dealer.chunk_list("doc_id", "tenant_id", ["kb_id"], max_count=200)

        # Should call search exactly 2 times (offset 0 and 128)
        self.assertEqual(mock_store.search.call_count, 2)
        self.assertEqual(len(chunks), 2)

if __name__ == '__main__':
    unittest.main()
