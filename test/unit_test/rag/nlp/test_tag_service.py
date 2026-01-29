import sys
from unittest.mock import MagicMock

sys.modules["tiktoken"] = MagicMock()
sys.modules["rag.nlp.rag_tokenizer"] = MagicMock()
sys.modules["valkey"] = MagicMock()
sys.modules["valkey.lock"] = MagicMock()
sys.modules["common.settings"] = MagicMock()

import unittest
from rag.nlp.tag_service import TagService
from common.constants import TAG_FLD


class TestTagService(unittest.TestCase):
    def test_all_tags(self):
        mock_store = MagicMock()
        mock_store.index_exist.return_value = True
        mock_store.get_aggregation.return_value = [("tag1", 10)]

        service = TagService(mock_store, MagicMock())

        res = service.all_tags("tenant1", ["kb1"])

        mock_store.search.assert_called_once()
        args, kwargs = mock_store.search.call_args
        # signature: search(selectFields, highlightFields, condition, matchText, orderBy, offset, limit, indexName, knowledgebaseIds, aggFields)
        # Using a safer assertion for indexName (arg 7) and knowledgebaseIds (arg 8)
        self.assertGreaterEqual(len(args), 9)
        self.assertEqual(args[7], "ragflow_tenant1")  # indexName
        self.assertEqual(args[8], ["kb1"])  # knowledgebaseIds
        self.assertEqual(res, [("tag1", 10)])

    def test_all_tags_empty(self):
        mock_store = MagicMock()
        service = TagService(mock_store, MagicMock())

        # Empty kb_ids
        self.assertEqual(service.all_tags("tenant1", []), [])
        mock_store.search.assert_not_called()

        # index not exist
        mock_store.index_exist.return_value = False
        self.assertEqual(service.all_tags("tenant1", ["kb1"]), [])
        # Should verify search not called when index doesn't exist
        mock_store.search.assert_not_called()

    def test_tag_content(self):
        mock_store = MagicMock()
        mock_store.get_aggregation.return_value = [("tag1", 10)]

        mock_qryr = MagicMock()
        mock_qryr.paragraph.return_value = "match_txt"

        service = TagService(mock_store, mock_qryr)

        doc = {"title_tks": "t", "content_ltks": "c"}
        all_tags = {"tag1": 0.0001}  # Use a valid value to avoid div0 if that was an issue, though logic handles it.

        service.tag_content("tenant1", ["kb1"], doc, all_tags, S=1)

        self.assertIn(TAG_FLD, doc)
        self.assertIn("tag1", doc[TAG_FLD])

    def test_tag_content_edge_cases(self):
        mock_store = MagicMock()
        mock_qryr = MagicMock()
        service = TagService(mock_store, mock_qryr)

        # Case: No aggregations found
        mock_store.get_aggregation.return_value = []
        doc = {"title_tks": "t", "content_ltks": "c"}
        self.assertFalse(service.tag_content("tenant1", ["kb1"], doc, {}))

        # Case: Tags with 0 score (filtered out)
        mock_store.get_aggregation.return_value = [("tag1", 0)]
        doc = {"title_tks": "t", "content_ltks": "c"}  # Fresh doc

        # Call with HIGH smoothing to force low score
        success = service.tag_content("tenant1", ["kb1"], doc, {"tag1": 0.1}, S=10000000)
        self.assertTrue(success)

        # Assert tag1 is NOT in doc because it should be filtered (score < 0.001)
        if TAG_FLD in doc:
            self.assertNotIn("tag1", doc[TAG_FLD])


if __name__ == "__main__":
    unittest.main()
