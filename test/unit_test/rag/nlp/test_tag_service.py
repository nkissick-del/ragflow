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
        self.assertEqual(res, [("tag1", 10)])

    def test_tag_content(self):
        mock_store = MagicMock()
        mock_store.get_aggregation.return_value = [("tag1", 10)]

        service = TagService(mock_store, MagicMock())
        service.qryr.paragraph.return_value = "match_txt"

        doc = {"title_tks": "t", "content_ltks": "c"}
        all_tags = {"tag1": 0.01}

        service.tag_content("tenant1", ["kb1"], doc, all_tags, S=1)

        self.assertIn(TAG_FLD, doc)
        self.assertIn("tag1", doc[TAG_FLD])


if __name__ == "__main__":
    unittest.main()
