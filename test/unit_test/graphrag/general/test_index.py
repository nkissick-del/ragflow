import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before import to avoid environment issues with werkzeug/flask
sys.modules['api.db.services.document_service'] = MagicMock()
sys.modules['api.db.services.task_service'] = MagicMock()
sys.modules['common.settings'] = MagicMock()
sys.modules['api.db.services.knowledgebase_service'] = MagicMock()
sys.modules['api.utils.api_utils'] = MagicMock()
sys.modules['common.connection_utils'] = MagicMock()
sys.modules['rag.nlp'] = MagicMock()
sys.modules['rag.utils.redis_conn'] = MagicMock()
sys.modules['graphrag.entity_resolution'] = MagicMock()
sys.modules['graphrag.general.community_reports_extractor'] = MagicMock()
sys.modules['graphrag.general.extractor'] = MagicMock()
sys.modules['graphrag.general.graph_extractor'] = MagicMock()
sys.modules['graphrag.light.graph_extractor'] = MagicMock()
sys.modules['graphrag.utils'] = MagicMock()
sys.modules['common.token_utils'] = MagicMock()
sys.modules['common.token_utils'].num_tokens_from_string.return_value = 100

# Setup timeout decorator mock
def mock_timeout_decorator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

sys.modules['common.connection_utils'].timeout = mock_timeout_decorator

import unittest
from unittest.mock import patch, AsyncMock
import asyncio

# Now import the module under test
from graphrag.general.index import run_graphrag_for_kb

class TestGraphRagIndex(unittest.IsolatedAsyncioTestCase):
    @patch('graphrag.general.index.settings')
    @patch('graphrag.general.index.DocumentService')
    @patch('graphrag.general.index.RedisDistributedLock')
    @patch('graphrag.general.index.generate_subgraph', new_callable=AsyncMock)
    @patch('graphrag.general.index.merge_subgraph', new_callable=AsyncMock)
    @patch('graphrag.general.index.resolve_entities', new_callable=AsyncMock)
    @patch('graphrag.general.index.extract_community', new_callable=AsyncMock)
    @patch('graphrag.general.index.has_canceled', return_value=False)
    @patch('graphrag.general.index.does_graph_contains', new_callable=AsyncMock)
    async def test_run_graphrag_for_kb_chunk_loading(
        self, mock_contains, mock_canceled, mock_extract, mock_resolve, mock_merge,
        mock_gen_subgraph, mock_lock, mock_doc_service, mock_settings
    ):
        # Mock inputs
        row = {
            "tenant_id": "tenant_1",
            "kb_id": "kb_1",
            "id": "task_1"
        }
        doc_ids = ["doc_1"]
        language = "English"
        kb_parser_config = {"graphrag": {"method": "general"}}
        chat_model = MagicMock()
        embedding_model = MagicMock()
        callback = MagicMock()

        # Mock settings.retriever.chunk_list to return a generator
        def mock_chunk_list_gen(*args, **kwargs):
            yield {"content_with_weight": "chunk1"}
            yield {"content_with_weight": "chunk2"}

        mock_settings.retriever.chunk_list.side_effect = mock_chunk_list_gen

        # Mock Redis lock
        mock_lock_instance = MagicMock()
        mock_lock_instance.spin_acquire = AsyncMock()
        mock_lock.return_value = mock_lock_instance

        # Mock subgraphs
        mock_gen_subgraph.return_value = MagicMock(nodes=lambda: ["node1"])
        mock_merge.return_value = MagicMock()
        mock_contains.return_value = False

        # Run the function
        result = await run_graphrag_for_kb(
            row, doc_ids, language, kb_parser_config,
            chat_model, embedding_model, callback,
            with_resolution=False, with_community=False
        )

        # Verification
        # Check if chunk_list was called with max_count=sys.maxsize
        mock_settings.retriever.chunk_list.assert_called()
        args, kwargs = mock_settings.retriever.chunk_list.call_args
        self.assertEqual(kwargs['max_count'], sys.maxsize)
        self.assertEqual(args[0], "doc_1")

        # Check result
        # Total chunks should be 1 because "chunk1" and "chunk2" are combined (token count 100 < 4096)
        self.assertEqual(result['total_chunks'], 1)
        self.assertEqual(result['total_docs'], 1)
        self.assertEqual(result['ok_docs'], ["doc_1"])

if __name__ == '__main__':
    unittest.main()
