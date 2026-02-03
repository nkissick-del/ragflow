from unittest.mock import MagicMock, patch
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.db_models import DB


class TestKnowledgebaseServicePerf:
    @patch.object(DB, "connect")
    @patch.object(DB, "close")
    @patch.object(KnowledgebaseService.model, "select")
    def test_get_all_kb_by_tenant_ids_pagination(self, mock_select, mock_close, mock_connect):
        # Arrange
        tenant_id = "tenant-1"
        user_id = "user-1"

        # Mock the query chain
        mock_query = MagicMock()
        mock_select.return_value = mock_query
        mock_query.where.return_value = mock_query

        # Mock order_by return value
        mock_sorted_query = MagicMock()
        mock_sorted_query.dicts.return_value = [{"id": "1"}]  # The result
        mock_query.order_by.return_value = mock_sorted_query

        # Mock kbs (the result of where())
        mock_kbs = mock_query

        # We don't expect offset or limit to be called anymore,
        # so we don't strictly need to mock their side effects for success,
        # but the assertion will check they are NOT called.

        # Act
        res = KnowledgebaseService.get_all_kb_by_tenant_ids([tenant_id], user_id)

        # Assert
        assert len(res) == 1
        assert res[0]["id"] == "1"

        # Verify offset was NOT called (proving the loop is gone)
        assert mock_kbs.offset.call_count == 0
        assert mock_kbs.limit.call_count == 0

        # Verify correct chaining
        mock_query.order_by.assert_called_once()
        mock_sorted_query.dicts.assert_called_once()
