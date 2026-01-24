
import unittest
from unittest.mock import MagicMock, patch
import sys

class TestKbPaginationOptimization(unittest.TestCase):
    def test_get_all_kb_by_tenant_ids_optimization(self):
        # Prepare mocks
        mock_peewee = MagicMock()
        mock_db_models = MagicMock()
        mock_db = MagicMock()

        def pass_through_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

        mock_db.connection_context.side_effect = pass_through_decorator
        mock_db_models.DB = mock_db
        mock_db_models.Knowledgebase = MagicMock()
        mock_db_models.Document = MagicMock()

        mock_settings = MagicMock()

        # Context manager to patch sys.modules safely
        modules_to_patch = {
            'peewee': mock_peewee,
            'api.db.db_models': mock_db_models,
            'common.settings': mock_settings
        }

        with patch.dict(sys.modules, modules_to_patch):
            # We must remove the module from sys.modules if it's already there,
            # to force re-import with our mocks.
            if 'api.db.services.knowledgebase_service' in sys.modules:
                del sys.modules['api.db.services.knowledgebase_service']

            # Now import inside the patched environment
            # Note: This import relies on the mocked modules above.
            from api.db.services.knowledgebase_service import KnowledgebaseService

            # Patch the model attribute on the service class
            with patch.object(KnowledgebaseService, 'model') as MockModel:
                mock_query = MagicMock()
                mock_query._name = "MockQuery"
                MockModel.select.return_value = mock_query
                mock_query.where.return_value = mock_query

                # Setup mocks for methods that should be called
                mock_query.order_by.return_value = mock_query

                # Setup mocks for methods that should NOT be called
                mock_query.offset.return_value = mock_query
                mock_query.limit.return_value = mock_query

                # Mock dicts() to return a list
                mock_query.dicts.return_value = []

                KnowledgebaseService.get_all_kb_by_tenant_ids(["t1"], "u1")

                # Verify offset/limit are NOT called
                self.assertFalse(mock_query.offset.called, "offset() should not be called in optimized version")
                self.assertFalse(mock_query.limit.called, "limit() should not be called in optimized version")

                # Verify order_by is called
                self.assertTrue(mock_query.order_by.called, "order_by() should be called")

        # Outside the context manager, sys.modules is restored.
        # But `api.db.services.knowledgebase_service` might still be in sys.modules
        # referring to the version imported with mocks.
        # We clean it up to avoid polluting subsequent tests.
        if 'api.db.services.knowledgebase_service' in sys.modules:
            del sys.modules['api.db.services.knowledgebase_service']

if __name__ == '__main__':
    unittest.main()
