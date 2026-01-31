import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Mock dependencies before any imports
_original_modules = {}
_MOCKED_MODULES = [
    "peewee",
    "anthropic",
    "api.db.db_models",
    "api.db.services.common_service",
    "api.db.services.document_service",
    "api.db.services.doc_metadata_service",
    "api.db.services.user_service",
    "common.misc_utils",
    "common.constants",
    "common.time_utils",
    "common.exceptions",
]

for mod in _MOCKED_MODULES:
    if mod in sys.modules:
        _original_modules[mod] = sys.modules[mod]
    sys.modules[mod] = MagicMock()

# Setup peewee and anthropic properly for imports
import types

peewee_mock = types.ModuleType("peewee")
peewee_mock.SQL = MagicMock()
peewee_mock.fn = MagicMock()
sys.modules["peewee"] = peewee_mock

anthropic_mock = types.ModuleType("anthropic")
anthropic_mock.BaseModel = MagicMock()
sys.modules["anthropic"] = anthropic_mock


# Setup CommonService to be a class
class MockCommonService:
    @classmethod
    def query(cls, *args, **kwargs):
        pass

    @classmethod
    def save(cls, *args, **kwargs):
        pass

    @classmethod
    def filter_update(cls, *args, **kwargs):
        pass

    @classmethod
    def update_by_id(cls, *args, **kwargs):
        pass


sys.modules["api.db.services.common_service"].CommonService = MockCommonService


# Setup TaskStatus
class MockTaskStatus:
    SCHEDULE = "SCHEDULE"
    RUNNING = "RUNNING"
    CANCEL = "CANCEL"


sys.modules["common.constants"].TaskStatus = MockTaskStatus


# Setup ConnectorError
class ConnectorError(Exception):
    def __init__(self, msg):
        self.msg = msg


sys.modules["common.exceptions"].ConnectorError = ConnectorError

# Now import the service
# We need to refresh the module to pick up the mocks if it was already imported elsewhere
# in some contexts, but here it's fresh.
from api.db.services.connector_service import Connector2KbService


class TestConnectorService(unittest.TestCase):
    def setUp(self):
        sys.modules["common.misc_utils"].get_uuid.return_value = "uuid_123"

    @patch("api.db.services.connector_service.SyncLogsService")
    def test_link_connectors_reordering(self, mock_sync_logs_service):
        """Test that cancellation happens before scheduling new task"""
        # Setup
        kb_id = "kb_1"
        tenant_id = "tenant_1"
        connectors = [{"id": "conn_1", "auto_parse": "1"}]

        # Mock class methods
        mock_sync_logs_service.filter_update = MagicMock()
        mock_sync_logs_service.schedule = MagicMock()

        with patch.object(Connector2KbService, "query") as mock_query, patch.object(Connector2KbService, "save") as mock_save:
            mock_query.return_value = []  # no old connectors

            # Use a manager to track call order across multiple mocks
            manager = MagicMock()
            manager.attach_mock(mock_sync_logs_service.filter_update, "filter_update")
            manager.attach_mock(mock_save, "save")
            manager.attach_mock(mock_sync_logs_service.schedule, "schedule")

            Connector2KbService.link_connectors(kb_id, connectors, tenant_id)

            # Expected order of operations:
            # 1. filter_update (to cancel old ones)
            # 2. save (to link connector to kb)
            # 3. schedule (to start sync)

            relevant_calls = [c for c in manager.mock_calls if any(x in c[0] for x in ["filter_update", "save", "schedule"])]

            self.assertEqual(len(relevant_calls), 3)
            self.assertIn("filter_update", relevant_calls[0][0])
            self.assertIn("save", relevant_calls[1][0])
            self.assertIn("schedule", relevant_calls[2][0])

            # Check arguments for filter_update specifically for cancellation
            mock_sync_logs_service.filter_update.assert_called_with(unittest.mock.ANY, {"status": MockTaskStatus.CANCEL})

    def test_link_connectors_exception_mapping(self):
        """Test that ValueError is caught and wrapped in ConnectorError"""
        with patch.object(Connector2KbService, "query") as mock_query:
            mock_query.side_effect = ValueError("specific_error")

            with self.assertRaises(ConnectorError) as cm:
                Connector2KbService.link_connectors("kb_1", [{"id": "c1"}], "t1")

            self.assertEqual(cm.exception.msg, "specific_error")

    def test_link_connectors_unexpected_exception(self):
        """Test that unexpected exceptions are re-raised"""
        with patch.object(Connector2KbService, "query") as mock_query:
            mock_query.side_effect = RuntimeError("unexpected")

            with self.assertRaises(RuntimeError):
                Connector2KbService.link_connectors("kb_1", [{"id": "c1"}], "t1")


if __name__ == "__main__":
    unittest.main()
