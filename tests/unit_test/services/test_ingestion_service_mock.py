import unittest
from unittest.mock import MagicMock
import sys
import os
import types

# Adjust path to import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Mock dependencies before any imports
# Store original modules if they exist to restore later
_original_modules = {}
_MOCKED_MODULES = [
    "xxhash",
    "rag.nlp",
    "rag.utils",
    "rag.utils.redis_conn",
    "rag.app",
    "graphrag.general.mind_map_extractor",
    "api.db.db_models",
    "api.db.db_utils",
    "api.db.services.common_service",
    "api.db.services.document_service",
    "api.db.services.knowledgebase_service",
    "api.db.services.task_service",
    "api.db.services.file_service",
    "api.db.services.dialog_service",
    "api.db.services.conversation_service",
    "api.db.services.api_service",
    "api.db.services.llm_service",
    "api.db.services.user_service",
    "common.time_utils",
    "common.constants",
    "common.settings",
    "common.misc_utils",
    "common",
    "rag",
]

for mod in _MOCKED_MODULES:
    if mod in sys.modules:
        _original_modules[mod] = sys.modules[mod]

sys.modules["xxhash"] = MagicMock()
sys.modules["rag.nlp"] = MagicMock()
sys.modules["rag.utils"] = MagicMock()
sys.modules["rag.utils.redis_conn"] = MagicMock()
sys.modules["rag.app"] = MagicMock()
sys.modules["graphrag.general.mind_map_extractor"] = MagicMock()

# Mock internal DB and Service dependencies to avoid loading their heavy logic
sys.modules["api.db.db_models"] = MagicMock()
sys.modules["api.db.db_utils"] = MagicMock()
# Mocking these specific service modules so they are not loaded from disk
sys.modules["api.db.services.common_service"] = MagicMock()
sys.modules["api.db.services.document_service"] = MagicMock()
sys.modules["api.db.services.knowledgebase_service"] = MagicMock()
sys.modules["api.db.services.task_service"] = MagicMock()
sys.modules["api.db.services.file_service"] = MagicMock()
sys.modules["api.db.services.dialog_service"] = MagicMock()
sys.modules["api.db.services.conversation_service"] = MagicMock()
sys.modules["api.db.services.api_service"] = MagicMock()
sys.modules["api.db.services.llm_service"] = MagicMock()
sys.modules["api.db.services.user_service"] = MagicMock()

sys.modules["common.time_utils"] = MagicMock()
sys.modules["common.constants"] = MagicMock()
sys.modules["common.settings"] = MagicMock()
sys.modules["common.misc_utils"] = MagicMock()
sys.modules["common"] = MagicMock()

# Link mocks so 'from common import settings' gets the same object as sys.modules['common.settings']
sys.modules["common"].settings = sys.modules["common.settings"]
sys.modules["common"].misc_utils = sys.modules["common.misc_utils"]
sys.modules["common"].constants = sys.modules["common.constants"]
sys.modules["common"].time_utils = sys.modules["common.time_utils"]

# Link rag mocks
if "rag" not in sys.modules:
    sys.modules["rag"] = types.ModuleType("rag")
sys.modules["rag"].nlp = sys.modules["rag.nlp"]
sys.modules["rag"].utils = sys.modules["rag.utils"]

# Setup CommonService to be a class so IngestionService can inherit from it
sys.modules["api.db.services.common_service"].CommonService = MagicMock

# Setup DB atomic mock
mock_db = MagicMock()
mock_db.atomic.return_value.__enter__.return_value = None
mock_db.atomic.return_value.__exit__.return_value = None


# Ensure connection_context acts as a transparent decorator
def identity_decorator(func):
    return func


mock_db.connection_context.return_value = identity_decorator

sys.modules["api.db.db_models"].DB = mock_db

# Now import the class under test
# We rely on existing modules for 'api', 'api.db', 'api.db.services' to be loaded normally from disk/cache
# but the specific submodules we mocked above will be returned from sys.modules
try:
    from api.db.services.ingestion_service import IngestionService
except ImportError:
    # If this fails, we might need to be more aggressive or check pythonpath
    raise


class TestIngestionService(unittest.TestCase):
    def setUp(self):
        self.doc_id = "doc_123"
        self.kb_id = "kb_123"
        self.tenant_id = "tenant_123"
        mock_db.atomic.reset_mock()

        # Reset persistent mocks
        sys.modules["api.db.services.document_service"].DocumentService.reset_mock()
        sys.modules["api.db.services.task_service"].TaskService.reset_mock()
        sys.modules["api.db.services.knowledgebase_service"].KnowledgebaseService.reset_mock()
        sys.modules["common.settings"].reset_mock()

        # Mock TaskStatus with an object that has .value attributes
        self.mock_task_status = MagicMock()
        self.mock_task_status.RUNNING = MagicMock(value="1")
        self.mock_task_status.CANCEL = MagicMock(value="0")
        self.mock_task_status.DONE = MagicMock(value="2")

        # Inject our mock TaskStatus into common.constants
        sys.modules["common.constants"].TaskStatus = self.mock_task_status

        # Save original sys.modules keys to restore later

    def test_handle_run_transaction(self):
        """Test that delete/cancel operations happen inside the atomic block"""
        # Get mocks from sys.modules
        mock_doc_service = sys.modules["api.db.services.document_service"].DocumentService
        mock_task_service = sys.modules["api.db.services.task_service"].TaskService
        mock_settings = sys.modules["common.settings"]

        # Setup
        mock_doc_service.accessible.return_value = True
        mock_doc_service.accessible4deletion.return_value = True
        mock_doc_service.get_tenant_id.return_value = self.tenant_id

        mock_doc = MagicMock()
        mock_doc.kb_id = self.kb_id
        mock_doc.run = "2"  # DONE
        mock_doc.id = self.doc_id
        mock_doc.to_dict.return_value = {"id": self.doc_id, "kb_id": self.kb_id, "parser_id": "naive"}
        mock_doc_service.get_by_id.return_value = (True, mock_doc)

        mock_settings.docStoreConn.index_exist.return_value = True

        IngestionService.handle_run(
            doc_ids=[self.doc_id],
            run_status="1",  # RUNNING
            delete_flag=True,
            apply_kb_flag=False,
            user_id="user_1",
        )

        # Verification
        self.assertTrue(mock_db.atomic.called)

        # Verify calls
        mock_task_service.filter_delete.assert_called()
        mock_settings.docStoreConn.index_exist.assert_called()
        mock_settings.docStoreConn.delete.assert_called()
        mock_doc_service.update_by_id.assert_called()

    def test_doc_upload_null_dialog(self):
        """Test LookupError when Dialog is missing"""
        mock_conv_service = sys.modules["api.db.services.conversation_service"].ConversationService
        mock_dialog_service = sys.modules["api.db.services.dialog_service"].DialogService

        mock_conv_service.get_by_id.return_value = (True, MagicMock(dialog_id="d_1"))
        mock_dialog_service.get_by_id.return_value = (False, None)  # e=False

        with self.assertRaisesRegex(LookupError, "Dialog not found"):
            IngestionService.doc_upload_and_parse("conv_1", [], "user_1")


def tearDownModule():
    # Restore original modules
    for mod in _MOCKED_MODULES:
        if mod in _original_modules:
            sys.modules[mod] = _original_modules[mod]
        else:
            sys.modules.pop(mod, None)


if __name__ == "__main__":
    unittest.main()
