import sys
from unittest.mock import MagicMock

# 1. Mock api.db.db_models BEFORE importing services
mock_db_models = MagicMock()
sys.modules['api.db.db_models'] = mock_db_models

# Configure DB.connection_context to be a passthrough decorator
def connection_context_decorator():
    def decorator(func):
        return func
    return decorator

# When @DB.connection_context() is called, it calls this side_effect, which returns the decorator
mock_db_models.DB.connection_context.side_effect = connection_context_decorator

# 2. Mock other problematic modules
sys.modules['api.utils.api_utils'] = MagicMock()
sys.modules['deepdoc'] = MagicMock()
sys.modules['deepdoc.parser'] = MagicMock()
sys.modules['deepdoc.parser.excel_parser'] = MagicMock()

from unittest.mock import patch

# 3. Import services AFTER mocking
from api.db.services.document_service import DocumentService
from api.db.services.task_service import TaskService
from common.constants import TaskStatus

@patch('api.db.services.document_service.DocumentService.update_by_id')
@patch('api.db.services.document_service.DocumentService.model')
@patch.object(TaskService, 'get_tasks_progress_by_doc_ids')
@patch('api.db.services.document_service.get_queue_length')
def test_sync_progress_batching(mock_get_queue_length, mock_get_tasks, mock_doc_model, mock_update_by_id):
    # Setup mocks
    mock_get_queue_length.return_value = 5

    # Test data
    doc1 = {
        "id": "doc_1",
        "process_begin_at": None,
        "run": TaskStatus.RUNNING.value,
        "progress": 0.1
    }
    doc2 = {
        "id": "doc_2",
        "process_begin_at": None,
        "run": TaskStatus.RUNNING.value,
        "progress": 0.2
    }
    docs = [doc1, doc2]

    # Mock tasks for doc1
    task1_1 = {
        "doc_id": "doc_1",
        "task_type": "normal",
        "progress": 0.5,
        "progress_msg": "Parsing...",
        "priority": 1
    }
    task1_2 = {
        "doc_id": "doc_1",
        "task_type": "normal",
        "progress": 1.0, # This one finished
        "progress_msg": "Done part 1",
        "priority": 1
    }

    # Mock tasks for doc2
    task2_1 = {
        "doc_id": "doc_2",
        "task_type": "normal",
        "progress": 0.8,
        "progress_msg": "Almost done",
        "priority": 2
    }

    # Return flattened list of tasks as get_tasks_progress_by_doc_ids would
    mock_get_tasks.return_value = [task1_1, task1_2, task2_1]

    # Mock update execute
    mock_doc_model.update.return_value.where.return_value.execute.return_value = 1

    # Run the method
    DocumentService._sync_progress(docs)

    # Assertions

    # 1. Verify get_tasks_progress_by_doc_ids called once with all doc IDs
    mock_get_tasks.assert_called_once()
    call_args = mock_get_tasks.call_args[0][0]
    assert set(call_args) == {"doc_1", "doc_2"}

    # 2. Verify model.update called for each doc (via the chain)
    # We expect 2 updates
    assert mock_doc_model.update.call_count == 2

    # Check update values for doc1
    # task1_1 (0.5) + task1_2 (1.0) = 1.5 / 2 = 0.75
    # We need to capture the calls
    calls = mock_doc_model.update.call_args_list

    found_doc1_update = False
    found_doc2_update = False

    for call in calls:
        args, _ = call
        info = args[0]
        if abs(info.get('progress', 0) - 0.75) < 0.001:
            found_doc1_update = True
        if abs(info.get('progress', 0) - 0.8) < 0.001: # task2_1 (0.8) / 1 = 0.8
            found_doc2_update = True

    assert found_doc1_update, "Did not find update for doc1 with expected progress 0.75"
    assert found_doc2_update, "Did not find update for doc2 with expected progress 0.8"

@patch.object(TaskService, 'get_tasks_progress_by_doc_ids')
def test_sync_progress_empty(mock_get_tasks):
    DocumentService._sync_progress([])
    mock_get_tasks.assert_not_called()
