from unittest.mock import MagicMock, patch
from api.db.services.evaluation.dataset_service import EvaluationDatasetService
from common.constants import StatusEnum


class TestDatasetServiceLogic:
    @patch("api.db.services.evaluation.dataset_service.EvaluationDataset")
    def test_update_dataset_race_condition_fix(self, MockEvaluationDataset):
        """
        Verify that update_dataset adds the status check to the WHERE clause
        and does not check existence first (to avoid TOCTOU).
        """
        # Setup
        dataset_id = "ds_123"

        # Configure mock to return valid result for execute()
        mock_update = MagicMock()
        mock_where = MagicMock()

        # Setup class method return values
        MockEvaluationDataset.update.return_value = mock_update
        mock_update.where.return_value = mock_where
        mock_where.execute.return_value = 1

        # Execute
        result = EvaluationDatasetService.update_dataset(dataset_id, name="New Name")

        # Assert
        assert result is True

        # Verify get_or_none was NOT called (we removed it)
        MockEvaluationDataset.get_or_none.assert_not_called()

        # Verify update call structure
        MockEvaluationDataset.update.assert_called_once()
        mock_update.where.assert_called_once()

    @patch("api.db.services.evaluation.dataset_service.DB")
    @patch("api.db.services.evaluation.dataset_service.EvaluationCase")
    @patch("api.db.services.evaluation.dataset_service.EvaluationDataset")
    def test_delete_dataset_atomicity(self, MockEvaluationDataset, MockEvaluationCase, MockDB):
        """
        Verify delete_dataset uses a transaction and atomic block.
        """
        dataset_id = "ds_delete_123"

        # Reset mocks
        mock_atomic_ctx = MagicMock()
        MockDB.atomic.return_value = mock_atomic_ctx
        mock_atomic_ctx.__enter__.return_value = None
        mock_atomic_ctx.__exit__.return_value = None

        # Setup Update mocks
        MockEvaluationDataset.update.return_value.where.return_value.execute.return_value = 1
        MockEvaluationCase.update.return_value.where.return_value.execute.return_value = 5

        # Execute
        result = EvaluationDatasetService.delete_dataset(dataset_id)

        # Assert
        assert result is True

        # Verify transaction usage
        MockDB.atomic.assert_called_once()

        # Verify dataset update
        MockEvaluationDataset.update.assert_called_once()
        call_kwargs = MockEvaluationDataset.update.call_args[1]
        assert call_kwargs["status"] == StatusEnum.INVALID.value
        assert "update_time" in call_kwargs

        # Verify cascading case update
        MockEvaluationCase.update.assert_called_once()
        case_kwargs = MockEvaluationCase.update.call_args[1]
        assert case_kwargs["status"] == StatusEnum.INVALID.value
        assert "update_time" in case_kwargs

        # Verify timestamp consistency (same timestamp used)
        ds_time = call_kwargs["update_time"]
        case_time = case_kwargs["update_time"]
        assert ds_time == case_time

    @patch("api.db.services.evaluation.dataset_service.EvaluationCase")
    @patch("api.db.services.evaluation.dataset_service.EvaluationDataset")
    def test_import_validation_and_counting(self, MockEvaluationDataset, MockEvaluationCase):
        """
        Verify import_test_cases checks dataset validity and uses ID-based counting.
        """
        dataset_id = "ds_import_123"
        cases = [{"question": "q1", "reference_answer": "a1"}]

        # Mock dataset existence check
        MockEvaluationDataset.get_or_none.return_value = MagicMock(id=dataset_id, status=StatusEnum.VALID.value)

        # Mock bulk_create
        # EvaluationCase.bulk_create = MagicMock() # Handled by patch

        # Mock success count query
        # logic: EvaluationCase.select().where().count()
        mock_select = MagicMock()
        mock_select.where.return_value.count.return_value = 1
        MockEvaluationCase.select.return_value = mock_select

        # Execute
        s, f = EvaluationDatasetService.import_test_cases(dataset_id, cases)

        # Verify dataset check
        MockEvaluationDataset.get_or_none.assert_called_once()

        # Verify bulk_create called
        MockEvaluationCase.bulk_create.assert_called_once()

        # Verify counting
        assert s == 1
        assert f == 0
