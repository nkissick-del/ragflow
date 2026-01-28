import unittest
from unittest.mock import MagicMock, patch


class TestDatasetServiceLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need to be careful: mocking a parent as MagicMock makes it NOT a package
        # So we create ModuleType objects instead for parents

        # Patch sys.modules with essential mocks
        cls.mocks_patcher = patch.dict(
            "sys.modules",
            {
                "peewee": MagicMock(),
            },
        )
        cls.mocks_patcher.start()

        try:
            # Setup mock_utils mocks
            from test.mocks.mock_utils import setup_mocks

            setup_mocks()
        except Exception:
            cls.mocks_patcher.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.mocks_patcher.stop()

    def test_update_dataset_race_condition_fix(self):
        """
        Verify that update_dataset adds the status check to the WHERE clause
        and does not check existence first (to avoid TOCTOU).
        """
        from api.db.services.evaluation.dataset_service import EvaluationDatasetService

        with patch("api.db.services.evaluation.dataset_service.EvaluationDataset") as MockEvaluationDataset:
            dataset_id = "ds_123"
            mock_update = MagicMock()
            mock_where = MagicMock()

            MockEvaluationDataset.update.return_value = mock_update
            mock_update.where.return_value = mock_where
            mock_where.execute.return_value = 1

            # Configure the status field mock to return an expression with column="status"
            # when used in a comparison (peewee style)
            status_expression = MagicMock()
            status_expression.column = "status"
            # Ensure bitwise operations preserve the identity/info for our assertion
            status_expression.__and__.return_value = status_expression
            status_expression.__rand__.return_value = status_expression
            MockEvaluationDataset.status.__eq__.return_value = status_expression

            result = EvaluationDatasetService.update_dataset(dataset_id, name="New Name")

            self.assertTrue(result)
            MockEvaluationDataset.get_or_none.assert_not_called()
            MockEvaluationDataset.update.assert_called_once()

            # Verify WHERE clause includes the status condition
            mock_update.where.assert_called()
            # Safely guard for None
            call_args = mock_update.where.call_args
            where_args = call_args[0] if call_args else []
            self.assertGreater(len(where_args), 0)

            # Check if any of the where arguments involve "status"
            found_status = False
            for arg in where_args:
                # Direct check for configured attribute
                # Since we configured __and__/__rand__ to return status_expression,
                # arg might BE status_expression.
                if hasattr(arg, "column") and arg.column == "status":
                    found_status = True
                    break
                if hasattr(arg, "name") and arg.name == "status":
                    found_status = True
                    break
                # Check for bitwise AND combinations if they wrapped our status_expression
                if hasattr(arg, "lhs") and (getattr(arg.lhs, "column", None) == "status" or getattr(arg.lhs, "name", None) == "status"):
                    found_status = True
                    break
                if hasattr(arg, "rhs") and (getattr(arg.rhs, "column", None) == "status" or getattr(arg.rhs, "name", None) == "status"):
                    found_status = True
                    break

            self.assertTrue(found_status, "WHERE clause should include 'status' condition")


if __name__ == "__main__":
    unittest.main()
