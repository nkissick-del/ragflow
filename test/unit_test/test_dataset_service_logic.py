import unittest
from unittest.mock import MagicMock, patch


class TestDatasetServiceLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need to be careful: mocking a parent as MagicMock makes it NOT a package
        # So we create ModuleType objects instead for parents

        # Patch sys.modules
        cls.mocks_patcher = patch.dict(
            "sys.modules",
            {
                "peewee": MagicMock(),
                "api.db.db_models": MagicMock(),
                "common": MagicMock(),
                "common.constants": MagicMock(),
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

            result = EvaluationDatasetService.update_dataset(dataset_id, name="New Name")

            self.assertTrue(result)
            MockEvaluationDataset.get_or_none.assert_not_called()
            MockEvaluationDataset.update.assert_called_once()

            # Verify WHERE clause includes the status condition
            # The exact implementation of the WHERE expression in peewee mocks can vary,
            # but we check that it's called with arguments.
            where_args = mock_update.where.call_args[0]
            self.assertGreater(len(where_args), 0)

            # Check if any of the where arguments involve "status"
            # In peewee, expressions often have a .column attribute or similar
            # Since we're using MagicMocks, we'll look for 'status' in the string representation
            # if the mock allows, or just check that multiple conditions are passed.
            found_status = False
            for arg in where_args:
                if "status" in str(arg).lower():
                    found_status = True
                    break
            self.assertTrue(found_status, "WHERE clause should include 'status' condition")


if __name__ == "__main__":
    unittest.main()
