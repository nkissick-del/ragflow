
import unittest
from unittest.mock import MagicMock, patch
import json
import csv
import io
import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Mock api.db.db_models before importing EvaluationService
# This needs to be done before importing CommonService as well
sys.modules["api.db.db_models"] = MagicMock()

# Setup DB mock to support connection_context decorator
mock_db = MagicMock()
# When @DB.connection_context() is used, it calls __call__ on the return value of connection_context()
# So we need DB.connection_context() -> returns decorator -> decorator(func) -> returns wrapped func
def mock_decorator(func):
    return func
mock_db.connection_context.return_value = mock_decorator
sys.modules["api.db.db_models"].DB = mock_db

# Mock api.db.services.dialog_service to avoid import chain issues
sys.modules["api.db.services.dialog_service"] = MagicMock()

# Now import the service
# We need to ensure we don't import the real DialogService inside EvaluationService if it does so
# But sys.modules mock should handle it.

from api.db.services.evaluation_service import EvaluationService
from api.db.db_models import EvaluationDataset, EvaluationCase, EvaluationRun, EvaluationResult

class TestEvaluationCSVExport(unittest.TestCase):

    @patch("api.db.services.evaluation_service.EvaluationRun")
    @patch("api.db.services.evaluation_service.EvaluationResult")
    @patch("api.db.services.evaluation_service.EvaluationCase")
    def test_get_run_results_csv(self, MockEvaluationCase, MockEvaluationResult, MockEvaluationRun):
        # Setup run
        run_id = "run_123"
        mock_run = MagicMock()
        MockEvaluationRun.get_by_id.return_value = mock_run

        # Setup results
        # We need to mock the query chain: select -> join -> where -> order_by
        mock_query = MagicMock()
        MockEvaluationResult.select.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.where.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        # Create mock result items
        # Item 1
        item1 = MagicMock()
        item1.to_dict.return_value = {
            "id": "res_1",
            "generated_answer": "Answer 1",
            "execution_time": 1.5,
            "retrieved_chunks": [{"chunk_id": "c1", "content_with_weight": "Content 1"}],
            "metrics": {"precision": 1.0, "recall": 0.5}
        }
        item1.case_id.to_dict.return_value = {
            "id": "case_1",
            "question": "Question 1",
            "reference_answer": "Ref Answer 1",
            "relevant_chunk_ids": ["c1", "c2"]
        }

        # Item 2 (different metrics to test dynamic columns)
        item2 = MagicMock()
        item2.to_dict.return_value = {
            "id": "res_2",
            "generated_answer": "Answer 2",
            "execution_time": 2.0,
            "retrieved_chunks": [],
            "metrics": {"precision": 0.0, "f1_score": 0.0}
        }
        item2.case_id.to_dict.return_value = {
            "id": "case_2",
            "question": "Question 2",
            "reference_answer": "Ref Answer 2",
            "relevant_chunk_ids": []
        }

        # Make query iterable
        mock_query.__iter__.return_value = iter([item1, item2])

        # Execute
        csv_output = EvaluationService.get_run_results_csv(run_id)

        # Verify
        self.assertIsNotNone(csv_output)

        # Parse CSV to check content
        f = io.StringIO(csv_output)
        reader = csv.DictReader(f)
        rows = list(reader)

        self.assertEqual(len(rows), 2)

        # Check columns
        expected_columns = {
            "Question", "Reference Answer", "Generated Answer", "Execution Time",
            "Retrieved Chunks", "Relevant Chunk IDs",
            "metric_precision", "metric_recall", "metric_f1_score"
        }
        self.assertTrue(expected_columns.issubset(set(reader.fieldnames)))

        # Check Row 1
        row1 = rows[0]
        self.assertEqual(row1["Question"], "Question 1")
        self.assertEqual(row1["metric_precision"], "1.0")
        self.assertEqual(row1["metric_recall"], "0.5")
        self.assertEqual(row1["metric_f1_score"], "") # Missing in item1

        # Check Row 2
        row2 = rows[1]
        self.assertEqual(row2["Question"], "Question 2")
        self.assertEqual(row2["metric_precision"], "0.0")
        self.assertEqual(row2["metric_recall"], "") # Missing in item2
        self.assertEqual(row2["metric_f1_score"], "0.0")

        # Check JSON fields
        chunks1 = json.loads(row1["Retrieved Chunks"])
        self.assertEqual(len(chunks1), 1)
        self.assertEqual(chunks1[0]["chunk_id"], "c1")

        rel_ids1 = json.loads(row1["Relevant Chunk IDs"])
        self.assertEqual(rel_ids1, ["c1", "c2"])

if __name__ == "__main__":
    unittest.main()
