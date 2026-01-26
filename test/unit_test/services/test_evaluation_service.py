#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Unit tests for EvaluationService.
"""
import sys
from unittest.mock import MagicMock

# Mock dependencies that cause import errors or are not needed for this test
mock_api_utils = MagicMock()
mock_api_utils.get_parser_config = MagicMock()
mock_api_utils.get_data_error_result = MagicMock()
sys.modules['api.utils.api_utils'] = mock_api_utils

mock_quart = MagicMock()
sys.modules['quart'] = mock_quart

# Mock DialogService to avoid importing deep dependency chain (FileService -> TaskService -> deepdoc)
mock_dialog_service_module = MagicMock()
sys.modules['api.db.services.dialog_service'] = mock_dialog_service_module

import pytest
from unittest.mock import patch
# Now import the service under test
from api.db.services.evaluation_service import EvaluationService

@pytest.mark.p1
class TestEvaluationCompareRuns:
    """Test compare_runs logic in EvaluationService."""

    @patch('api.db.services.evaluation_service.EvaluationRun')
    def test_compare_runs_success(self, mock_evaluation_run):
        """Test successful comparison of runs."""
        # Arrange
        run_ids = ['run1', 'run2']

        run1 = MagicMock()
        run1.id = 'run1'
        run1.dataset_id_id = 'ds1'
        run1.metrics_summary = {'avg_precision': 0.8, 'avg_recall': 0.6}
        run1.to_dict.return_value = {
            'id': 'run1', 'dataset_id': 'ds1',
            'metrics_summary': {'avg_precision': 0.8, 'avg_recall': 0.6}
        }

        run2 = MagicMock()
        run2.id = 'run2'
        run2.dataset_id_id = 'ds1'
        run2.metrics_summary = {'avg_precision': 0.9, 'avg_recall': 0.5}
        run2.to_dict.return_value = {
            'id': 'run2', 'dataset_id': 'ds1',
            'metrics_summary': {'avg_precision': 0.9, 'avg_recall': 0.5}
        }

        # Mock the query chain: EvaluationRun.select().where()
        mock_query = MagicMock()
        mock_query.__iter__.return_value = [run1, run2]
        mock_evaluation_run.select.return_value.where.return_value = mock_query

        # Act
        success, result = EvaluationService.compare_runs(run_ids)

        # Assert
        assert success is True
        assert 'runs' in result
        assert len(result['runs']) == 2
        assert 'comparison' in result

        # Check if pivoted metrics are correct
        # Structure: comparison[metric][run_id] = value
        assert 'avg_precision' in result['comparison']
        assert result['comparison']['avg_precision']['run1'] == 0.8
        assert result['comparison']['avg_precision']['run2'] == 0.9

        assert 'avg_recall' in result['comparison']
        assert result['comparison']['avg_recall']['run1'] == 0.6
        assert result['comparison']['avg_recall']['run2'] == 0.5

    @patch('api.db.services.evaluation_service.EvaluationRun')
    def test_compare_runs_missing_run(self, mock_evaluation_run):
        """Test error when a run ID is missing."""
        run_ids = ['run1', 'run2']

        run1 = MagicMock()
        run1.id = 'run1'
        run1.dataset_id_id = 'ds1'

        # Only return run1
        mock_query = MagicMock()
        mock_query.__iter__.return_value = [run1]

        mock_evaluation_run.select.return_value.where.return_value = mock_query

        success, result = EvaluationService.compare_runs(run_ids)

        assert success is False
        assert "not found" in result

    @patch('api.db.services.evaluation_service.EvaluationRun')
    def test_compare_runs_different_datasets(self, mock_evaluation_run):
        """Test error when runs belong to different datasets."""
        run_ids = ['run1', 'run2']

        run1 = MagicMock()
        run1.id = 'run1'
        run1.dataset_id_id = 'ds1'

        run2 = MagicMock()
        run2.id = 'run2'
        run2.dataset_id_id = 'ds2' # Different dataset

        mock_query = MagicMock()
        mock_query.__iter__.return_value = [run1, run2]

        mock_evaluation_run.select.return_value.where.return_value = mock_query

        success, result = EvaluationService.compare_runs(run_ids)

        assert success is False
        assert "different datasets" in result
