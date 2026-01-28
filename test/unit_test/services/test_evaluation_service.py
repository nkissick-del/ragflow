#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import pytest
from unittest.mock import MagicMock, patch

# Define the patch dict globally or in a fixture
# We need to make sure we cover all imports that might fail or need mocking


@pytest.fixture
def mock_env():
    # Create mocks
    mock_dialog_service = MagicMock()
    mock_db_models = MagicMock()

    mock_common_service = MagicMock()
    mock_common_service.CommonService = object

    mock_constants = MagicMock()
    mock_constants.LLMType.IMAGE2TEXT = "image2text"
    mock_constants.LLMType.CHAT = "chat"

    mock_llm_service = MagicMock()
    mock_tenant_llm_service = MagicMock()
    mock_template = MagicMock()
    mock_generator = MagicMock()
    mock_json_repair = MagicMock()

    # Mock quart and api_utils for other tests
    mock_api_utils = MagicMock()
    mock_quart = MagicMock()

    # Config generator
    mock_generator.PROMPT_JINJA_ENV.from_string.return_value.render.return_value = "Rendered Prompt"
    mock_generator.message_fit_in.return_value = (100, [{"role": "user", "content": "Rendered Prompt"}])

    # Config LLM Service
    mock_bundle_instance = mock_llm_service.LLMBundle.return_value
    mock_bundle_instance.max_length = 4096

    # Patch dictionary
    patches = {
        "api.db.services.dialog_service": mock_dialog_service,
        "api.db.db_models": mock_db_models,
        "api.db.services.common_service": mock_common_service,
        "common.constants": mock_constants,
        "common.misc_utils": MagicMock(),
        "common.time_utils": MagicMock(),
        "api.db.services.llm_service": mock_llm_service,
        "api.db.services.tenant_llm_service": mock_tenant_llm_service,
        "rag.prompts.template": mock_template,
        "rag.prompts.generator": mock_generator,
        "json_repair": mock_json_repair,
        "api.utils.api_utils": mock_api_utils,
        "quart": mock_quart,
    }

    with patch.dict(sys.modules, patches):
        # We also need to remove evaluation_service from sys.modules if it exists
        # to force re-import with mocked dependencies
        if "api.db.services.evaluation_service" in sys.modules:
            del sys.modules["api.db.services.evaluation_service"]

        yield {"llm_service": mock_llm_service, "tenant_llm_service": mock_tenant_llm_service, "template": mock_template, "json_repair": mock_json_repair, "db_models": mock_db_models}


def test_evaluate_with_llm(mock_env):
    from api.db.services.evaluation.metrics_service import EvaluationMetricsService

    # Setup
    dialog = MagicMock()
    dialog.tenant_id = "tenant_1"
    dialog.llm_id = "llm_1"

    mock_env["tenant_llm_service"].TenantLLMService.llm_id2llm_type.return_value = "chat"

    # Mock LLM response
    mock_bundle = mock_env["llm_service"].LLMBundle.return_value
    mock_bundle.chat.return_value = '{"faithfulness": 0.9}'  # simplified response string

    # Mock json_repair
    mock_env["json_repair"].loads.return_value = {"faithfulness": 0.9, "context_relevance": 0.8, "answer_relevance": 0.95, "semantic_similarity": 0.85}

    # Execute
    metrics = EvaluationMetricsService.evaluate_with_llm("Q", "A", "Ref", [{"content": "Ctx"}], dialog)

    # Verify
    assert metrics["faithfulness"] == 0.9
    assert metrics["context_relevance"] == 0.8

    mock_env["llm_service"].LLMBundle.assert_called()
    mock_bundle.chat.assert_called()


def test_evaluate_with_llm_error(mock_env):
    from api.db.services.evaluation.metrics_service import EvaluationMetricsService

    dialog = MagicMock()
    mock_bundle = mock_env["llm_service"].LLMBundle.return_value
    mock_bundle.chat.return_value = "Invalid"

    mock_env["json_repair"].loads.side_effect = Exception("Fail")

    metrics = EvaluationMetricsService.evaluate_with_llm("Q", "A", "Ref", [{"content": "Ctx"}], dialog)

    assert metrics["faithfulness"] == 0.0
    assert metrics["semantic_similarity"] is None
    assert metrics["evaluation_status"] == "failed"


@pytest.mark.p1
class TestEvaluationCompareRuns:
    """Test compare_runs logic in EvaluationService."""

    def test_compare_runs_success(self, mock_env):
        from api.db.services.evaluation_service import EvaluationService

        # We need to mock EvaluationRun within the mocked environment
        # mock_env['db_models'] is the mocked api.db.db_models
        mock_evaluation_run = mock_env["db_models"].EvaluationRun

        # Arrange
        run_ids = ["run1", "run2"]

        run1 = MagicMock()
        run1.id = "run1"
        run1.dataset_id_id = "ds1"  # Peewee FK usage
        run1.metrics_summary = {"avg_precision": 0.8, "avg_recall": 0.6}
        run1.to_dict.return_value = {"id": "run1", "dataset_id": "ds1", "metrics_summary": {"avg_precision": 0.8, "avg_recall": 0.6}}

        run2 = MagicMock()
        run2.id = "run2"
        run2.dataset_id_id = "ds1"
        run2.metrics_summary = {"avg_precision": 0.9, "avg_recall": 0.5}
        run2.to_dict.return_value = {"id": "run2", "dataset_id": "ds1", "metrics_summary": {"avg_precision": 0.9, "avg_recall": 0.5}}

        # Mock the query chain: EvaluationRun.select().where()
        mock_query = MagicMock()
        mock_query.__iter__.return_value = [run1, run2]
        mock_evaluation_run.select.return_value.where.return_value = mock_query

        # Act
        success, result = EvaluationService.compare_runs(run_ids)

        # Assert
        assert success is True
        assert "runs" in result
        assert len(result["runs"]) == 2
        assert "comparison" in result

        # Check if pivoted metrics are correct
        # Structure: comparison[metric][run_id] = value
        assert "avg_precision" in result["comparison"]
        assert result["comparison"]["avg_precision"]["run1"] == 0.8
        assert result["comparison"]["avg_precision"]["run2"] == 0.9

        assert "avg_recall" in result["comparison"]
        assert result["comparison"]["avg_recall"]["run1"] == 0.6
        assert result["comparison"]["avg_recall"]["run2"] == 0.5

    def test_compare_runs_missing_run(self, mock_env):
        """Test error when a run ID is missing."""
        from api.db.services.evaluation_service import EvaluationService

        mock_evaluation_run = mock_env["db_models"].EvaluationRun

        run_ids = ["run1", "run2"]

        run1 = MagicMock()
        run1.id = "run1"
        run1.dataset_id_id = "ds1"

        # Only return run1
        mock_query = MagicMock()
        mock_query.__iter__.return_value = [run1]

        mock_evaluation_run.select.return_value.where.return_value = mock_query

        success, result = EvaluationService.compare_runs(run_ids)

        assert success is False
        assert "Runs not found: run2" in result  # Stronger assertion

    def test_compare_runs_different_datasets(self, mock_env):
        """Test error when runs belong to different datasets."""
        from api.db.services.evaluation_service import EvaluationService

        mock_evaluation_run = mock_env["db_models"].EvaluationRun
        run_ids = ["run1", "run2"]

        run1 = MagicMock()
        run1.id = "run1"
        run1.dataset_id_id = "ds1"

        run2 = MagicMock()
        run2.id = "run2"
        run2.dataset_id_id = "ds2"  # Different dataset

        mock_query = MagicMock()
        mock_query.__iter__.return_value = [run1, run2]

        mock_evaluation_run.select.return_value.where.return_value = mock_query

        success, result = EvaluationService.compare_runs(run_ids)

        assert success is False
        assert "different datasets" in result
