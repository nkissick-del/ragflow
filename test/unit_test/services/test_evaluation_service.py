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
        "json_repair": mock_json_repair
    }

    with patch.dict(sys.modules, patches):
        # We also need to remove evaluation_service from sys.modules if it exists
        # to force re-import with mocked dependencies
        if "api.db.services.evaluation_service" in sys.modules:
             del sys.modules["api.db.services.evaluation_service"]

        yield {
            "llm_service": mock_llm_service,
            "tenant_llm_service": mock_tenant_llm_service,
            "template": mock_template,
            "json_repair": mock_json_repair
        }

def test_evaluate_with_llm(mock_env):
    from api.db.services.evaluation_service import EvaluationService

    # Setup
    dialog = MagicMock()
    dialog.tenant_id = "tenant_1"
    dialog.llm_id = "llm_1"

    mock_env["tenant_llm_service"].TenantLLMService.llm_id2llm_type.return_value = "chat"

    # Mock LLM response
    mock_bundle = mock_env["llm_service"].LLMBundle.return_value
    mock_bundle._run_coroutine_sync.return_value = '{"faithfulness": 0.9}' # simplified response string

    # Mock json_repair
    mock_env["json_repair"].loads.return_value = {
        "faithfulness": 0.9,
        "context_relevance": 0.8,
        "answer_relevance": 0.95,
        "semantic_similarity": 0.85
    }

    # Execute
    metrics = EvaluationService._evaluate_with_llm(
        "Q", "A", "Ref", [{"content": "Ctx"}], dialog
    )

    # Verify
    assert metrics["faithfulness"] == 0.9
    assert metrics["context_relevance"] == 0.8

    mock_env["llm_service"].LLMBundle.assert_called()
    mock_bundle.async_chat.assert_called()

def test_evaluate_with_llm_error(mock_env):
    from api.db.services.evaluation_service import EvaluationService

    dialog = MagicMock()
    mock_bundle = mock_env["llm_service"].LLMBundle.return_value
    mock_bundle._run_coroutine_sync.return_value = "Invalid"

    mock_env["json_repair"].loads.side_effect = Exception("Fail")

    metrics = EvaluationService._evaluate_with_llm(
        "Q", "A", "Ref", [{"content": "Ctx"}], dialog
    )

    assert metrics["faithfulness"] == 0.0
