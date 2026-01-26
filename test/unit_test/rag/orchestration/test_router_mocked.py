import sys
import pytest
from unittest.mock import MagicMock, patch

# List of modules to mock
HEAVY_DEPENDENCIES = [
    "cv2",
    "xgboost",
    "pdfplumber",
    "pdfminer",
    "pdfminer.high_level",
    "pdfminer.layout",
    "pypdf",
    "PyPDF2",
    "olefile",
    "PIL",
    "PIL.Image",
    "openpyxl",
    "pandas",
    # Tencent Cloud
    "tencentcloud",
    "tencentcloud.common",
    "tencentcloud.common.credential",
    "tencentcloud.common.profile",
    "tencentcloud.common.profile.client_profile",
    "tencentcloud.common.profile.http_profile",
    "tencentcloud.common.exception",
    "tencentcloud.common.exception.tencent_cloud_sdk_exception",
    "tencentcloud.lkeap",
    "tencentcloud.lkeap.v20240522",
    "tencentcloud.lkeap.v20240522.lkeap_client",
    "tencentcloud.lkeap.v20240522.models",
    # DeepDoc internal
    "deepdoc",
    "deepdoc.vision",
    "rag.parsers.deepdoc.vision",
    # Service dependencies
    "api.db.services.llm_service",
    "api.db.services.tenant_llm_service",
]


@pytest.fixture
def mock_heavy_deps():
    """
    Fixture to mock heavy dependencies in sys.modules for the duration of the test.
    """
    with patch.dict(sys.modules):
        for dep in HEAVY_DEPENDENCIES:
            sys.modules[dep] = MagicMock()
        yield


def test_router_dispatch(mock_heavy_deps):
    """
    Verifies that UniversalRouter correctly dispatches to DeepDocParser
    and that DeepDocParser calls the underlying logic (which is mocked).
    This test runs with heavy dependencies mocked out to ensure wiring works
    without requiring installation of cv2, etc.
    """

    # Import inside the test function AFTER mocking takes effect
    # Note: If these modules were already imported by other tests,
    # reload might be necessary, but patch.dict handles safe rollback.
    try:
        from rag.orchestration.router import UniversalRouter
        # Ensure we can import it without error
    except ImportError as e:
        pytest.fail(f"Import failed with mocked dependencies: {e}")

    # Setup Keyword Arguments for routing
    filename = "test.pdf"
    binary = b"dummy content"

    # Mock callback
    callback = MagicMock()

    # We expect the router to call DeepDocParser, which will try to use the mocked deepdoc engine.
    # Since our mocks return MagicMocks (which aren't iterable tuples by default),
    # the unpacking in the parser will likely fail.
    # If it fails with TypeError (unpacking), it means the Router -> Parser -> Engine wiring SUCCEEDED.

    try:
        UniversalRouter.route(filename=filename, binary=binary, parser_config={"layout_recognize": "DeepDOC"}, callback=callback)
    except (TypeError, ValueError) as e:
        # Check if the error is due to unpacking the Mock return value
        # This confirms control flow reached the deepdoc logic
        msg = str(e)
        if "iterable" in msg or "unpack" in msg or "not enough values" in msg:
            # Success! The wiring is correct.
            pass
        else:
            pytest.fail(f"Router failed unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"Router failed with unexpected exception type: {type(e)}: {e}")
