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

        # UniversalRouter is imported successfully
        # Ensure we can import it without error
    except ImportError as e:
        pytest.fail(f"Import failed with mocked dependencies: {e}")

    # Setup Keyword Arguments for routing
    filename = "test.pdf"
    binary = b"dummy content"

    # Mock callback
    callback = MagicMock()

    # Verify that UniversalRouter correctly routes to DeepDocParser.
    # We patch the DeepDocParser class in rag.orchestration.router since it is imported there.
    # We must ensure rag.orchestration.router is imported first (which we did above).
    with patch("rag.orchestration.router.DeepDocParser") as MockDeepDocParser:
        # Configure the mock to return a known structure (sections, tables, pdf_parser)
        expected_sections = ["mock_section"]
        expected_tables = ["mock_table"]
        expected_parser = MagicMock()
        MockDeepDocParser.return_value.parse_pdf.return_value = (expected_sections, expected_tables, expected_parser)

        res = UniversalRouter.route(filename=filename, binary=binary, parser_config={"layout_recognize": "DeepDOC"}, callback=callback)

        # Assertions
        # 1. Check if DeepDocParser was instantiated and used
        MockDeepDocParser.assert_called()
        # 2. Check if parse_pdf was called with correct arguments
        MockDeepDocParser.return_value.parse_pdf.assert_called_once()
        call_args = MockDeepDocParser.return_value.parse_pdf.call_args
        assert call_args[0] == (filename, binary, callback) or (
            call_args.kwargs.get("filepath") == filename and call_args.kwargs.get("binary") == binary and call_args.kwargs.get("callback") == callback
        )

        # 3. Check if the result from router matches what the parser returned
        assert res.sections == expected_sections
        assert res.tables == expected_tables
        assert res.pdf_parser == expected_parser
