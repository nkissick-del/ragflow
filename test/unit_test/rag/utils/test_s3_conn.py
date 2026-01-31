import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Mock dependencies before importing RAGFlowS3
from test.mocks.mock_utils import setup_mocks, teardown_mocks


class TestS3Connection(unittest.TestCase):
    def setUp(self):
        # Mock dependencies that have side effects on import
        sys.modules["boto3"] = MagicMock()
        sys.modules["botocore"] = MagicMock()
        sys.modules["botocore.exceptions"] = MagicMock()
        sys.modules["botocore.config"] = MagicMock()

        # Mock config_utils to avoid loading real YAML files
        mock_config_utils = MagicMock()
        mock_config_utils.read_config.return_value = {}
        mock_config_utils.get_base_config.side_effect = lambda n, d=None: d
        sys.modules["common.config_utils"] = mock_config_utils

        # Ensure we start fresh for these modules to apply decorator changes
        for mod in ["common.decorator", "rag.utils.s3_conn", "common.settings"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import rag.utils.s3_conn as real_s3_conn

        self.original_modules = setup_mocks()

        # Restore the real s3_conn for this test
        sys.modules["rag.utils.s3_conn"] = real_s3_conn

        # Patch settings for testing
        import common.settings as settings

        settings.S3 = {"access_key": "test_ak", "secret_key": "test_sk", "bucket": "test_bucket"}
        settings.S3_MAX_RETRIES = 3

    def _get_s3_class(self):
        from rag.utils.s3_conn import RAGFlowS3

        if hasattr(RAGFlowS3, "__wrapped__"):
            return RAGFlowS3.__wrapped__
        if hasattr(RAGFlowS3, "__closure__") and RAGFlowS3.__closure__:
            return RAGFlowS3.__closure__[0].cell_contents
        return RAGFlowS3

    def tearDown(self):
        teardown_mocks(self.original_modules)

    @patch("rag.utils.s3_conn.boto3.client")
    def test_init_and_open_success(self, mock_boto_client):
        s3_cls = self._get_s3_class()

        s3 = s3_cls()
        self.assertIsNotNone(s3.conn)
        self.assertEqual(len(s3.conn), 1)
        mock_boto_client.assert_called_once()

    @patch("rag.utils.s3_conn.boto3.client")
    def test_open_failure_sets_conn_none(self, mock_boto_client):
        mock_boto_client.side_effect = Exception("Connection failed")
        s3_cls = self._get_s3_class()

        s3 = s3_cls()
        self.assertIsNone(s3.conn)

    @patch("rag.utils.s3_conn.boto3.client")
    def test_methods_raise_runtime_error_when_no_conn(self, mock_boto_client):
        mock_boto_client.side_effect = Exception("Connection failed")
        s3_cls = self._get_s3_class()

        s3 = s3_cls()
        self.assertIsNone(s3.conn)

        with self.assertRaises(RuntimeError) as cm:
            s3.get("bucket", "key")
        self.assertEqual(str(cm.exception), "S3 connection not available")

        with self.assertRaises(RuntimeError) as cm:
            s3.put("bucket", "key", b"data")
        self.assertEqual(str(cm.exception), "S3 connection not available")

    @patch("rag.utils.s3_conn.boto3.client")
    def test_get_supports_max_retries_override(self, mock_boto_client):
        # First attempt fails, second succeeds
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Mock get_object to fail then succeed
        mock_s3.get_object.side_effect = [Exception("Failed 1"), {"Body": MagicMock(read=lambda: b"data")}]

        s3_cls = self._get_s3_class()
        s3 = s3_cls()
        # Reset call count after init
        mock_boto_client.reset_mock()

        # Test with override
        s3.get("bucket", "key", max_retries=2)

        self.assertEqual(mock_s3.get_object.call_count, 2)

    @patch("rag.utils.s3_conn.boto3.client")
    def test_get_presigned_url_hardening(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        s3_cls = self._get_s3_class()

        s3 = s3_cls()
        # Reset after init
        mock_boto_client.reset_mock()

        # Simulate connection loss before call
        s3.conn = None

        s3.get_presigned_url("bucket", "key", 3600)

    @patch("rag.utils.s3_conn.boto3.client")
    @patch("rag.utils.s3_conn.time.sleep")
    def test_get_presigned_url_retries_and_raises(self, mock_sleep, mock_boto_client):
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        # Simulate Forbidden error (credential error)
        error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
        mock_s3.generate_presigned_url.side_effect = ClientError(error_response, "generate_presigned_url")

        s3_cls = self._get_s3_class()
        s3 = s3_cls()
        s3.max_retries = 2
        mock_boto_client.reset_mock()

        with self.assertRaises(ClientError):
            s3.get_presigned_url("bucket", "key", 3600)

        # 1 initial try + 1 retry = 2 calls
        self.assertEqual(mock_s3.generate_presigned_url.call_count, 2)
        # Should call __open__ (mock_boto_client) again for credential error
        self.assertEqual(mock_boto_client.call_count, 1)
        mock_sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
