import unittest
from unittest.mock import patch, MagicMock
import sys
import os


# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


class TestSemanticLogging(unittest.TestCase):
    """Test logging behavior for semantic template."""

    def test_tiktoken_fallback_logging(self):
        """
        Verify that a warning is logged when tiktoken is missing.
        """
        # We need to force a reload of the module with tiktoken missing

        # Mock dependencies that might import tiktoken themselves
        mock_tokenizer = MagicMock()
        mock_doc = MagicMock()

        with patch.dict(sys.modules, {"tiktoken": None, "rag.nlp": MagicMock(), "rag.nlp.rag_tokenizer": mock_tokenizer, "common.token_utils": MagicMock(), "rag.app.standardized_document": mock_doc}):
            # Mock logging to capture calls
            with patch("logging.warning") as mock_log:
                # Save original module to restore later
                old_module = sys.modules.get("rag.app.templates.semantic")

                # Remove module from sys.modules if present to force re-import
                if "rag.app.templates.semantic" in sys.modules:
                    del sys.modules["rag.app.templates.semantic"]

                try:
                    import rag.app.templates.semantic
                except ImportError as e:
                    if "tiktoken" in str(e):
                        self.skipTest("tiktoken not installed")
                    raise
                finally:
                    # Restore original module to avoid test pollution
                    if old_module:
                        sys.modules["rag.app.templates.semantic"] = old_module
                    else:
                        sys.modules.pop("rag.app.templates.semantic", None)

                # Check if specific message was logged
                # We expect something mentioning "num_tokens", "tiktoken", "fallback"
                found = any(any("tiktoken" in str(arg) and "num_tokens" in str(arg) for arg in call.args + tuple(call.kwargs.values())) for call in mock_log.call_args_list)

                self.assertTrue(found, "Did not find expected log message about tiktoken fallback in logging.warning calls")


if __name__ == "__main__":
    unittest.main()
