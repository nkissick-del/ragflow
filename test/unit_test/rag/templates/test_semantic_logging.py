import unittest
from unittest.mock import patch
import sys

# Project root is automatically added to sys.path by test/unit_test/conftest.py


class TestSemanticLogging(unittest.TestCase):
    """Test logging behavior for semantic template."""

    def test_tiktoken_fallback_logging(self):
        """
        Verify that a warning is logged when tiktoken is missing.
        """
        # We need to force a reload of the module with tiktoken missing

        from test.mocks.mock_utils import setup_mocks

        setup_mocks()

        # We need to manually patch tiktoken to None as the test expects it to be missing
        # setup_mocks mocks it, so we overwrite it
        with patch.dict(sys.modules, {"tiktoken": None}):
            # Mock logging to capture calls
            with patch("logging.warning") as mock_log:
                # Save original module to restore later
                old_module = sys.modules.get("rag.templates.semantic")

                # Remove module from sys.modules if present to force re-import
                if "rag.templates.semantic" in sys.modules:
                    del sys.modules["rag.templates.semantic"]

                try:
                    import rag.templates.semantic  # noqa: F401
                except ImportError as e:
                    if "tiktoken" in str(e):
                        self.skipTest("tiktoken not installed")
                    raise
                finally:
                    # Restore original module to avoid test pollution
                    if old_module:
                        sys.modules["rag.templates.semantic"] = old_module
                    else:
                        sys.modules.pop("rag.templates.semantic", None)

                # Check if specific message was logged
                # We expect something mentioning "num_tokens", "tiktoken", "fallback"
                found = False
                debug_details = []

                for i, call in enumerate(mock_log.call_args_list):
                    args_str = [str(arg) for arg in call.args]
                    kwargs_str = [str(v) for v in call.kwargs.values()]
                    all_args = args_str + kwargs_str

                    has_tiktoken = any("tiktoken" in arg for arg in all_args)
                    has_num_tokens = any("num_tokens" in arg for arg in all_args)

                    debug_details.append(f"Call {i}: tiktoken={has_tiktoken}, num_tokens={has_num_tokens}, args={all_args}")

                    if has_tiktoken and has_num_tokens:
                        found = True
                        break

                self.assertTrue(found, f"Did not find expected log message about tiktoken fallback. Details: {debug_details}")


if __name__ == "__main__":
    unittest.main()
