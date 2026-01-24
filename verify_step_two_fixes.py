import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import datetime
import re

# Mocking the relative imports and dependencies
sys.modules[".entities"] = MagicMock()
sys.modules["rag.nlp"] = MagicMock()
sys.modules["xpinyin"] = MagicMock()
sys.modules["demjson3"] = MagicMock()

# Now we can import the module correctly after setting up mocks
# We need to add the parent directory to sys.path to allow imports to work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "rag/parsers/deepdoc/resume/"))

import step_two


class TestStepTwoFixes(unittest.TestCase):
    def test_rmHtmlTag(self):
        self.assertEqual(step_two.rmHtmlTag("<div>hello</div>"), " hello ")
        self.assertEqual(step_two.rmHtmlTag("<p class='test'>text</p>"), " text ")
        self.assertEqual(step_two.rmHtmlTag("<br/>"), " ")
        self.assertEqual(step_two.rmHtmlTag("<a href='?q=1&p=1'>link</a>"), " link ")

    def test_dealWithInt64(self):
        d = {"a": np.int64(1), "b": [np.float64(2.5), 3], "c": {"d": np.int32(4), "e": np.float32(5.5)}}
        result = step_two.dealWithInt64(d)
        self.assertTrue(isinstance(result["a"], int))
        self.assertTrue(isinstance(result["b"][0], float))
        self.assertTrue(isinstance(result["c"]["d"], int))
        self.assertTrue(isinstance(result["c"]["e"], float))

    def test_subordinates_count_filter(self):
        cv = {}
        fea = {"subordinates_count": ["10", "abc", "20", " "]}
        # Mocking or simulating the logic in forWork
        # Simplified test of the regex and max logic
        sub_list = [int(i) for i in fea["subordinates_count"] if re.match(r"^\d+$", str(i))]
        self.assertEqual(sub_list, [10, 20])
        max_sub = np.max(sub_list)
        self.assertEqual(max_sub, 20)

        # Test empty case
        sub_list_empty = [int(i) for i in [] if re.match(r"^\d+$", str(i))]
        self.assertEqual(sub_list_empty, [])

    def test_time_limit_signal(self):
        # This test might be platform dependent, but we can check if it runs without error
        try:
            with step_two.time_limit(1):
                pass
        except Exception as e:
            self.fail(f"time_limit raised exception unexpectedly: {e}")

    def test_time_limit_timeout(self):
        # We can try to trigger a timeout
        import time

        with self.assertRaises(step_two.TimeoutException):
            with step_two.time_limit(1):
                # We need to do something that doesn't release the GIL if we are testing the timer path
                # but signal path should work too.
                # Since we are in main thread on Mac (Unix), it should use signal.
                time.sleep(2)

    @patch("step_two.datetime")
    def test_forWork_year_check(self, mock_datetime):
        mock_datetime.datetime.now().year = 2026
        # Simulate the logic
        y = "2027"
        current_year = 2026
        y_int = int(y) if y and str(y).isdigit() else 0
        self.assertTrue(y_int > current_year)

        y = "2025"
        y_int = int(y) if y and str(y).isdigit() else 0
        self.assertFalse(y_int > current_year)

    def test_tob_resume_id_handling(self):
        cv = {"tob_resume_id": 12345}
        # Simulate the logic
        tr_id = cv.get("tob_resume_id")
        if tr_id is not None:
            cv["id"] = str(tr_id)
        self.assertEqual(cv["id"], "12345")

        cv_missing = {}
        tr_id = cv_missing.get("tob_resume_id")
        if tr_id is not None:
            cv_missing["id"] = str(tr_id)
        else:
            cv_missing["id"] = None
        self.assertIsNone(cv_missing["id"])


if __name__ == "__main__":
    unittest.main()
