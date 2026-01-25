from tests.mock_utils import setup_mocks

# Set up system mocks
setup_mocks()

import unittest
import json
import os
import pandas as pd

# Adjust import paths based on project structure
from rag.parsers.deepdoc.resume.entities import corporations, regions, schools
from rag.parsers.deepdoc.resume import step_one


class TestResumeFixes(unittest.TestCase):
    def setUp(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_corp_tag_empty_keys(self):
        """Test corp_tag logic handling empty keys."""
        # Mock CORP_TAG to contain an empty key if possible or force specific scenario
        # corporations.CORP_TAG is loaded from file. We can mock it.
        original_tag = corporations.CORP_TAG
        try:
            # Inject an empty key and a short key
            corporations.CORP_TAG = {"": ["TAG_EMPTY"], "ab": ["TAG_SHORT"], "abc": ["TAG_NORMAL"]}

            # This should not crash or return TAG_EMPTY for simple query
            res = corporations.corp_tag("somecorp")
            self.assertEqual(res, [])

            # Test ratio check with short key "ab" (len 2)
            # "abcd" (len 4). 4/2 = 2. Should be skipped if len(n)<3 and len(nm)/len(n)>=2
            res = corporations.corp_tag("abcd")
            # "ab" matches "abcd". len("ab")=2 < 3. len("abcd")/len("ab") = 2 >= 2. -> continue
            self.assertEqual(res, [])

            # Test exact match
            res = corporations.corp_tag("abc")
            self.assertEqual(res, ["TAG_NORMAL"])
        finally:
            corporations.CORP_TAG = original_tag

    def test_is_name_regex(self):
        """Test regions.is_name regex improvements."""
        # Mock NM_SET to include test names
        original_nm_set = regions.NM_SET
        regions.NM_SET = {"广西", "内蒙古", "新疆"}
        try:
            self.assertTrue(regions.is_name("广西"))
            self.assertTrue(regions.is_name("广西壮族自治区"))
            self.assertTrue(regions.is_name("内蒙古自治区"))
            self.assertTrue(regions.is_name("新疆维吾尔自治区"))
            self.assertFalse(regions.is_name("广西某某"))
        finally:
            regions.NM_SET = original_nm_set

    def test_step_one_validation(self):
        """Test validation in step_one.refactor."""
        # Case 1: Empty DF
        df_empty = pd.DataFrame()
        with self.assertRaises(ValueError) as cm:
            step_one.refactor(df_empty)
        self.assertIn("exactly one row", str(cm.exception))

        # Case 2: Multi-row DF
        df_multi = pd.DataFrame([{"a": 1}, {"a": 2}])
        with self.assertRaises(ValueError) as cm:
            step_one.refactor(df_multi)
        self.assertIn("exactly one row", str(cm.exception))

        # TODO: implement test for column mismatch in step_one — needs mocking or crafted payload

    @unittest.skip("Depends on full flow or mocked components - not fully implemented")
    def test_step_one_phone_vectorization(self):
        """Test phone vectorization logic."""
        # Since logic is inside refactor and hard to isolate without full flow,
        # we can verify the pandas logic snippet independently or mock.
        # Let's try to run refactor with a minimal valid input if possible.
        # step_one.refactor expects a DF with 'resume_content'.
        pass

    def test_good_sch_json_content(self):
        """Verify good_sch.json content fixes."""
        sch_path = os.path.join(os.path.dirname(schools.__file__), "res/good_sch.json")
        with open(sch_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertNotIn("cdut ", data)
        self.assertIn("cdut", data)
        self.assertNotIn("neu 或 nu", data)
        self.assertIn("neu", data)
        self.assertIn("nu", data)

        self.assertNotIn("西大", data)

    def test_good_corp_json_content(self):
        """Verify good_corp.json content fixes."""
        corp_path = os.path.join(os.path.dirname(corporations.__file__), "res/good_corp.json")
        with open(corp_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertNotIn(" lu", data)
        self.assertIn("lu", data)
        self.assertNotIn("linkedln", data)
        self.assertIn("linkedin", data)
        self.assertNotIn("hqg , limited", data)
        self.assertIn("hqg, limited", data)
        matches = [c for c in data if "ping an insurance group of china" in c and " ," not in c]
        self.assertTrue(matches, f"Expected message not found in data. Matches found: {matches}")


if __name__ == "__main__":
    unittest.main()
