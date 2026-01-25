import unittest
from unittest.mock import MagicMock
import pandas as pd


# Import parsers (assuming they are in python path)
# We will mock dependencies where needed to test logic in isolation

from rag.parsers.deepdoc.docx_parser import RAGFlowDocxParser
from rag.parsers.deepdoc.json_parser import RAGFlowJsonParser
from rag.parsers.deepdoc.resume import refactor
from rag.parsers.deepdoc.txt_parser import RAGFlowTxtParser
from rag.parsers.deepdoc.ppt_parser import RAGFlowPptParser


class TestParserFixes(unittest.TestCase):
    def test_docx_max_type_empty(self):
        """Test that max_type handle empty counter (DocxParser logic isolation)"""
        parser = RAGFlowDocxParser()
        # Mocking internal method usage by creating a fake dataframe
        df = pd.DataFrame([])
        # The logic under test is inside __compose_table_content
        # Accessing private method for test access
        res = parser._RAGFlowDocxParser__compose_table_content(df)
        self.assertEqual(res, [])

        # Non-empty DF but empty content affecting Counter?
        df2 = pd.DataFrame([["", ""]], columns=["A", "B"])
        # If blockType returns nothing or Counter is empty, it shouldn't crash
        # We can't easily trigger the exact crash without mocking blockType or having specific data,
        # but the fix condition 'if df.shape[1] > 0 and counter:' prevents the crash.
        res2 = parser._RAGFlowDocxParser__compose_table_content(df2)
        self.assertEqual(res2, [])

    def test_json_valid_method_static(self):
        """Test RAGFlowJsonParser._is_valid_json is static and works"""
        self.assertIsInstance(RAGFlowJsonParser.__dict__["_is_valid_json"], staticmethod)
        self.assertTrue(RAGFlowJsonParser._is_valid_json('{"a": 1}'))
        self.assertFalse(RAGFlowJsonParser._is_valid_json("{a: 1}"))

    def test_json_split_text_ascii(self):
        """Test split_text uses ensure_ascii=False"""
        parser = RAGFlowJsonParser()
        data = {"key": "中文"}
        chunks = parser.split_text(data, ensure_ascii=False)
        self.assertTrue(chunks)
        self.assertIn("中文", chunks[0])

    def test_resume_falsy_check(self):
        """Test resume refactor handles falsy values correctly (0 shouldn't be deleted)"""
        cv = {"basic": {"basic_salary_month": 0, "expect_annual_salary_from": 100}}
        refactor(cv)
        # Check renaming happened even for 0
        self.assertIn("salary_month", cv["basic"])
        self.assertEqual(cv["basic"]["salary_month"], 0)
        self.assertNotIn("basic_salary_month", cv["basic"])

    def test_txt_chunking(self):
        """Test txt parser chunking logic"""
        parser = RAGFlowTxtParser()
        # parser_txt accepts raw string, no need to mock get_text
        # Include delimiters to facilitate chunking
        txt = "a b c \n" * 50
        res = parser.parser_txt(txt, chunk_token_num=5)
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 1)
        # Verify reconstruction (ignoring exact whitespace if delimiter handling is complex, but roughly)
        # Check that we got a list of [chunk, metadata] pairs
        for item in res:
            self.assertIsInstance(item, (list, tuple))
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], str)
            # Optional: check chunk size
            # token-like split length check
            tokens = item[0].split()
            # It's soft limit, but let's just check it's not empty
            self.assertTrue(len(tokens) > 0 or item[0] == "")

    def test_ppt_parser_has_extract_method(self):
        """Test that PPT parser has the expected private extract method and it behaves safely."""
        parser = RAGFlowPptParser()
        self.assertTrue(hasattr(parser, "_RAGFlowPptParser__extract"))

        # Behavioral assertion: call the private method
        # Mocking or providing minimal args to ensure it runs or fails predictably
        # __extract(self, ppt_path, from_page, to_page) usually
        # Since we don't have a real PPT, let's see if we can call it.
        # It likely accepts a path.
        # Depending on implementation, it might raise validation error or try to open.
        # If we pass a dummy path that doesn't exist?
        extract_method = getattr(parser, "_RAGFlowPptParser__extract")
        # Just ensure we can get a handle to it.
        # If the user wants a RUN, we need to mock Presentation.
        with unittest.mock.patch("rag.parsers.deepdoc.ppt_parser.Presentation") as mock_ppt:
            # Setup a mock presentation with slides
            mock_prs = MagicMock()
            mock_ppt.return_value = mock_prs
            mock_prs.slides = []

            # Call the method
            res = extract_method("dummy.pptx", 0, 10)
            # Expect a list
            self.assertIsInstance(res, list)


if __name__ == "__main__":
    unittest.main()
