import unittest
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from rag.app.templates import q_and_a


class TestQAndA(unittest.TestCase):
    def test_chunk_excel(self):
        # Simulate Excel extraction logic
        # The q_and_a.chunk function calls excel_parser(filename, binary, callback) and iterates
        # over the returned (question, answer) tuples using: for ii, (q, a) in enumerate(excel_parser(...))
        with patch("rag.app.templates.q_and_a.Excel") as MockParser:
            mock_instance = MockParser.return_value
            # The Excel.__call__ returns a list of (question, answer) tuples directly
            # q_and_a.chunk iterates over this list via: for ii, (q, a) in enumerate(excel_parser(...))
            mock_instance.return_value = [("What is this?", "A test answer.")]

            res = q_and_a.chunk("test.xlsx", b"content", callback=lambda p, m: None)

            # Verify that the result contains the expected Q&A chunk data
            self.assertEqual(len(res), 1)
            # Check the chunk has the expected question/answer content
            self.assertIn("content_with_weight", res[0])
            self.assertIn("What is this?", res[0]["content_with_weight"])
            self.assertIn("A test answer", res[0]["content_with_weight"])

    def test_chunk_txt(self):
        # Text file Q&A logic
        # Data format must be delimited (tab or comma)
        filename = "test.txt"
        binary = b"Question,Answer"

        with patch("rag.app.templates.q_and_a.get_text") as mock_get:
            mock_get.return_value = "Question,Answer"

            res = q_and_a.chunk(filename, binary, callback=lambda p, m: None)

            self.assertEqual(len(res), 1)
            # The q_and_a template logic for txt is:
            # 1. Split by \n
            # 2. Join
            # 3. Tokenize.


if __name__ == "__main__":
    unittest.main()
