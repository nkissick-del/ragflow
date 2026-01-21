import unittest
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from rag.app.templates import q_and_a


class TestQAndA(unittest.TestCase):
    def test_chunk_excel(self):
        # Simulate Excel extraction logic
        # Since Excel parsing is complex/binary dependent, we verify the dispatcher logic behavior
        # or we mock the ExcelParser class
        with patch("rag.app.templates.q_and_a.Excel") as MockParser:
            mock_instance = MockParser.return_value
            # return mocked rows. q_and_a iterates over the instance directly.
            mock_instance.return_value = [("Q: What is this?", "A: A test.")]

            res = q_and_a.chunk("test.xlsx", b"content", callback=lambda p, m: None)

            # Verification logic for Excel depends on how q_and_a processes the result of excel_parser.html
            # In qa.py: sections = excel_parser.html(binary, ...)
            # Then tokenize(doc, "\n".join(sections), ...)

            self.assertEqual(len(res), 1)

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
