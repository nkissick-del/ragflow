"""Unit tests for rag.app.templates.q_and_a module.

Tests the Q&A parsing functionality for various file formats including
Excel, TXT, CSV, and edge cases.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys


class TestQAndATemplate(unittest.TestCase):
    """Tests for the Q&A template."""

    def setUp(self):
        # Setup mocks for modules that q_and_a might import
        self.mock_modules = {
            "rag.parsers": MagicMock(),
            "rag.parsers.deepdoc_client": MagicMock(),
            "rag.parsers.PdfParser": MagicMock(),
            "rag.parsers.ExcelParser": MagicMock(),
            "rag.parsers.DocxParser": MagicMock(),
            "deepdoc": MagicMock(),
            "deepdoc.parser": MagicMock(),
            "common": MagicMock(),
            "common.token_utils": MagicMock(),
            "bs4": MagicMock(),
            "openpyxl": MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import q_and_a after mocks are set up
        from rag.templates import q_and_a

        self.q_and_a = q_and_a

        self.callback = MagicMock()

    def tearDown(self):
        self.patcher.stop()

    def test_chunk_excel(self):
        """Test Excel file parsing returns expected Q&A pairs."""
        with patch.object(self.q_and_a, "Excel") as MockParser:
            mock_instance = MockParser.return_value
            # The Excel.__call__ returns a list of (question, answer) tuples
            # Note: rmPrefix strips "A " prefix (matches "A:" pattern) so use answer without A prefix
            mock_instance.return_value = [("What is this?", "This is a test answer.")]

            res = self.q_and_a.chunk("test.xlsx", b"content", callback=self.callback)

            # Verify that the result contains the expected Q&A chunk data
            self.assertEqual(len(res), 1)
            # Check the chunk has the expected question/answer content
            self.assertIn("content_with_weight", res[0])
            self.assertIn("What is this?", res[0]["content_with_weight"])
            self.assertIn("test answer", res[0]["content_with_weight"])
            # Verify tokenization fields are present
            self.assertIn("content_ltks", res[0])
            self.assertIn("content_sm_ltks", res[0])

    def test_chunk_txt(self):
        """Test TXT file parsing with comma-delimited Q&A pairs."""
        filename = "test.txt"
        binary = b"Question,Answer"

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = "Question,Answer"

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            # Verify mock was called with the expected arguments
            mock_get.assert_called_once_with(filename, binary)

            # Verify result structure
            self.assertEqual(len(res), 1)
            # Verify the chunk contains expected Q&A content
            self.assertIn("content_with_weight", res[0])
            self.assertIn("Question", res[0]["content_with_weight"])
            self.assertIn("Answer", res[0]["content_with_weight"])
            # Should have question prefix
            self.assertRegex(res[0]["content_with_weight"], r"(问题：|Question:)")

    def test_chunk_txt_tab_delimited(self):
        """Test TXT file parsing with tab-delimited Q&A pairs."""
        filename = "test.txt"
        binary = b"Q1\tA1\nQ2\tA2"

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = "Q1\tA1\nQ2\tA2"

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            mock_get.assert_called_once_with(filename, binary)
            # Should detect tab delimiter and parse both Q&A pairs
            self.assertEqual(len(res), 2)

    def test_empty_input_txt(self):
        """Test handling of empty TXT input."""
        filename = "empty.txt"
        binary = b""

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = ""

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            # Empty input should return empty result
            self.assertEqual(len(res), 0)

    def test_malformed_q_and_a_txt(self):
        """Test handling of malformed Q&A format (not exactly 2 columns)."""
        filename = "malformed.txt"
        binary = b"only_one_column\n"

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = "only_one_column\n"

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            # Malformed lines should be skipped
            self.assertEqual(len(res), 0)

    def test_missing_delimiters_txt(self):
        """Test handling of lines without proper delimiters."""
        filename = "no_delimiter.txt"
        binary = b"No delimiter here\nAnother line without delimiter"

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = "No delimiter here\nAnother line without delimiter"

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            # Lines without exactly 2 fields should be skipped
            self.assertEqual(len(res), 0)

    def test_mixed_valid_invalid_lines_txt(self):
        """Test handling of mix of valid and invalid lines."""
        filename = "mixed.txt"
        # Line 1: valid, Line 2: invalid (no delimiter), Line 3: valid
        text = "Q1,A1\nInvalid line\nQ2,A2"
        binary = text.encode()

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = text

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            # Should parse 2 valid Q&A pairs
            self.assertEqual(len(res), 2)

    def test_chunk_content_validation_txt(self):
        """Test that chunk content is properly structured."""
        filename = "test.txt"
        question = "What is Python?"
        answer = "A programming language"
        binary = f"{question},{answer}".encode()

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = f"{question},{answer}"

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            self.assertEqual(len(res), 1)
            chunk = res[0]

            # Verify required fields exist
            required_fields = ["docnm_kwd", "title_tks", "content_with_weight", "content_ltks", "content_sm_ltks"]
            for field in required_fields:
                self.assertIn(field, chunk, f"Missing required field: {field}")

            # Verify content contains the original Q&A (minus any prefix stripping)
            self.assertIn("Python", chunk["content_with_weight"])
            self.assertIn("programming language", chunk["content_with_weight"])

    def test_unsupported_file_format(self):
        """Test that unsupported file format raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            self.q_and_a.chunk("test.unsupported", b"content", callback=self.callback)

        self.assertIn("supported", str(context.exception).lower())

    def test_multiline_answer_txt(self):
        """Test handling of answers that span multiple lines."""
        filename = "multiline.txt"
        # First line starts Q&A, subsequent lines continue the answer
        text = "Question1,Answer line 1\nContinuation of answer"
        binary = text.encode()

        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = text

            res = self.q_and_a.chunk(filename, binary, callback=self.callback)

            # The continuation should be appended to the answer
            self.assertEqual(len(res), 1)
            # The answer should contain the continuation
            self.assertIn("Continuation", res[0]["content_with_weight"])

    def test_rmprefix_function(self):
        """Test the rmPrefix helper function."""
        # Test various prefixes that should be stripped
        test_cases = [
            ("Q: What is this?", "What is this?"),
            ("Question: What is this?", "What is this?"),
            ("问题：这是什么？", "这是什么？"),
            ("A: This is the answer", "This is the answer"),
            ("Answer: This is the answer", "This is the answer"),
            ("回答：这是答案", "这是答案"),
            ("user: Hello", "Hello"),
            ("assistant: Hi there", "Hi there"),
            ("问：简单问题", "简单问题"),
            ("答：简单答案", "简单答案"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.q_and_a.rmPrefix(input_text)
                self.assertEqual(result, expected)

    def test_rmprefix_no_prefix(self):
        """Test rmPrefix with text that has no matching prefix."""
        text = "Regular text without prefix"
        result = self.q_and_a.rmPrefix(text)
        self.assertEqual(result, text)

    def test_english_language_flag(self):
        """Test that English language flag affects output formatting."""
        with patch.object(self.q_and_a, "get_text") as mock_get:
            mock_get.return_value = "Question,Answer"

            # Test with English
            res_en = self.q_and_a.chunk("test.txt", b"data", lang="English", callback=self.callback)
            self.assertEqual(len(res_en), 1)
            # English should use "Question:" and "Answer:" prefixes
            self.assertIn("Question:", res_en[0]["content_with_weight"])

            # Test with Chinese (default)
            mock_get.return_value = "Question,Answer"
            res_zh = self.q_and_a.chunk("test.txt", b"data", lang="Chinese", callback=self.callback)
            self.assertEqual(len(res_zh), 1)
            # Chinese should use "问题：" and "回答：" prefixes
            self.assertIn("问题：", res_zh[0]["content_with_weight"])


class TestMdQuestionLevel(unittest.TestCase):
    """Test cases for the mdQuestionLevel helper function."""

    def setUp(self):
        # We need q_and_a module here too
        # Reuse patch.dict or just import if no side effects...
        # mdQuestionLevel depends on token_utils? No, it's regex.
        # But q_and_a import triggers others.
        # So we must mock.
        self.mock_modules = {
            "rag.parsers": MagicMock(),
            "deepdoc": MagicMock(),
            "common": MagicMock(),
            "common.token_utils": MagicMock(),
            "bs4": MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()
        from rag.templates import q_and_a

        self.q_and_a = q_and_a

    def tearDown(self):
        self.patcher.stop()

    def test_heading_levels(self):
        """Test that markdown heading levels are correctly detected."""
        test_cases = [
            ("# Heading 1", (1, "Heading 1")),
            ("## Heading 2", (2, "Heading 2")),
            ("### Heading 3", (3, "Heading 3")),
            ("#### Heading 4", (4, "Heading 4")),
            ("##### Heading 5", (5, "Heading 5")),
            ("###### Heading 6", (6, "Heading 6")),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = self.q_and_a.mdQuestionLevel(input_text)
                self.assertEqual(result, expected)

    def test_no_heading(self):
        """Test that non-heading text returns level 0."""
        result = self.q_and_a.mdQuestionLevel("Regular text")
        self.assertEqual(result[0], 0)

    def test_empty_heading(self):
        """Test heading with no text after hashes."""
        result = self.q_and_a.mdQuestionLevel("### ")
        self.assertEqual(result[0], 3)
        self.assertEqual(result[1].strip(), "")


if __name__ == "__main__":
    unittest.main()
