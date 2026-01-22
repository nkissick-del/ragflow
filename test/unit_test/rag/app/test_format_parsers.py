import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from rag.app import format_parsers


class TestFormatParsers(unittest.TestCase):
    def test_parser_existence(self):
        # Verify that key parser classes exist and are importable
        self.assertTrue(hasattr(format_parsers, "Docx"))
        self.assertTrue(hasattr(format_parsers, "Pdf"))
        self.assertTrue(hasattr(format_parsers, "Markdown"))

    def test_parser_instantiation(self):
        # Test basic instantiation if classes allow it without complex init
        docx_parser = format_parsers.Docx()
        self.assertIsNotNone(docx_parser)

        # Pdf parser inherits from PdfParser - validate inheritance and interface
        from deepdoc.parser import PdfParser

        self.assertTrue(issubclass(format_parsers.Pdf, PdfParser))

        # Verify essential methods exist on the Pdf class
        self.assertTrue(hasattr(format_parsers.Pdf, "__call__"))
        self.assertTrue(callable(getattr(format_parsers.Pdf, "__call__", None)))

        # Verify the Pdf class can be instantiated
        pdf_parser = format_parsers.Pdf()
        self.assertIsInstance(pdf_parser, format_parsers.Pdf)
        self.assertIsInstance(pdf_parser, PdfParser)


if __name__ == "__main__":
    unittest.main()
