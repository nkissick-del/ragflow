from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .deepdoc.docx_parser import RAGFlowDocxParser as DocxParser
    from .deepdoc.excel_parser import RAGFlowExcelParser as ExcelParser
    from .deepdoc.html_parser import RAGFlowHtmlParser as HtmlParser
    from .deepdoc.json_parser import RAGFlowJsonParser as JsonParser
    from .deepdoc.markdown_parser import RAGFlowMarkdownParser as MarkdownParser, MarkdownElementExtractor
    from .deepdoc.pdf_parser import RAGFlowPdfParser as PdfParser, PlainParser
    from .deepdoc.ppt_parser import RAGFlowPptParser as PptParser
    from .deepdoc.txt_parser import RAGFlowTxtParser as TxtParser

__all__ = [
    "DocxParser",
    "ExcelParser",
    "HtmlParser",
    "JsonParser",
    "MarkdownParser",
    "MarkdownElementExtractor",
    "PdfParser",
    "PlainParser",
    "PptParser",
    "TxtParser",
]

_LAZY_MODULES = {
    "DocxParser": (".deepdoc.docx_parser", "RAGFlowDocxParser"),
    "ExcelParser": (".deepdoc.excel_parser", "RAGFlowExcelParser"),
    "HtmlParser": (".deepdoc.html_parser", "RAGFlowHtmlParser"),
    "JsonParser": (".deepdoc.json_parser", "RAGFlowJsonParser"),
    "MarkdownParser": (".deepdoc.markdown_parser", "RAGFlowMarkdownParser"),
    "MarkdownElementExtractor": (".deepdoc.markdown_parser", "MarkdownElementExtractor"),
    "PdfParser": (".deepdoc.pdf_parser", "RAGFlowPdfParser"),
    "PlainParser": (".deepdoc.pdf_parser", "PlainParser"),
    "PptParser": (".deepdoc.ppt_parser", "RAGFlowPptParser"),
    "TxtParser": (".deepdoc.txt_parser", "RAGFlowTxtParser"),
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module_path, class_name = _LAZY_MODULES[name]
        try:
            # Import relatively from the current package (rag.parsers)
            import importlib

            module = importlib.import_module(module_path, package=__package__)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not lazy load {name} from {module_path}: {e}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
