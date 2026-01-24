import logging
from rag.parsers import PdfParser, DocxParser, MarkdownParser
from rag.parsers.deepdoc.figure_parser import vision_figure_parser_pdf_wrapper


class DeepDocParser:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def check_installation(self) -> bool:
        # DeepDoc is internal, assume always available if imported
        return True

    def parse_pdf(
        self,
        filepath: str,
        binary: bytes = None,
        callback=None,
        from_page=0,
        to_page=100000,
        **kwargs,
    ):
        """
        Parse PDF using DeepDoc RAGFlowPdfParser.
        """
        try:
            pdf_parser = PdfParser()
            sections, tables = pdf_parser(filepath if not binary else binary, from_page=from_page, to_page=to_page, callback=callback)

            # Apply figure parsing wrapper as done in router.py
            tables = vision_figure_parser_pdf_wrapper(
                tbls=tables,
                sections=sections,
                callback=callback,
                **kwargs,
            )
            return sections, tables, pdf_parser
        except Exception as e:
            self.logger.error(f"[DeepDoc] PDF parsing failed: {e}")
            if callback:
                callback(-1, f"[DeepDoc] PDF parsing failed: {e}")
            return [], [], None

    def parse_docx(
        self,
        filepath: str,
        binary: bytes = None,
        callback=None,
        **kwargs,
    ):
        """
        Parse DOCX using DeepDoc RAGFlowDocxParser.
        """
        try:
            docx_parser = DocxParser()
            sections = docx_parser(filepath, binary)
            return sections, None
        except Exception as e:
            self.logger.error(f"[DeepDoc] DOCX parsing failed: {e}")
            if callback:
                callback(-1, f"[DeepDoc] DOCX parsing failed: {e}")
            return [], None

    def parse_markdown(
        self,
        filepath: str,
        binary: bytes = None,
        callback=None,
        **kwargs,
    ):
        """
        Parse Markdown using DeepDoc RAGFlowMarkdownParser.
        """
        parser_config = kwargs.get("parser_config", {})
        try:
            chunk_token_num = int(parser_config.get("chunk_token_num", 128))
            markdown_parser = MarkdownParser(chunk_token_num)

            sections, tables, section_images = markdown_parser(
                filepath,
                binary,
                separate_tables=False,
                delimiter=parser_config.get("delimiter", "\n!?;。；！？"),
                return_section_images=True,
            )
            return sections, tables, section_images
        except Exception as e:
            self.logger.error(f"[DeepDoc] Markdown parsing failed: {e}")
            if callback:
                callback(-1, f"[DeepDoc] Markdown parsing failed: {e}")
            return [], [], []
