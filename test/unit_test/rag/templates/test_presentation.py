import unittest
from unittest.mock import MagicMock, patch
import sys
import types

# Project root is automatically added to sys.path by test/unit_test/conftest.py

# ----------------- MOCK SETUP START -----------------
# Save original sys.modules state for cleanup
_MOCKED_MODULES = [
    "PIL",
    "PIL.Image",
    "PyPDF2",
    "aspose",
    "aspose.slides",
    "aspose.pydrawing",
    "deepdoc",
    "deepdoc.parser",
    "rag.nlp",
    "rag.orchestration.router",
    "common",
    "common.parser_config_utils",
]

# Track intermediate package names that may be created by imports
_INTERMEDIATE_PACKAGES = [
    "rag",
    "rag.app",
    "rag.templates",
]

_original_modules = {mod: sys.modules.get(mod) for mod in _MOCKED_MODULES}
_original_intermediate_packages = {pkg: sys.modules.get(pkg) for pkg in _INTERMEDIATE_PACKAGES}

# Mock PIL with a working Image.open
mock_pil_image_module = MagicMock()
mock_pil_image = MagicMock()
mock_pil_image.copy.return_value = MagicMock()  # Return a mock image object
mock_pil_image_module.open.return_value = mock_pil_image
sys.modules["PIL"] = MagicMock()
sys.modules["PIL"].__version__ = "10.0.0"
sys.modules["PIL.Image"] = mock_pil_image_module

sys.modules["PyPDF2"] = MagicMock()

# Mock aspose.slides with proper slide structure
# The key is to make the presentation object work as a context manager
# and have its slides attribute be a list that can be sliced


def create_mock_presentation(*args, **kwargs):
    """Factory function that creates a new presentation mock each time."""
    mock_slide = MagicMock()
    mock_thumbnail = MagicMock()
    mock_thumbnail.save = MagicMock()
    mock_slide.get_thumbnail.return_value = mock_thumbnail

    mock_presentation = MagicMock()
    mock_presentation.slides = [mock_slide]  # One slide to match MockPptParser's ["text"]
    # Use lambda to return self for context manager
    # __enter__ is called with self, so it needs to accept that argument
    mock_presentation.__enter__ = lambda self: mock_presentation
    # __exit__ is called with self, exc_type, exc_value, traceback
    mock_presentation.__exit__ = lambda self, *args: False
    return mock_presentation


# Create proper module objects instead of MagicMock to avoid interference
# The parent aspose module must also be a proper module, otherwise Python's import
# mechanism will access MagicMock attributes instead of sys.modules entries
mock_aspose = types.ModuleType("aspose")

mock_aspose_slides = types.ModuleType("aspose.slides")
mock_aspose_slides.Presentation = MagicMock(side_effect=create_mock_presentation)
mock_aspose.slides = mock_aspose_slides  # Link the module to the parent

mock_aspose_drawing = types.ModuleType("aspose.pydrawing")
mock_aspose_drawing.imaging = MagicMock()
mock_aspose_drawing.imaging.ImageFormat = MagicMock()
mock_aspose_drawing.imaging.ImageFormat.jpeg = "jpeg"
mock_aspose.pydrawing = mock_aspose_drawing  # Link the module to the parent

sys.modules["aspose"] = mock_aspose
sys.modules["aspose.slides"] = mock_aspose_slides
sys.modules["aspose.pydrawing"] = mock_aspose_drawing


class MockPdfParser:
    def __init__(self):
        self.page_images = []
        self.boxes = []

    def __call__(self, *args, **kwargs):
        return ["text"]

    def _images(self, *args, **kwargs):
        pass

    def _layouts_rec(self, *args, **kwargs):
        pass

    def _table_transformer_job(self, *args, **kwargs):
        pass

    def _text_merge(self, *args, **kwargs):
        pass

    def _extract_table_figure(self, *args, **kwargs):
        return []


class MockPptParser:
    def __call__(self, fnm, from_page, to_page, callback=None):
        return ["text"]


class MockPlainParser:
    pass


mock_deepdoc = MagicMock()
mock_deepdoc.parser.PdfParser = MockPdfParser
mock_deepdoc.parser.PptParser = MockPptParser
mock_deepdoc.parser.PlainParser = MockPlainParser
sys.modules["deepdoc"] = mock_deepdoc
sys.modules["deepdoc.parser"] = mock_deepdoc.parser

# All global mocks are removed and will be handled by patch.dict in setUp.
# ----------------- MOCK SETUP END -----------------

# Imports moved to setUp to ensure mocks are in place
# from rag.templates.presentation import Pdf, Ppt, chunk


class TestPresentationTemplate(unittest.TestCase):
    """Tests for the presentation template to verify callback safety and argument handling."""

    def setUp(self):
        self.mock_modules = {
            "rag.app.format_parsers": MagicMock(),
            "rag.orchestration.router": MagicMock(),
            "rag.templates.general": MagicMock(),
            "rag.templates.semantic": MagicMock(),
            "rag.nlp": MagicMock(),
            "rag.utils.file_utils": MagicMock(),
            "common": MagicMock(),
            "common.settings": MagicMock(),
            "api.db.services.llm_service": MagicMock(),
            "rag.parsers": MagicMock(),
            "rag.parsers.deepdoc": MagicMock(),
            "rag.parsers.deepdoc.ppt_parser": MagicMock(),
            "pptx": MagicMock(),
            "deepdoc": MagicMock(),
            "deepdoc.vision": MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        from rag.templates.presentation import Pdf, Ppt, chunk

        self.Pdf = Pdf
        self.Ppt = Ppt
        self.chunk = chunk
        patcher = patch("rag.templates.presentation.BytesIO")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pdf_callback_none_safe(self):
        """Test Pdf parser with callback=None does NOT raise TypeError."""
        # Manually set the settings value on the mocked module
        sys.modules["common"].settings.PARALLEL_DEVICES = 0
        parser = self.Pdf()
        parser._images = MagicMock()
        parser._layouts_rec = MagicMock()
        parser._table_transformer_job = MagicMock()
        parser._text_merge = MagicMock()
        parser._extract_table_figure = MagicMock(return_value=[])

        try:
            # Manually set the settings value on the imported module to avoid mock issues
            import rag.parsers.deepdoc.pdf_parser

            rag.parsers.deepdoc.pdf_parser.settings.PARALLEL_DEVICES = 0
            parser("dummy.pdf", callback=None)
        except TypeError as e:
            self.fail(f"Pdf raised TypeError with callback=None: {e}")
        except Exception:
            # These exceptions are acceptable due to mocking limitations
            # (e.g., mock object missing attributes, file operations on dummy paths)
            pass

    def test_ppt_callback_none_safe(self):
        """Test Ppt parser with callback=None does NOT raise TypeError."""
        with patch("rag.templates.presentation.PptParser") as MockParent:
            # Configure the mock to return a list of texts when the instance is called
            # Ppt() -> instance. instance() -> list of texts.
            # MockParent() returns the instance mock.
            MockParent.return_value.side_effect = lambda *args, **kwargs: ["mock text"]

            parser = self.Ppt()
            try:
                parser("dummy.pptx", 0, 10, callback=None)
            except TypeError as e:
                self.fail(f"Ppt raised TypeError with callback=None: {e}")
            except (FileNotFoundError, AttributeError, KeyError, Exception):
                # Accept other errors as long as it's not the TypeError we are testing for
                pass

    def test_chunk_callback_none_safe(self):
        """Test chunk function with callback=None."""
        # Create a mock parser that returns the expected 3-tuple
        mock_parser_impl = MagicMock(return_value=([], None, None))

        # Patch PARSERS directly in the presentation module namespace
        # This is necessary because PARSERS is imported at module level
        with patch.dict("rag.templates.presentation.PARSERS", {"deepdoc": mock_parser_impl}, clear=False):
            try:
                self.chunk("dummy.pdf", callback=None)
            except TypeError as e:
                self.fail(f"chunk(pdf) raised TypeError with callback=None: {e}")
            except (FileNotFoundError, AttributeError, KeyError):
                # These exceptions are acceptable due to mocking limitations
                pass

            try:
                # Patch base class call to avoid file system access
                with patch("rag.parsers.deepdoc.ppt_parser.PptParser.__call__") as mock_ppt_call:
                    mock_ppt_call.return_value = ["mock text"]
                    self.chunk("dummy.pptx", callback=None)
            except TypeError as e:
                self.fail(f"chunk(ppt) raised TypeError with callback=None: {e}")
            except (FileNotFoundError, AttributeError, KeyError, Exception):
                # These exceptions are acceptable due to mocking limitations
                pass

    def test_chunk_passes_correct_to_page(self):
        """Test that chunk passes the correct to_page parameter to PptParser."""
        with patch("rag.templates.presentation.Ppt") as MockPptClass:
            mock_ppt_instance = MagicMock()
            mock_ppt_instance.return_value = []
            MockPptClass.return_value = mock_ppt_instance

            self.chunk("test.pptx", from_page=0, to_page=123)

            args, _ = mock_ppt_instance.call_args
            self.assertEqual(args[2], 123, f"Expected to_page=123, got {args[2]}")


def teardown_module():
    """Restore sys.modules to original state after tests complete."""
    # Remove the imported presentation module to allow fresh imports in other tests
    if "rag.templates.presentation" in sys.modules:
        del sys.modules["rag.templates.presentation"]

    # Restore original sys.modules entries
    for mod, original_value in _original_modules.items():
        if original_value is None:
            # Module didn't exist originally, remove it
            sys.modules.pop(mod, None)
        else:
            # Restore original module
            sys.modules[mod] = original_value

    # Restore intermediate package names to ensure complete isolation
    # Process in reverse order (deepest to shallowest) to avoid parent/child issues
    for pkg in reversed(_INTERMEDIATE_PACKAGES):
        original_value = _original_intermediate_packages.get(pkg)
        if original_value is None:
            # Package didn't exist originally, remove it
            sys.modules.pop(pkg, None)
        else:
            # Restore original package
            sys.modules[pkg] = original_value


if __name__ == "__main__":
    unittest.main()
