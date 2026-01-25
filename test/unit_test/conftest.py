"""
Unit test configuration file for RAGFlow.

This file handles mocking of heavy dependencies (like vision libraries, PDF parsers, etc.)
that are difficult to install or unnecessary for unit testing logic.

IMPORTANT: Mocks are organized and ordered specifically to ensure submodules are mocked
before their parent packages are imported or mocked, to avoid ImportErrors.
"""

import sys
from unittest.mock import MagicMock
import warnings
import types
from pathlib import Path

# Suppress FutureWarning from google packages (protobuf deprecation warnings)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")


# =============================================================================
# Project path setup
# =============================================================================
def _setup_project_path():
    """Add project root to sys.path by finding repo marker files.

    Walks up from this file's directory to find the repository root
    (identified by pyproject.toml, setup.cfg, or .git), then adds it
    to sys.path if not already present.
    """
    current = Path(__file__).resolve().parent
    repo_markers = ["pyproject.toml", "setup.cfg", ".git"]

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check if any repo marker exists in this directory
        if any((parent / marker).exists() for marker in repo_markers):
            repo_root = str(parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            return repo_root

    # Fallback: couldn't find repo root
    raise RuntimeError(f"Could not find repository root from {__file__}. Looked for: {', '.join(repo_markers)}")


# Set up project path before any imports
_setup_project_path()

# =============================================================================
# Mock heavy dependencies that are difficult to install on Mac
# These mocks allow unit tests to run without Docker
#
# IMPORTANT: We must mock submodules BEFORE importing parent packages
# =============================================================================

# 0. Mock beartype FIRST (type checking library used by deepdoc)
# This must happen before deepdoc is imported
mock_beartype = types.ModuleType("beartype")
mock_beartype_claw = types.ModuleType("beartype.claw")
mock_beartype_claw.beartype_this_package = lambda: None
mock_beartype.claw = mock_beartype_claw
sys.modules["beartype"] = mock_beartype
sys.modules["beartype.claw"] = mock_beartype_claw

# 0a. Mock deepdoc.vision package FIRST (prevents heavy CV/OCR imports)
# The deepdoc.vision module has many heavy dependencies (cv2, onnxruntime, etc.)
mock_deepdoc_vision = MagicMock()
sys.modules["deepdoc.vision"] = mock_deepdoc_vision

# 0b. Mock shapely (geometry library used by deepdoc vision)
sys.modules["shapely"] = MagicMock()
sys.modules["shapely.geometry"] = MagicMock()

# 0b2. Mock pyclipper (geometry clipping library used by deepdoc vision)
sys.modules["pyclipper"] = MagicMock()

# 0c. Mock sklearn (machine learning library used by deepdoc pdf_parser)
mock_sklearn = MagicMock()
sys.modules["sklearn"] = mock_sklearn
sys.modules["sklearn.cluster"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()

# 0d. Mock cv2 (OpenCV library used by deepdoc vision)
sys.modules["cv2"] = MagicMock()

# 0c3. Mock python-pptx library (pptx, pptx.enum, pptx.enum.shapes)
sys.modules["pptx"] = MagicMock()
sys.modules["pptx.enum"] = MagicMock()
sys.modules["pptx.enum.shapes"] = MagicMock()

# 0e. Mock onnxruntime (used by deepdoc vision)
sys.modules["onnxruntime"] = MagicMock()

# 0e. Mock pdfplumber (heavy PDF library used by deepdoc)
sys.modules["pdfplumber"] = MagicMock()

# 0f. Mock openpyxl (Excel library used by deepdoc)
mock_openpyxl = types.ModuleType("openpyxl")
mock_openpyxl.Workbook = MagicMock()
mock_openpyxl.load_workbook = MagicMock()
sys.modules["openpyxl"] = mock_openpyxl

# 0h. Mock PIL (image library) - need ImageFont for pptx
mock_pil = types.ModuleType("PIL")
mock_pil.Image = MagicMock()
mock_pil.ImageFont = MagicMock()
mock_pil.__version__ = "10.0.0"  # pypdf checks this at import time
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image
sys.modules["PIL.ImageFont"] = mock_pil.ImageFont

# 0i. Mock markdown library
sys.modules["markdown"] = MagicMock()

# 0k. Mock bs4 (BeautifulSoup)
sys.modules["bs4"] = MagicMock()

# 0l. Mock huggingface_hub
sys.modules["huggingface_hub"] = MagicMock()

# 0m. Mock pypdf
sys.modules["pypdf"] = MagicMock()


# 0n. Mock peewee
class MockField:
    def __init__(self, *args, **kwargs):
        pass


sys.modules["peewee"] = MagicMock()
sys.modules["peewee"].Model = MagicMock
sys.modules["peewee"].CharField = MockField
sys.modules["peewee"].DateTimeField = MockField
sys.modules["peewee"].FloatField = MockField
sys.modules["peewee"].IntegerField = MockField
sys.modules["peewee"].TextField = MockField
sys.modules["peewee"].Field = MockField

# 0o. Mock werkzeug
sys.modules["werkzeug"] = MagicMock()
sys.modules["werkzeug.security"] = MagicMock()

# 0p. Mock playhouse
mock_playhouse = types.ModuleType("playhouse")
sys.modules["playhouse"] = mock_playhouse
sys.modules["playhouse.pool"] = MagicMock()
sys.modules["playhouse.shortcuts"] = MagicMock()
sys.modules["playhouse.migrate"] = MagicMock()

# 0q. Mock xxhash
sys.modules["xxhash"] = MagicMock()

# 0r. Mock itsdangerous
sys.modules["itsdangerous"] = MagicMock()
sys.modules["itsdangerous.url_safe"] = MagicMock()

# 0s. Mock quart_auth
sys.modules["quart_auth"] = MagicMock()
sys.modules["quart_auth"].AuthUser = MagicMock

# 0t. Mock tenacity
sys.modules["tenacity"] = MagicMock()

# 0u. Mock langfuse
sys.modules["langfuse"] = MagicMock()

# 0v. Mock json_repair
sys.modules["json_repair"] = MagicMock()

# 0w. Mock litellm
sys.modules["litellm"] = MagicMock()

# 0x. Mock openai
mock_openai = types.ModuleType("openai")
mock_openai.AsyncOpenAI = MagicMock()
mock_openai.OpenAI = MagicMock()
sys.modules["openai"] = mock_openai
sys.modules["openai.lib"] = MagicMock()
sys.modules["openai.lib.azure"] = MagicMock()

# 0y. Mock jinja2
sys.modules["jinja2"] = MagicMock()

# 0z. Mock LLM providers
sys.modules["dashscope"] = MagicMock()
sys.modules["zhipuai"] = MagicMock()
sys.modules["volcengine"] = MagicMock()
sys.modules["volcengine.maas"] = MagicMock()
sys.modules["ollama"] = MagicMock()
sys.modules["boto3"] = MagicMock()
sys.modules["qianfan"] = MagicMock()

# 0z2. Mock httpx
sys.modules["httpx"] = MagicMock()

# 0z3. Mock yarl
sys.modules["yarl"] = MagicMock()

# 0z4. Mock ormsgpack
sys.modules["ormsgpack"] = MagicMock()

# 0z5. Mock websocket
sys.modules["websocket"] = MagicMock()

# 0z6. Mock PyPDF2
sys.modules["PyPDF2"] = MagicMock()

# 0z7. Mock olefile
sys.modules["olefile"] = MagicMock()

# 0z8. Mock deepdoc.parser submodules to prevent heavy imports
# These are imported by deepdoc.parser.__init__ and have heavy dependencies
sys.modules["deepdoc.parser.pdf_parser"] = MagicMock()
sys.modules["deepdoc.parser.ppt_parser"] = MagicMock()
sys.modules["deepdoc.parser.docx_parser"] = MagicMock()
sys.modules["deepdoc.parser.excel_parser"] = MagicMock()
sys.modules["deepdoc.parser.html_parser"] = MagicMock()

# 1. Mock opendal FIRST (Rust-based storage library, hard to install on Mac)
# This must happen before common.settings is imported
mock_opendal = MagicMock()
sys.modules["opendal"] = mock_opendal

# 2. Mock infinity FIRST (RAGFlow's tokenizer library)
# Create mock module structure
mock_infinity = types.ModuleType("infinity")
mock_infinity_rag_tokenizer = types.ModuleType("infinity.rag_tokenizer")


class MockInfinityRagTokenizer:
    """Mock tokenizer that returns input as-is."""

    def tokenize(self, line):
        return line if line else ""

    def fine_grained_tokenize(self, tks):
        return tks if tks else ""

    def tag(self, text):
        return []

    def freq(self, word):
        return 0

    def _tradi2simp(self, text):
        return text

    def _strQ2B(self, text):
        return text


mock_infinity_rag_tokenizer.RagTokenizer = MockInfinityRagTokenizer
mock_infinity_rag_tokenizer.is_chinese = lambda s: False
mock_infinity_rag_tokenizer.is_number = lambda s: False
mock_infinity_rag_tokenizer.is_alphabet = lambda s: s.isalpha() if s else False
mock_infinity_rag_tokenizer.naive_qie = lambda txt: txt.split() if txt else []
mock_infinity.rag_tokenizer = mock_infinity_rag_tokenizer

sys.modules["infinity"] = mock_infinity
sys.modules["infinity.rag_tokenizer"] = mock_infinity_rag_tokenizer

# 3. Mock Google Generative AI to avoid FutureWarning at import time
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai
sys.modules["google.genai"] = MagicMock()

# 4. Mock XGBoost to avoid libomp dependency issues on Mac
mock_xgb = MagicMock()
sys.modules["xgboost"] = mock_xgb

# 5. Mock file_utils which is imported by token_utils
mock_file_utils = types.ModuleType("common.file_utils")
mock_file_utils.get_project_base_directory = lambda: "/tmp"
sys.modules["common.file_utils"] = mock_file_utils

# 6. Mock common.settings submodule (depends on opendal via OpenDALStorage)
mock_settings = types.ModuleType("common.settings")
mock_settings.DOC_ENGINE_INFINITY = False
mock_settings.REDIS_CONN = None
mock_settings.ELASTICSEARCH_CONN = None
mock_settings.DATABASE_TYPE = "mysql"
mock_settings.RAG_FLOW_EDITION = "community"
mock_settings.ES_CONN = MagicMock()
mock_settings.docStoreConn = MagicMock()
mock_settings.DATABASE = {
    "name": "ragflow",
    "user": "root",
    "password": "mock_password",
    "host": "127.0.0.1",
    "port": 3306,
}
mock_settings.PAGER = None  # Mock pager config
mock_settings.PARALLEL_DEVICES = 0  # Mock parallel devices config
sys.modules["common.settings"] = mock_settings

# 7. Mock common.config_utils
mock_config_utils = MagicMock()

mock_config_utils.read_config.return_value = {
    "ragflow": {},
    "service": {"ports": {"8000": 8000}},
    "mysql": {"name": "ragflow", "user": "root", "password": "", "host": "127.0.0.1", "port": 3306},
    "postgres": {"name": "ragflow", "user": "root", "password": "", "host": "127.0.0.1", "port": 5432},
}


def side_effect_get_base(name, default=None):
    return default


mock_config_utils.get_base_config.side_effect = side_effect_get_base


def side_effect_decrypt(name=None):
    """Return different DB configs based on the name parameter."""
    configs = {
        "mysql": {
            "host": "127.0.0.1",
            "port": 3306,
            "name": "ragflow",
            "user": "root",
            "password": "mock_password",
            "engine": "mysql",
        },
        "postgres": {
            "host": "127.0.0.1",
            "port": 5432,
            "name": "ragflow",
            "user": "postgres",
            "password": "mock_password",
            "engine": "postgres",
        },
    }
    # Return specific config if name matches, otherwise return default (mysql-like)
    if name and name.lower() in configs:
        return configs[name.lower()]
    return {
        "host": "127.0.0.1",
        "port": 3306,
        "name": "mock_db",
        "user": "mock_user",
        "password": "mock_password",
    }


mock_config_utils.decrypt_database_config.side_effect = side_effect_decrypt

sys.modules["common.config_utils"] = mock_config_utils

# 8. Mock rag.utils connections (OpenDAL, etc.)
sys.modules["rag.utils.ob_conn"] = MagicMock()
sys.modules["rag.utils.opendal_conn"] = MagicMock()

# 9. Mock tencentcloud (SDK)
mock_tencentcloud = MagicMock()
sys.modules["tencentcloud"] = mock_tencentcloud
sys.modules["tencentcloud.common"] = MagicMock()
sys.modules["tencentcloud.common.credential"] = MagicMock()
sys.modules["tencentcloud.common.profile"] = MagicMock()
sys.modules["tencentcloud.common.profile.client_profile"] = MagicMock()
sys.modules["tencentcloud.common.profile.http_profile"] = MagicMock()
sys.modules["tencentcloud.common.exception"] = MagicMock()
sys.modules["tencentcloud.common.exception.tencent_cloud_sdk_exception"] = MagicMock()
sys.modules["tencentcloud.ocr"] = MagicMock()
sys.modules["tencentcloud.ocr.v20181119"] = MagicMock()
sys.modules["tencentcloud.ocr.v20181119.ocr_client"] = MagicMock()
sys.modules["tencentcloud.ocr.v20181119.models"] = MagicMock()
sys.modules["tencentcloud.lkeap"] = MagicMock()
sys.modules["tencentcloud.lkeap.v20240522"] = MagicMock()
sys.modules["tencentcloud.lkeap.v20240522.lkeap_client"] = MagicMock()
sys.modules["tencentcloud.lkeap.v20240522.models"] = MagicMock()
