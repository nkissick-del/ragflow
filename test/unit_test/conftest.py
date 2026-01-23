import sys
from unittest.mock import MagicMock
import warnings
import types
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")


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
    raise RuntimeError(
        f"Could not find repository root from {__file__}. "
        f"Looked for: {', '.join(repo_markers)}"
    )


# Set up project path before any imports
_setup_project_path()

# =============================================================================
# Mock heavy dependencies that are difficult to install on Mac
# These mocks allow unit tests to run without Docker
#
# IMPORTANT: We must mock submodules BEFORE importing parent packages
# =============================================================================

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
    return {"host": "127.0.0.1", "port": 3306, "name": "mock_db", "user": "mock_user", "password": "mock_password"}


mock_config_utils.decrypt_database_config.side_effect = side_effect_decrypt

sys.modules["common.config_utils"] = mock_config_utils

# 8. Mock rag.utils connections (OpenDAL, etc.)
sys.modules["rag.utils.ob_conn"] = MagicMock()
sys.modules["rag.utils.opendal_conn"] = MagicMock()
