import sys
from unittest.mock import MagicMock
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Mock Google Generative AI to avoid FutureWarning at import time
mock_genai = MagicMock()
# Do NOT mock sys.modules["google"] as it breaks other google.* imports (e.g. google.cloud)
sys.modules["google.generativeai"] = mock_genai
sys.modules["google.genai"] = MagicMock()

# 1. Mock XGBoost to avoid libomp dependency issues on Mac
mock_xgb = MagicMock()
sys.modules["xgboost"] = mock_xgb

# 2. Mock common.config_utils
# We need to ensure that when settings.py calls `read_config` or `decrypt_database_config` it gets valid data.
mock_config_utils = MagicMock()

# Mock return for read_config() -> CONFIGS
mock_config_utils.read_config.return_value = {
    # Add keys that might be accessed
    "ragflow": {},
    "service": {"ports": {"8000": 8000}},
    "mysql": {"name": "ragflow", "user": "root", "password": "", "host": "127.0.0.1", "port": 3306},
    "postgres": {"name": "ragflow", "user": "root", "password": "", "host": "127.0.0.1", "port": 5432},
}


# Mock get_base_config
def side_effect_get_base(name, default=None):
    return default


mock_config_utils.get_base_config.side_effect = side_effect_get_base


# Mock decrypt_database_config
# It usually expects 'name' argument.
def side_effect_decrypt(name=None):
    return {"host": "127.0.0.1", "port": 3306, "name": "mock_db", "user": "mock_user", "password": "mock_password"}


mock_config_utils.decrypt_database_config.side_effect = side_effect_decrypt

sys.modules["common.config_utils"] = mock_config_utils
