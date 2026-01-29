import sys
from unittest.mock import MagicMock
import types


def setup_mocks():
    """
    Sets up a comprehensive set of mocks for system modules to allow unit tests to run
    in an environment with missing dependencies.

    Returns:
        dict: A dictionary of the original sys.modules entries that were replaced,
              to be used with teardown_mocks.
    """
    # Save original modules to restore later
    original_modules = sys.modules.copy()

    def mock_package(name):
        m = sys.modules.setdefault(name, types.ModuleType(name))
        if not hasattr(m, "__path__"):
            m.__path__ = []
        return m

    # Base system mocks
    sys.modules["opendal"] = MagicMock()
    # Mock opendal submodules that might be accessed
    sys.modules["opendal.layers"] = MagicMock()
    sys.modules["rag.utils.opendal_conn"] = MagicMock()
    sys.modules["requests"] = MagicMock()
    sys.modules["boto3"] = MagicMock()
    sys.modules["botocore"] = MagicMock()
    sys.modules["minio"] = MagicMock()
    sys.modules["minio.commonconfig"] = MagicMock()
    sys.modules["minio.error"] = MagicMock()

    # Provides lightweight mock implementations of heavy dependencies
    # for local unit testing without Docker.
    #
    # NOTE: If your test requires mocking a new system dependency, please add it here
    # rather than mocking it locally in your test file. This ensures a single source
    # of truth for mocks.
    # RAG Utils Wrappers
    # Mock file_utils which is imported by token_utils
    mock_file_utils = types.ModuleType("common.file_utils")
    mock_file_utils.get_project_base_directory = lambda: "/tmp"
    sys.modules["common.file_utils"] = mock_file_utils

    sys.modules["common.misc_utils"] = types.ModuleType("common.misc_utils")
    sys.modules["common.misc_utils"].get_uuid = lambda: "mock_uuid"

    sys.modules["common.time_utils"] = types.ModuleType("common.time_utils")
    sys.modules["common.time_utils"].current_timestamp = lambda: 1234567890
    sys.modules["common.time_utils"].datetime_format = lambda ts: "2026-01-28 22:00:00"

    sys.modules["common.float_utils"] = MagicMock()
    sys.modules["rag.utils.s3_conn"] = MagicMock()
    sys.modules["rag.utils.minio_conn"] = MagicMock()
    sys.modules["rag.utils.infinity_conn"] = MagicMock()
    sys.modules["rag.utils.azure_spn_conn"] = MagicMock()
    sys.modules["rag.utils.oss_conn"] = MagicMock()

    # Database & Storage
    sys.modules["elasticsearch"] = MagicMock()
    sys.modules["elasticsearch_dsl"] = MagicMock()
    sys.modules["opensearchpy"] = MagicMock()
    sys.modules["oss2"] = MagicMock()
    sys.modules["azure.storage.blob"] = MagicMock()
    sys.modules["azure.storage.blob._generated"] = MagicMock()
    sys.modules["azure.storage.blob._generated.models"] = MagicMock()
    sys.modules["azure.storage.blob._models"] = MagicMock()
    sys.modules["azure.storage.filedatalake"] = MagicMock()
    sys.modules["google.cloud.storage"] = MagicMock()
    sys.modules["redis"] = MagicMock()
    sys.modules["valkey"] = MagicMock()
    sys.modules["valkey.lock"] = MagicMock()

    # NLP & ML Libraries
    sys.modules["infinity"] = MagicMock()
    sys.modules["infinity.common"] = MagicMock()
    sys.modules["infinity.errors"] = MagicMock()
    sys.modules["infinity.index"] = MagicMock()
    sys.modules["infinity.rag_tokenizer"] = MagicMock()
    sys.modules["huggingface_hub"] = MagicMock()
    sys.modules["nltk"] = MagicMock()
    sys.modules["nltk.corpus"] = MagicMock()
    sys.modules["nltk.tokenize"] = MagicMock()
    mock_tiktoken = MagicMock()
    mock_encoder = MagicMock()
    # Mock encode to return a list whose length is roughly number of words
    mock_encoder.encode.side_effect = lambda x: x.split() if x else []
    mock_tiktoken.encoding_for_model.return_value = mock_encoder
    sys.modules["tiktoken"] = mock_tiktoken

    sys.modules["xgboost"] = MagicMock()
    sys.modules["sklearn"] = MagicMock()
    sys.modules["sklearn.cluster"] = MagicMock()
    sys.modules["sklearn.metrics"] = MagicMock()
    sys.modules["sklearn.linear_model"] = MagicMock()
    sys.modules["sklearn.feature_extraction"] = MagicMock()
    sys.modules["sklearn.feature_extraction.text"] = MagicMock()
    sys.modules["deepdoc"] = MagicMock()
    sys.modules["deepdoc.vision"] = MagicMock()

    # Document Processing
    sys.modules["openpyxl"] = MagicMock()
    sys.modules["bs4"] = MagicMock()
    sys.modules["markdown"] = MagicMock()
    sys.modules["jinja2"] = MagicMock()
    sys.modules["json_repair"] = MagicMock()
    sys.modules["pdfplumber"] = MagicMock()
    sys.modules["pypdf"] = MagicMock()
    sys.modules["fitz"] = MagicMock()
    sys.modules["pptx"] = MagicMock()
    sys.modules["pptx.enum"] = MagicMock()
    sys.modules["pptx.enum.shapes"] = MagicMock()
    sys.modules["cv2"] = MagicMock()
    sys.modules["PIL"] = MagicMock()
    sys.modules["tabulate"] = MagicMock()
    sys.modules["tqdm"] = MagicMock()

    # XML/LXML
    sys.modules["lxml"] = MagicMock()
    sys.modules["lxml.etree"] = MagicMock()
    sys.modules["lxml.html"] = MagicMock()

    # Docx (Deep mocking due to complex hierarchy)
    sys.modules["docx"] = MagicMock()
    sys.modules["docx.image"] = MagicMock()
    sys.modules["docx.image.exceptions"] = MagicMock()
    sys.modules["docx.table"] = MagicMock()
    sys.modules["docx.oxml"] = MagicMock()
    sys.modules["docx.oxml.table"] = MagicMock()
    sys.modules["docx.text"] = MagicMock()
    sys.modules["docx.text.paragraph"] = MagicMock()
    sys.modules["docx.oxml.text"] = MagicMock()
    sys.modules["docx.oxml.text.paragraph"] = MagicMock()
    sys.modules["docx.document"] = MagicMock()
    sys.modules["docx.opc"] = MagicMock()
    sys.modules["docx.opc.oxml"] = MagicMock()
    sys.modules["docx.opc.pkgreader"] = MagicMock()
    sys.modules["docx.parts"] = MagicMock()
    sys.modules["docx.parts.document"] = MagicMock()
    sys.modules["docx.parts.customprops"] = MagicMock()
    sys.modules["docx.parts.numbering"] = MagicMock()
    sys.modules["docx.parts.styles"] = MagicMock()
    sys.modules["docx.shared"] = MagicMock()
    sys.modules["docx.enum"] = MagicMock()
    sys.modules["docx.enum.text"] = MagicMock()
    sys.modules["docx.enum.style"] = MagicMock()
    sys.modules["docx.enum.table"] = MagicMock()

    # DB & API
    sys.modules["peewee"] = MagicMock()
    sys.modules["api.db.services.llm_service"] = MagicMock()
    sys.modules["api.db.services.user_service"] = MagicMock()
    sys.modules["api.db.db_models"] = MagicMock()

    # Cloud Providers
    sys.modules["tencentcloud"] = MagicMock()
    sys.modules["tencentcloud.common"] = MagicMock()
    sys.modules["tencentcloud.common.profile"] = MagicMock()
    sys.modules["tencentcloud.common.profile.client_profile"] = MagicMock()
    sys.modules["tencentcloud.common.profile.http_profile"] = MagicMock()
    sys.modules["tencentcloud.common.exception"] = MagicMock()
    sys.modules["tencentcloud.common.exception.tencent_cloud_sdk_exception"] = MagicMock()
    sys.modules["tencentcloud.ocr"] = MagicMock()
    sys.modules["tencentcloud.ocr.v20181119"] = MagicMock()
    sys.modules["tencentcloud.ocr.v20181119.ocr_client"] = MagicMock()
    sys.modules["tencentcloud.lkeap"] = MagicMock()
    sys.modules["tencentcloud.lkeap.v20240522"] = MagicMock()
    sys.modules["tencentcloud.lkeap.v20240522.lkeap_client"] = MagicMock()
    sys.modules["tencentcloud.lkeap.v20240522.models"] = MagicMock()

    # Mock common.token_utils
    sys.modules["common.token_utils"] = MagicMock()
    sys.modules["common.constants"] = MagicMock()
    sys.modules["common.parser_config_utils"] = MagicMock()

    # Special logic for rag_tokenizer to return strings instead of Mocks
    rag_tokenizer = MagicMock()
    rag_tokenizer.tradi2simp.side_effect = lambda x: x
    rag_tokenizer.strQ2B.side_effect = lambda x: x
    rag_tokenizer.tokenize.side_effect = lambda x: x.split() if x else []
    sys.modules["rag.nlp.rag_tokenizer"] = rag_tokenizer

    # Mock common.config_utils
    mock_config_utils = MagicMock()
    # read_config simulates the stored/encrypted config with blank password fields
    mock_config_utils.read_config.return_value = {
        "ragflow": {},
        "service": {"ports": {"8000": 8000}},
        "mysql": {"name": "ragflow", "user": "root", "password": "", "host": "127.0.0.1", "port": 3306},
        "postgres": {"name": "ragflow", "user": "root", "password": "", "host": "127.0.0.1", "port": 5432},
    }
    mock_config_utils.get_base_config.side_effect = lambda name, default=None: default

    # side_effect_decrypt simulates the decrypted secrets (hence "mock_password")
    def side_effect_decrypt(name=None):
        configs = {
            "mysql": {"host": "127.0.0.1", "port": 3306, "name": "ragflow", "user": "root", "password": "mock_password", "engine": "mysql"},
            "postgres": {"host": "127.0.0.1", "port": 5432, "name": "ragflow", "user": "postgres", "password": "mock_password", "engine": "postgres"},
        }
        if name and name.lower() in configs:
            return configs[name.lower()]
        return {"host": "127.0.0.1", "port": 3306, "name": "mock_db", "user": "mock_user", "password": "mock_password"}

    mock_config_utils.decrypt_database_config.side_effect = side_effect_decrypt
    sys.modules["common.config_utils"] = mock_config_utils

    # Mock common package
    # We must preserve submodules we explicitly mocked, but mocking the parent helps with attribute access
    sys.modules["common"] = MagicMock()
    # Link mocked submodules to common mock
    sys.modules["common"].file_utils = sys.modules["common.file_utils"]
    sys.modules["common"].float_utils = sys.modules["common.float_utils"]
    sys.modules["common"].config_utils = sys.modules["common.config_utils"]
    sys.modules["common"].token_utils = sys.modules["common.token_utils"]
    sys.modules["common"].constants = sys.modules["common.constants"]
    sys.modules["common"].parser_config_utils = sys.modules["common.parser_config_utils"]
    sys.modules["common"].misc_utils = sys.modules["common.misc_utils"]
    sys.modules["common"].time_utils = sys.modules["common.time_utils"]

    # Mock rag.nlp package
    mock_rag_nlp = MagicMock()
    mock_rag_nlp.rag_tokenizer = rag_tokenizer
    mock_rag_nlp.bullets_category.return_value = []
    mock_rag_nlp.title_frequency.return_value = (0, [])
    mock_rag_nlp.tokenize.side_effect = lambda x, eng=True: x.split() if isinstance(x, str) else []
    mock_rag_nlp.tokenize_table.return_value = []
    mock_rag_nlp.add_positions.side_effect = lambda d, p: None
    mock_rag_nlp.tokenize_chunks.return_value = []
    mock_rag_nlp.attach_media_context.side_effect = lambda r, t, i: None

    sys.modules["rag.nlp"] = mock_rag_nlp

    # Utils
    sys.modules["PyPDF2"] = MagicMock()
    sys.modules["olefile"] = MagicMock()

    # Web & API Helpers
    mock_package("werkzeug")
    sys.modules["werkzeug.security"] = MagicMock()

    mock_package("playhouse")
    sys.modules["playhouse.pool"] = MagicMock()
    sys.modules["playhouse.migrate"] = MagicMock()

    sys.modules["itsdangerous"] = MagicMock()
    mock_its_url = MagicMock()
    sys.modules["itsdangerous.url_safe"] = mock_its_url

    class MockSerializer:
        def dumps(self, obj):
            return str(obj)

        def loads(self, token, **kwargs):
            return token

    mock_its_url.URLSafeTimedSerializer = MockSerializer

    mock_q_auth = mock_package("quart_auth")

    class MockAuthUser:
        pass

    mock_q_auth.AuthUser = MockAuthUser

    mock_package("quart")
    sys.modules["flask"] = MagicMock()
    sys.modules["flask_login"] = MagicMock()
    sys.modules["flask_cors"] = MagicMock()
    sys.modules["xxhash"] = MagicMock()

    # tenacity decorator mock
    mock_tenacity = mock_package("tenacity")
    mock_tenacity.retry = lambda **kwargs: (lambda f: f)
    mock_tenacity.stop_after_attempt = MagicMock()
    mock_tenacity.wait_exponential = MagicMock()
    mock_tenacity.retry_if_exception_type = MagicMock()

    # peewee exceptions if mocked
    import peewee

    # Guard check for peewee mock - intentional defensive code for future refactors
    if isinstance(peewee, MagicMock):
        peewee.InterfaceError = type("InterfaceError", (Exception,), {})
        peewee.OperationalError = type("OperationalError", (Exception,), {})
        peewee.DoesNotExist = type("DoesNotExist", (Exception,), {})

    return original_modules


def teardown_mocks(original_modules):
    """
    Restores sys.modules to its state before setup_mocks was called.

    Args:
        original_modules (dict): The dictionary returned by setup_mocks.
    """
    if not original_modules:
        return

    # Remove keys added by setup_mocks
    current_keys = list(sys.modules.keys())
    for key in current_keys:
        if key not in original_modules:
            del sys.modules[key]

    # Restore keys that were modified
    sys.modules.update(original_modules)
