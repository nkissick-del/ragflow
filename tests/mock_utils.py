import sys
from unittest.mock import MagicMock


def setup_mocks():
    """
    Sets up a comprehensive set of mocks for system modules to allow unit tests to run
    in an environment with missing dependencies.
    """
    # Base system mocks
    sys.modules["opendal"] = MagicMock()
    sys.modules["rag.utils.opendal_conn"] = MagicMock()
    sys.modules["boto3"] = MagicMock()
    sys.modules["botocore"] = MagicMock()
    sys.modules["minio"] = MagicMock()
    sys.modules["minio.commonconfig"] = MagicMock()
    sys.modules["minio.error"] = MagicMock()

    # RAG Utils Wrappers (mocking these avoids importing their heavy dependencies)
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
    sys.modules["tiktoken"] = MagicMock()
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
    sys.modules["api.db"] = MagicMock()
    sys.modules["api.db.services"] = MagicMock()
    sys.modules["api.db.services.llm_service"] = MagicMock()
    sys.modules["api.db.services.user_service"] = MagicMock()

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

    # Utils
    sys.modules["PyPDF2"] = MagicMock()
    sys.modules["olefile"] = MagicMock()

    # Special logic for rag_tokenizer to return strings instead of Mocks
    rag_tokenizer = MagicMock()
    rag_tokenizer.tradi2simp.side_effect = lambda x: x
    rag_tokenizer.strQ2B.side_effect = lambda x: x
    rag_tokenizer.tokenize.side_effect = lambda x: x
    sys.modules["rag.nlp.rag_tokenizer"] = rag_tokenizer
