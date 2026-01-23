import sys
import os
from unittest.mock import MagicMock

# Add root to python path
sys.path.append(os.getcwd())

# Mock tiktoken
mock_tiktoken = MagicMock()
mock_tiktoken.get_encoding.return_value = MagicMock()
sys.modules['tiktoken'] = mock_tiktoken

# Mock common.token_utils
mock_token_utils = MagicMock()
mock_token_utils.num_tokens_from_string.return_value = 10
sys.modules['common.token_utils'] = mock_token_utils

# Mock rag.nlp.rag_tokenizer
mock_rag_tokenizer = MagicMock()
mock_rag_tokenizer.tokenize.return_value = "Hello world This is a test string with some special characters @ # % !"
mock_rag_tokenizer.tag.return_value = "n"
mock_rag_tokenizer.freq.return_value = 100
mock_rag_tokenizer.fine_grained_tokenize.return_value = "Hello world"
sys.modules['rag.nlp.rag_tokenizer'] = mock_rag_tokenizer

# Mock common.file_utils
mock_file_utils = MagicMock()
mock_file_utils.get_project_base_directory.return_value = os.getcwd()
sys.modules['common.file_utils'] = mock_file_utils

# Mock infinity.rag_tokenizer
sys.modules['infinity.rag_tokenizer'] = MagicMock()

# Mock other dependencies imported in rag/nlp/__init__.py
sys.modules['roman_numbers'] = MagicMock()
sys.modules['word2number'] = MagicMock()
sys.modules['cn2an'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['chardet'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import Dealer
try:
    from rag.nlp.term_weight import Dealer
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def verify_pretoken():
    dealer = Dealer()
    text = "Hello, world! This is a test string with some special characters: @#%!"
    res = dealer.pretoken(text)
    print(f"Result: {res}")

if __name__ == "__main__":
    verify_pretoken()
