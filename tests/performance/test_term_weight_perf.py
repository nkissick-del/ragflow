import sys
import os
import time
from unittest.mock import MagicMock

# Add root to python path
sys.path.append(os.getcwd())

# Mock dependencies
sys.modules['tiktoken'] = MagicMock()
sys.modules['common.token_utils'] = MagicMock()

mock_rag_tokenizer = MagicMock()
# Return many tokens to stress the loop
mock_rag_tokenizer.tokenize.return_value = "token " * 1000
sys.modules['rag.nlp.rag_tokenizer'] = mock_rag_tokenizer

sys.modules['common.file_utils'] = MagicMock()
sys.modules['infinity.rag_tokenizer'] = MagicMock()
sys.modules['roman_numbers'] = MagicMock()
sys.modules['word2number'] = MagicMock()
sys.modules['cn2an'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['chardet'] = MagicMock()
sys.modules['numpy'] = MagicMock()

try:
    from rag.nlp.term_weight import Dealer
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_pretoken_perf():
    dealer = Dealer()
    # A string with some special characters to trigger the regexes
    text = "Hello, world! @#%! " * 100

    start_time = time.time()
    for _ in range(100):
        dealer.pretoken(text)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    test_pretoken_perf()
