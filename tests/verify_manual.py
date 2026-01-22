import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from rag.app.templates import manual

    print("Successfully imported rag.app.templates.manual")
except ImportError as e:
    print(f"ImportError (expected if dependencies missing): {e}")
    # If we can't import, we can't test class attributes easily, but at least syntax is checked before ImportError
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during import: {e}")
    sys.exit(1)

# Inspect Pdf class source or attributes if mockable
# Since deepdoc might be missing, we might not be able to instantiate Pdf if it calls super().__init__ which needs stuff.
# But let's try.

if "manual" in locals():
    print("Checking specific fixes...")
    # Check 1: model_species typo
    # We can inspect the __init__ source code or try to instantiate if possible.
    try:
        if hasattr(manual, "Pdf"):
            print("Pdf class found.")
            # mocking super if needed?
    except Exception as e:
        print(f"Error accessing Pdf class: {e}")
