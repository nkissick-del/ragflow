import sys
import os

print(f"Values in sys.path: {sys.path}", file=sys.stderr)
print(f"CWD: {os.getcwd()}", file=sys.stderr)

try:
    print("Attempting to import rag...", file=sys.stderr)
    import rag

    print(f"rag imported: {rag}", file=sys.stderr)
except Exception as e:
    print(f"Failed to import rag: {e}", file=sys.stderr)
    sys.exit(1)

try:
    print("Attempting to import rag.templates...", file=sys.stderr)
    import rag.templates

    print(f"rag.templates imported: {rag.templates}", file=sys.stderr)
except Exception as e:
    print(f"Failed to import rag.templates: {e}", file=sys.stderr)
    sys.exit(1)

try:
    print("Attempting to import rag.templates.semantic...", file=sys.stderr)
    import rag.templates.semantic

    print(f"rag.templates.semantic imported: {rag.templates.semantic}", file=sys.stderr)
except Exception as e:
    print(f"Failed to import rag.templates.semantic: {e}", file=sys.stderr)
    sys.exit(1)

print("All imports successful", file=sys.stderr)
