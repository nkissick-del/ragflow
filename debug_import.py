import sys
import os

print(f"Values in sys.path: {sys.path}", file=sys.stderr)
print(f"CWD: {os.getcwd()}", file=sys.stderr)

import importlib


def try_import(module_path):
    print(f"Attempting to import {module_path}...", file=sys.stderr)
    try:
        mod = importlib.import_module(module_path)
        print(f"{module_path} imported: {mod}", file=sys.stderr)
        return mod
    except Exception as e:
        print(f"Failed to import {module_path}: {e}", file=sys.stderr)
        sys.exit(1)


try_import("rag")
try_import("rag.templates")
try_import("rag.templates.semantic")

print("All imports successful", file=sys.stderr)
