import sys
import os

# Mimic the test setup
print("Initial sys.path:", sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

print("Modified sys.path:", sys.path)

try:
    import docx

    print("Imported docx:", docx)
    if hasattr(docx, "__file__"):
        print("File:", docx.__file__)
    else:
        print("No __file__ attribute (Namespace package?)")
        if hasattr(docx, "__path__"):
            print("Path:", docx.__path__)
except ImportError as e:
    print("ImportError:", e)
