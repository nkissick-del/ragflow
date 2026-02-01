import sys
import traceback

print(f"Python: {sys.version}")

print("\n--- Trying to import quart.render_template_string ---")
try:
    from quart import render_template_string

    print("SUCCESS: quart.render_template_string imported")
except ImportError as e:
    print(f"FAILURE: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n--- Trying to import rag.utils.es_conn ---")
try:
    import rag.utils.es_conn

    print("SUCCESS: rag.utils.es_conn imported")
except ImportError as e:
    print(f"FAILURE: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
