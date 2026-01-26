import sys
import os
import time
import json
import random
from peewee import SqliteDatabase
from unittest.mock import MagicMock

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, repo_root)

# Mock redis
sys.modules['rag.utils.redis_conn'] = MagicMock()
sys.modules['rag.utils.redis_conn'].REDIS_CONN = MagicMock()

# Mock settings
sys.modules['common.settings'] = MagicMock()
sys.modules['common.settings'].DATABASE_TYPE = 'sqlite'
sys.modules['common.settings'].check_db_type = MagicMock(return_value=True)

# Mock heavy dependencies to avoid installing them
sys.modules['api.db.services.tenant_llm_service'] = MagicMock()
sys.modules['rag.llm'] = MagicMock()
sys.modules['rag.nlp'] = MagicMock() # Also likely heavy

# Create file DB to persist across connection closes
if os.path.exists("test.db"):
    os.remove("test.db")
test_db = SqliteDatabase('test.db')

# Add lock decorator mock
def lock_decorator(name, timeout):
    def decorator(func):
        return func
    return decorator

test_db.lock = lock_decorator

# Create a mock module for api.db.connection
connection_mock = MagicMock()
connection_mock.DB = test_db
connection_mock.ensure_database_exists = MagicMock()

# Inject into sys.modules
sys.modules['api.db.connection'] = connection_mock

# Import models
from api.db.models.knowledge import Document, File, File2Document

# Bind models to test_db explicitly
models = [Document, File, File2Document]
test_db.bind(models)
test_db.connect()
test_db.create_tables(models)

# Patch DB in api.db.db_models
import api.db.db_models
api.db.db_models.DB = test_db

# Import DocumentService
from api.db.services.document_service import DocumentService

def populate_db(count=10000):
    print(f"Populating DB with {count} documents...")
    random.seed(42) # Deterministic seed for consistency
    kb_id = "test_kb"
    docs = []
    files = []
    f2ds = []

    suffixes = ['pdf', 'docx', 'txt', 'md', 'html']
    run_statuses = ['0', '1', '2']

    for i in range(count):
        doc_id = f"doc_{i}"
        file_id = f"file_{i}"

        suffix = random.choice(suffixes)
        run = random.choice(run_statuses)

        # Random metadata
        meta = {}
        if random.random() > 0.2:
            meta["author"] = f"author_{random.randint(0, 10)}"
            if random.random() > 0.5:
                 meta["tag"] = [f"tag_{random.randint(0, 5)}", f"tag_{random.randint(0, 5)}"]
            else:
                 meta["tag"] = f"tag_{random.randint(0, 5)}"

        docs.append({
            "id": doc_id,
            "kb_id": kb_id,
            "parser_id": "naive",
            "source_type": "local",
            "type": suffix,
            "created_by": "user1",
            "name": f"doc_{i}.{suffix}",
            "suffix": suffix,
            "run": run,
            "meta_fields": meta
        })

        files.append({
            "id": file_id,
            "parent_id": "root",
            "tenant_id": "tenant1",
            "created_by": "user1",
            "name": f"doc_{i}.{suffix}",
            "type": suffix,
            "source_type": "local"
        })

        f2ds.append({
            "id": f"f2d_{i}",
            "file_id": file_id,
            "document_id": doc_id
        })

    with test_db.atomic():
        # Insert in chunks to avoid sqlite limits
        chunk_size = 500
        for i in range(0, len(docs), chunk_size):
            Document.insert_many(docs[i:i+chunk_size]).execute()
            File.insert_many(files[i:i+chunk_size]).execute()
            File2Document.insert_many(f2ds[i:i+chunk_size]).execute()

    print("Population done.")

def benchmark():
    kb_id = "test_kb"
    start_time = time.time()
    result, total = DocumentService.get_filter_by_kb_id(kb_id, keywords=None, run_status=None, types=None, suffix=None)
    end_time = time.time()

    print(f"Total documents: {total}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    # Sort keys for deterministic output comparison
    def sort_dict(d):
        if isinstance(d, dict):
            return {k: sort_dict(v) for k, v in sorted(d.items())}
        return d

    return end_time - start_time, sort_dict(result)

if __name__ == "__main__":
    populate_db(50000)
    print("Running benchmark...")
    t1, r1 = benchmark()
    t2, r2 = benchmark()

    print(f"Average time: {(t1+t2)/2:.4f}")

    # Save result to file for later comparison
    if not os.path.exists("benchmark_result.json"):
        print("Saving baseline result...")
        with open("benchmark_result.json", "w") as f:
            json.dump(r1, f, indent=2)
    else:
        with open("benchmark_result.json", "r") as f:
            old_result = json.load(f)

        # Normalize keys to strings just in case
        r1_s = json.dumps(r1, sort_keys=True)
        old_s = json.dumps(old_result, sort_keys=True)

        if r1_s == old_s:
             print("SUCCESS: Results match!")
        else:
             print("FAILURE: Results do not match!")

    # Cleanup
    test_db.close()
    if os.path.exists("test.db"):
        os.remove("test.db")
