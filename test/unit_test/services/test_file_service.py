
import pytest
from peewee import SqliteDatabase
from api.db.services.file_service import FileService
from api.db.db_models import File
from api.db import FileType
from api.db.connection import DB
import uuid

# Use an in-memory SQLite database for testing
db_instance = SqliteDatabase(':memory:')

class TestFileService:
    def setup_method(self):
        # Save the original database to restore later
        self.original_db = File._meta.database
        # Bind the model to the test database
        File._meta.database = db_instance
        db_instance.bind([File])
        db_instance.connect()
        db_instance.create_tables([File])

    def teardown_method(self):
        db_instance.drop_tables([File])
        db_instance.close()
        # Restore the original database connection
        File._meta.database = self.original_db

    def test_get_folder_size(self, monkeypatch):
        # Patch the DB object in file_service to use our test db or a mock that does nothing on connection_context
        # The easiest way is to mock connection_context to return a dummy context manager

        class MockContext:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_val, exc_tb): return None
            def __call__(self, func):
                def wrapper(*args, **kwargs):
                    with self:
                        return func(*args, **kwargs)
                return wrapper

        # We need to patch the DB object in api.db.services.file_service
        # Because @DB.connection_context() is evaluated at import time (decorator application),
        # patching it NOW (at runtime) won't change the decorator already applied to the function.

        # ! IMPORTANT !
        # Since the decorator is already applied, we cannot easily unwrap it or patch the DB it uses
        # (unless the decorator uses `self` or a global that we can patch).
        # Peewee's `connection_context` is usually an instance method of the database object.

        # However, `FileService.get_folder_size` is already decorated.
        # If we want to test the logic without the decorator interference, we might need to access the original function
        # if it was wrapped with `functools.wraps` (which usually preserves `__wrapped__`).

        # Let's check if we can access `__wrapped__`.
        if hasattr(FileService.get_folder_size, '__wrapped__'):
             # Call the unwrapped function directly
             # But we need to make sure we pass 'cls' if it's a classmethod.
             # Wait, `get_folder_size` is a classmethod.
             # The decorator `@classmethod` is usually the outer one?
             # Order:
             # @classmethod
             # @DB.connection_context()
             # def get_folder_size(cls, ...):

             # So `FileService.get_folder_size` is the bound class method.
             # Its `__func__` might be the thing decorated by `connection_context`.

             pass

        # Alternative: We can mock the `DB` object that the decorator holds a reference to?
        # No, the decorator holds a reference to the method `connection_context` of the DB object,
        # or the DB object itself.

        # If `DB` is `api.db.connection.DB`, let's try to verify if we can patch what the decorator does.
        # But patching after import is hard for decorators.

        # BUT, `api.db.services.file_service.py` does:
        # from api.db.db_models import DB
        # ...
        # @DB.connection_context()

        # So the decorator uses the `DB` instance imported at that time.

        # Workaround:
        # Since we cannot easily bypass the decorator, we can try to make the `DB` instance it refers to
        # behave nicely. But we can't change the instance easily if it's a global.
        # BUT, peewee's `PooledDatabase` (which DB is) has a `connect()` method.
        # If we can patch the `connect` method of the GLOBAL `DB` object to be a no-op or connect to our sqlite db?
        # `DB` is imported from `api.db.connection`.

        from api.db.connection import DB as real_DB

        # Patch the connect method of the real DB object to do nothing
        # We use a context manager to ensure we revert it
        with monkeypatch.context() as m:
            m.setattr(real_DB, 'connect', lambda *args, **kwargs: None)
            m.setattr(real_DB, 'close', lambda *args, **kwargs: None)
            # We also need `commit`, `rollback` etc?
            # connection_context() calls connect() on enter and close() or similar on exit.

            # Note: The decorator might create a transaction or lock.

            # Let's try to just mock `connect` and `close`.

            root_id = str(uuid.uuid4()).replace('-', '')

            # Create root folder
            File.create(id=root_id, parent_id=root_id, name="root", size=0, type=FileType.FOLDER.value, tenant_id="t1", created_by="u1")

            # Create subfolder 1
            sub1_id = str(uuid.uuid4()).replace('-', '')
            File.create(id=sub1_id, parent_id=root_id, name="sub1", size=0, type=FileType.FOLDER.value, tenant_id="t1", created_by="u1")

            # Create file in root (size 100)
            f1_id = str(uuid.uuid4()).replace('-', '')
            File.create(id=f1_id, parent_id=root_id, name="f1.txt", size=100, type=FileType.PDF.value, tenant_id="t1", created_by="u1")

            # Create file in sub1 (size 200)
            f2_id = str(uuid.uuid4()).replace('-', '')
            File.create(id=f2_id, parent_id=sub1_id, name="f2.txt", size=200, type=FileType.PDF.value, tenant_id="t1", created_by="u1")

            # Create subfolder 2 in sub1
            sub2_id = str(uuid.uuid4()).replace('-', '')
            File.create(id=sub2_id, parent_id=sub1_id, name="sub2", size=0, type=FileType.FOLDER.value, tenant_id="t1", created_by="u1")

            # Create file in sub2 (size 300)
            f3_id = str(uuid.uuid4()).replace('-', '')
            File.create(id=f3_id, parent_id=sub2_id, name="f3.txt", size=300, type=FileType.PDF.value, tenant_id="t1", created_by="u1")

            # Total size should be 100 + 200 + 300 = 600

            size = FileService.get_folder_size(root_id)
            assert size == 600

            # Test checking size of sub1
            size_sub1 = FileService.get_folder_size(sub1_id)
            assert size_sub1 == 500 # 200 + 300
