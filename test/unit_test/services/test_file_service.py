
import pytest
from unittest.mock import MagicMock, patch
import sys
from peewee import SqliteDatabase, Model, CharField, IntegerField, DateTimeField
import datetime

# Define a mock model and DB
mock_sqlite_db = SqliteDatabase(':memory:')

class MockFile(Model):
    id = CharField(primary_key=True)
    tenant_id = CharField()
    parent_id = CharField()
    type = CharField()
    name = CharField()
    size = IntegerField(default=0)
    location = CharField(default='')
    create_time = DateTimeField(default=datetime.datetime.now)

    class Meta:
        database = mock_sqlite_db

@pytest.fixture
def setup_db():
    mock_sqlite_db.bind([MockFile])
    mock_sqlite_db.connect()
    mock_sqlite_db.create_tables([MockFile])
    yield
    mock_sqlite_db.drop_tables([MockFile])
    mock_sqlite_db.close()

class TestFileServiceOptimization:

    def test_get_all_file_ids_by_tenant_id(self, setup_db):
        # Prepare mocks for api.db.connection
        mock_connection = MagicMock()
        mock_db_obj = MagicMock()

        # Make connection_context a pass-through decorator
        def mock_connection_context():
            def decorator(func):
                return func
            return decorator

        mock_db_obj.connection_context = mock_connection_context
        mock_connection.DB = mock_db_obj
        mock_connection.BaseDataBase = MagicMock()
        mock_connection.close_connection = MagicMock()

        # Also need to mock other things imported by api.db.db_models
        # api.db.pool, api.db.locks, api.db.fields, api.db.base, api.db.models, api.db.migrations

        # Ideally we only mock api.db.connection because that's where DB comes from.
        # But api.db.db_models imports a lot.

        # If we patch sys.modules['api.db.connection'], subsequent imports of it will get the mock.

        # We need to ensure api.db.db_models is NOT already loaded, or we reload it.
        # Since pytest might have loaded it, let's assume we need to patch it in sys.modules and maybe reload db_models if needed.

        with patch.dict(sys.modules, {'api.db.connection': mock_connection}):
            # We need to import FileService.
            # If api.db.db_models was already imported by pytest collection (unlikely if we didn't import it in this file's top level),
            # then it has a reference to the real api.db.connection module (or whatever was loaded).

            # Since I am running ONLY this test file, and I removed top-level imports,
            # api.db.db_models should NOT be loaded yet (unless pytest plugins load it).

            # Let's import FileService now.
            # It will import api.db.db_models.
            # api.db.db_models will import api.db.connection.
            # Since api.db.connection is in sys.modules as our mock, it should use it.

            # We also need to mock api.db.models because db_models imports * from it?
            # No, db_models imports specific models from api.db.models.
            # api.db.models imports ... everything.

            # We might run into issues with api.db.models importing real stuff.
            # api.db.models/__init__.py imports all models.
            # models/file.py imports DB.

            # If models/file.py imports DB from api.db.db_models (circular?) or api.db.connection.
            # api/db/models/__init__.py imports file.py.

            # Let's hope replacing api.db.connection is enough.

            # If api.db.db_models is already loaded, we are in trouble.
            if 'api.db.db_models' in sys.modules:
                del sys.modules['api.db.db_models']
            if 'api.db.services.file_service' in sys.modules:
                del sys.modules['api.db.services.file_service']

            from api.db.services.file_service import FileService

            # Setup data
            tenant_id = "tenant_1"
            other_tenant_id = "tenant_2"

            for i in range(150):
                MockFile.create(
                    id=f"f{i}",
                    tenant_id=tenant_id,
                    parent_id="root",
                    type="file",
                    name=f"file_{i}",
                    create_time=datetime.datetime.now() + datetime.timedelta(seconds=i)
                )

            MockFile.create(
                id="other_f1",
                tenant_id=other_tenant_id,
                parent_id="root",
                type="file",
                name="other_file",
                create_time=datetime.datetime.now()
            )

            # Patch FileService.model
            # Note: FileService.model is the real File model (which is now likely broken or half-imported).
            # We replace it with MockFile.
            original_model = FileService.model
            FileService.model = MockFile

            try:
                # Act
                results = FileService.get_all_file_ids_by_tenant_id(tenant_id)

                # Assert
                assert len(results) == 150

                ids = [r['id'] for r in results]
                assert 'f0' in ids
                assert 'f149' in ids
                assert 'other_f1' not in ids

                # Verify order strictly
                # We expect ascending order by create_time
                for i in range(150):
                     assert results[i]['id'] == f"f{i}", f"Expected f{i} at index {i}, but got {results[i]['id']}"

            finally:
                FileService.model = original_model
