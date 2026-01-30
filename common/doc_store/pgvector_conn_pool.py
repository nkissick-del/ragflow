#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import threading
from contextlib import contextmanager
from psycopg2 import pool


logger = logging.getLogger("ragflow.pgvector_conn")


class PGVectorConnPool:
    """
    Connection pool for PGVector connections.
    Uses psycopg2's ThreadedConnectionPool for thread-safe connection management.
    """

    _instance = None
    _pool = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if PGVectorConnPool._pool is not None:
            return
        with PGVectorConnPool._lock:
            if PGVectorConnPool._pool is None:
                self._init_pool()

    def _init_pool(self):
        """Initialize the connection pool from settings."""
        from common import settings

        # Get pgvector-specific config, fallback to postgres config
        pgvector_config = settings.get_base_config("pgvector", {})
        postgres_config = settings.get_base_config("postgres", {})

        # Merge configs - pgvector takes precedence
        host = pgvector_config.get("host") or postgres_config.get("host") or os.getenv("PGVECTOR_HOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = pgvector_config.get("port") or postgres_config.get("port") or os.getenv("PGVECTOR_PORT") or os.getenv("POSTGRES_PORT", "5432")
        dbname = pgvector_config.get("name") or postgres_config.get("name") or os.getenv("PGVECTOR_DBNAME") or os.getenv("POSTGRES_DBNAME", "ragflow")
        user = pgvector_config.get("user") or postgres_config.get("user") or os.getenv("PGVECTOR_USER") or os.getenv("POSTGRES_USER", "ragflow")
        password = pgvector_config.get("password") or postgres_config.get("password") or os.getenv("PGVECTOR_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "")

        min_conn = int(pgvector_config.get("min_connections", 2))
        max_conn = int(pgvector_config.get("max_connections", 20))

        try:
            PGVectorConnPool._pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
            )
            logger.info(f"PGVector connection pool initialized: {host}:{port}/{dbname}")
        except Exception as e:
            logger.error(f"Failed to initialize PGVector connection pool: {e}")
            raise

    def get_conn(self):
        """Get a connection from the pool."""
        if PGVectorConnPool._pool is None:
            with PGVectorConnPool._lock:
                if PGVectorConnPool._pool is None:
                    self._init_pool()
        return PGVectorConnPool._pool.getconn()

    def put_conn(self, conn):
        """Return a connection to the pool."""
        if PGVectorConnPool._pool is not None:
            PGVectorConnPool._pool.putconn(conn)

    @contextmanager
    def connection(self):
        """Context manager for getting and returning connections."""
        conn = self.get_conn()
        try:
            yield conn
        finally:
            self.put_conn(conn)

    @contextmanager
    def cursor(self, commit=True):
        """Context manager for getting a cursor with auto-commit."""
        with self.connection() as conn:
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()

    def close_all(self):
        """Close all connections in the pool."""
        if PGVectorConnPool._pool is not None:
            PGVectorConnPool._pool.closeall()
            PGVectorConnPool._pool = None


def get_pgvector_conn():
    """Factory function for PGVectorConnPool singleton."""
    return PGVectorConnPool()
