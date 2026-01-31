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
        def get_config_val(key, primary_env, secondary_env=None, default=None):
            if key in pgvector_config and pgvector_config[key] is not None:
                return pgvector_config[key]
            if key in postgres_config and postgres_config[key] is not None:
                return postgres_config[key]
            val = os.getenv(primary_env)
            if val is not None:
                return val
            if secondary_env:
                val = os.getenv(secondary_env)
                if val is not None:
                    return val
            return default

        host = get_config_val("host", "PGVECTOR_HOST", "POSTGRES_HOST", "localhost")
        port = str(get_config_val("port", "PGVECTOR_PORT", "POSTGRES_PORT", "5432"))
        dbname = get_config_val("name", "PGVECTOR_DBNAME", "POSTGRES_DBNAME", "ragflow")
        user = get_config_val("user", "PGVECTOR_USER", "POSTGRES_USER", "ragflow")
        password = get_config_val("password", "PGVECTOR_PASSWORD", "POSTGRES_PASSWORD", "")

        try:
            min_conn = int(get_config_val("min_connections", "PGVECTOR_MIN_CONNECTIONS", "POSTGRES_MIN_CONNECTIONS", "2"))
        except (ValueError, TypeError):
            logger.warning("Invalid min_connections config, falling back to 2")
            min_conn = 2

        try:
            max_conn = int(get_config_val("max_connections", "PGVECTOR_MAX_CONNECTIONS", "POSTGRES_MAX_CONNECTIONS", "20"))
        except (ValueError, TypeError):
            logger.warning("Invalid max_connections config, falling back to 20")
            max_conn = 20

        # Validate connection pool sizes
        if min_conn <= 0:
            logger.warning(f"min_conn ({min_conn}) must be positive. Falling back to 2.")
            min_conn = 2
        if max_conn <= 0:
            logger.warning(f"max_conn ({max_conn}) must be positive. Falling back to 20.")
            max_conn = 20
        if min_conn > max_conn:
            logger.warning(f"min_conn ({min_conn}) exceeds max_conn ({max_conn}). Adjusting min_conn to {max_conn}.")
            min_conn = max_conn

        try:
            PGVectorConnPool._pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
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

        # Get connection and validate it
        max_retries = 3
        conn = None
        for attempt in range(max_retries):
            try:
                conn = PGVectorConnPool._pool.getconn()
                # Check if connection is closed or executing
                if conn.closed:
                    logger.warning(f"Got closed connection from pool (attempt {attempt + 1}/{max_retries}), discarding")
                    PGVectorConnPool._pool.putconn(conn, close=True)
                    conn = None
                    continue

                # Perform lightweight health check
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()

                return conn
            except Exception as e:
                logger.warning(f"Connection obtain/validation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if conn is not None:
                    try:
                        PGVectorConnPool._pool.putconn(conn, close=True)
                    except Exception:
                        pass
                    conn = None

                if attempt == max_retries - 1:
                    logger.error(f"Failed to get valid PG connection after {max_retries} attempts: {e}")
                    raise

    def put_conn(self, conn, close=False):
        """Return a connection to the pool."""
        if PGVectorConnPool._pool is not None:
            PGVectorConnPool._pool.putconn(conn, close=close)

    @contextmanager
    def connection(self):
        """Context manager for getting and returning connections."""
        conn = self.get_conn()
        success = False
        try:
            yield conn
            success = True
        except Exception:
            try:
                if not conn.closed:
                    conn.rollback()
            except Exception as rollback_err:
                logger.warning(f"Rollback failed during connection cleanup: {rollback_err}")
            raise
        finally:
            # Only put back if connection is open and block succeeded
            if success and not conn.closed:
                self.put_conn(conn)
            else:
                try:
                    self.put_conn(conn, close=True)
                except Exception:
                    pass

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
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.warning(f"Rollback failed: {rollback_error}")
                raise
            finally:
                cur.close()

    def close_all(self):
        """Close all connections in the pool."""
        with PGVectorConnPool._lock:
            if PGVectorConnPool._pool is not None:
                PGVectorConnPool._pool.closeall()
                PGVectorConnPool._pool = None


def get_pgvector_conn():
    """Factory function for PGVectorConnPool singleton."""
    return PGVectorConnPool()
