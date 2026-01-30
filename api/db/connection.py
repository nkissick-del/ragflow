#
# Database connection initialization and utilities
#
from __future__ import annotations
import sys

print("DEBUG: Importing api.db.connection...", file=sys.stderr, flush=True)

import logging

print("DEBUG: Importing common.settings...", file=sys.stderr, flush=True)
from common import settings

print("DEBUG: Importing common.decorator...", file=sys.stderr, flush=True)
from common.decorator import singleton

# Import all pooling, locking, and diagnostic components
# These are handled lazily within functions to avoid import-time side effects or failures
print("DEBUG: api.db.connection basic imports complete.", file=sys.stderr, flush=True)

from typing import TYPE_CHECKING

# Type hints only - actual imports happen at module end for proper exports
if TYPE_CHECKING:
    from playhouse.pool import PooledMySQLDatabase, PooledPostgresqlDatabase
    from api.db.pool import PooledDatabase

DB: PooledMySQLDatabase | PooledPostgresqlDatabase


def get_database_config():
    """
    Extract and normalize database configuration from settings.

    Returns dict with keys: 'type', 'name', 'host', 'port', 'user', 'password'
    """
    database_config = (settings.DATABASE or {}).copy()
    # Guard against None DATABASE_TYPE
    db_type_raw = settings.DATABASE_TYPE
    if not db_type_raw:
        raise ValueError("DATABASE_TYPE setting is required. Must be 'postgres' or 'mysql'. Set via environment or service_conf.yaml")
    db_type = db_type_raw.lower()

    return {
        "type": db_type,
        "name": database_config.get("name"),
        "host": database_config.get("host", "localhost"),
        "port": database_config.get("port", 5432 if db_type == "postgres" else 3306),
        "user": database_config.get("user"),
        "password": database_config.get("password"),
    }


def ensure_database_exists():
    """
    Create the target database if it doesn't exist.

    Uses the configured database user credentials (expected to be superuser for initial setup).
    Mirrors MySQL approach: assumes user has CREATE DATABASE permission.

    For PostgreSQL: Connects to 'postgres' system database to create target DB.
    For MySQL: Connects without database to create target DB.

    Idempotent—safe to call multiple times. Non-blocking: logs warnings on failure.

    Security Note:
        By default, expects superuser credentials (postgres/root) for database creation.
        For sandboxed environments with restricted users, see docs/POSTGRESQL_SECURITY.md
        on pre-creating the database or granting CREATE DATABASE permission.
    """
    try:
        config = get_database_config()
        db_type = config["type"]
        db_name = config["name"]
        db_host = config["host"]
        db_port = config["port"]
        db_user = config["user"]
        db_pass = config["password"]

        if db_type == "postgres":
            try:
                import psycopg2
                from psycopg2 import sql
            except ImportError:
                logging.warning("psycopg2 not available; skipping database creation for PostgreSQL")
                return

            try:
                # Connect to postgres system database using configured credentials
                # Add connect_timeout to prevent indefinite hanging during startup
                print(f"Connecting to PostgreSQL at {db_host}:{db_port} to ensure database '{db_name}' exists...", flush=True)
                conn = psycopg2.connect(host=db_host, port=db_port, user=db_user, password=db_pass, database="postgres", connect_timeout=10)
                conn.autocommit = True
                cursor = conn.cursor()

                # Check if database exists first (idempotent)
                cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), (db_name,))

                if cursor.fetchone() is None:
                    # Database doesn't exist, create it
                    print(f"Database '{db_name}' not found. Creating...", flush=True)
                    cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                    logging.info(f"Created PostgreSQL database '{db_name}' at {db_host}:{db_port}")
                else:
                    logging.info(f"PostgreSQL database '{db_name}' already exists at {db_host}:{db_port}")
                    print(f"PostgreSQL database '{db_name}' already exists.", flush=True)

                cursor.close()
                conn.close()

            except Exception as e:
                print(f"Error ensuring PostgreSQL database exists: {e}", flush=True)
                logging.warning(
                    f"Failed to create PostgreSQL database '{db_name}': {e}. "
                    f"If using restricted user, ensure database is pre-created or user has CREATE DATABASE permission. "
                    f"See docs/POSTGRESQL_SECURITY.md for sandboxed setup."
                )

        elif db_type == "mysql":
            try:
                import mysql.connector
            except ImportError:
                logging.warning("mysql.connector not available; skipping pre-flight DB creation for MySQL")
                return

            try:
                # Validate identifier to prevent SQL injection (must be alphanumeric or underscore)
                if not db_name or not all(c.isalnum() or c == "_" for c in db_name):
                    raise ValueError(f"Invalid database name: {db_name}. Database names must contain only alphanumeric characters and underscores.")

                conn = mysql.connector.connect(host=db_host, port=db_port, user=db_user, password=db_pass)
                cursor = conn.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
                cursor.close()
                conn.close()
                logging.info(f"Ensured MySQL database '{db_name}' exists at {db_host}:{db_port}")
            except Exception as e:
                logging.warning(f"Failed to pre-create MySQL database '{db_name}': {e}. Migrations may handle creation.")

        else:
            logging.warning(f"Unknown database type '{db_type}'; skipping pre-flight DB creation")

    except Exception as e:
        logging.warning(f"Unexpected error in ensure_database_exists: {e}")


class TransactionLogger:
    """
    Backward compatibility re-export.

    TransactionLogger moved to api.db.transaction module.
    """

    @staticmethod
    def log_transaction_state(db, operation="begin", extra_info=None):
        from api.db.transaction import TransactionLogger as TL

        return TL.log_transaction_state(db, operation, extra_info)

    @staticmethod
    def log_transaction_error(db, exception, context=None):
        from api.db.transaction import TransactionLogger as TL

        return TL.log_transaction_error(db, exception, context)


@singleton
class BaseDataBase:
    def __init__(self):
        # Ensure database exists before creating connection pool
        ensure_database_exists()

        # Import at runtime to avoid circular dependency issues
        from api.db.pool import PooledDatabase

        database_config = (settings.DATABASE or {}).copy()
        db_name = database_config.pop("name")

        pool_config = {
            "max_retries": 5,
            "retry_delay": 1,
        }
        database_config.update(pool_config)
        db_type_upper = settings.DATABASE_TYPE.upper()
        self.database_connection = PooledDatabase[db_type_upper].value(db_name, **database_config)

        # Lazy import locks
        from api.db.locks import DatabaseLock

        self.database_connection.lock = DatabaseLock[db_type_upper].value  # type: ignore[attr-defined]

        # Log initial pool configuration
        max_conn = database_config.get("max_connections", 32)
        logging.info(f"Initialized {db_type_upper} connection pool: max_connections={max_conn}, max_retries={pool_config['max_retries']}, retry_delay={pool_config['retry_delay']}s")

        # Lazy import diagnostics
        try:
            from api.db.diagnostics import PoolDiagnostics

            # Log initial pool stats
            stats = PoolDiagnostics.get_pool_stats(self.database_connection)
            logging.info(f"Connection pool stats: {stats}")

            # Start background health monitoring
            PoolDiagnostics.start_health_monitoring(self.database_connection)
        except ImportError:
            logging.warning("api.db.diagnostics not available; skipping pool monitoring")

        logging.info("Database connection pool initialized")


# Lazy initialization: getter function instead of module-level instantiation
_db_instance = None


def init_db():
    """Initialize the database connection pool (lazy initialization).

    Called automatically on first access via get_db().
    Can be called explicitly to control initialization timing.
    Idempotent—safe to call multiple times.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = BaseDataBase().database_connection
    return _db_instance


def get_db():
    """Get the database connection pool, initializing if needed (lazy initialization).

    This prevents side effects at import time. Connections are not opened,
    background health checks are not started, and locks are not acquired
    until this function is explicitly called.

    Returns:
        PooledMySQLDatabase | PooledPostgresqlDatabase: The initialized connection pool
    """
    return init_db()


# For backward compatibility, also provide DB as module-level variable
# It is initialized via get_db() to ensure the pool is created
DB = get_db()


def close_connection():
    """Close stale database connections."""
    try:
        db = get_db()
        if db:
            db.close_stale(age=30)
    except Exception:
        logging.exception("Failed to close stale DB connections")


def log_connection_stats():
    """
    Log current connection pool statistics

    This is a convenience function that can be called from anywhere
    to check the current state of the connection pool.
    """
    try:
        from api.db.diagnostics import PoolDiagnostics

        if DB:
            PoolDiagnostics.log_pool_health(DB)
    except (ImportError, Exception) as e:
        logging.error(f"Failed to log connection stats: {e}")


def wait_for_schema_ready(max_retries: int = 30, retry_delay: float = 0.5):
    """
    Wait for database schema to be ready before accessing tables.

    This ensures init_database_tables() has completed before any code
    tries to access all critical tables. Prevents race conditions
    during startup.

    Args:
        max_retries: Maximum number of retry attempts (30 retries * 0.5s = 15s timeout)
        retry_delay: Delay in seconds between retries

    Raises:
        RuntimeError: If schema is not ready after max_retries
    """
    import time

    critical_tables = ["user", "sync_logs", "system_settings"]
    # Use portable identifier quoting across DBs: Postgres uses double quotes, MySQL uses backticks
    db_type = settings.DATABASE_TYPE.lower()
    quote_char = '"' if db_type == "postgres" else "`"

    for attempt in range(max_retries):
        try:
            # Try to query all critical tables to verify schema exists
            for table in critical_tables:
                cursor = None
                try:
                    # Wrap check in atomic() so that if table doesn't exist yet (ProgrammingError),
                    # the transaction is rolled back and connection remains usable for next retry.
                    with DB.atomic():
                        cursor = DB.execute_sql(f"SELECT 1 FROM {quote_char}{table}{quote_char} LIMIT 1")
                finally:
                    if cursor:
                        cursor.close()
            logging.info(f"✓ Database schema is ready (attempt {attempt + 1}/{max_retries})")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logging.debug(f"Schema not ready yet (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logging.error(f"✗ Database schema still not ready after {max_retries} attempts")
                raise RuntimeError(f"Database schema initialization timeout. Critical tables {critical_tables} not accessible after {max_retries * retry_delay}s") from e


# Backward compatibility: re-export lock instances from locks module
# Handled via property or lazy access if needed, or just import here
def get_locks():
    from api.db.locks import MysqlDatabaseLock, PostgresDatabaseLock

    return MysqlDatabaseLock, PostgresDatabaseLock


# Also export playhouse pooled database classes for tests
# We wrap this in a try-except because playhouse might be missing in some builds
try:
    from playhouse.pool import PooledMySQLDatabase, PooledPostgresqlDatabase
except ImportError:
    # Minimal stubs for type checking if missing
    class PooledMySQLDatabase:
        pass

    class PooledPostgresqlDatabase:
        pass


# Backward compatibility: re-export pool classes from pool module
from api.db.pool import (  # noqa: E402, F401
    PooledDatabase,
    RetryingPooledMySQLDatabase,
    RetryingPooledPostgresqlDatabase,
    with_retry,
)

__all__ = [
    "BaseDataBase",
    "DB",
    "PooledDatabase",
    "RetryingPooledMySQLDatabase",
    "RetryingPooledPostgresqlDatabase",
    "PooledMySQLDatabase",
    "PooledPostgresqlDatabase",
    "with_retry",
    "get_database_config",
    "ensure_database_exists",
    "wait_for_schema_ready",
    "close_connection",
    "log_connection_stats",
    "TransactionLogger",
]
