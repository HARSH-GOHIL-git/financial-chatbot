import os
import sqlite3
import psycopg
from contextlib import contextmanager
from app.core.config import DB_URI
from app.core.logger import get_logger

logger = get_logger(__name__)

# Global state to track active database engine: "postgres" or "sqlite"
_db_engine = "sqlite"
_db_uri_clean = None
_sqlite_path = "chatbot.db"

def init_db():
    """
    Attempts to initialize the database connection.
    If PostgreSQL is configured:
      1. Tries to connect to default database to auto-create the target database if missing.
      2. Tries to connect to the target database.
    If PostgreSQL connection/creation fails or is not configured, gracefully falls back to SQLite.
    """
    global _db_engine, _db_uri_clean, _sqlite_path
    
    if not DB_URI:
        logger.info("[DB] No DB_URI provided. Initialising local SQLite database: chatbot.db")
        _db_engine = "sqlite"
        return
        
    _db_uri_clean = DB_URI.strip('"').strip("'")
    
    # Check if the URI is a postgres URI
    if not (_db_uri_clean.startswith("postgresql://") or _db_uri_clean.startswith("postgres://")):
        logger.info(f"[DB] DB_URI is not a Postgres connection string. Using SQLite: {_sqlite_path}")
        _db_engine = "sqlite"
        return

    # Try to connect to Postgres and auto-create the database if it doesn't exist
    try:
        # Parse connection params to connect to default 'postgres' database first
        # to check/create the target database
        conn_params = psycopg.conninfo.conninfo_to_dict(_db_uri_clean)
        target_db = conn_params.get("dbname", "chatbot_db")
        
        # Connect to 'postgres' default database to check if target exists
        conn_params["dbname"] = "postgres"
        
        logger.info(f"[DB] Connecting to PostgreSQL host to verify database '{target_db}'...")
        # We set autocommit=True because CREATE DATABASE cannot run in a transaction block
        with psycopg.connect(**conn_params, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
                exists = cur.fetchone()
                if not exists:
                    logger.info(f"[DB] Database '{target_db}' does not exist. Creating it automatically...")
                    # Quote identifier safely
                    cur.execute(f"CREATE DATABASE {psycopg.sql.Identifier(target_db).as_string(conn)}")
                    logger.info(f"[DB] Database '{target_db}' created successfully.")
                else:
                    logger.info(f"[DB] Database '{target_db}' verified (already exists).")
        
        # Test connection to the target database
        with psycopg.connect(_db_uri_clean) as conn:
            pass
            
        logger.info("[DB] Successfully connected to PostgreSQL database!")
        _db_engine = "postgres"
        
    except Exception as e:
        logger.warning(
            f"[DB] PostgreSQL connection or auto-creation failed: {e}\n"
            f"[DB] Gracefully falling back to zero-configuration local SQLite database: {_sqlite_path}"
        )
        _db_engine = "sqlite"

# Run initialization once at module import
init_db()

def get_engine_type() -> str:
    return _db_engine

def get_db_uri() -> str:
    if _db_engine == "postgres":
        return _db_uri_clean
    return _sqlite_path

class SQLiteConnectionWrapper:
    """A wrapper around sqlite3.Connection to mimic psycopg.Connection behavior."""
    def __init__(self, conn):
        self._conn = conn
        
    def cursor(self):
        return SQLiteCursorWrapper(self._conn.cursor())
        
    def commit(self):
        self._conn.commit()
        
    def rollback(self):
        self._conn.rollback()
        
    def close(self):
        self._conn.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        self.close()

class SQLiteCursorWrapper:
    """A wrapper around sqlite3.Cursor to translate query syntax from PostgreSQL to SQLite."""
    def __init__(self, cursor):
        self._cursor = cursor
        
    def execute(self, query: str, params=None):
        # 1. Translate PostgreSQL %s placeholder to SQLite ? placeholder
        translated_query = query.replace("%s", "?")
        
        # 2. Replace Postgres-specific NOW() with SQLite-compatible CURRENT_TIMESTAMP
        translated_query = translated_query.replace("NOW()", "CURRENT_TIMESTAMP")
        
        # 3. Replace TIMESTAMPTZ with TIMESTAMP/TEXT
        translated_query = translated_query.replace("TIMESTAMPTZ", "TIMESTAMP")
        
        if params is not None:
            self._cursor.execute(translated_query, params)
        else:
            self._cursor.execute(translated_query)
        return self
        
    def fetchone(self):
        return self._cursor.fetchone()
        
    def fetchall(self):
        return self._cursor.fetchall()
        
    def close(self):
        self._cursor.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

@contextmanager
def get_db_connection():
    """
    Context manager that yields a database connection.
    Automatically handles PostgreSQL or SQLite depending on the active engine,
    and translates queries transparently when using SQLite.
    """
    if _db_engine == "postgres":
        with psycopg.connect(_db_uri_clean) as conn:
            yield conn
    else:
        conn = sqlite3.connect(_sqlite_path)
        yield SQLiteConnectionWrapper(conn)

def get_checkpointer():
    """
    Returns the appropriate checkpointer class and context manager.
    If using Postgres: returns (PostgresSaver, PostgresSaver.from_conn_string(DB_URI))
    If using SQLite: returns (SqliteSaver, SqliteSaver.from_conn_string(sqlite_path))
    """
    if _db_engine == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        cm = PostgresSaver.from_conn_string(_db_uri_clean)
        return PostgresSaver, cm
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        cm = SqliteSaver.from_conn_string(_sqlite_path)
        return SqliteSaver, cm
