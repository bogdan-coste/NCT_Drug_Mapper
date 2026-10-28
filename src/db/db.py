import asyncio
import os
from pathlib import Path

import asyncpg
from asyncpg import Pool
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

class DatabaseData(BaseModel):
    host: str = os.getenv("DB_HOST") or "localhost"
    port: int = int(os.getenv("DB_PORT") or 5432)
    database: str = os.getenv("DB_NAME") or "NCT_Drug_Mapper"
    user: str = os.getenv("DB_USER") or "postgres"
    password: str = os.getenv("DB_PASSWORD") or ""
    min_size: int = int(os.getenv("DB_POOL_MIN") or 5)
    max_size: int = int(os.getenv("DB_POOL_MAX") or 10)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL   PRIMARY KEY,
    username      TEXT        NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id           BIGSERIAL   PRIMARY KEY,
    user_id      BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash   BYTEA       NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    user_agent   TEXT
);

CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON sessions (expires_at);

"""

_pool: Pool | None = None


async def initialize_pool() -> Pool:
    """
        Open the connection pool and create the schema if it is missing.
    """

    global _pool
    if _pool is not None:
        return _pool

    db_data = DatabaseData()
    print(
        f"Connecting to PostgreSQL at "
        f"{db_data.host}:{db_data.port}/{db_data.database} as {db_data.user}",
        flush=True,
    )

    pool = await asyncio.wait_for(
        asyncpg.create_pool(
            host=db_data.host,
            port=db_data.port,
            database=db_data.database,
            user=db_data.user,
            password=db_data.password,
            min_size=db_data.min_size,
            max_size=db_data.max_size,
        ),
        timeout=10.0,
    )

    try:
        async with pool.acquire() as conn:
            await conn.execute(_SCHEMA)
    except Exception:
        await pool.close()
        raise

    _pool = pool
    return pool


async def close_pool() -> None:
    global _pool
    if _pool is None:
        return
    await _pool.close()
    _pool = None


def get_pool() -> Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not open")
    return _pool
