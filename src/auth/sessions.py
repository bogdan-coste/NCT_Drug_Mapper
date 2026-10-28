import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from ..db import get_pool

SESSION_TTL = timedelta(days=7)
LAST_SEEN_REFRESH = timedelta(hours=1)

def _hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()

async def create_session(user_id: int, user_agent: str | None = None) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC) + SESSION_TTL

    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (user_id, token_hash, expires_at, user_agent)
            VALUES ($1, $2, $3, $4)
            """,
            user_id,
            _hash_token(token),
            expires_at,
            user_agent,
        )

    return token

async def get_session_user(token: str) -> str | None:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT s.id, s.last_seen_at, u.username
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = $1 AND s.expires_at > now()
            """,
            _hash_token(token),
        )

        if row is None:
            return None

        if datetime.now(UTC) - row["last_seen_at"] > LAST_SEEN_REFRESH:
            await conn.execute(
                "UPDATE sessions SET last_seen_at = now() WHERE id = $1",
                row["id"],
            )

    return row["username"]

async def revoke_session(token: str) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(
            "DELETE FROM sessions WHERE token_hash = $1",
            _hash_token(token),
        )


async def revoke_all_sessions(user_id: int) -> None:
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM sessions WHERE user_id = $1", user_id)


async def purge_expired_sessions() -> None:
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM sessions WHERE expires_at < now()")
