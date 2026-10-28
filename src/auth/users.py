import asyncio

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from asyncpg.exceptions import UniqueViolationError

from ..db import get_pool
from ..schemas.user import User

_hasher = PasswordHasher()


def _hash_password(password: str) -> str:
    """
    Hash a password. Argon2 generates a random salt internally and embeds it,
    together with the cost parameters, in the returned PHC string.
    """
    return _hasher.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerificationError:
        return False


async def register_user(user: User) -> None:
    password_hash = await asyncio.to_thread(_hash_password, user.password)
    try:
        async with get_pool().acquire() as conn:
            await conn.execute(
                "INSERT INTO users (username, password_hash) VALUES ($1, $2)",
                user.username,
                password_hash,
            )
    except UniqueViolationError as exc:
        raise ValueError(f"User {user.username!r} already exists") from exc


async def authenticate_user(username: str, password: str) -> int | None:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, password_hash FROM users WHERE username = $1",
            username,
        )

    if row is None:
        return None

    valid = await asyncio.to_thread(_verify_password, password, row["password_hash"])
    return row["id"] if valid else None
