from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from .sessions import get_session_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    username = await get_session_user(token)
    if username is None:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
