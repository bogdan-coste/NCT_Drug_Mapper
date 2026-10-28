import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import fastapi
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.db.db import close_pool, initialize_pool

from ..auth.authentication import get_current_user, oauth2_scheme
from ..auth.sessions import (
    create_session,
    purge_expired_sessions,
    revoke_session,
)
from ..auth.users import authenticate_user, register_user
from ..embedder.sapbert_embedder import SapBERTEmbedder
from ..ingestion.nct_loader import NCTLoader
from ..llm.llm_client import LLMClient
from ..processing.modules.thereapeutic_cathegory_module import TherapeuticCategory
from ..retrieval.retriever import TherapeuticCategoryRetriever
from ..schemas.models import AppConfig, EmbedRequest, LLMRequest
from ..schemas.user import User
from ..vectordb.vector_store import VectorStore
from .sapbert_server import SapBERTServer

UI_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
NCT_BASE_URL = "https://clinicaltrials.gov/api/v2"

MAX_CONCURRENT_RUNS = 4

class PipelineRequest(BaseModel):
    term: str
    nct_ids: list[str]

_results: dict[str, dict[str, Any]] = {}

def _safe_serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return _safe_serialize(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    return obj

def _patch(run_id: str, updates: dict) -> None:
    target = _results.get(run_id)
    if target is not None:
        target.update(_safe_serialize(updates))

class Server:
    def __init__(self, config: AppConfig):
        self._sapbert_server = SapBERTServer(config)
        self._sapbert = SapBERTEmbedder(config.sapbert)
        self._llm = LLMClient(config.llm)
        self._retriever = TherapeuticCategoryRetriever(
            embedder=self._sapbert,
            vector_store=VectorStore(config.qdrant),
        )
        self._pipeline = TherapeuticCategory(
            nct_loader=NCTLoader(base_url=NCT_BASE_URL),
            definition_llm=LLMClient(config.llm),
            superclass_llm=LLMClient(config.llm),
            mapping_llm=LLMClient(config.llm),
            retriever=self._retriever,
            term_extractor_llm=LLMClient(config.llm),
        )

        self._pipeline_gate = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
        self._run_tasks: dict[str, asyncio.Task[None]] = {}

        self.server = fastapi.FastAPI(
            title="Therapeutic Category API",
            lifespan=self._lifespan,
        )
        self.server.mount(
            "/ui",
            StaticFiles(directory=UI_DIR, html=True, check_dir=False),
            name="ui",
        )
        self._create_endpoints()

    @asynccontextmanager
    async def _lifespan(self, _app: fastapi.FastAPI):

        try:
            await initialize_pool()
            await purge_expired_sessions()
            yield
        finally:
            tasks = list(self._run_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await close_pool()
            await self._sapbert.aclose()

    def _track(
        self,
        run_id: str,
        phase: Callable[[], Awaitable[None]],
    ) -> None:
        """
            Run one pipeline phase as a task, recording its outcome.
        """

        async def runner() -> None:
            try:
                async with self._pipeline_gate:
                    await phase()
                _results[run_id]["_done"] = True
            except asyncio.CancelledError:
                _results.setdefault(run_id, {})["_error"] = "cancelled"
                _results[run_id]["_done"] = True
                raise
            except Exception as exc:
                _results[run_id]["_error"] = str(exc)
                _results[run_id]["_done"] = True
            finally:
                self._run_tasks.pop(run_id, None)

        self._run_tasks[run_id] = asyncio.create_task(runner())

    def _create_endpoints(self):

        @self.server.post("/signup", status_code=201)
        async def signup(user: User) -> dict[str, str]:
            try:
                 await register_user(user)

            except ValueError as exc:
                raise fastapi.HTTPException(
                    status_code=409, detail=str(exc)
                ) from exc
            return {"username": user.username}

        @self.server.post("/login")
        async def login(
            request: fastapi.Request,
            form: OAuth2PasswordRequestForm = fastapi.Depends(),
        ) -> dict[str, str]:
            user_id = await authenticate_user(form.username, form.password)
            if user_id is None:
                raise fastapi.HTTPException(
                    status_code=401,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            token = await create_session(user_id, request.headers.get("user-agent"))
            return {"access_token": token, "token_type": "bearer"}

        @self.server.post("/logout", status_code=204)
        async def logout(token: str = fastapi.Depends(oauth2_scheme)) -> None:
            await revoke_session(token)

        @self.server.get("/users/me")
        async def read_current_user(
            username: str = fastapi.Depends(get_current_user),
        ) -> dict[str, str]:
            return {"username": username}

        @self.server.post("/embed")
        async def embed(request: EmbedRequest) -> list[float] | list[list[float]]:
            return await self._sapbert.embed(request.query)

        @self.server.post("/ask_llm")
        def llm_ask(request: LLMRequest) -> str | None:
            return self._llm.ask_llm(request.query)

        @self.server.post("/start_pipeline")
        async def start_pipeline(request: PipelineRequest,
                                username : str = fastapi.Depends(get_current_user)) -> dict:
            run_id = str(uuid.uuid4())
            _results[run_id] = {}
            pipeline = self._pipeline

            def phase():
                return pipeline._run_staged_async(
                    term=request.term,
                    nct_ids=request.nct_ids,
                    on_stage=lambda updates: _patch(run_id, updates),
                )

            self._track(run_id, phase)
            return {"run_id": run_id}

        @self.server.get("/result/{run_id}")
        async def get_result(run_id: str):
            if run_id not in _results:
                raise fastapi.HTTPException(status_code=404, detail="Unknown run_id")
            snapshot = dict(_results[run_id])
            if snapshot.get("_done"):
                _results.pop(run_id, None)
            return fastapi.responses.JSONResponse(content=snapshot)

        @self.server.get("/", include_in_schema=False)
        async def ui_root():
            index = UI_DIR / "index.html"
            if not index.exists():
                return fastapi.responses.HTMLResponse(
                    "<h1>Frontend not built</h1>"
                    "<p>Run <code>npm install &amp;&amp; npm run build</code> inside "
                    "<code>frontend/react</code>, or use the Vite dev server "
                    "(<code>npm run dev</code> &rarr; http://localhost:5173).</p>",
                    status_code=503,
                )
            return FileResponse(index)

        @self.server.get("/health")
        async def health() -> dict:
            return {"status": "ok"}
