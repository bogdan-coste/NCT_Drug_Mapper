import threading
import uuid
from pathlib import Path
from typing import Any

import fastapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..embedder.sapbert_embedder import SapBERTEmbedder
from ..ingestion.nct_loader import NCTLoader
from ..llm.llm_client import LLMClient
from ..processing.modules.thereapeutic_cathegory_module import TherapeuticCategory
from ..processing.modules.drug_class_module import DrugClass
from ..retrieval.retriever import TherapeuticCategoryRetriever, DrugClassRetriever
from ..schemas.models import AppConfig, EmbedRequest, LLMRequest
from ..vectordb.vector_store import VectorStore
from .sapbert_server import SapBERTServer

# The Vue frontend (see frontend/) is bundled to frontend/dist by `npm run build`.
UI_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
NCT_BASE_URL = "https://clinicaltrials.gov/api/v2"

class PipelineRequest(BaseModel):
    term: str
    nct_ids: list[str]

_results: dict[str, Any] = {}

def _safe_serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return _safe_serialize(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    return obj

def _patch(run_id: str, updates: dict) -> None:
    _results[run_id].update(_safe_serialize(updates))

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
        self._dc_retriever = DrugClassRetriever(
            embedder=self._sapbert,
            vector_store=VectorStore(config.qdrant),
        )

        self._dc_pipeline = DrugClass(
            nct_loader=NCTLoader(base_url=NCT_BASE_URL),
            definition_llm=LLMClient(config.llm),
            query_llm=LLMClient(config.llm),
            mapping_llm=LLMClient(config.llm),
            retriever=self._dc_retriever,
            term_extractor_llm=LLMClient(config.llm),
        )

        self.server = fastapi.FastAPI(title="Therapeutic Category API")
        self.server.mount("/ui", StaticFiles(directory=UI_DIR, html=True, check_dir=False), name="ui")
        self._create_endpoints()

    def _create_endpoints(self):

        @self.server.post("/embed")
        async def embed(request: EmbedRequest) -> list[float] | list[list[float]]:
            return self._sapbert.embed(request.query)

        @self.server.post("/ask_llm")
        async def llm_ask(request: LLMRequest) -> str | None:
            return self._llm.ask_llm(request.query)

        @self.server.post("/start_pipeline")
        async def start_pipeline(request: PipelineRequest) -> dict:
            run_id = str(uuid.uuid4())
            pipeline = self._pipeline
            _results[run_id] = {}

            def _run():
                try:
                    pipeline._run_staged(
                        term=request.term,
                        nct_ids=request.nct_ids,
                        on_stage=lambda updates: _patch(run_id, updates),
                    )
                    _results[run_id]["_done"] = True
                except Exception as exc:
                    _results[run_id]["_error"] = str(exc)
                    _results[run_id]["_done"] = True

            threading.Thread(target=_run, daemon=True).start()
            return {"run_id": run_id}

        @self.server.post("/start_dc_pipeline")
        async def start_dc_pipeline(request: PipelineRequest) -> dict:
            run_id = str(uuid.uuid4())
            pipeline = self._dc_pipeline
            _results[run_id] = {}

            def _run():
                try:
                    pipeline._run_staged(
                        term=request.term,
                        nct_ids=request.nct_ids,
                        on_stage=lambda updates: _patch(run_id, updates),
                    )
                    _results[run_id]["_done"] = True
                except Exception as exc:
                    _results[run_id]["_error"] = str(exc)
                    _results[run_id]["_done"] = True

            threading.Thread(target=_run, daemon=True).start()
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
                    "<code>frontend/</code>, or use the Vite dev server "
                    "(<code>npm run dev</code> &rarr; http://localhost:5173).</p>",
                    status_code=503,
                )
            return FileResponse(index)

        @self.server.get("/health")
        async def health() -> dict:
            return {"status": "ok"}
