import threading

import fastapi
import torch
import uvicorn
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from ..schemas.models import AppConfig


class SapBERTServer:
    def __init__(self, config: AppConfig):

        self.port = config.sapbert.sapbert_port
        self.app = fastapi.FastAPI(title="SapBERT Embedding Server")

        print(f"Loading SapBERT from {config.paths.sapbert_model_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.paths.sapbert_model_path)
        self.model = AutoModel.from_pretrained(config.paths.sapbert_model_path)
        self.model.eval()
        print("SapBERT model loaded.", flush=True)

        self._register_routes()
        self.start()

    def _register_routes(self):

        class EmbedRequest(BaseModel):
            input: str | list[str]
            model: str = "sapbert"

        @self.app.post("/v1/embeddings")
        def embeddings(request: EmbedRequest):
            phrases = [request.input] if isinstance(request.input, str) else request.input
            inputs = self.tokenizer(
                phrases,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            vecs = outputs.last_hidden_state[:, 0, :].tolist()
            return {
                "object": "list",
                "model": request.model,
                "data": [
                    {"object": "embedding", "index": i, "embedding": vec}
                    for i, vec in enumerate(vecs)
                ]
            }

    def start(self):
        thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": self.app, "host": "0.0.0.0", "port": self.port},
            daemon=True
        )
        thread.start()
        print(f"SapBERT server running on port {self.port}", flush=True)
