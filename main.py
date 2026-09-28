import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

print("Starting up...", flush=True)

from src.schemas.models import AppConfig


def _ensure_model(model_path: Path) -> None:
    """
    Check that SapBERT weights exist at model_path.
    If not (or incomplete), download them from HuggingFace before the server
    tries to load them.
    """
    REQUIRED = "config.json"
    HAS_WEIGHTS = any(
        model_path.glob("*.safetensors")
    ) or any(
        model_path.glob("pytorch_model*.bin")
    )

    if model_path.exists() and (model_path / REQUIRED).exists() and HAS_WEIGHTS:
        return  # already present

    print(
        f"SapBERT weights not found at {model_path}.\n"
        "Downloading from HuggingFace (one-time, ~400 MB)...",
        flush=True,
    )

    from transformers import AutoModel, AutoTokenizer

    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    model_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    print(f"SapBERT weights saved to {model_path.resolve()}", flush=True)

NCT_BASE_URL = "https://clinicaltrials.gov/api/v2"


def main():
    config = AppConfig.from_env()

    # ── Ensure SapBERT weights are present (download if missing) ───────────────
    _ensure_model(config.paths.sapbert_model_path)

    # ── Start SapBERT server ──────────────────────────────────────────────────
    print("Loading SapBERT model (this may take a moment)...", flush=True)
    from src.api.sapbert_server import SapBERTServer
    SapBERTServer(config)

    # ── Wire pipeline components ──────────────────────────────────────────────
    print("Connecting to Qdrant...", flush=True)
    from src.embedder.sapbert_embedder import SapBERTEmbedder
    from src.retrieval.retriever import TherapeuticCategoryRetriever
    from src.vectordb.vector_store import VectorStore
    embedder  = SapBERTEmbedder(config.sapbert)
    retriever = TherapeuticCategoryRetriever(embedder, VectorStore(config.qdrant))

    print("Wiring LLM pipeline...", flush=True)
    from src.ingestion.nct_loader import NCTLoader
    from src.llm.llm_client import LLMClient
    from src.processing.modules.thereapeutic_cathegory_module import TherapeuticCategory
    pipeline = TherapeuticCategory(
        nct_loader=NCTLoader(base_url=NCT_BASE_URL),
        definition_llm=LLMClient(config.llm),
        superclass_llm=LLMClient(config.llm),
        mapping_llm=LLMClient(config.llm),
        retriever=retriever,
    )

    print("\nPipeline ready.", flush=True)
    print("Type 'exit' at any prompt to quit.\n", flush=True)

    # ── Interactive loop ──────────────────────────────────────────────────────
    while True:
        try:
            term = input("Drug term:  ").strip()
            if term.lower() == "exit":
                break
            if not term:
                continue

            nct_raw = input("NCT IDs (comma-separated):  ").strip()
            if nct_raw.lower() == "exit":
                break
            if not nct_raw:
                continue

            nct_ids = [n.strip() for n in nct_raw.split(",") if n.strip()]

        except (KeyboardInterrupt, EOFError):
            break

        print(f"\n{'=' * 60}")
        print(f"Term:    {term}")
        print(f"NCT IDs: {', '.join(nct_ids)}")
        print(f"{'=' * 60}\n")

        try:
            print("Fetching NCT trial and generating definition...", flush=True)
            result = pipeline.run(term=term, nct_ids=nct_ids)

        except Exception as e:
            print(f"ERROR: {e}\n", flush=True)
            continue

        if "error" in result:
            print(f"ERROR: {result['error']}\n", flush=True)
            continue

        print("\n── Step 1: Definition ──────────────────────────────────────")
        print(result.get("definition", ""))

        print("\n── Step 2: Superclass labels ───────────────────────────────")
        print(json.dumps(result.get("superclass_labels", {}), indent=2))

        print("\n── SapBERT candidates (Qdrant) ─────────────────────────────")
        for i, c in enumerate(result.get("candidates", []), 1):
            print(f"  {i:2}. {c['name']}")

        print("\n── Step 3: Ontology mapping ────────────────────────────────")
        print(json.dumps(result.get("ontology_mapping", {}), indent=2))

        print(f"\n{'=' * 60}\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
