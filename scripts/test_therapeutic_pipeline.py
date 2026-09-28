"""
Test script for the TherapeuticCategory pipeline.

Steps tested:
    1. Fetch NCT trial and generate a drug definition.
    2. Suggest therapeutic superclass labels from the definition.
    3. Embed superclass labels with SapBERT, retrieve candidates from Qdrant,
       and map to an ontology term.

Usage:
    uv run scripts/test_therapeutic_pipeline.py

Requires the following .env variables:
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_APP_ID
    SAPBERT_MODEL, SAPBERT_BASE_URL, SAPBERT_API_KEY
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY
"""

import json
import sys
from pathlib import Path

# ── Make src importable when running from project root ────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.embedder.sapbert_embedder import SapBERTEmbedder
from src.ingestion.nct_loader import NCTLoader
from src.llm.llm_client import LLMClient
from src.processing.modules.thereapeutic_cathegory_module import TherapeuticCategory
from src.retrieval.retriever import TherapeuticCategoryRetriever
from src.schemas.models import LLMParameters, QdrantParameters, SapBERTParameters
from src.vectordb.vector_store import VectorStore

# ── Config ────────────────────────────────────────────────────────────────────

NCT_BASE_URL = "https://clinicaltrials.gov/api/v2"

# Each entry: (term, [nct_ids])
TEST_CASES = [
    ("IEV407", ["NCT07604571"]),
]

# ── Wiring ────────────────────────────────────────────────────────────────────

llm_params     = LLMParameters.from_env()
sapbert_params = SapBERTParameters.from_env()
qdrant_params  = QdrantParameters.from_env()

embedder   = SapBERTEmbedder(sapbert_params)
retriever  = TherapeuticCategoryRetriever(embedder, VectorStore(qdrant_params))

pipeline = TherapeuticCategory(
    nct_loader=NCTLoader(base_url=NCT_BASE_URL),
    definition_llm=LLMClient(llm_params),
    superclass_llm=LLMClient(llm_params),
    mapping_llm=LLMClient(llm_params),
    retriever=retriever,
)

# ── Run ───────────────────────────────────────────────────────────────────────

for TERM, NCT_IDS in TEST_CASES:
    print(f"\n{'=' * 60}")
    print(f"Term:    {TERM}")
    print(f"NCT IDs: {', '.join(NCT_IDS)}")
    print(f"{'=' * 60}\n")

    result = pipeline.run(term=TERM, nct_ids=NCT_IDS)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        continue

    print("── Step 1: Definition ──────────────────────────────────────")
    print(result.get("definition", ""))

    print("\n── Step 2: Superclass labels ───────────────────────────────")
    print(json.dumps(result.get("superclass_labels", {}), indent=2))

    print("\n── SapBERT candidates (Qdrant) ─────────────────────────────")
    for i, c in enumerate(result.get("candidates", []), 1):
        print(f"  {i:2}. {c['name']}")

    print("\n── Step 3: Ontology mapping ────────────────────────────────")
    print(json.dumps(result.get("ontology_mapping", {}), indent=2))

    print(f"\n{'=' * 60}")

print("Done.")
