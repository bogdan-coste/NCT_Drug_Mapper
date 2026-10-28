"""Fill the SapBERT embedding fields of a therapeutic-category hierarchy JSON.

Reads a hierarchy file (``therapeutic_definitions.json``), embeds every
``metadata.display_name`` through the project's SapBERT server, and writes the
same structure back out with these fields populated:

* ``metadata.display_name_embedding`` — vector of the entry's own display name.
* ``children_embeddings`` — one vector per entry in ``children``.
* ``parents_embeddings`` — one vector per entry in ``parents``.

Child and parent vectors are produced from the referenced entity's display name,
which is what the Qdrant loader mean-pools into the ``children`` / ``parents``
vectors.

Usage (from the project root, with the SapBERT server running):

    uv run scripts/db_enrichment/embed_therapeutic_definitions.py

    uv run scripts/db_enrichment/embed_therapeutic_definitions.py \
        scripts/db_enrichment/therapeutic_definitions.json \
        --output scripts/db_enrichment/therapeutic_definitions_embedded.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import openai
from dotenv import load_dotenv

load_dotenv()

from src.embedder.sapbert_embedder import SapBERTEmbedder
from src.schemas.models import SapBERTParameters

DEFAULT_INPUT = Path(__file__).resolve().with_name("therapeutic_definitions.json")
DEFAULT_BASE_URL = "http://localhost:7997/v1"
DEFAULT_MODEL = "sapbert"
DEFAULT_API_KEY = "none"

RELATION_FIELDS = (
    ("children", "children_embeddings", "child"),
    ("parents", "parents_embeddings", "parent"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate the SapBERT embedding fields of a therapeutic-category "
            "hierarchy JSON."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Hierarchy JSON to read. Default: {DEFAULT_INPUT.name}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the embedded JSON. Default: <input>_embedded.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Texts embedded per request. Default: 32",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "SapBERT server URL. Default: $SAPBERT_BASE_URL or "
            f"{DEFAULT_BASE_URL}"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Embedding model name. Default: $SAPBERT_MODEL or {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="SapBERT API key. Default: $SAPBERT_API_KEY or 'none'",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    return args


def build_embedder(args: argparse.Namespace) -> SapBERTEmbedder:
    env = SapBERTParameters.from_env()
    return SapBERTEmbedder(
        SapBERTParameters(
            sapbert_model=args.model or env.sapbert_model or DEFAULT_MODEL,
            sapbert_base_url=args.base_url
            or env.sapbert_base_url
            or DEFAULT_BASE_URL,
            sapbert_api_key=args.api_key or env.sapbert_api_key or DEFAULT_API_KEY,
            sapbert_port=env.sapbert_port,
        )
    )


def display_names(hierarchy: dict[str, Any]) -> dict[str, str]:
    """Map every member ID to the text that should be embedded for it."""

    names: dict[str, str] = {}
    for member_id, entry in hierarchy.items():
        if not isinstance(entry, dict):
            raise TypeError(
                f"Member {member_id!r} is {type(entry).__name__}, expected a JSON object."
            )

        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        name = metadata.get("display_name")
        if not isinstance(name, str) or not name.strip():
            fallback = metadata.get("object_name") or member_id
            print(
                f"Warning: {member_id!r} has no metadata.display_name; "
                f"embedding {fallback!r} instead.",
                flush=True,
            )
            name = fallback

        names[member_id] = name

    for member_id, entry in hierarchy.items():
        for relation, _, label in RELATION_FIELDS:
            for reference in entry.get(relation) or []:
                if reference not in names:
                    raise ValueError(
                        f"Member {member_id!r} references unknown {label} "
                        f"{reference!r}."
                    )

    return names


def unique_texts(names: dict[str, str]) -> list[str]:
    """Deduplicate display names while keeping a stable order."""

    return list(dict.fromkeys(names.values()))


def embed_texts(
    embedder: SapBERTEmbedder,
    texts: list[str],
    batch_size: int,
) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        end = start + len(batch)
        print(f"Embedding {start + 1}-{end} of {total}...", flush=True)

        batch_vectors = embedder.embed_many(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"SapBERT returned {len(batch_vectors)} vectors for {len(batch)} inputs."
            )

        for text, vector in zip(batch, batch_vectors):
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"SapBERT returned an empty vector for {text!r}.")
            vectors[text] = vector

    return vectors


def vector_size(vectors: dict[str, list[float]]) -> int:
    sizes = {len(vector) for vector in vectors.values()}
    if len(sizes) > 1:
        raise ValueError(
            f"SapBERT returned inconsistent vector sizes: {sorted(sizes)}. "
            "Check the server and the model it is running."
        )
    if not sizes:
        raise ValueError("No vectors were produced; the hierarchy is empty.")
    return sizes.pop()


def fill_embeddings(
    hierarchy: dict[str, Any],
    names: dict[str, str],
    vectors: dict[str, list[float]],
) -> None:
    for member_id, entry in hierarchy.items():
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            entry["metadata"] = metadata

        metadata["display_name_embedding"] = vectors[names[member_id]]

        for relation, embedding_field, _ in RELATION_FIELDS:
            references = entry.get(relation) or []
            entry[embedding_field] = [vectors[names[ref]] for ref in references]


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input.resolve()}")

    print(f"Reading: {args.input.resolve()}", flush=True)
    with args.input.open("r", encoding="utf-8") as file:
        hierarchy = json.load(file)

    if not isinstance(hierarchy, dict):
        raise TypeError("Expected the JSON root to be an object keyed by member ID.")

    names = display_names(hierarchy)
    texts = unique_texts(names)
    embedder = build_embedder(args)
    base_url = embedder.client.base_url

    print(f"Members: {len(hierarchy)}", flush=True)
    print(f"Unique display names: {len(texts)}", flush=True)
    print(f"SapBERT server: {base_url}", flush=True)

    try:
        vectors = embed_texts(embedder, texts, args.batch_size)
    except openai.OpenAIError as exc:
        raise SystemExit(
            f"Could not use the SapBERT server at {base_url}: {exc}\n"
            "Start it with `uv run main.py`, or point --base-url at a running "
            "SapBERT instance."
        ) from exc

    size = vector_size(vectors)
    fill_embeddings(hierarchy, names, vectors)

    output = args.output or args.input.with_name(f"{args.input.stem}_embedded.json")
    with output.open("w", encoding="utf-8") as file:
        json.dump(hierarchy, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Vector size: {size}", flush=True)
    print(f"Written: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
