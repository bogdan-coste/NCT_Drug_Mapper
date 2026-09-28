from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient, models


DISPLAY_EMBEDDING_FIELDS = (
    "display_name_embedding",
    "display_name_embeddings",
    "display_name_sapbert_embedding",
    "sapbert_embedding",
    "name_embedding",
)

DOCUMENTATION_EMBEDDING_FIELDS = (
    "documentation_embedding",
    "documentation_embeddings",
    "documentation_biolord_embedding",
    "biolord_embedding",
    "description_embedding",
)

CHILDREN_EMBEDDING_FIELDS = (
    "children_embedding",
    "children_embeddings",
    "children_sapbert_embedding",
    "children_sapbert_embeddings",
    "child_embeddings",
)

PARENTS_EMBEDDING_FIELDS = (
    "parents_embedding",
    "parents_embeddings",
    "parents_sapbert_embedding",
    "parents_sapbert_embeddings",
    "parent_embeddings",
)

ALL_EMBEDDING_FIELDS = {
    *DISPLAY_EMBEDDING_FIELDS,
    *DOCUMENTATION_EMBEDDING_FIELDS,
    *CHILDREN_EMBEDDING_FIELDS,
    *PARENTS_EMBEDDING_FIELDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an embedded hierarchy JSON into Qdrant."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON file containing the hierarchy and embeddings.",
    )
    parser.add_argument(
        "--collection",
        default="drug_hierarchy",
        help="Qdrant collection name. Default: drug_hierarchy",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant URL. Default: http://localhost:6333",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Qdrant API key, normally needed for Qdrant Cloud.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Points uploaded per batch. Default: 64",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection if it already exists.",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(is_number(item) for item in value)
    )


def is_vector_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(is_vector(item) for item in value)
    )


def normalize_vector(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Expected a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Vector contains NaN or infinite values.")
    return array.tolist()


def mean_pool(vectors: list[list[float]]) -> list[float]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Expected one or more vectors for mean pooling.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Vectors contain NaN or infinite values.")
    return matrix.mean(axis=0).tolist()


def zero_vector(size: int) -> list[float]:
    return [0.0] * size


def find_value(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    candidates: tuple[str, ...],
) -> Any:
    for field_name in candidates:
        if field_name in entry:
            return entry[field_name]
        if field_name in metadata:
            return metadata[field_name]
    return None


def extract_vector(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    candidates: tuple[str, ...],
) -> list[float] | None:
    value = find_value(entry, metadata, candidates)
    if value is None:
        return None
    if is_vector(value):
        return normalize_vector(value)
    if is_vector_list(value):
        return mean_pool(value)
    return None


def stable_uuid(member_id: str) -> str:
    digest = hashlib.sha256(member_id.encode("utf-8")).digest()[:16]
    mutable = bytearray(digest)
    mutable[6] = (mutable[6] & 0x0F) | 0x40
    mutable[8] = (mutable[8] & 0x3F) | 0x80
    return str(UUID(bytes=bytes(mutable)))


def without_embedding_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_embedding_fields(item)
            for key, item in value.items()
            if key not in ALL_EMBEDDING_FIELDS
        }
    if isinstance(value, list):
        return [without_embedding_fields(item) for item in value]
    return value


def validate_dimension(
    vector_name: str,
    vector: list[float],
    expected_size: int,
    member_id: str,
) -> None:
    if len(vector) != expected_size:
        raise ValueError(
            f"Member {member_id!r}: vector {vector_name!r} has dimension "
            f"{len(vector)}, expected {expected_size}."
        )


def find_dimensions(hierarchy: dict[str, Any]) -> tuple[int, int]:
    sapbert_size: int | None = None
    biolord_size: int | None = None

    for entry in hierarchy.values():
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        if sapbert_size is None:
            vector = extract_vector(entry, metadata, DISPLAY_EMBEDDING_FIELDS)
            if vector is None:
                vector = extract_vector(entry, metadata, CHILDREN_EMBEDDING_FIELDS)
            if vector is None:
                vector = extract_vector(entry, metadata, PARENTS_EMBEDDING_FIELDS)
            if vector is not None:
                sapbert_size = len(vector)

        if biolord_size is None:
            vector = extract_vector(entry, metadata, DOCUMENTATION_EMBEDDING_FIELDS)
            if vector is not None:
                biolord_size = len(vector)

        if sapbert_size is not None and biolord_size is not None:
            return sapbert_size, biolord_size

    raise ValueError(
        "Could not determine the SapBERT and BioLORD vector dimensions. "
        "Inspect the JSON embedding field names and add them to the candidate "
        "field tuples near the top of this script."
    )


def point_iterator(
    hierarchy: dict[str, Any],
    sapbert_size: int,
    biolord_size: int,
) -> Iterable[models.PointStruct]:
    for member_id, entry in hierarchy.items():
        if not isinstance(entry, dict):
            print(f"Skipping {member_id!r}: entry is not a JSON object.")
            continue

        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        vectors = {
            "display_name": extract_vector(
                entry, metadata, DISPLAY_EMBEDDING_FIELDS
            ) or zero_vector(sapbert_size),
            "documentation": extract_vector(
                entry, metadata, DOCUMENTATION_EMBEDDING_FIELDS
            ) or zero_vector(biolord_size),
            "children": extract_vector(
                entry, metadata, CHILDREN_EMBEDDING_FIELDS
            ) or zero_vector(sapbert_size),
            "parents": extract_vector(
                entry, metadata, PARENTS_EMBEDDING_FIELDS
            ) or zero_vector(sapbert_size),
        }

        validate_dimension(
            "display_name", vectors["display_name"], sapbert_size, member_id
        )
        validate_dimension(
            "documentation", vectors["documentation"], biolord_size, member_id
        )
        validate_dimension(
            "children", vectors["children"], sapbert_size, member_id
        )
        validate_dimension(
            "parents", vectors["parents"], sapbert_size, member_id
        )

        payload = without_embedding_fields(entry)
        payload["member_id"] = member_id
        payload["display_name"] = metadata.get("display_name")
        payload["documentation"] = metadata.get("documentation")
        payload["durable_id"] = metadata.get("durable_id")
        payload["object_name"] = metadata.get("object_name")
        payload["children"] = entry.get("children", [])
        payload["parents"] = entry.get("parents", [])

        yield models.PointStruct(
            id=stable_uuid(member_id),
            vector=vectors,
            payload=payload,
        )


def create_collection(
    client: QdrantClient,
    collection_name: str,
    sapbert_size: int,
    biolord_size: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)

    if exists and recreate:
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        exists = False

    if exists:
        info = client.get_collection(collection_name)
        configured = info.config.params.vectors
        expected = {
            "display_name": sapbert_size,
            "documentation": biolord_size,
            "children": sapbert_size,
            "parents": sapbert_size,
        }
        for name, size in expected.items():
            if name not in configured or configured[name].size != size:
                raise ValueError(
                    f"Existing collection has an incompatible {name!r} vector. "
                    "Run again with --recreate to rebuild the collection."
                )
        print(f"Using existing collection: {collection_name}")
        return

    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "display_name": models.VectorParams(
                size=sapbert_size, distance=models.Distance.COSINE
            ),
            "documentation": models.VectorParams(
                size=biolord_size, distance=models.Distance.COSINE
            ),
            "children": models.VectorParams(
                size=sapbert_size, distance=models.Distance.COSINE
            ),
            "parents": models.VectorParams(
                size=sapbert_size, distance=models.Distance.COSINE
            ),
        },
    )

    for field_name in ("member_id", "durable_id", "object_name"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input.resolve()}")

    print(f"Reading: {args.input.resolve()}")
    with args.input.open("r", encoding="utf-8") as file:
        hierarchy = json.load(file)

    if not isinstance(hierarchy, dict):
        raise ValueError("Expected the JSON root to be an object keyed by member ID.")

    sapbert_size, biolord_size = find_dimensions(hierarchy)
    print(f"Members: {len(hierarchy)}")
    print(f"SapBERT dimension: {sapbert_size}")
    print(f"BioLORD dimension: {biolord_size}")

    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=120)
    create_collection(
        client,
        args.collection,
        sapbert_size,
        biolord_size,
        args.recreate,
    )

    print("Uploading points...")
    client.upload_points(
        collection_name=args.collection,
        points=point_iterator(hierarchy, sapbert_size, biolord_size),
        batch_size=args.batch_size,
        parallel=1,
        max_retries=3,
        wait=True,
    )

    info = client.get_collection(args.collection)
    print("Upload complete.")
    print(f"Collection: {args.collection}")
    print(f"Points: {info.points_count}")
    print(f"Dashboard: {args.url.rstrip('/')}/dashboard")


if __name__ == "__main__":
    main()
