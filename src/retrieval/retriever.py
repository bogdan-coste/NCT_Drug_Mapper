import re
import unicodedata
from typing import Any

from ..embedder.sapbert_embedder import SapBERTEmbedder
from ..vectordb.vector_store import VectorStore


class OntologyRetriever:
    """
    Shared SapBERT/Qdrant retriever for one configured ontology axis.

    Each retrieval query produces an independent ranking. Rankings are fused
    through Reciprocal Rank Fusion, exact display-name matches receive a
    deterministic bonus, and the highest-ranked candidates can be enriched
    with direct parents and children.

    Axis filtering is applied defensively from each Qdrant payload using
    ``app_drug_category``. The value may be stored either at payload root or
    under ``metadata``.
    """

    def __init__(
        self,
        embedder: SapBERTEmbedder,
        vector_store: VectorStore,
        app_drug_category: str,
        search_oversample_factor: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._app_drug_category = self._clean_text(app_drug_category)
        self._search_oversample_factor = search_oversample_factor

        if not self._app_drug_category:
            raise ValueError("app_drug_category cannot be empty.")

        if search_oversample_factor <= 0:
            raise ValueError(
                "search_oversample_factor must be greater than zero."
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Collapse whitespace and strip a value converted to text."""
        return " ".join(str(value or "").split()).strip()

    @staticmethod
    def _normalize_label(value: str) -> str:
        """
        Normalize a label for deterministic lexical comparison.

        This normalization is separate from embedding generation. The
        original query is still sent unchanged to SapBERT.
        """
        normalized = unicodedata.normalize("NFKC", str(value)).casefold()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    @classmethod
    def _payload_value(
        cls,
        payload: dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        """Read a field from payload root, then nested metadata."""
        metadata = payload.get("metadata") or {}

        root_value = payload.get(key)
        if root_value is not None:
            return root_value

        if isinstance(metadata, dict):
            metadata_value = metadata.get(key)
            if metadata_value is not None:
                return metadata_value

        return default

    @classmethod
    def _candidate_axis(cls, payload: dict[str, Any]) -> str:
        """Extract the candidate's configured application category."""
        return cls._clean_text(
            cls._payload_value(payload, "app_drug_category", "")
        )

    def _belongs_to_axis(self, payload: dict[str, Any]) -> bool:
        """Return True only for payloads belonging to this retriever axis."""
        return (
            self._candidate_axis(payload).casefold()
            == self._app_drug_category.casefold()
        )

    @classmethod
    def _candidate_key(
        cls,
        payload: dict[str, Any],
        name: str,
    ) -> str:
        """Build a stable candidate key from available ontology identifiers."""
        return str(
            cls._payload_value(payload, "durable_id")
            or cls._payload_value(payload, "object_name")
            or name.casefold()
        )

    @classmethod
    def _object_name(cls, payload: dict[str, Any]) -> str:
        return cls._clean_text(
            cls._payload_value(payload, "object_name", "")
        )

    @staticmethod
    def _deduplicate_queries(labels: list[str]) -> list[str]:
        """Clean and deduplicate retrieval queries while preserving order."""
        result: list[str] = []
        seen: set[str] = set()

        for label in labels:
            cleaned = " ".join(str(label).split()).strip()
            key = cleaned.casefold()

            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)

        return result

    @staticmethod
    def _validate_parameters(
        per_label_k: int,
        final_k: int,
        rrf_k: int,
        exact_match_weight: float,
        enrich_top_n: int,
    ) -> None:
        if per_label_k <= 0:
            raise ValueError("per_label_k must be greater than zero.")
        if final_k <= 0:
            raise ValueError("final_k must be greater than zero.")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative.")
        if exact_match_weight < 0:
            raise ValueError("exact_match_weight cannot be negative.")
        if enrich_top_n < 0:
            raise ValueError("enrich_top_n cannot be negative.")

    def _search_axis_hits(
        self,
        vector: list[float],
        desired_k: int,
    ) -> list[Any]:
        """
        Search Qdrant and retain only hits from the configured axis.

        The current VectorStore API has no required filter argument, so this
        method oversamples and filters locally. If VectorStore later supports
        a native Qdrant filter, apply the same axis condition there and keep
        this local check as a defense-in-depth validation.
        """
        search_k = max(
            desired_k,
            desired_k * self._search_oversample_factor,
        )
        hits = self._store.search(vector=vector, top_k=search_k)

        filtered: list[Any] = []
        for hit in hits:
            payload = hit.payload or {}
            if self._belongs_to_axis(payload):
                filtered.append(hit)
            if len(filtered) >= desired_k:
                break

        return filtered

    def retrieve(self, term: str, top_k: int = 20) -> list[str]:
        """Return top ontology names for one query on this retriever axis."""
        query = self._clean_text(term)

        if not query or top_k <= 0:
            return []

        vector = self._embedder.embed_one(query)
        hits = self._search_axis_hits(vector=vector, desired_k=top_k)

        names: list[str] = []
        seen: set[str] = set()

        for hit in hits:
            payload = hit.payload or {}
            name = self._clean_text(
                self._payload_value(payload, "display_name", "")
            )
            key = name.casefold()

            if name and key not in seen:
                seen.add(key)
                names.append(name)

        return names[:top_k]

    def retrieve_for_labels(
        self,
        labels: list[str],
        per_label_k: int = 20,
        final_k: int = 20,
        rrf_k: int = 60,
        exact_match_weight: float = 1.0,
        enrich_top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search once per query, fuse with RRF, and enrich top candidates.

        Returned candidate dictionaries include ``app_drug_category`` and
        ontology identifiers so downstream deterministic validation can verify
        both candidate identity and mapping axis.
        """
        self._validate_parameters(
            per_label_k=per_label_k,
            final_k=final_k,
            rrf_k=rrf_k,
            exact_match_weight=exact_match_weight,
            enrich_top_n=enrich_top_n,
        )

        cleaned_labels = self._deduplicate_queries(labels)
        if not cleaned_labels:
            return []

        # Pre-compute normalised forms of all query labels so we can detect
        # exact matches against Qdrant candidate names during the ranking loop.
        normalized_labels = {
            self._normalize_label(label) for label in cleaned_labels
        }

        # Accumulators shared across all per-label searches.
        # Each candidate is keyed by a stable string derived from its payload.
        rrf_scores: dict[str, float] = {}          # fused RRF score per candidate
        names: dict[str, str] = {}                 # display name per candidate
        docs: dict[str, str] = {}                  # documentation text (first seen)
        object_names: dict[str, str] = {}          # internal ontology object name
        durable_ids: dict[str, str] = {}           # stable cross-version identifier
        parent_ids: dict[str, list[str]] = {}      # parent node object names
        child_ids: dict[str, list[str]] = {}       # child node object names
        exact_match_keys: set[str] = set()         # candidates whose name matches a label exactly
        provenance: dict[str, list[dict[str, Any]]] = {}  # which queries surfaced each candidate

        # ── Step 1: embed each label and collect per-label Qdrant hits ────────
        # Each label (specific, broad, root category) is searched independently.
        # Results from all labels are merged below via RRF.
        for label in cleaned_labels:
            print(
                f"  [retriever:{self._app_drug_category}] "
                f"embedding query: {label!r}",
                flush=True,
            )
            vector = self._embedder.embed_one(label)
            print(
                f"  [retriever:{self._app_drug_category}] "
                f"vector dim: {len(vector)}, searching Qdrant...",
                flush=True,
            )

            hits = self._search_axis_hits(
                vector=vector,
                desired_k=per_label_k,
            )
            print(
                f"  [retriever:{self._app_drug_category}] "
                f"got {len(hits)} axis-filtered hits",
                flush=True,
            )

            seen_in_ranking: set[str] = set()

            # ── Step 2: accumulate RRF scores across all per-label rankings ──
            # For each hit returned by this label's search, add its reciprocal
            # rank contribution: score += 1 / (rrf_k + rank).
            # rrf_k=60 is a standard damping constant that prevents the top
            # rank from dominating too heavily.
            # A candidate that appears in multiple label searches accumulates
            # contributions from each, rising above single-label-only hits.
            for rank, hit in enumerate(hits, start=1):
                payload = hit.payload or {}

                # Keep the local axis check even after _search_axis_hits().
                if not self._belongs_to_axis(payload):
                    continue

                name = self._clean_text(
                    self._payload_value(payload, "display_name", "")
                )
                if not name:
                    continue

                key = self._candidate_key(payload, name)
                if key in seen_in_ranking:
                    continue
                seen_in_ranking.add(key)

                rrf_scores[key] = (
                    rrf_scores.get(key, 0.0)
                    + 1.0 / (rrf_k + rank)
                )
                names[key] = name

                # Store metadata on first encounter only; it describes the
                # ontology node itself and does not change across labels.
                if key not in docs:
                    docs[key] = self._clean_text(
                        self._payload_value(payload, "documentation", "")
                    )
                    object_names[key] = self._object_name(payload)
                    durable_ids[key] = self._clean_text(
                        self._payload_value(payload, "durable_id", "")
                    )
                    parent_ids[key] = list(
                        self._payload_value(payload, "parents", []) or []
                    )
                    child_ids[key] = list(
                        self._payload_value(payload, "children", []) or []
                    )

                provenance.setdefault(key, []).append(
                    {"query": label, "rank": rank}
                )

                # ── Step 3 (detection): flag exact-name matches ─────────────
                # If this candidate's display name exactly matches one of the
                # query labels (case-insensitive), mark it for a score boost
                # applied after the loop.
                if self._normalize_label(name) in normalized_labels:
                    exact_match_keys.add(key)

        if not rrf_scores:
            return []

        # ── Step 3 (application): boost exact-match candidates ───────────────
        # A candidate whose name exactly matches a query label gets a one-off
        # score addition equivalent to a rank-1 hit, so it reliably surfaces
        # above semantically similar but non-identical neighbours.
        exact_match_boost = exact_match_weight / (rrf_k + 1)
        for key in exact_match_keys:
            if key in rrf_scores:
                rrf_scores[key] += exact_match_boost

        # ── Step 4: sort by fused score and trim to final_k ─────────────────
        # Primary sort: descending RRF score.
        # Tiebreaker: alphabetical by name so the output is deterministic.
        ranked = sorted(
            rrf_scores,
            key=lambda key: (
                -rrf_scores[key],
                names[key].casefold(),
            ),
        )[:final_k]

        print(
            f"  [retriever:{self._app_drug_category}] "
            f"{len(rrf_scores)} unique candidates, "
            f"returning top {len(ranked)}",
            flush=True,
        )

        for index, key in enumerate(ranked, start=1):
            marker = " exact-match" if key in exact_match_keys else ""
            print(
                f"    {index:>2}. {names[key]} "
                f"(rrf={rrf_scores[key]:.6f}{marker})",
                flush=True,
            )

        # ── Step 5: enrich top-N candidates with ontology hierarchy ─────────
        # For the top enrich_top_n candidates (default 5), fetch their parent
        # and child nodes from Qdrant. This gives the OntologyMapper LLM
        # hierarchy context so it can see where each candidate sits in the
        # ontology tree, improving mapping accuracy.
        top_keys = ranked[: min(enrich_top_n, len(ranked))]
        related_object_names: list[str] = []

        for key in top_keys:
            related_object_names.extend(parent_ids.get(key, []))
            related_object_names.extend(child_ids.get(key, []))

        related = self._store.fetch_by_object_names(
            list(dict.fromkeys(related_object_names))
        )

        def resolve(requested_names: list[str]) -> list[dict[str, Any]]:
            resolved: list[dict[str, Any]] = []

            for object_name in requested_names:
                item = related.get(object_name)
                if not item:
                    continue

                # fetch_by_object_names may return flattened payloads or
                # complete payload dictionaries. Support both shapes.
                payload = item.get("payload") or item
                if not isinstance(payload, dict):
                    continue
                if not self._belongs_to_axis(payload):
                    continue

                display_name = self._clean_text(
                    self._payload_value(payload, "display_name", "")
                )
                if not display_name:
                    continue

                resolved.append(
                    {
                        "name": display_name,
                        "documentation": self._clean_text(
                            self._payload_value(
                                payload,
                                "documentation",
                                "",
                            )
                        ),
                        "object_name": self._object_name(payload),
                        "app_drug_category": self._candidate_axis(payload),
                    }
                )

            return resolved

        # ── Assemble final result list ─────────────────────────────────────
        # Each entry carries identity fields, RRF score, exact-match flag,
        # and retrieval provenance. Parents and children are attached only
        # for the top enrich_top_n entries.
        results: list[dict[str, Any]] = []

        for index, key in enumerate(ranked):
            entry: dict[str, Any] = {
                "name": names[key],
                "documentation": docs[key],
                "object_name": object_names.get(key, ""),
                "durable_id": durable_ids.get(key, ""),
                "app_drug_category": self._app_drug_category,
                "rrf_score": rrf_scores[key],
                "exact_match": key in exact_match_keys,
                "retrieval_provenance": provenance.get(key, []),
            }

            if index < enrich_top_n:
                entry["parents"] = resolve(parent_ids.get(key, []))
                entry["children"] = resolve(child_ids.get(key, []))

            results.append(entry)

        return results


class TherapeuticCategoryRetriever(OntologyRetriever):
    """Retriever restricted to the therapeutic-category ontology axis."""

    def __init__(
        self,
        embedder: SapBERTEmbedder,
        vector_store: VectorStore,
        search_oversample_factor: int = 5,
    ) -> None:
        super().__init__(
            embedder=embedder,
            vector_store=vector_store,
            app_drug_category="therapeutic category",
            search_oversample_factor=search_oversample_factor,
        )


class DrugClassRetriever(OntologyRetriever):
    """Retriever restricted to the molecular/compositional drug-class axis."""

    def __init__(
        self,
        embedder: SapBERTEmbedder,
        vector_store: VectorStore,
        search_oversample_factor: int = 5,
    ) -> None:
        super().__init__(
            embedder=embedder,
            vector_store=vector_store,
            app_drug_category="drug class",
            search_oversample_factor=search_oversample_factor,
        )
