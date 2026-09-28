import json
import unicodedata
from typing import Any

from pydantic import ValidationError

from ...generation.drug_class_grounded_answer import (
    DrugClassDefinitionGenerator,
    DrugClassOntologyMapper,
    DrugClassRetrievalQuerySuggester,
)
from ...generation.term_extract_grounded_answer import NCItCandidateListExtractor
from ...ingestion.nct_loader import NCTLoader
from ...llm.llm_client import LLMClient
from ...retrieval.retriever import DrugClassRetriever
from ...schemas.models import (
    DrugClassRetrievalSuggestion,
    InternalDrugClassMapping,
    NCItIdentityResolution,
    NCTDrugContext,
)


class DrugClass:
    """
    Orchestrates the drug-class mapping pipeline using the same LLM
    identity-resolution flow as TherapeuticCategory.

    Step 1 — NCT fetch + LLM identity resolution:
        Fetch NCT records, resolve the author term against intervention
        records via LLM (NCItCandidateListExtractor), verify matched
        names exist verbatim, filter by role.

    Step 2 — Composition-centered definition:
        Generate a definition focused on molecular, structural,
        biochemical, compositional, or origin-based identity.

    Step 3 — Retrieval-query suggestion:
        Convert the definition into one primary and up to two
        complementary drug-class retrieval queries.

    Step 4 — Candidate retrieval:
        SapBERT + Qdrant RRF across all queries, restricted to the
        drug-class ontology axis.

    Step 5 — Ontology mapping:
        Select the best compatible drug class from candidates.
    """

    _DRUG_CLASS_AXIS = "drug class"

    _VALID_MAPPING_COMBINATIONS: dict[str, set[str]] = {
        "directly_compatible": {
            "exact",
            "equivalent_label",
        },
        "broader_compatible": {
            "broader_available",
        },
        "related_compatible": {
            "nearest_available",
        },
        "none": {
            "none",
        },
    }

    def __init__(
        self,
        nct_loader: NCTLoader,
        definition_llm: LLMClient,
        query_llm: LLMClient,
        mapping_llm: LLMClient,
        retriever: DrugClassRetriever,
        term_extractor_llm: LLMClient | None = None,
    ) -> None:
        self._loader = nct_loader
        self._definition_generator = DrugClassDefinitionGenerator(definition_llm)
        self._query_suggester = DrugClassRetrievalQuerySuggester(query_llm)
        self._ontology_mapper = DrugClassOntologyMapper(mapping_llm)
        self._retriever = retriever
        self._candidate_extractor = (
            NCItCandidateListExtractor(term_extractor_llm)
            if term_extractor_llm is not None
            else None
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _err(term: str, message: str, **extra: Any) -> dict[str, Any]:
        return {"term": term, "error": message, **extra}

    @staticmethod
    def _none_mapping(reason: str) -> dict[str, str]:
        return {
            "selected_term": "NONE_OF_THE_ABOVE",
            "match_type": "none",
            "reason": reason,
        }

    @staticmethod
    def _normalize_labels(labels: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for label in labels:
            normalized = " ".join(str(label).split()).strip()
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result

    @staticmethod
    def _identity_key(value: str) -> str:
        return " ".join(
            unicodedata.normalize("NFKC", str(value)).casefold().split()
        ).strip()

    @staticmethod
    def _is_unknown_definition(definition: str) -> bool:
        normalized = " ".join(str(definition).casefold().split())
        markers = (
            "don't know",
            "do not know",
            "insufficient information",
            "not enough information",
            "cannot determine",
            "unable to determine",
            "term was not found",
            "intervention was not found",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _get_candidate_name(candidate: Any) -> str:
        if isinstance(candidate, dict):
            return str(
                candidate.get("name")
                or candidate.get("label")
                or candidate.get("term")
                or ""
            ).strip()
        return str(
            getattr(candidate, "name", "")
            or getattr(candidate, "label", "")
            or getattr(candidate, "term", "")
            or ""
        ).strip()

    @staticmethod
    def _get_candidate_axis(candidate: Any) -> str:
        if candidate is None:
            return ""
        if isinstance(candidate, dict):
            metadata = candidate.get("metadata") or {}
            metadata_axis = (
                metadata.get("app_drug_category")
                if isinstance(metadata, dict)
                else None
            )
            value = candidate.get("app_drug_category") or metadata_axis or ""
            return " ".join(str(value).split()).strip()
        metadata = getattr(candidate, "metadata", None) or {}
        metadata_axis = (
            metadata.get("app_drug_category")
            if isinstance(metadata, dict)
            else getattr(metadata, "app_drug_category", "")
        )
        value = (
            getattr(candidate, "app_drug_category", "")
            or metadata_axis
            or ""
        )
        return " ".join(str(value).split()).strip()

    @classmethod
    def _validate_internal_mapping(
        cls,
        internal_mapping: InternalDrugClassMapping,
        candidates: list[Any],
    ) -> dict[str, str]:
        allowed_match_types = cls._VALID_MAPPING_COMBINATIONS.get(
            internal_mapping.compatibility, set()
        )
        candidate_by_name = {
            name.casefold(): candidate
            for candidate in candidates
            if (name := cls._get_candidate_name(candidate))
        }
        selected_is_none = internal_mapping.selected_term == "NONE_OF_THE_ABOVE"
        none_state_is_consistent = (
            selected_is_none
            and internal_mapping.compatibility == "none"
            and internal_mapping.match_type == "none"
        )
        if none_state_is_consistent:
            return cls._none_mapping(internal_mapping.reason)

        selected_candidate = candidate_by_name.get(
            internal_mapping.selected_term.casefold()
        )
        selected_axis = cls._get_candidate_axis(selected_candidate)
        real_selection_is_consistent = (
            not selected_is_none
            and selected_candidate is not None
            and selected_axis.casefold() == cls._DRUG_CLASS_AXIS.casefold()
            and internal_mapping.compatibility
            in {"directly_compatible", "broader_compatible", "related_compatible"}
            and internal_mapping.match_type in allowed_match_types
        )
        if not real_selection_is_consistent:
            return cls._none_mapping(
                "None of the supplied ontology candidates has a compatible "
                "drug-class meaning."
            )
        return {
            "selected_term": internal_mapping.selected_term,
            "match_type": internal_mapping.match_type,
            "reason": internal_mapping.reason,
        }

    # ------------------------------------------------------------------
    # Identity resolution (shared with TherapeuticCategory)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_raw_intervention_context(trials: list[NCTDrugContext]) -> str:
        sections: list[str] = []
        for trial in trials:
            lines: list[str] = []
            if trial.nct_id:
                lines.append(f"NCT ID: {trial.nct_id}")
            if trial.official_title:
                lines.append(f"Official title: {trial.official_title}")
            if trial.brief_title:
                lines.append(f"Brief title: {trial.brief_title}")
            for index, intervention in enumerate(trial.interventions, start=1):
                name = " ".join(str(intervention.name or "").split()).strip()
                if not name:
                    continue
                lines.append(f"Intervention {index} exact name: {name}")
                intervention_type = " ".join(
                    str(intervention.type or "").split()
                ).strip()
                if intervention_type:
                    lines.append(f"Intervention {index} type: {intervention_type}")
                aliases = [
                    " ".join(str(alias).split()).strip()
                    for alias in (getattr(intervention, "other_names", None) or [])
                    if str(alias).strip()
                ]
                if aliases:
                    lines.append(
                        f"Intervention {index} alternative names: "
                        + ", ".join(aliases)
                    )
                description = " ".join(
                    str(intervention.description or "").split()
                ).strip()
                if description:
                    lines.append(
                        f"Intervention {index} description: {description}"
                    )
            sections.append("\n".join(f"- {line}" for line in lines))
        return "\n\n---\n\n".join(sections)

    @classmethod
    def _extract_identity_matches(cls, extracted: Any) -> list[dict[str, str]]:
        """Normalize structured extractor output into match dictionaries.

        Plain strings are rejected: the extractor must return structured dicts
        so that matched_trial_name can be verified against the NCT records.
        """
        # Primary path: typed Pydantic output.
        if isinstance(extracted, NCItIdentityResolution):
            return [m.model_dump() for m in extracted.matches]

        # Legacy path: raw dict from old code or test stubs.
        if extracted is None:
            return []
        if isinstance(extracted, dict):
            items = extracted.get("matches") or extracted.get("candidates") or []
        else:
            items = getattr(extracted, "matches", None)
            if items is None:
                items = getattr(extracted, "candidates", None)
            if items is None:
                items = extracted if isinstance(extracted, list) else []

        matches: list[dict[str, str]] = []
        for item in items:
            if isinstance(item, str):
                print(
                    f"  [dc-identity] rejected unstructured candidate: {item!r}",
                    flush=True,
                )
                continue
            if isinstance(item, dict):
                get = item.get
            else:
                get = lambda key, default="", _item=item: getattr(_item, key, default)
            search_term = " ".join(
                str(get("search_term") or get("term") or "").split()
            ).strip()
            matched_name = " ".join(
                str(get("matched_trial_name") or get("trial_name") or "").split()
            ).strip()
            relationship = " ".join(
                str(get("relationship") or "normalized_name").split()
            ).strip()
            role = " ".join(str(get("role") or "").split()).strip()
            evidence = " ".join(str(get("evidence") or "").split()).strip()
            if search_term and matched_name:
                entry: dict[str, str] = {
                    "search_term": search_term,
                    "matched_trial_name": matched_name,
                    "relationship": relationship,
                }
                if role:
                    entry["role"] = role
                if evidence:
                    entry["evidence"] = evidence
                matches.append(entry)
        return matches

    @classmethod
    def _validate_identity_matches(
        cls,
        *,
        extracted: Any,
        trials: list[NCTDrugContext],
    ) -> list[tuple[NCTDrugContext, Any, dict[str, str]]]:
        index: dict[str, list[tuple[NCTDrugContext, Any]]] = {}
        for trial in trials:
            for intervention in trial.interventions:
                names = [
                    str(intervention.name or "").strip(),
                    *[
                        str(alias).strip()
                        for alias in (
                            getattr(intervention, "other_names", None) or []
                        )
                        if str(alias).strip()
                    ],
                ]
                for name in names:
                    key = cls._identity_key(name)
                    if key:
                        index.setdefault(key, []).append((trial, intervention))

        validated: list[tuple[NCTDrugContext, Any, dict[str, str]]] = []
        seen: set[tuple[str, str, str]] = set()
        for match in cls._extract_identity_matches(extracted):
            key = cls._identity_key(match["matched_trial_name"])
            records = index.get(key, [])
            if not records:
                print(
                    "  [dc-identity] rejected unknown trial name: "
                    f"{match['matched_trial_name']!r}",
                    flush=True,
                )
                continue
            for trial, intervention in records:
                record_key = (
                    str(trial.nct_id or ""),
                    cls._identity_key(str(intervention.name or "")),
                    cls._identity_key(match["search_term"]),
                )
                if record_key in seen:
                    continue
                seen.add(record_key)
                validated.append((trial, intervention, match))
        return validated

    @classmethod
    def _filter_resolved_records(
        cls,
        *,
        term: str,
        resolution_kind: str,
        resolved_records: list[tuple[NCTDrugContext, Any, dict[str, str]]],
    ) -> list[tuple[NCTDrugContext, Any, dict[str, str]]]:
        allowed_roles: dict[str, set[str]] = {
            "single_product": {"requested_identity"},
            "single_procedure": {"requested_identity"},
            "regimen": {"requested_regimen_component"},
            "composite_product": {"requested_composite_component"},
            "class_request": {"requested_identity"},
        }
        required = allowed_roles.get(resolution_kind, set())

        role_filtered = [
            record
            for record in resolved_records
            if record[2].get("role", "") in required
        ]
        if role_filtered:
            return role_filtered

        # Fallback: roles absent — keep non-combination records for single_product
        if resolution_kind == "single_product":
            author_key = cls._identity_key(term)
            non_component = {
                "exact_name", "normalized_name", "alias", "generic_name",
                "brand_name", "development_code", "active_substance", "explicit_alias",
            }
            fallback = [
                record
                for record in resolved_records
                if (
                    cls._identity_key(record[2]["search_term"]) == author_key
                    or record[2].get("relationship", "") in non_component
                )
            ]
            if fallback:
                print(
                    "  [dc-identity] role absent; fell back to key/relationship "
                    f"heuristic ({len(fallback)} record(s) kept).",
                    flush=True,
                )
                return fallback

        print(
            "  [dc-identity] role absent and heuristic produced nothing; "
            "returning all records.",
            flush=True,
        )
        return resolved_records

    @classmethod
    def _resolution_kind(cls, extracted: Any) -> str:
        if isinstance(extracted, NCItIdentityResolution):
            return extracted.resolution_kind
        if isinstance(extracted, dict):
            value = extracted.get("resolution_kind", "")
        else:
            value = getattr(extracted, "resolution_kind", "")
        return str(value or "single_product").strip().casefold()

    @staticmethod
    def _build_resolved_context(
        records: list[tuple[NCTDrugContext, Any, dict[str, str]]],
    ) -> str:
        sections: list[str] = []
        for trial, intervention, match in records:
            identification: list[str] = []
            evidence: list[str] = []
            supporting: list[str] = []

            if trial.nct_id:
                identification.append(f"NCT ID: {trial.nct_id}")
            if trial.official_title:
                identification.append(f"Official title: {trial.official_title}")
            if trial.brief_title:
                identification.append(f"Brief title: {trial.brief_title}")
            sponsor = getattr(trial, "sponsor", None)
            if sponsor:
                identification.append(f"Sponsor: {sponsor}")

            evidence.append(
                f"Resolved intervention identity: {match['search_term']}"
            )
            evidence.append(
                f"Matched ClinicalTrials.gov intervention: {intervention.name}"
            )
            evidence.append(f"Identity relationship: {match['relationship']}")
            intervention_type = str(intervention.type or "").strip()
            if intervention_type:
                evidence.append(f"Intervention type: {intervention_type}")
            aliases = [
                str(alias).strip()
                for alias in (getattr(intervention, "other_names", None) or [])
                if str(alias).strip()
            ]
            if aliases:
                evidence.append(
                    "Recorded alternative names: " + ", ".join(aliases)
                )
            description = str(intervention.description or "").strip()
            if description:
                evidence.append(f"Intervention description: {description}")

            if trial.brief_summary:
                supporting.append(f"Brief summary: {trial.brief_summary}")
            if trial.detailed_description:
                supporting.append(
                    f"Detailed description: {trial.detailed_description}"
                )
            if trial.conditions:
                supporting.append("Conditions: " + ", ".join(trial.conditions))
            if trial.intervention_mesh_terms:
                supporting.append(
                    "Intervention MeSH terms: "
                    + ", ".join(trial.intervention_mesh_terms)
                )
            if trial.mesh_terms:
                supporting.append(
                    "Disease MeSH terms: " + ", ".join(trial.mesh_terms)
                )

            parts = [
                "TRIAL IDENTIFICATION:\n"
                + "\n".join(f"- {item}" for item in identification),
                "DIRECT INTERVENTION EVIDENCE:\n"
                + "\n".join(f"- {item}" for item in evidence),
            ]
            if supporting:
                parts.append(
                    "SUPPORTING TRIAL CONTEXT:\n"
                    + "\n".join(f"- {item}" for item in supporting)
                )
            sections.append("\n\n".join(parts))
        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_staged(
        self,
        term: str,
        nct_ids: list[str],
        on_stage,  # callable(dict) — called after each stage with partial results
    ) -> None:
        """
        Execute the drug-class pipeline stage-by-stage, calling on_stage(partial_dict)
        after each stage so callers can stream partial results to clients.
        Raises ValueError on unrecoverable errors.
        """
        emit = on_stage

        term = " ".join(str(term).split()).strip()
        if not term:
            raise ValueError("Drug or therapy term cannot be empty.")

        normalized_nct_ids = list(
            dict.fromkeys(
                str(nct_id).strip().upper()
                for nct_id in nct_ids
                if str(nct_id).strip()
            )
        )
        if not normalized_nct_ids:
            raise ValueError("At least one valid NCT ID must be supplied.")
        if self._candidate_extractor is None:
            raise ValueError("No LLM intervention-identity resolver is configured.")

        # ── Stage 1: Fetch NCT records + LLM identity resolution ─────────────
        fetched_trials = []
        context_warnings = []
        for nct_id in normalized_nct_ids:
            try:
                raw = self._loader.fetch_study(nct_id)
                trial = NCTDrugContext.from_api(raw)
                fetched_trials.append(trial)
                print(f"  [dc-pipeline] {nct_id}: {len(trial.interventions)} intervention(s) found.", flush=True)
            except Exception as exc:
                context_warnings.append(f"{nct_id}: failed to load or parse record: {exc}")

        if not fetched_trials:
            raise ValueError("No NCT records could be loaded.")

        raw_identity_context = self._build_raw_intervention_context(fetched_trials)
        extracted = self._candidate_extractor.extract(
            term=term,
            nct_ids=normalized_nct_ids,
            context=raw_identity_context,
        )
        identity_matches = self._extract_identity_matches(extracted)
        resolved_records = self._validate_identity_matches(
            extracted=extracted,
            trials=fetched_trials,
        )
        if not resolved_records:
            raise ValueError("The requested intervention could not be resolved against the NCT intervention records.")

        resolution_kind = self._resolution_kind(extracted)
        resolved_records = self._filter_resolved_records(
            term=term,
            resolution_kind=resolution_kind,
            resolved_records=resolved_records,
        )
        if not resolved_records:
            raise ValueError("The requested intervention could not be isolated after role filtering.")

        emit({
            "term": term,
            "nct_ids": normalized_nct_ids,
            "identity_resolution": identity_matches,
            "context_warnings": context_warnings,
            "_stage": 1,
        })

        # ── Stage 2: Composition-centered definition ──────────────────────────
        definition_context = self._build_resolved_context(resolved_records)
        definition = self._definition_generator.generate(
            term=term,
            nct_ids=normalized_nct_ids,
            context=definition_context,
        )
        if not definition or self._is_unknown_definition(definition):
            raise ValueError("Could not generate a composition-centered definition.")

        emit({
            "definition": definition,
            "_stage": 2,
        })

        # ── Stage 3: Retrieval-query suggestion ───────────────────────────────
        retrieval_query_raw = self._query_suggester.suggest(definition=definition)
        if not retrieval_query_raw:
            raise ValueError("Drug-class retrieval-query generation failed.")
        try:
            retrieval_suggestion = DrugClassRetrievalSuggestion.model_validate_json(retrieval_query_raw)
        except ValidationError as exc:
            raise ValueError(f"Drug-class retrieval-query validation failed: {exc}")

        retrieval_query_data = retrieval_suggestion.model_dump()
        retrieval_queries = self._normalize_labels([
            retrieval_suggestion.primary_query,
            *retrieval_suggestion.alternative_queries,
        ])

        emit({
            "retrieval_queries": retrieval_query_data,
            "_stage": 3,
        })

        # ── Stage 4: Candidate retrieval + ontology mapping ───────────────────
        candidates = self._retriever.retrieve_for_labels(
            labels=retrieval_queries,
            per_label_k=20,
            final_k=20,
        )
        if not candidates:
            raise ValueError("No compatible drug-class candidates were retrieved.")

        import json as _json
        retrieval_query_context = _json.dumps(
            {
                **retrieval_query_data,
                "query_role": "These values are drug-class retrieval aids. They do not assert ontology parent-child relationships.",
                "retrieval_scope": "The candidate pool is restricted to ontology nodes marked app_drug_category=drug class.",
            },
            ensure_ascii=False,
            indent=2,
        )

        mapping_raw = self._ontology_mapper.map(
            definition=definition,
            retrieval_queries=retrieval_query_context,
            candidates=candidates,
        )
        if not mapping_raw:
            raise ValueError("Drug-class ontology mapping failed.")
        try:
            internal_mapping = InternalDrugClassMapping.model_validate_json(mapping_raw)
        except ValidationError as exc:
            raise ValueError(f"Drug-class ontology mapping validation failed: {exc}")

        drug_class_mapping = self._validate_internal_mapping(
            internal_mapping=internal_mapping,
            candidates=candidates,
        )

        emit({
            "candidates": candidates,
            "drug_class_mapping": drug_class_mapping,
            "_stage": 4,
        })

    def run(self, term: str, nct_ids: list[str]) -> dict[str, Any]:
        term = " ".join(str(term).split()).strip()
        if not term:
            return self._err(term, "Drug or therapy term cannot be empty.")

        normalized_nct_ids = list(
            dict.fromkeys(
                str(nct_id).strip().upper()
                for nct_id in nct_ids
                if str(nct_id).strip()
            )
        )
        if not normalized_nct_ids:
            return self._err(term, "At least one valid NCT ID must be supplied.")

        if self._candidate_extractor is None:
            return self._err(
                term,
                "No LLM intervention-identity resolver is configured.",
            )

        # ── Step 1: Fetch NCT records ─────────────────────────────────────────
        fetched_trials: list[NCTDrugContext] = []
        context_warnings: list[str] = []

        for nct_id in normalized_nct_ids:
            try:
                raw = self._loader.fetch_study(nct_id)
                trial = NCTDrugContext.from_api(raw)
                fetched_trials.append(trial)
                print(
                    f"  [dc-pipeline] {nct_id}: "
                    f"{len(trial.interventions)} intervention(s) found.",
                    flush=True,
                )
            except Exception as exc:
                context_warnings.append(
                    f"{nct_id}: failed to load or parse record: {exc}"
                )

        if not fetched_trials:
            return self._err(
                term,
                "No NCT records could be loaded.",
                context_warnings=context_warnings,
            )

        # ── Step 2: LLM identity resolution ──────────────────────────────────
        raw_identity_context = self._build_raw_intervention_context(fetched_trials)
        extracted = self._candidate_extractor.extract(
            term=term,
            nct_ids=normalized_nct_ids,
            context=raw_identity_context,
        )

        identity_matches = self._extract_identity_matches(extracted)

        print(
            "  [dc-identity] raw extractor output: "
            + (
                repr(extracted.model_dump())
                if isinstance(extracted, NCItIdentityResolution)
                else repr(extracted)
            ),
            flush=True,
        )
        print(
            "  [dc-identity] normalized matches: "
            f"{identity_matches!r}",
            flush=True,
        )

        resolved_records = self._validate_identity_matches(
            extracted=extracted,
            trials=fetched_trials,
        )

        print(
            "  [dc-identity] validated matches: "
            + ", ".join(
                f"{trial.nct_id}: {intervention.name!r} -> {match['search_term']!r}"
                for trial, intervention, match in resolved_records
            ),
            flush=True,
        )

        if not resolved_records:
            return self._err(
                term,
                "The requested intervention could not be resolved against the "
                "NCT intervention records.",
                identity_resolution=identity_matches,
                context_warnings=context_warnings,
            )

        resolution_kind = self._resolution_kind(extracted)

        resolved_records = self._filter_resolved_records(
            term=term,
            resolution_kind=resolution_kind,
            resolved_records=resolved_records,
        )

        if not resolved_records:
            return self._err(
                term,
                "The requested intervention could not be isolated from other "
                "trial interventions after role filtering.",
                identity_resolution=identity_matches,
                context_warnings=context_warnings,
            )

        definition_context = self._build_resolved_context(resolved_records)

        # ── Step 3: Composition-centered definition ───────────────────────────
        definition = self._definition_generator.generate(
            term=term,
            nct_ids=normalized_nct_ids,
            context=definition_context,
        )

        if not definition or self._is_unknown_definition(definition):
            return {
                "term": term,
                "nct_ids": normalized_nct_ids,
                "definition": definition or "",
                "identity_resolution": identity_matches,
                "retrieval_queries": {
                    "primary_query": "",
                    "alternative_queries": [],
                },
                "candidates": [],
                "drug_class_mapping": self._none_mapping(
                    "The supplied NCT evidence does not establish a reusable "
                    "molecular, structural, biochemical, compositional, or "
                    "origin-based drug class."
                ),
                "context_warnings": context_warnings,
            }

        # ── Step 4: Retrieval-query suggestion ────────────────────────────────
        retrieval_query_raw = self._query_suggester.suggest(definition=definition)
        if not retrieval_query_raw:
            return self._err(
                term,
                "Drug-class retrieval-query generation failed.",
                definition=definition,
                context_warnings=context_warnings,
            )

        try:
            retrieval_suggestion = DrugClassRetrievalSuggestion.model_validate_json(
                retrieval_query_raw
            )
        except ValidationError as exc:
            return self._err(
                term,
                f"Drug-class retrieval-query validation failed: {exc}",
                definition=definition,
                retrieval_query_raw=retrieval_query_raw,
                context_warnings=context_warnings,
            )

        retrieval_query_data = retrieval_suggestion.model_dump()
        retrieval_queries = self._normalize_labels(
            [
                retrieval_suggestion.primary_query,
                *retrieval_suggestion.alternative_queries,
            ]
        )

        print(
            f"  [dc-pipeline] retrieval queries: {retrieval_queries}",
            flush=True,
        )

        if not retrieval_queries:
            return {
                "term": term,
                "nct_ids": normalized_nct_ids,
                "definition": definition,
                "identity_resolution": identity_matches,
                "retrieval_queries": retrieval_query_data,
                "candidates": [],
                "drug_class_mapping": self._none_mapping(
                    "The intervention definition does not establish a reusable "
                    "molecular, structural, biochemical, compositional, or "
                    "origin-based drug class."
                ),
                "context_warnings": context_warnings,
            }

        retrieval_query_context = json.dumps(
            {
                **retrieval_query_data,
                "query_role": (
                    "These values are drug-class retrieval aids. They do not "
                    "assert ontology parent-child relationships."
                ),
                "retrieval_scope": (
                    "The candidate pool is restricted to ontology nodes "
                    "marked app_drug_category=drug class."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

        # ── Step 5: Candidate retrieval ───────────────────────────────────────
        candidates = self._retriever.retrieve_for_labels(
            labels=retrieval_queries,
            per_label_k=20,
            final_k=20,
        )

        if not candidates:
            return {
                "term": term,
                "nct_ids": normalized_nct_ids,
                "definition": definition,
                "identity_resolution": identity_matches,
                "retrieval_queries": retrieval_query_data,
                "candidates": [],
                "drug_class_mapping": self._none_mapping(
                    "No compatible drug-class candidates were retrieved from "
                    "the configured ontology axis."
                ),
                "context_warnings": context_warnings,
            }

        # ── Step 6: Ontology mapping ──────────────────────────────────────────
        mapping_raw = self._ontology_mapper.map(
            definition=definition,
            retrieval_queries=retrieval_query_context,
            candidates=candidates,
        )

        if not mapping_raw:
            return self._err(
                term,
                "Drug-class ontology mapping failed.",
                definition=definition,
                identity_resolution=identity_matches,
                retrieval_queries=retrieval_query_data,
                context_warnings=context_warnings,
            )

        try:
            internal_mapping = InternalDrugClassMapping.model_validate_json(
                mapping_raw
            )
        except ValidationError as exc:
            return self._err(
                term,
                f"Drug-class ontology mapping validation failed: {exc}",
                definition=definition,
                identity_resolution=identity_matches,
                retrieval_queries=retrieval_query_data,
                mapping_raw=mapping_raw,
                context_warnings=context_warnings,
            )

        drug_class_mapping = self._validate_internal_mapping(
            internal_mapping=internal_mapping,
            candidates=candidates,
        )

        return {
            "term": term,
            "nct_ids": normalized_nct_ids,
            "definition": definition,
            "identity_resolution": identity_matches,
            "retrieval_queries": retrieval_query_data,
            "candidates": candidates,
            "drug_class_mapping": drug_class_mapping,
            "context_warnings": context_warnings,
        }
