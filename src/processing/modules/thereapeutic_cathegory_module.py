import asyncio
import unicodedata
from typing import Any, ClassVar

from pydantic import ValidationError

from ...generation.term_extract_grounded_answer import NCItCandidateListExtractor
from ...generation.therapeutic_grounded_answer import (
    DefinitionGenerator,
    OntologyMapper,
    TherapeuticCategorySuggester,
)
from ...ingestion.ncit_evs_loader import EVSLoader
from ...ingestion.nct_loader import NCTLoader
from ...ingestion.ols_loader import OLSLoader
from ...llm.llm_client import LLMClient
from ...retrieval.retriever import TherapeuticCategoryRetriever
from ...schemas.models import (
    InternalOntologyMapping,
    NCItIdentityResolution,
    NCTDrugContext,
    TherapeuticCategorySuggestion,
)


class TherapeuticCategory:
    """
        Map a drug or therapy term to a therapeutic ontology category.
    """

    _VALID_MAPPING_COMBINATIONS: ClassVar[dict[str, set[str]]] = {
        "directly_compatible": {
            "exact",
            "equivalent_label",
        },
        "broader_compatible": {
            "broader_available",
        },
        "related_compatible": {
            "nearest_available",
            "cross_axis_therapeutic_match",
        },
        "none": {
            "none",
        },
    }

    def __init__(
        self,
        nct_loader: NCTLoader,
        definition_llm: LLMClient,
        superclass_llm: LLMClient,
        mapping_llm: LLMClient,
        retriever: TherapeuticCategoryRetriever,
        term_extractor_llm: LLMClient | None = None,
        evs_loader: EVSLoader | None = None,
        ols_loader: OLSLoader | None = None
    ) -> None:
        self._loader = nct_loader
        self._definition_generator = DefinitionGenerator(definition_llm)
        self._category_suggester = TherapeuticCategorySuggester(superclass_llm)
        self._ontology_mapper = OntologyMapper(mapping_llm)
        self._retriever = retriever
        self._candidate_extractor = (
            NCItCandidateListExtractor(term_extractor_llm)
            if term_extractor_llm is not None
            else None
        )
        self._evs_loader = evs_loader or EVSLoader()
        self._ols_loader = ols_loader or OLSLoader()

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
        ).strip()

    @classmethod
    def _validate_internal_mapping(
        cls,
        internal_mapping: InternalOntologyMapping,
        candidates: list[Any],
    ) -> dict[str, str]:
        allowed_match_types = cls._VALID_MAPPING_COMBINATIONS.get(
            internal_mapping.compatibility,
            set(),
        )
        candidate_names = {
            name.casefold()
            for candidate in candidates
            if (name := cls._get_candidate_name(candidate))
        }
        selected_is_none = internal_mapping.selected_term == "NONE_OF_THE_ABOVE"
        selected_term_is_invalid = (
            not selected_is_none
            and internal_mapping.selected_term.casefold() not in candidate_names
        )
        none_state_is_consistent = (
            selected_is_none
            and internal_mapping.compatibility == "none"
            and internal_mapping.match_type == "none"
        )
        real_selection_is_consistent = (
            not selected_is_none
            and internal_mapping.compatibility
            in {"directly_compatible", "broader_compatible", "related_compatible"}
            and internal_mapping.match_type in allowed_match_types
            and not selected_term_is_invalid
        )
        if none_state_is_consistent:
            return {
                "selected_term": "NONE_OF_THE_ABOVE",
                "compatibility": "none",
                "match_type": "none",
                "reason": internal_mapping.reason,
            }
        if not real_selection_is_consistent:
            return {
                "selected_term": "NONE_OF_THE_ABOVE",
                "compatibility": "none",
                "match_type": "none",
                "reason": (
                    "None of the supplied ontology candidates has a "
                    "compatible therapeutic scope."
                ),
            }
        return {
            "selected_term": internal_mapping.selected_term,
            "compatibility": internal_mapping.compatibility,
            "match_type": internal_mapping.match_type,
            "reason": internal_mapping.reason,
        }

    @staticmethod
    def _expand_variants(term: str) -> list[str]:
        import re

        variants: list[str] = []
        seen: set[str] = set()
        seen.add(term.casefold())

        _CELL_TYPE_SUFFIXES = re.compile(
            r"\b(CAR[-\s]T|CAR\s+T[-\s]cell|CAR\s+T[-\s]lymphocyte|"
            r"T[-\s]cell|T[-\s]lymphocyte|NK\s+cell|stem\s+cell|"
            r"dendritic\s+cell|macrophage)s?\b",
            re.IGNORECASE,
        )

        def add(candidate: str) -> None:
            c = candidate.strip()
            key = c.casefold()
            if not c or key in seen:
                return

            if len(c) <= 3:
                return

            if re.match(r'^\d', c):
                return
            seen.add(key)
            variants.append(c)

        def split_list(phrase: str) -> list[str]:
            """
                Split on ', ' and ' and ' boundaries, trim each segment.
            """
            normalised = re.sub(r"\band\b", ",", phrase)
            return [
                seg.strip().strip(",")
                for seg in normalised.split(",")
                if seg.strip().strip(",")
            ]

        if "/" in term and _CELL_TYPE_SUFFIXES.search(term):
            slash_to_space = term.replace("/", " ")
            add(slash_to_space)

            add(" ".join(slash_to_space.split()))
            return variants

        paren_matches = list(re.finditer(r"\(([^()]+)\)", term))

        if paren_matches:
            fully_stripped = re.sub(r"\s*\([^()]+\)", "", term).strip()
            add(fully_stripped)

            for m in paren_matches:
                inside = m.group(1).strip()
                add(inside)

                if "," in inside or re.search(r"\band\b", inside):
                    for segment in split_list(inside):
                        add(segment)

            return variants

        if "," in term or re.search(r"\band\b", term):
            segments = split_list(term)
            if len(segments) > 1:
                for segment in segments:
                    add(segment)

        return variants

    _SOURCE_LOOKUP_TIMEOUT: ClassVar[float] = 30.0

    def _ontology_sources(
        self,
    ) -> dict[str, tuple[EVSLoader | OLSLoader, str, dict[str, Any]]]:
        """
            Describe the sources the unified lookup races.

            Each entry pairs a loader with its display label and the extra
            keyword arguments its search_results accepts. Adding a source means
            adding one entry here; no other lookup code has to change.
        """
        return {
            "ncit": (self._evs_loader, "NCIt", {"match_type": EVSLoader.MATCH}),
            "ols": (self._ols_loader, "OLS", {}),
        }

    @staticmethod
    def _definition_text(record: dict[str, Any]) -> str:
        """
            Best available definition text in a loader record.

            OLS normalises its payload to a plain definition string, while EVS
            returns a definitions list whose entries carry a source. NCI's own
            text is preferred when present.
        """
        definition = str(record.get("definition") or "").strip()
        if definition:
            return definition

        definitions = record.get("definitions") or []
        if not isinstance(definitions, list):
            return ""

        fallback = ""
        for entry in definitions:
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("definition") or "").strip()
            if not value:
                continue
            if str(entry.get("source") or "").strip().upper() == "NCI":
                return value
            fallback = fallback or value
        return fallback

    @staticmethod
    def _record_names(value: Any) -> list[str]:
        """Flatten a synonym or parent list of strings or dicts into names."""

        names: list[str] = []
        for item in value or []:
            text = (
                str(item.get("name") or "").strip()
                if isinstance(item, dict)
                else str(item).strip()
            )
            if text:
                names.append(text)
        return names

    @classmethod
    def _pseudo_definition(
        cls,
        label: str,
        code: str,
        preferred_name: str,
        record: dict[str, Any],
    ) -> str:
        """
            Synthesize a definition from synonyms and parents when a source
            carries no definition text of its own.
        """
        synonyms = [
            name
            for name in cls._record_names(record.get("synonyms"))
            if name.casefold() != preferred_name.casefold()
        ][:6]
        parents = cls._record_names(record.get("parents"))[:4]

        parts = [f"{preferred_name} ({label}: {code})."]
        if synonyms:
            parts.append(f"Also known as: {', '.join(synonyms)}.")
        if parents:
            parts.append(f"Subtype of: {', '.join(parents)}.")
        return " ".join(parts) if len(parts) > 1 else ""

    async def _search_source(
        self,
        source: str,
        loader: EVSLoader | OLSLoader,
        search_str: str,
        *,
        label: str,
        **search_kwargs: Any,
    ) -> tuple[str, str, str, str] | None:

        try:
            hits: list[dict[str, Any]] = await loader.search_results(
                search_str,
                limit=5,
                **search_kwargs,
            )
        except Exception as exc:
            print(
                f"  [{source}] search failed for {search_str!r}: {exc}",
                flush=True,
            )
            return None

        if not hits:
            print(f"  [{source}] no hits for {search_str!r}.", flush=True)
            return None

        print(f"  [{source}] {len(hits)} hit(s) for {search_str!r}.", flush=True)

        without_definition: list[tuple[str, str, dict[str, Any]]] = []
        for hit in hits:
            if hit.get("is_obsolete"):
                continue
            code = str(hit.get("code") or hit.get("Code") or "").strip()
            preferred_name = str(hit.get("name") or "").strip()
            if not code or not preferred_name:
                continue

            record = hit
            definition = self._definition_text(hit)
            if not definition:
                try:
                    record = await loader.get_concept(code) or {}
                except Exception as exc:
                    print(
                        f"  [{source}] get_concept({code}) failed: {exc}",
                        flush=True,
                    )
                    continue
                definition = self._definition_text(record)

            if definition:
                print(
                    f"  [{source}] definition on {code} "
                    f"({preferred_name}): {definition[:80]!r}",
                    flush=True,
                )
                return definition, code, preferred_name, source

            without_definition.append((code, preferred_name, record))

        for code, preferred_name, record in without_definition:
            pseudo = self._pseudo_definition(label, code, preferred_name, record)
            if not pseudo:
                continue
            print(
                f"  [{source}] no definition on {code} ({preferred_name}); "
                f"synthesized pseudo-definition.",
                flush=True,
            )
            return pseudo, code, preferred_name, source

        print(
            f"  [{source}] hit(s) for {search_str!r} but none carried a definition.",
            flush=True,
        )
        return None

    async def _lookup_term(
        self,
        candidate_term: str,
    ) -> tuple[str, str, str, str] | None:

        term_clean = " ".join(candidate_term.split()).strip()
        if not term_clean:
            return None

        variants = [term_clean, *self._expand_variants(term_clean)]

        for search_str in variants:
            if search_str != term_clean:
                print(
                    f"  [lookup] retrying with variant: {search_str!r} "
                    f"(from {term_clean!r})",
                    flush=True,
                )
            else:
                print(
                    f"  [lookup] searching sources for: {search_str!r}",
                    flush=True,
                )

            pending: dict[asyncio.Task, str] = {
                asyncio.create_task(
                    self._search_source(
                        source,
                        loader,
                        search_str,
                        label=label,
                        **search_kwargs,
                    )
                ): source
                for source, (loader, label, search_kwargs) in (
                    self._ontology_sources().items()
                )
            }

            try:
                async with asyncio.timeout(self._SOURCE_LOOKUP_TIMEOUT):
                    while pending:
                        done, _ = await asyncio.wait(
                            pending,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            source = pending.pop(task)
                            try:
                                result = task.result()
                            except Exception as exc:
                                print(
                                    f"  [lookup] {source} failed for "
                                    f"{search_str!r}: {exc}",
                                    flush=True,
                                )
                                continue

                            if result is None:
                                print(
                                    f"  [lookup] {source} found nothing for "
                                    f"{search_str!r}.",
                                    flush=True,
                                )
                                continue

                            definition, code, preferred_name, winner = result
                            print(
                                f"  [lookup] {winner} answered first for "
                                f"{search_str!r} with {code} ({preferred_name}).",
                                flush=True,
                            )
                            return definition, code, preferred_name, winner
            except TimeoutError:
                print(
                    f"  [lookup] no source answered within "
                    f"{self._SOURCE_LOOKUP_TIMEOUT:.0f}s for {search_str!r}.",
                    flush=True,
                )
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        print(
            f"  [lookup] no definition found for {term_clean!r} "
            f"(tried {len(variants)} variant(s)).",
            flush=True,
        )
        return None


    async def _fetch_cohort_component_definitions_async(
        self,
        components: list[str],
    ) -> list[dict[str, str]]:
        """
            Async twin of _fetch_cohort_component_definitions.

            Components are independent, so they are resolved concurrently
            instead of one after another.
        """
        unique: list[str] = []
        seen: set[str] = set()
        for component in components:
            component_clean = " ".join(str(component).split()).strip()
            key = component_clean.casefold()
            if not component_clean or key in seen:
                continue
            seen.add(key)
            unique.append(component_clean)

        results = await asyncio.gather(
            *(self._lookup_term(component) for component in unique),
            return_exceptions=True,
        )

        resolved: list[dict[str, str]] = []
        for component_clean, result in zip(unique, results):
            if isinstance(result, BaseException) or result is None:
                print(
                    f"  [lookup] no component definition for: {component_clean!r}",
                    flush=True,
                )
                continue

            definition, code, preferred_name, _source = result
            resolved.append(
                {
                    "search_term": component_clean,
                    "code": code,
                    "preferred_name": preferred_name,
                    "definition": definition,
                }
            )

        return resolved

    async def _generate_multi_component_definition_async(
        self,
        *,
        term: str,
        nct_ids: list[str],
        resolution_kind: str,
        component_concepts: list[dict[str, str]],
        trial_context: str,
        unresolved_components: list[str] | None = None,
    ) -> str | None:
        """Async twin of _generate_multi_component_definition."""
        return await self._definition_generator.generate_async(
            term=term,
            nct_ids=nct_ids,
            context=self._multi_component_context(
                term=term,
                resolution_kind=resolution_kind,
                component_concepts=component_concepts,
                trial_context=trial_context,
                unresolved_components=unresolved_components,
            ),
        )

    @staticmethod
    def _multi_component_context(
        *,
        term: str,
        resolution_kind: str,
        component_concepts: list[dict[str, str]],
        trial_context: str,
        unresolved_components: list[str] | None = None,
    ) -> str:
        """Render the prompt context for a regimen/composite definition."""
        component_evidence = "\n\n".join(
            (
                f"COMPONENT SEARCH TERM: {concept['search_term']}\n"
                f"NCIT PREFERRED NAME: {concept['preferred_name']}\n"
                f"NCIT CODE: {concept['code']}\n"
                f"NCIT DEFINITION: {concept['definition']}"
            )
            for concept in component_concepts
        )
        unresolved_note = ""
        if unresolved_components:
            unresolved_note = (
                "\n\nUNRESOLVED COMPONENTS (no NCIt definition available; "
                "include by name only):\n"
                + "\n".join(f"- {c}" for c in unresolved_components)
            )
        return (
            "REQUESTED MULTI-COMPONENT INTERVENTION:\n"
            f"{term}\n\n"
            "RESOLUTION KIND:\n"
            f"{resolution_kind}\n\n"
            "AUTHORITATIVE NCIT COMPONENT DEFINITIONS:\n"
            f"{component_evidence}"
            f"{unresolved_note}\n\n"
            "TRIAL-SPECIFIC CONTEXT:\n"
            f"{trial_context}\n\n"
            "Write one concise ontology-style definition of the complete "
            "intervention. If the resolution kind is 'regimen', describe it "
            "as a combination regimen of separately identified therapeutic "
            "components. If the resolution kind is 'composite_product', "
            "describe it as one product containing multiple constituent "
            "components. Name every component (resolved and unresolved). "
            "Use only the supplied NCIt definitions and trial context. "
            "Do not describe the complete intervention as a single active "
            "substance. Do not treat components as aliases. Do not claim "
            "established efficacy, safety, superiority, or clinical benefit. "
            "Return only the definition."
        )

    @staticmethod
    def _identity_key(value: str) -> str:
        """
            Mechanical comparison key; the LLM decides biomedical equivalence.
        """
        return " ".join(
            unicodedata.normalize("NFKC", str(value)).casefold().split()
        ).strip()

    @staticmethod
    def _build_raw_intervention_context(
        trials: list[NCTDrugContext],
    ) -> str:
        sections: list[str] = []
        for trial in trials:
            lines: list[str] = []
            if trial.nct_id:
                lines.append(f"NCT ID: {trial.nct_id}")
            if trial.official_title:
                t = str(trial.official_title)
                if len(t) > 200:
                    t = t[:200] + "…"
                lines.append(f"Official title: {t}")
            if trial.brief_title:
                t = str(trial.brief_title)
                if len(t) > 200:
                    t = t[:200] + "…"
                lines.append(f"Brief title: {t}")
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
                    if len(description) > 400:
                        description = description[:400] + "…"
                    lines.append(f"Intervention {index} description: {description}")
            sections.append("\n".join(f"- {line}" for line in lines))
        return "\n\n---\n\n".join(sections)

    @classmethod
    def _extract_identity_matches(cls, extracted: Any) -> list[dict[str, str]]:
        """
            Normalize structured extractor output into match dictionaries.

            Plain strings are rejected: the extractor must return structured dicts
            so that matched_trial_name can be verified against the NCT records.
        """

        if isinstance(extracted, NCItIdentityResolution):
            return [m.model_dump() for m in extracted.matches]

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
                    f"  [identity] rejected unstructured candidate: {item!r}",
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
            role = " ".join(
                str(get("role") or "").split()
            ).strip()
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
        """
            Accept only matches whose claimed trial name exists exactly.
        """
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
                    "  [identity] rejected unknown exact trial name: "
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

    @staticmethod
    def _build_resolved_context(
        records: list[tuple[NCTDrugContext, Any, dict[str, str]]],
    ) -> str:
        """
            Collecting data from the NCT trial for the LLM to
            generate the definition.
        """
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
                "Matched ClinicalTrials.gov intervention: "
                f"{intervention.name}"
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
                evidence.append("Recorded alternative names: " + ", ".join(aliases))
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
                supporting.append("Disease MeSH terms: " + ", ".join(trial.mesh_terms))

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

    @classmethod
    def _filter_resolved_records(
        cls,
        *,
        term: str,
        resolution_kind: str,
        resolved_records: list[tuple[NCTDrugContext, Any, dict[str, str]]],
    ) -> list[tuple[NCTDrugContext, Any, dict[str, str]]]:
        """
            Keep only records whose role is appropriate for the resolution_kind.

            Falls back to a key-based identity heuristic when the LLM omitted
            roles, so records are never silently dropped on a prompt error.
        """
        allowed_roles: dict[str, set[str]] = {
            "single_product": {"requested_identity"},
            "class_request": {"requested_identity"},
            "regimen": {"requested_regimen_component"},
            "composite_product": {"requested_composite_component"},
        }
        required = allowed_roles.get(resolution_kind, set())

        role_filtered = [
            record
            for record in resolved_records
            if record[2].get("role", "") in required
        ]
        if role_filtered:
            return role_filtered

        if resolution_kind == "single_product":
            author_key = cls._identity_key(term)
            non_component_relationships = {
                "exact_name",
                "normalized_name",
                "alias",
                "generic_name",
                "brand_name",
                "development_code",
                "active_substance",
                "explicit_alias",
            }
            fallback = [
                record
                for record in resolved_records
                if (
                    cls._identity_key(record[2]["search_term"]) == author_key
                    or record[2].get("relationship", "") in non_component_relationships
                )
            ]
            if fallback:
                print(
                    "  [identity] role field absent; fell back to key/relationship "
                    f"heuristic for single_product ({len(fallback)} record(s) kept).",
                    flush=True,
                )
                return fallback

        print(
            "  [identity] role field absent and heuristic produced no records; "
            "returning all resolved records unfiltered.",
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

    async def _run_staged_async(
        self,
        term: str,
        nct_ids: list[str],
        on_stage,
    ) -> None:
        """
        Execute the pipeline stage-by-stage, calling on_stage(partial_dict) after
        each stage so callers can stream partial results to clients.
        Raises on unrecoverable errors so the caller can set _error.
        """
        from typing import Callable
        emit: Callable[[dict], None] = on_stage

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

        fetched_trials = []
        context_warnings = []
        raw_records = await asyncio.gather(
            *(
                asyncio.to_thread(self._loader.fetch_study, nct_id)
                for nct_id in normalized_nct_ids
            ),
            return_exceptions=True,
        )
        for nct_id, raw in zip(normalized_nct_ids, raw_records):
            if isinstance(raw, BaseException):
                context_warnings.append(
                    f"{nct_id}: failed to load or parse record: {raw}"
                )
                continue
            try:
                trial = NCTDrugContext.from_api(raw)
            except Exception as exc:
                context_warnings.append(
                    f"{nct_id}: failed to load or parse record: {exc}"
                )
                continue
            fetched_trials.append(trial)
            print(
                f"  [pipeline] {nct_id}: "
                f"{len(trial.interventions)} intervention(s) found.",
                flush=True,
            )

        if not fetched_trials:
            raise ValueError("No NCT records could be loaded.")

        raw_identity_context = self._build_raw_intervention_context(fetched_trials)
        extracted = await self._candidate_extractor.extract_async(
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
            raise ValueError("The requested intervention could not be isolated from other trial interventions after role filtering.")

        resolved_search_terms = self._normalize_labels(
            [match["search_term"] for _, _, match in resolved_records]
        )

        emit({
            "term": term,
            "normalized_term": self._identity_key(term),
            "nct_ids": normalized_nct_ids,
            "identity_resolution": identity_matches,
            "resolved_search_terms": resolved_search_terms,
            "context_warnings": context_warnings,
            "_stage": 1,
        })

        # ── Stage 2: get definition (NCIt or LLM) ─────────────────────────────
        ncit_search_term = None
        ncit_code = None
        ncit_preferred_name = None
        ncit_component_concepts = []
        regimen_components = []
        llm_definition = None
        definition_source = "llm"
        definition_context = None

        if resolution_kind == "class_request":
            class_search_terms = self._normalize_labels(
                [match["search_term"] for _, _, match in resolved_records]
            )
            definition = None
            for cst in class_search_terms:
                result = await self._lookup_term(cst)
                if result is not None:
                    definition, ncit_code, ncit_preferred_name, _source = result
                    ncit_search_term = cst
                    break
            if not definition:
                raise ValueError("class_request: could not find NCIt definition for any exemplar entity in the trial record.")
            definition_source = "ncit_class_exemplar"
            llm_definition = definition
            resolved_search_terms = class_search_terms
        else:
            definition_context = self._build_resolved_context(resolved_records)
            definition = await self._definition_generator.generate_async(
                term=term,
                nct_ids=normalized_nct_ids,
                context=definition_context,
            )
            if not definition or self._is_unknown_definition(definition):
                raise ValueError("Could not generate a definition.")
            llm_definition = definition

            if (
                resolution_kind in {"regimen", "composite_product"}
                and len(resolved_search_terms) > 1
            ):
                author_key = self._identity_key(term)
                if all(self._identity_key(s) == author_key for s in resolved_search_terms):
                    resolution_kind = "single_product"

            if (
                resolution_kind in {"regimen", "composite_product"}
                and len(resolved_search_terms) > 1
            ):
                regimen_components = resolved_search_terms
                ncit_component_concepts = await self._fetch_cohort_component_definitions_async(regimen_components)
                min_resolved = 2 if resolution_kind == "composite_product" else len(resolved_search_terms)
                if len(ncit_component_concepts) >= min_resolved:
                    synthesized = await self._generate_multi_component_definition_async(
                        term=term,
                        nct_ids=normalized_nct_ids,
                        resolution_kind=resolution_kind,
                        component_concepts=ncit_component_concepts,
                        trial_context=definition_context,
                        unresolved_components=None,
                    )
                    if synthesized and not self._is_unknown_definition(synthesized):
                        definition = synthesized
                        definition_source = "ncit_components_plus_llm"
            else:
                expanded_resolved = []
                for rst in resolved_search_terms:
                    expanded_resolved.append(rst)
                    for variant in self._expand_variants(rst):
                        expanded_resolved.append(variant)
                term_variants = self._expand_variants(term)
                term_safety_net = [term, *term_variants] if not term_variants else term_variants
                search_candidates = self._normalize_labels([*expanded_resolved, *term_safety_net])
                for search_candidate in search_candidates:
                    result = await self._lookup_term(search_candidate)
                    if result is None:
                        continue
                    definition, ncit_code, ncit_preferred_name, _source = result
                    ncit_search_term = search_candidate
                    definition_source = "ncit"
                    break

        emit({
            "definition": definition,
            "definition_source": definition_source,
            "ncit_search_term": ncit_search_term,
            "ncit_code": ncit_code,
            "ncit_preferred_name": ncit_preferred_name,
            "ncit_component_concepts": ncit_component_concepts,
            "regimen_components": regimen_components,
            "llm_definition": llm_definition,
            "_stage": 2,
        })

        # ── Stage 3: suggest therapeutic categories ───────────────────────────
        category_raw = await self._category_suggester.suggest_async(definition=definition)
        if not category_raw:
            raise ValueError("Therapeutic-category suggestion failed.")
        try:
            category_suggestion = TherapeuticCategorySuggestion.model_validate_json(category_raw)
        except ValidationError as exc:
            raise ValueError(f"Therapeutic-category suggestion validation failed: {exc}")

        category_labels = category_suggestion.model_dump()
        emit({
            "category_labels": category_labels,
            "superclass_labels": category_labels,
            "_stage": 3,
        })

        # ── Stage 4: retrieve Qdrant candidates & map ontology term ───────────
        import json as _json
        category_context = _json.dumps(
            {**category_labels, "retrieval_scope": "The candidate hierarchy contains functional therapeutic and pharmacological drug categories."},
            ensure_ascii=False, indent=2,
        )
        labels = self._normalize_labels([
            category_labels["specific_category"],
            category_labels["broad_category"],
            category_labels["root_category"],
        ])
        if not labels:
            fallback = [term]
            if ncit_preferred_name and ncit_preferred_name.casefold() != term.casefold():
                fallback.append(ncit_preferred_name)
            labels = self._normalize_labels(fallback)

        candidates = await self._retriever.retrieve_for_labels(
            labels=labels,
            per_label_k=20,
            final_k=20,
        )
        if not candidates:
            raise ValueError("Qdrant retrieval returned no candidates.")

        mapping_raw = await self._ontology_mapper.map_async(
            definition=definition,
            superclass=category_context,
            candidates=candidates,
        )
        if not mapping_raw:
            raise ValueError("Ontology mapping failed.")
        try:
            internal_mapping = InternalOntologyMapping.model_validate_json(mapping_raw)
        except ValidationError as exc:
            raise ValueError(f"Ontology mapping validation failed: {exc}")

        ontology_mapping = self._validate_internal_mapping(
            internal_mapping=internal_mapping,
            candidates=candidates,
        )
        emit({
            "candidates": candidates,
            "ontology_mapping": ontology_mapping,
            "_stage": 4,
        })
