import json
import unicodedata
from typing import Any

from pydantic import ValidationError

from ...generation.therapeutic_grounded_answer import (
    DefinitionGenerator,
    OntologyMapper,
    TherapeuticCategorySuggester,
)
from ...generation.term_extract_grounded_answer import NCItCandidateListExtractor
from ...ingestion.evs_loader import EVSLoader
from ...ingestion.nct_loader import NCTLoader
from ...llm.llm_client import LLMClient
from ...retrieval.retriever import TherapeuticCategoryRetriever
from ...schemas.models import (
    InternalOntologyMapping,
    NCItIdentityResolution,
    NCTDrugContext,
    TherapeuticCategorySuggestion,
)


class TherapeuticCategory:
    """Map a drug or therapy term to a therapeutic ontology category."""

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

    @staticmethod
    def _err(term: str, message: str, **extra: Any) -> dict[str, Any]:
        return {"term": term, "error": message, **extra}

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
    # Variations of the prompt so we
    # can identify the drug
    def _expand_ncit_variants(term: str) -> list[str]:
        """
        When the primary NCIt search for a term returns no results, this
        function generates simpler reformulations of the same term to try
        next — in order, until one hits.

        The reformulation rules are:

        1. Remove parentheses
           Strip anything in parentheses and search the bare name.
           Also try each parenthetical clause on its own, in case it
           is itself a known NCIt code or synonym.
           "Mosunetuzumab (SC) (RO7030816)"
             -> try "Mosunetuzumab", then "SC", then "RO7030816"
           "Avutometinib (VS-6766)"
             -> try "Avutometinib", then "VS-6766"

        2. Split comma / "and" lists
           When the term is a list of names joined by commas or "and",
           try each name individually.
           "LMP, BARF1 and EBNA1 specific CTLs"
             -> try "LMP", then "BARF1", then "EBNA1 specific CTLs"

        3. Combined
           If a parenthetical clause is itself a list, it is also split.

        4. CAR-T constructs — replace slashes with spaces, never split
           When the term ends with a cell-type suffix (CAR T, T-cell,
           T-lymphocyte, …), any slashes are part of the construct name
           and must not be treated as separators. Instead, a space-separated
           variant is tried because NCIt sometimes indexes these with spaces.
           "EGFRt/19-28z/IL-12 CAR T-lymphocyte"
             -> try "EGFRt 19-28z IL-12 CAR T-lymphocyte"
        """
        import re

        variants: list[str] = []
        seen: set[str] = set()
        seen.add(term.casefold())

        # Cell-type suffixes that signal a CAR-T / engineered-cell product.
        # If the term ends with one of these words, slashes are construct
        # separators, not drug-boundary separators.
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
            # Skip tokens that are too short to be useful NCIt search terms.
            if len(c) <= 3:
                return
            # Skip tokens that start with a digit — these are chemistry formula
            # fragments (e.g. "1H", "3H" from "2,4(1H, 3H)-pyrimidinedione")
            # rather than valid NCIt concept names.
            if re.match(r'^\d', c):
                return
            seen.add(key)
            variants.append(c)

        def split_list(phrase: str) -> list[str]:
            """Split on ', ' and ' and ' boundaries, trim each segment."""
            normalised = re.sub(r"\band\b", ",", phrase)
            return [
                seg.strip().strip(",")
                for seg in normalised.split(",")
                if seg.strip().strip(",")
            ]

        # Strategy 4 first: detect CAR-T construct terms with slashes.
        # These must NOT be split; instead add a slash→space variant.
        if "/" in term and _CELL_TYPE_SUFFIXES.search(term):
            slash_to_space = term.replace("/", " ")
            add(slash_to_space)
            # Also try with the space-normalised form stripped of extra spaces.
            add(" ".join(slash_to_space.split()))
            return variants

        # Find all parenthetical groups anywhere in the term.
        paren_matches = list(re.finditer(r"\(([^()]+)\)", term))

        if paren_matches:
            # Strategy 1a: fully-stripped base (all parentheticals removed).
            fully_stripped = re.sub(r"\s*\([^()]+\)", "", term).strip()
            add(fully_stripped)

            # Strategy 1b: each inside clause as its own candidate.
            for m in paren_matches:
                inside = m.group(1).strip()
                add(inside)
                # Strategy 3: split list-valued inside clauses.
                if "," in inside or re.search(r"\band\b", inside):
                    for segment in split_list(inside):
                        add(segment)

            return variants

        # Strategy 2: list splitting (no parenthetical present).
        if "," in term or re.search(r"\band\b", term):
            segments = split_list(term)
            if len(segments) > 1:
                for segment in segments:
                    add(segment)

        return variants

    def _try_ncit_term(
        self,
        candidate_term: str,
    ) -> tuple[str, str, str] | None:
        """
        Return (definition, code, preferred name) for one NCIt term.

        When the primary search returns no hits and the term contains a
        parenthetical annotation (e.g. "Avutometinib (VS-6766)"), automatically
        retries with the base name and the code inside the parenthesis before
        giving up.
        """
        term_clean = " ".join(candidate_term.split()).strip()
        if not term_clean:
            return None

        # Build the ordered list of strings to try: primary term first,
        # then any parenthetical variants.
        candidates_to_try = [term_clean, *self._expand_ncit_variants(term_clean)]

        for search_str in candidates_to_try:
            result = self._try_ncit_one(search_str, original_term=term_clean)
            if result is not None:
                return result

        print(
            f"  [ncit] no definition found for {term_clean!r} "
            f"(tried {len(candidates_to_try)} variant(s)).",
            flush=True,
        )
        return None

    def _try_ncit_one(
        self,
        search_str: str,
        *,
        original_term: str,
    ) -> tuple[str, str, str] | None:
        """
            Single EVS lookup for one search string. Returns None on miss.
        """
        if search_str != original_term:
            print(
                f"  [ncit] retrying with variant: {search_str!r}"
                f" (from {original_term!r})",
                flush=True,
            )
        else:
            print(f"  [ncit] searching NCIt for: {search_str!r}", flush=True)

        try:
            hits: list[dict] = self._evs_loader.search(
                term=search_str,
                limit=5,
                match_type=EVSLoader.MATCH,
            )
        except Exception as exc:
            print(
                f"  [ncit] EVS match search failed for {search_str!r}: {exc}",
                flush=True,
            )
            return None

        if not hits:
            print(f"  [ncit] no hits for {search_str!r}.", flush=True)
            return None

        print(f"  [ncit] {len(hits)} hit(s) for {search_str!r}.", flush=True)

        # First pass: fetch each concept and look for a real definition.
        fetched_concepts: list[tuple[str, str, dict]] = []  # (code, preferred_name, concept)
        for hit in hits:
            code = str(hit.get("code") or hit.get("Code") or "").strip()
            preferred_name = str(hit.get("name") or "").strip()
            if not code:
                continue
            try:
                concept = self._evs_loader.get_concept(code)
            except Exception as exc:
                print(f"  [ncit] get_concept({code}) failed: {exc}", flush=True)
                continue

            fetched_concepts.append((code, preferred_name, concept))

            definitions = concept.get("definitions") or []
            if not isinstance(definitions, list):
                continue

            for defn in definitions:
                if not isinstance(defn, dict):
                    continue
                source = str(defn.get("source") or "").strip().upper()
                value = str(defn.get("definition") or "").strip()
                if value and source == "NCI":
                    print(
                        f"  [ncit] NCI definition on {code} "
                        f"({preferred_name}): {value[:80]!r}",
                        flush=True,
                    )
                    return value, code, preferred_name

            for defn in definitions:
                if not isinstance(defn, dict):
                    continue
                value = str(defn.get("definition") or "").strip()
                if value:
                    print(
                        f"  [ncit] non-NCI definition on {code} "
                        f"({preferred_name}): {value[:80]!r}",
                        flush=True,
                    )
                    return value, code, preferred_name

        # Second pass: no real definition found; try to synthesize a
        # pseudo-definition from preferred name, synonyms, and parent classes.
        for code, preferred_name, concept in fetched_concepts:
            parts: list[str] = [f"{preferred_name} (NCIt: {code})."]

            synonyms = concept.get("synonyms") or []
            syn_names = list(dict.fromkeys(
                str(s.get("name") or "").strip()
                for s in synonyms
                if isinstance(s, dict)
                and str(s.get("name") or "").strip()
                and str(s.get("name") or "").strip().casefold() != preferred_name.casefold()
            ))[:6]
            if syn_names:
                parts.append(f"Also known as: {', '.join(syn_names)}.")

            parents = concept.get("parents") or []
            parent_names = [
                str(p.get("name") or "").strip()
                for p in parents
                if isinstance(p, dict) and str(p.get("name") or "").strip()
            ][:4]
            if parent_names:
                parts.append(f"Subtype of: {', '.join(parent_names)}.")

            if len(parts) > 1:
                pseudo_def = " ".join(parts)
                print(
                    f"  [ncit] no definition on {code} ({preferred_name}); "
                    f"synthesized pseudo-definition from synonyms/parents.",
                    flush=True,
                )
                return pseudo_def, code, preferred_name

        print(
            f"  [ncit] hit(s) for {search_str!r} but none carried a definition.",
            flush=True,
        )
        return None

    def _fetch_ncit_component_definitions(
        self,
        components: list[str],
    ) -> list[dict[str, str]]:
        """Resolve every regimen component independently in NCIt."""
        resolved: list[dict[str, str]] = []
        seen: set[str] = set()

        for component in components:
            component_clean = " ".join(str(component).split()).strip()
            key = component_clean.casefold()
            if not component_clean or key in seen:
                continue
            seen.add(key)

            result = self._try_ncit_term(component_clean)
            if result is None:
                print(
                    f"  [ncit] no component definition for: {component_clean!r}",
                    flush=True,
                )
                continue

            definition, code, preferred_name = result
            resolved.append(
                {
                    "search_term": component_clean,
                    "code": code,
                    "preferred_name": preferred_name,
                    "definition": definition,
                }
            )

        return resolved

    def _generate_multi_component_definition(
        self,
        *,
        term: str,
        nct_ids: list[str],
        resolution_kind: str,
        component_concepts: list[dict[str, str]],
        trial_context: str,
        unresolved_components: list[str] | None = None,
    ) -> str | None:
        """Synthesize a definition for a regimen or composite product."""
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
        context = (
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
        return self._definition_generator.generate(
            term=term,
            nct_ids=nct_ids,
            context=context,
        )

    @staticmethod
    def _identity_key(value: str) -> str:
        """Mechanical comparison key; the LLM decides biomedical equivalence."""
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
        """Accept only matches whose claimed trial name exists exactly."""
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
        """Keep only records whose role is appropriate for the resolution_kind.

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

        # Primary filter: use the role field when present.
        role_filtered = [
            record
            for record in resolved_records
            if record[2].get("role", "") in required
        ]
        if role_filtered:
            return role_filtered

        # Fallback: the LLM omitted roles.  For single_product, keep only
        # matches whose search_term shares an identity key with the author term
        # or whose relationship is not combination_component.
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

        # Last resort: return all records unchanged so the pipeline never
        # silently fails due to a prompt-instruction omission.
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

    def map_definition(
        self,
        term: str,
        definition: str,
    ) -> dict[str, Any]:
        """
            Run steps 3–5 (category suggestion → retrieval → mapping) on a pre-built definition.
        """
        term = " ".join(str(term).split()).strip()
        definition = " ".join(str(definition).split()).strip()
        if not term:
            return self._err(term, "term cannot be empty.")
        if not definition:
            return self._err(term, "definition cannot be empty.")

        category_raw = self._category_suggester.suggest(definition=definition)
        if not category_raw:
            return self._err(term, "Therapeutic-category suggestion failed.", definition=definition)
        try:
            category_suggestion = TherapeuticCategorySuggestion.model_validate_json(
                category_raw
            )
        except ValidationError as exc:
            return self._err(
                term,
                f"Therapeutic-category suggestion validation failed: {exc}",
                definition=definition,
            )

        category_labels = category_suggestion.model_dump()
        category_context = json.dumps(
            {
                **category_labels,
                "retrieval_scope": (
                    "The candidate hierarchy contains functional therapeutic "
                    "and pharmacological drug categories."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        labels = self._normalize_labels(
            [
                category_labels["specific_category"],
                category_labels["broad_category"],
                category_labels["root_category"],
            ]
        )
        if not labels:
            return self._err(
                term,
                "No usable labels were generated for ontology retrieval.",
                definition=definition,
                category_labels=category_labels,
            )

        candidates = self._retriever.retrieve_for_labels(
            labels=labels,
            per_label_k=20,
            final_k=20,
        )
        if not candidates:
            return self._err(
                term,
                "Qdrant retrieval returned no candidates.",
                definition=definition,
                category_labels=category_labels,
            )

        mapping_raw = self._ontology_mapper.map(
            definition=definition,
            superclass=category_context,
            candidates=candidates,
        )
        if not mapping_raw:
            return self._err(
                term,
                "Ontology mapping failed.",
                definition=definition,
                category_labels=category_labels,
            )
        try:
            internal_mapping = InternalOntologyMapping.model_validate_json(mapping_raw)
        except ValidationError as exc:
            return self._err(
                term,
                f"Ontology mapping validation failed: {exc}",
                definition=definition,
                category_labels=category_labels,
            )

        ontology_mapping = self._validate_internal_mapping(
            internal_mapping=internal_mapping,
            candidates=candidates,
        )
        return {
            "term": term,
            "definition": definition,
            "category_labels": category_labels,
            "candidates": candidates,
            "ontology_mapping": ontology_mapping,
        }

    def _run_staged(
        self,
        term: str,
        nct_ids: list[str],
        on_stage,   # callable(dict) — called after each stage with partial results
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

        # ── Stage 1: fetch NCT records & resolve identity ─────────────────────
        fetched_trials = []
        context_warnings = []
        for nct_id in normalized_nct_ids:
            try:
                raw = self._loader.fetch_study(nct_id)
                trial = NCTDrugContext.from_api(raw)
                fetched_trials.append(trial)
                print(f"  [pipeline] {nct_id}: {len(trial.interventions)} intervention(s) found.", flush=True)
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
                result = self._try_ncit_term(cst)
                if result is not None:
                    definition, ncit_code, ncit_preferred_name = result
                    ncit_search_term = cst
                    break
            if not definition:
                raise ValueError("class_request: could not find NCIt definition for any exemplar entity in the trial record.")
            definition_source = "ncit_class_exemplar"
            llm_definition = definition
            resolved_search_terms = class_search_terms
        else:
            definition_context = self._build_resolved_context(resolved_records)
            definition = self._definition_generator.generate(
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
                ncit_component_concepts = self._fetch_ncit_component_definitions(regimen_components)
                min_resolved = 2 if resolution_kind == "composite_product" else len(resolved_search_terms)
                if len(ncit_component_concepts) >= min_resolved:
                    synthesized = self._generate_multi_component_definition(
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
                    for variant in self._expand_ncit_variants(rst):
                        expanded_resolved.append(variant)
                term_variants = self._expand_ncit_variants(term)
                term_safety_net = [term, *term_variants] if not term_variants else term_variants
                search_candidates = self._normalize_labels([*expanded_resolved, *term_safety_net])
                for search_candidate in search_candidates:
                    result = self._try_ncit_term(search_candidate)
                    if result is None:
                        continue
                    definition, ncit_code, ncit_preferred_name = result
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
        category_raw = self._category_suggester.suggest(definition=definition)
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

        candidates = self._retriever.retrieve_for_labels(labels=labels, per_label_k=20, final_k=20)
        if not candidates:
            raise ValueError("Qdrant retrieval returned no candidates.")

        mapping_raw = self._ontology_mapper.map(
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

        fetched_trials: list[NCTDrugContext] = []
        context_warnings: list[str] = []
        for nct_id in normalized_nct_ids:
            try:
                raw = self._loader.fetch_study(nct_id)
                trial = NCTDrugContext.from_api(raw)
                fetched_trials.append(trial)
                print(
                    f"  [pipeline] {nct_id}: "
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

        raw_identity_context = self._build_raw_intervention_context(fetched_trials)
        extracted = self._candidate_extractor.extract(
            term=term,
            nct_ids=normalized_nct_ids,
            context=raw_identity_context,
        )

        identity_matches = self._extract_identity_matches(extracted)

        print(
            "  [identity] raw extractor output: "
            + (
                repr(extracted.model_dump())
                if isinstance(extracted, NCItIdentityResolution)
                else repr(extracted)
            ),
            flush=True,
        )
        print(
            "  [identity] normalized matches: "
            f"{identity_matches!r}",
            flush=True,
        )

        resolved_records = self._validate_identity_matches(
            extracted=extracted,
            trials=fetched_trials,
        )

        print(
            "  [identity] validated matches: "
            + ", ".join(
                (
                    f"{trial.nct_id}: "
                    f"{intervention.name!r} "
                    f"-> {match['search_term']!r}"
                )
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

        # ── Role-based record filtering ───────────────────────────────────────
        # Keep only records whose LLM-assigned role is appropriate for the
        # declared resolution_kind.  This must run before definition_context
        # is built so that co-intervention evidence is excluded from the
        # definition generation prompt.
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

        # ── class_request fast-path ───────────────────────────────────────────
        # The author term is a therapeutic-class description (e.g. "CD33
        # directed therapy", "anti-HER2 agent").  The LLM has identified one
        # exemplar drug from the trial record to anchor the NCIt lookup.
        # Skip definition generation entirely and go straight to NCIt so we
        # classify the class by its representative member's pharmacology.
        if resolution_kind == "class_request":
            class_search_terms = self._normalize_labels(
                [match["search_term"] for _, _, match in resolved_records]
            )
            print(
                f"  [ncit] class_request fast-path; searching NCIt for: "
                + ", ".join(repr(t) for t in class_search_terms),
                flush=True,
            )
            ncit_definition: str | None = None
            ncit_search_term: str | None = None
            ncit_code: str | None = None
            ncit_preferred_name: str | None = None
            for cst in class_search_terms:
                result = self._try_ncit_term(cst)
                if result is not None:
                    ncit_definition, ncit_code, ncit_preferred_name = result
                    ncit_search_term = cst
                    break
            if not ncit_definition:
                return self._err(
                    term,
                    "class_request: could not find NCIt definition for "
                    "any exemplar drug in the trial record.",
                    context_warnings=context_warnings,
                )
            definition = ncit_definition
            definition_source = "ncit_class_exemplar"
            llm_definition = definition
            ncit_component_concepts: list[dict[str, str]] = []
            regimen_components: list[str] = []
            resolved_search_terms = class_search_terms
        else:
            definition_context = self._build_resolved_context(resolved_records)
            definition = self._definition_generator.generate(
                term=term,
                nct_ids=normalized_nct_ids,
                context=definition_context,
            )
            if not definition or self._is_unknown_definition(definition):
                return self._err(
                    term,
                    "Could not generate a definition.",
                    context_warnings=context_warnings,
                )

            llm_definition = definition
            definition_source = "llm"
            ncit_search_term = None
            ncit_code = None
            ncit_preferred_name = None
            ncit_component_concepts = []
            regimen_components = []

            resolved_search_terms = self._normalize_labels(
                [
                    match["search_term"]
                    for _, _, match in resolved_records
                ]
            )

        # ── Deterministic single-product downgrade ────────────────────────────
        # Safety net: if the LLM still declared regimen but all surviving
        # matches share the same search_term key as the author term, it is
        # almost certainly a single product used inside a regimen trial.
        # Filter_resolved_records above handles the role-based case; this
        # handles the edge case where roles were set to combination_component
        # but filtering still let them through (last-resort path).

        if (
            resolution_kind in {"regimen", "composite_product"}
            and len(resolved_search_terms) > 1
        ):
            author_key = self._identity_key(term)
            only_author_term = all(
                self._identity_key(s) == author_key
                for s in resolved_search_terms
            )
            if only_author_term:
                print(
                    "  [identity] downgraded regimen to single_product: all "
                    "resolved search terms are aliases of the author term.",
                    flush=True,
                )
                resolution_kind = "single_product"

        # ── NCIt resolution branch ────────────────────────────────────────────
        if (
            resolution_kind in {"regimen", "composite_product"}
            and len(resolved_search_terms) > 1
        ):
            regimen_components = resolved_search_terms
            ncit_component_concepts = self._fetch_ncit_component_definitions(
                regimen_components
            )

            resolved_component_keys = {
                self._identity_key(concept["search_term"])
                for concept in ncit_component_concepts
            }
            unresolved_components = [
                component
                for component in resolved_search_terms
                if self._identity_key(component) not in resolved_component_keys
            ]
            # For composite_product, allow partial synthesis when at least 2
            # out of N components are resolved — linkers/chelators like DOTA
            # may lack NCIt definitions without affecting classification accuracy.
            min_resolved = 2 if resolution_kind == "composite_product" else len(resolved_search_terms)
            all_components_resolved = len(ncit_component_concepts) >= min_resolved

            if not all_components_resolved:
                print(
                    "  [pipeline] incomplete NCIt component resolution; "
                    "not synthesizing an NCIt-grounded multi-component "
                    "definition. Unresolved: "
                    + ", ".join(repr(c) for c in unresolved_components),
                    flush=True,
                )
            else:
                synthesized = self._generate_multi_component_definition(
                    term=term,
                    nct_ids=normalized_nct_ids,
                    resolution_kind=resolution_kind,
                    component_concepts=ncit_component_concepts,
                    trial_context=definition_context,
                    unresolved_components=unresolved_components or None,
                )
                if synthesized and not self._is_unknown_definition(synthesized):
                    definition = synthesized
                    definition_source = "ncit_components_plus_llm"
        else:
            # Build a flat, ordered list of NCIt search candidates.
            #
            # For each LLM-returned resolved search term, pre-expand it via
            # _expand_ncit_variants so that composite alias terms like
            # "LMP, BARF1 and EBNA1 specific CTLs" are split into individual
            # top-level candidates ("LMP", "BARF1", "EBNA1 specific CTLs")
            # before they reach _try_ncit_term.  Each top-level candidate is
            # then a clean single-concept query, which avoids NCIt matching a
            # multi-word phrase literally when the individual components are
            # indexed separately.
            #
            # The original (unexpanded) term is kept first so that an exact
            # NCIt match beats an expanded sub-term.
            #
            # A safety net of raw-term expansions is appended last so we
            # don't miss e.g. "Avutometinib" when the user typed
            # "Avutometinib (VS-6766)".
            expanded_resolved: list[str] = []
            for rst in resolved_search_terms:
                expanded_resolved.append(rst)
                for variant in self._expand_ncit_variants(rst):
                    expanded_resolved.append(variant)

            term_variants = self._expand_ncit_variants(term)
            term_safety_net = (
                [term, *term_variants] if not term_variants else term_variants
            )
            search_candidates = self._normalize_labels(
                [*expanded_resolved, *term_safety_net]
            )
            print(
                "  [ncit] single-product search order: "
                + ", ".join(repr(c) for c in search_candidates),
                flush=True,
            )
            for search_candidate in search_candidates:
                result = self._try_ncit_term(search_candidate)
                if result is None:
                    continue
                definition, ncit_code, ncit_preferred_name = result
                ncit_search_term = search_candidate
                definition_source = "ncit"
                break

        category_raw = self._category_suggester.suggest(definition=definition)
        if not category_raw:
            return self._err(
                term,
                "Therapeutic-category suggestion failed.",
                definition=definition,
                context_warnings=context_warnings,
            )
        try:
            category_suggestion = TherapeuticCategorySuggestion.model_validate_json(
                category_raw
            )
        except ValidationError as exc:
            return self._err(
                term,
                f"Therapeutic-category suggestion validation failed: {exc}",
                definition=definition,
                category_raw=category_raw,
                context_warnings=context_warnings,
            )

        category_labels = category_suggestion.model_dump()
        category_context = json.dumps(
            {
                **category_labels,
                "retrieval_scope": (
                    "The candidate hierarchy contains functional therapeutic "
                    "and pharmacological drug categories."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        labels = self._normalize_labels(
            [
                category_labels["specific_category"],
                category_labels["broad_category"],
                category_labels["root_category"],
            ]
        )
        if not labels:
            # The definition alone didn't support a functional category label.
            # Fall back to retrieving directly with the term and, when available,
            # the NCIt preferred name, so that the ontology mapper can still find
            # a match (e.g. a radiopharmaceutical whose NCIt definition is purely
            # physical/nuclear, or a vaccine whose NCIt entry resolves to a disease).
            fallback_label_candidates = [term]
            if ncit_preferred_name and ncit_preferred_name.casefold() != term.casefold():
                fallback_label_candidates.append(ncit_preferred_name)
            print(
                "  [pipeline] no functional category labels from definition; "
                f"falling back to term-name retrieval "
                f"({', '.join(repr(l) for l in fallback_label_candidates)}).",
                flush=True,
            )
            labels = self._normalize_labels(fallback_label_candidates)
        candidates = self._retriever.retrieve_for_labels(
            labels=labels,
            per_label_k=20,
            final_k=20,
        )
        if not candidates:
            return self._err(
                term,
                "Qdrant retrieval returned no candidates.",
                definition=definition,
                category_labels=category_labels,
                context_warnings=context_warnings,
            )
        mapping_raw = self._ontology_mapper.map(
            definition=definition,
            superclass=category_context,
            candidates=candidates,
        )
        if not mapping_raw:
            return self._err(
                term,
                "Ontology mapping failed.",
                definition=definition,
                category_labels=category_labels,
                context_warnings=context_warnings,
            )
        try:
            internal_mapping = InternalOntologyMapping.model_validate_json(
                mapping_raw
            )
        except ValidationError as exc:
            return self._err(
                term,
                f"Ontology mapping validation failed: {exc}",
                definition=definition,
                category_labels=category_labels,
                mapping_raw=mapping_raw,
                context_warnings=context_warnings,
            )
        ontology_mapping = self._validate_internal_mapping(
            internal_mapping=internal_mapping,
            candidates=candidates,
        )
        return {
            "term": term,
            "normalized_term": self._identity_key(term),
            "nct_ids": normalized_nct_ids,
            "definition": definition,
            "definition_source": definition_source,
            "ncit_search_term": ncit_search_term,
            "ncit_code": ncit_code,
            "ncit_preferred_name": ncit_preferred_name,
            "ncit_component_concepts": ncit_component_concepts,
            "regimen_components": regimen_components,
            "identity_resolution": identity_matches,
            "resolved_search_terms": resolved_search_terms,
            "llm_definition": llm_definition,
            "category_labels": category_labels,
            "superclass_labels": category_labels,
            "candidates": candidates,
            "ontology_mapping": ontology_mapping,
            "context_warnings": context_warnings,
        }
