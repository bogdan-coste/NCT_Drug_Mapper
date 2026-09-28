import json
from typing import Any

from ..llm.llm_client import LLMClient
from ..prompts.prompt_templates import PromptTemplates


class DrugClassDefinitionGenerator:
    """
    Call 1 wrapper for generating an evidence-grounded,
    composition-centered intervention definition.

    The generated definition is intended for entity-class mapping and should
    preserve only explicitly supported compositional, structural, or
    origin-based entity information.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._llm.system_prompt = PromptTemplates.NCT_DRUG_CLASS_DEFINITION_SYSTEM_PROMPT

    def generate(
        self,
        term: str,
        nct_ids: list[str],
        context: str,
    ) -> str:
        """Generate one composition-centered entity definition with a citation."""
        normalized_term = " ".join(str(term).split()).strip()
        normalized_nct_ids = [
            str(nct_id).strip().upper()
            for nct_id in nct_ids
            if str(nct_id).strip()
        ]
        normalized_context = str(context).strip()

        if not normalized_term:
            raise ValueError("term cannot be empty.")
        if not normalized_nct_ids:
            raise ValueError("At least one NCT ID must be supplied.")
        if not normalized_context:
            raise ValueError("context cannot be empty.")

        user_prompt = (
            PromptTemplates.NCT_DRUG_CLASS_DEFINITION_TEMPLATE.substitute(
                term=normalized_term,
                nct_ids=", ".join(normalized_nct_ids),
                context=normalized_context,
            )
        )

        response = self._llm.ask_llm(user_prompt)

        return str(response or "").strip()


class DrugClassRetrievalQuerySuggester:
    """
    Call 2 wrapper for generating one primary and up to two complementary
    entity-class retrieval queries.

    The returned JSON is validated later by DrugClassRetrievalSuggestion.
    These query values are retrieval aids and do not assert ontology
    parent-child relationships.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._llm.system_prompt = PromptTemplates.DRUG_CLASS_RETRIEVAL_QUERY_SYSTEM_PROMPT

    def suggest(self, definition: str) -> str:
        """Generate the raw two-field JSON entity-class retrieval-query response."""
        normalized_definition = str(definition).strip()

        if not normalized_definition:
            raise ValueError("definition cannot be empty.")

        user_prompt = (
            PromptTemplates.DRUG_CLASS_RETRIEVAL_QUERY_TEMPLATE.substitute(
                definition=normalized_definition,
            )
        )

        response = self._llm.ask_llm(user_prompt)

        return str(response or "").strip()


class DrugClassOntologyMapper:
    """
    Call 3 wrapper for selecting the best compatible supplied entity-class
    ontology candidate.

    The returned JSON is validated later by InternalDrugClassMapping and by
    the deterministic candidate and ontology-axis checks in the orchestrator.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._llm.system_prompt = PromptTemplates.DRUG_CLASS_MAPPING_SYSTEM_PROMPT

    @staticmethod
    def _serialize_candidates(candidates: list[Any]) -> str:
        """Serialize supplied candidates without silently omitting fields."""
        return json.dumps(
            candidates,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    def map(
        self,
        definition: str,
        retrieval_queries: str,
        candidates: list[Any],
    ) -> str:
        """Generate the raw four-field internal entity-class mapping JSON."""
        normalized_definition = str(definition).strip()
        normalized_retrieval_queries = str(retrieval_queries).strip()

        if not normalized_definition:
            raise ValueError("definition cannot be empty.")
        if not normalized_retrieval_queries:
            raise ValueError("retrieval_queries cannot be empty.")
        if not candidates:
            raise ValueError("At least one ontology candidate is required.")

        user_prompt = PromptTemplates.DRUG_CLASS_MAPPING_TEMPLATE.substitute(
            definition=normalized_definition,
            retrieval_queries=normalized_retrieval_queries,
            candidates=self._serialize_candidates(candidates),
        )

        response = self._llm.ask_llm(user_prompt)

        return str(response or "").strip()
