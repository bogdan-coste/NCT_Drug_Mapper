from ..llm.llm_client import LLMClient
from ..prompts.prompt_templates import PromptTemplates


class DefinitionGenerator:
    """
    Step 1: Generates an ontology-style definition for an intervention
    or entity term using evidence extracted from NCT records.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._llm.system_prompt = (
            PromptTemplates
            .NCT_DEFINITION_GENERATION_SYSTEM_PROMPT
        )

    def generate(
        self,
        term: str,
        nct_ids: list[str],
        context: str,
    ) -> str | None:
        """
        Generate an evidence-grounded definition for an intervention or entity.

        Args:
            term:
                The intervention or entity term to define.
            nct_ids:
                NCT identifiers used as evidence sources.
            context:
                Concatenated NCT context passages for the term.

        Returns:
            A definition in the format:
            'Subclass of PARENT. Definition. '
            '(ClinicalTrials.gov; NCT_IDENTIFIER)'

            Returns None when the LLM does not provide a response.
        """
        normalized_nct_ids = [
            str(nct_id).strip().upper()
            for nct_id in nct_ids
            if str(nct_id).strip()
        ]

        user_msg = (
            PromptTemplates
            .NCT_DEFINITION_GENERATION_TEMPLATE
            .substitute(
                term=term,
                nct_ids=", ".join(normalized_nct_ids),
                context=context,
            )
        )

        return self._llm.ask_llm(user_msg)


class TherapeuticCategorySuggester:
    """
    Generates therapeutic categories and a modality label from an
    evidence-grounded intervention definition.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._llm.system_prompt = PromptTemplates.THERAPEUTIC_CATEGORY_SUGGESTION_SYSTEM_PROMPT

    def suggest(self, definition: str) -> str | None:
        user_msg = (
            PromptTemplates
            .THERAPEUTIC_CATEGORY_SUGGESTION_TEMPLATE
            .substitute(definition=definition)
        )

        return self._llm.ask_llm(user_msg)

class OntologyMapper:
    """
    Step 3 — Given a definition, a suggested superclass, and 20 SapBERT
    candidate terms, selects the best matching ontology term.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._llm.system_prompt = PromptTemplates.ONTOLOGY_MAPPING_SYSTEM_PROMPT

    def map(
        self,
        definition: str,
        superclass: str,
        candidates: list[dict],
    ) -> str | None:
        """
        Select the best ontology term from SapBERT candidates.

        Args:
            definition:  The definition generated in Step 1.
            superclass:  The suggested superclass label from Step 2.
            candidates:  List of dicts with 'name' and 'documentation' keys.

        Returns:
            A JSON string with keys: selected_term, match_type, reason.
            or None if the LLM fails to respond.
        """

        lines = []
        for i, c in enumerate(candidates):
            line = f"{i + 1}. {c['name']}"
            if c.get("documentation"):
                line += f" — {c['documentation']}"
            if c.get("parents"):
                parents_str = "; ".join(
                    f"{p['name']} ({p['documentation']})" if p.get("documentation") else p["name"]
                    for p in c["parents"]
                )
                line += f"\n   Parents: {parents_str}"
            if c.get("children"):
                children_str = "; ".join(
                    f"{ch['name']} ({ch['documentation']})" if ch.get("documentation") else ch["name"]
                    for ch in c["children"]
                )
                line += f"\n   Children: {children_str}"
            lines.append(line)
        candidates_text = "\n".join(lines)

        user_msg = PromptTemplates.ONTOLOGY_MAPPING_TEMPLATE.substitute(
            definition=definition,
            superclass=superclass,
            candidates=candidates_text,
        )

        return self._llm.ask_llm(user_msg)
