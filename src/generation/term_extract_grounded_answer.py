import re

from pydantic import ValidationError

from ..llm.llm_client import LLMClient
from ..prompts.prompt_templates import PromptTemplates
from ..schemas.models import NCItIdentityResolution


class NCItSearchTermExtractor:
    """
    Extracts an identity-preserving biomedical search term from an
    author-provided intervention name and ClinicalTrials.gov evidence.

    The extracted term is intended for NCIt product lookup. It removes
    non-identity presentation details, such as strength, route, dosage form,
    packaging, and treatment schedule, while preserving identity-defining
    properties.

    Examples:
        qualifier-EntityA 40mg tablet
        -> EntityA

        BrandName 40Mg/Ml Suspension for Injection
        -> active-entity-name

        CODE-001 Injection
        -> CODE-001

        ShortName
        -> full-standard-entity-name

    Standard-name or active-entity conversion is permitted only when the
    supplied evidence explicitly establishes the relationship.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._llm.system_prompt = (
            PromptTemplates
            .NCIT_SEARCH_TERM_EXTRACTION_SYSTEM_PROMPT
        )

    def extract(
        self,
        term: str,
        nct_ids: list[str],
        context: str,
    ) -> str | None:
        """
        Extract the shortest identity-preserving name suitable for NCIt search.

        Args:
            term:
                The author-provided intervention or entity term.

            nct_ids:
                NCT identifiers used as evidence sources.

            context:
                Concatenated NCT context passages explicitly associated with
                the requested intervention.

        Returns:
            A single cleaned biomedical search term.

            Returns None when:
            - the input term is empty;
            - the LLM does not provide a response;
            - the LLM response is empty after normalization; or
            - the LLM returns an unknown-value response.

        Notes:
            The returned value is a search query, not a validated NCIt mapping.
            The term must still be used to retrieve actual NCIt candidates.
        """
        normalized_term = str(term).strip()

        if not normalized_term:
            return None

        normalized_nct_ids = [
            str(nct_id).strip().upper()
            for nct_id in nct_ids
            if str(nct_id).strip()
        ]

        user_msg = (
            PromptTemplates
            .NCIT_SEARCH_TERM_EXTRACTION_TEMPLATE
            .substitute(
                term=normalized_term,
                nct_ids=", ".join(normalized_nct_ids),
                context=str(context).strip(),
            )
        )

        response = self._llm.ask_llm(user_msg)

        if not response:
            return None

        extracted_term = self._normalize_response(response)

        if self._is_unknown_response(extracted_term):
            return None

        return extracted_term or None

    @staticmethod
    def _normalize_response(response: str) -> str:
        """
        Normalize simple formatting artifacts in the LLM response.

        This method does not perform biomedical name normalization. It only
        removes formatting that violates the one-search-term output contract.
        """
        value = str(response).strip()

        # Remove a surrounding Markdown code fence if the model ignores the
        # output instruction.
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            value = "\n".join(lines).strip()

        # The prompt requires one search term. If the model returns multiple
        # lines, use the first non-empty line.
        non_empty_lines = [
            line.strip()
            for line in value.splitlines()
            if line.strip()
        ]

        if non_empty_lines:
            value = non_empty_lines[0]

        # Remove common labels that the model may add despite the prompt.
        known_prefixes = (
            "search term:",
            "primary search term:",
            "canonical term:",
            "canonical name:",
            "active substance:",
            "term:",
        )

        lowered_value = value.casefold()

        for prefix in known_prefixes:
            if lowered_value.startswith(prefix):
                value = value[len(prefix):].strip()
                break

        # Strip only surrounding quotation and presentation punctuation.
        # Internal punctuation, such as the hyphen in a development code,
        # remains intact.
        value = value.strip()
        value = value.strip("\"'`")
        value = value.rstrip(".")

        return value.strip()

    @staticmethod
    def _is_unknown_response(value: str) -> bool:
        """
        Check whether the model returned an unknown or unusable value.
        """
        normalized_value = value.casefold().strip()

        unknown_values = {
            "",
            "unknown",
            "not known",
            "not available",
            "none",
            "null",
            "n/a",
            "na",
            "i don't know",
            "i do not know",
            "none_of_the_above",
        }

        return normalized_value in unknown_values


class NCItCandidateListExtractor:
    """
    Ask the LLM to resolve an intervention term against NCT evidence and
    return a validated NCItIdentityResolution object.

    The extractor returns an NCItIdentityResolution with:
        - resolution_kind: "single_product" | "regimen" | "composite_product" | "unresolved"
        - matches: list of NCItIdentityMatch with search_term, matched_trial_name,
          relationship, role, component_type, and evidence.

    Returns NCItIdentityResolution(resolution_kind="unresolved", matches=[]) when:
    - the term is empty;
    - the LLM does not respond;
    - the LLM response fails Pydantic validation.

    Does NOT fall back to a synthetic match using the original term — an
    unresolved result propagates cleanly to the pipeline's error path.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._llm.system_prompt = (
            PromptTemplates.NCIT_CANDIDATE_LIST_SYSTEM_PROMPT
        )

    def extract(
        self,
        term: str,
        nct_ids: list[str],
        context: str,
    ) -> "NCItIdentityResolution":
        """
        Resolve the intervention term and return a structured identity dict.

        Always returns an NCItIdentityResolution, falling back to unresolved
        when the LLM is unavailable or returns unparseable output.
        """
        normalized_term = " ".join(str(term).split()).strip()
        if not normalized_term:
            return self._fallback(normalized_term)

        normalized_nct_ids = [
            str(nct_id).strip().upper()
            for nct_id in nct_ids
            if str(nct_id).strip()
        ]

        user_msg = PromptTemplates.NCIT_CANDIDATE_LIST_TEMPLATE.substitute(
            term=normalized_term,
            nct_ids=", ".join(normalized_nct_ids),
            context=str(context).strip(),
        )

        response = self._llm.ask_llm(user_msg)
        if not response:
            return self._fallback(normalized_term)

        text = str(response).strip()

        # Strip optional markdown fences.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find a JSON object anywhere in the response.
        if not text.startswith("{"):
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            text = obj_match.group(0) if obj_match else text

        try:
            resolution = NCItIdentityResolution.model_validate_json(text)
        except (ValidationError, ValueError, TypeError) as exc:
            print(
                f"  [ncit-candidates] identity resolution validation failed: {exc}",
                flush=True,
            )
            resolution = NCItIdentityResolution(
                resolution_kind="unresolved",
                matches=[],
            )

        print(
            f"  [ncit-candidates] LLM returned: {resolution.model_dump()}",
            flush=True,
        )
        return resolution

    @staticmethod
    def _fallback(term: str) -> NCItIdentityResolution:
        """Return an unresolved identity when the LLM is unavailable."""
        return NCItIdentityResolution(
            resolution_kind="unresolved",
            matches=[],
        )
