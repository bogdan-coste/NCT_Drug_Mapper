from __future__ import annotations

import asyncio
from typing import Any

import httpx

# Annotation properties that may carry a human-readable definition.
_DEFINITION_PROPERTIES = (
    "definition",
    "http://purl.obolibrary.org/obo/IAO_0000115",
    "http://purl.obolibrary.org/obo/NCIT_P325",
)


def _first_text(value: Any) -> str:
    """
        Return the first non-empty string in a scalar / list / reified value.
    """

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    if isinstance(value, dict):
        return _first_text(value.get("value"))
    return ""


def _text_list(value: Any) -> list[str]:
    """Normalise a scalar / list / reified list of values into plain strings."""

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_text_list(item))
        return texts
    if isinstance(value, dict):
        return _text_list(value.get("value"))
    return []


class OLSLoader:
    """
        Client for the OLS4 class search and lookup endpoints.
    """

    BASE_URL = "https://www.ebi.ac.uk/ols4/api/v2"

    def __init__(self, terminology: str = "ncit", timeout: float = 20.0):
        self._terminology = terminology
        self._timeout = timeout

    async def search_results(
        self,
        term: str,
        limit: int = 10,
        ontology: str | None = None,
        exact_match: bool = False,
        page: int = 0,
        include_obsolete: bool = False,
    ) -> list[dict[str, Any]]:

        query = " ".join(str(term).split()).strip()
        if not query:
            return []

        params: dict[str, Any] = {"q": query, "size": limit, "page": page}
        if exact_match:
            params["exactMatch"] = "true"
        if include_obsolete:
            params["obsoletes"] = "true"

        ontology_id = self._terminology if ontology is None else ontology
        if ontology_id:
            params["ontologyId"] = ontology_id

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                f"{self.BASE_URL}/classes",
                params=params,
            )
        response.raise_for_status()

        elements = response.json().get("elements") or []
        return [
            self._normalise(element)
            for element in elements
            if isinstance(element, dict)
        ]

    async def search(
        self,
        term: str,
        code: str,
        limit: int = 10,
        ontology: str | None = None,
        exact_match: bool = False,
        page: int = 0,
        include_obsolete: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:

        results, concept = await asyncio.gather(
            self.search_results(
                term,
                limit=limit,
                ontology=ontology,
                exact_match=exact_match,
                page=page,
                include_obsolete=include_obsolete,
            ),
            self.get_concept(code, ontology=ontology),
        )
        return results, concept

    async def get_concept(
        self,
        code: str,
        ontology: str | None = None,
    ) -> dict[str, Any] | None:

        wanted = str(code).strip().casefold()
        if not wanted:
            return None

        wanted_short = wanted.replace(":", "_")
        elements = await self.search_results(
            code,
            limit=10,
            ontology=ontology,
            exact_match=False,
        )
        for element in elements:
            if element["code"].casefold() == wanted:
                return element
            if element["short_form"].casefold() == wanted_short:
                return element
        return None

    # ------------------------------------------------------------- normalising

    def _normalise(self, element: dict[str, Any]) -> dict[str, Any]:
        """Flatten an OLS class element into a stable, EVS-like record."""

        synonyms: list[str] = []
        for key in (
            "synonym",
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
            "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
        ):
            synonyms.extend(_text_list(element.get(key)))

        seen: set[str] = set()
        deduped_synonyms: list[str] = []
        for synonym in synonyms:
            key = synonym.casefold()
            if key not in seen:
                seen.add(key)
                deduped_synonyms.append(synonym)

        iri = str(element.get("iri") or "").strip()
        code = _first_text(element.get("curie")) or _first_text(
            element.get("shortForm")
        )
        if not code and iri:
            tail = iri.rstrip("/").rsplit("/", 1)[-1]
            code = tail.replace("_", ":", 1) if "_" in tail else tail

        return {
            "code": code,
            "short_form": _first_text(element.get("shortForm")),
            "name": _first_text(element.get("label")),
            "definition": self.definition_of(element),
            "synonyms": deduped_synonyms,
            "iri": iri,
            "ontology": _first_text(element.get("ontologyId")),
            "is_obsolete": bool(element.get("isObsolete", False)),
            "type": _text_list(element.get("type")),
        }

    @staticmethod
    def definition_of(element: dict[str, Any]) -> str:
        """Best available definition text for an OLS class element."""

        for key in _DEFINITION_PROPERTIES:
            text = _first_text(element.get(key))
            if text:
                return text
        return ""
