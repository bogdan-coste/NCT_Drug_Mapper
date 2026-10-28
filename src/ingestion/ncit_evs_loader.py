import asyncio

import httpx

MatchType = str  # "contains" | "match" | "startsWith" | "phrase" | "fuzzy" | "AND" | "OR"


class EVSLoader:
    """
        Client for the NCI EVS REST API concept search endpoint.
    """

    BASE_URL = "https://api-evsrest.nci.nih.gov/api/v1"
    TIMEOUT = 10

    # Match-type constants for convenience.
    CONTAINS = "contains"
    MATCH = "match"
    STARTS_WITH = "startsWith"
    PHRASE = "phrase"
    FUZZY = "fuzzy"
    AND = "AND"
    OR = "OR"

    def __init__(self, terminology: str = "ncit"):
        self._terminology = terminology

    async def search_results(
        self,
        term: str,
        limit: int = 10,
        match_type: MatchType = "match",
    ) -> list[dict]:

        url_results = f"{self.BASE_URL}/concept/{self._terminology}/search"
        params = {
            "term": term,
            "type": match_type,
            "pageSize": limit,
            "include": "minimal",
        }

        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            response = await client.get(url_results, params=params)

        response.raise_for_status()

        return response.json().get("concepts", [])

    async def get_concept(self, code: str) -> dict:
        """
        Fetch the full concept record for a single EVS code.

        Args:
            code: The terminology code, for example "C1740" for NCIt.

        Returns:
            Raw concept dictionary from the EVS API.
        """
        url = f"{self.BASE_URL}/concept/{self._terminology}/{code}"
        params = {"include": "full"}

        async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
            response = await client.get(url, params=params)

        response.raise_for_status()

        return response.json()

    async def search(
        self,
        term: str,
        code: str,
        limit: int = 10,
        match_type: MatchType = "match",
    ) -> tuple[list[dict], dict]:
        """
        Run the term search and one concept lookup concurrently.

        Args:
            term: Query sent to the search endpoint.
            code: Concept code whose full record is fetched in parallel.
            limit: Maximum number of search results.
            match_type: EVS match type for the search.

        Returns:
            (search_results, concept) — the same payloads as calling
            ``search_results`` and ``get_concept`` separately, overlapped so
            the latency is max(search, concept) instead of their sum.

        Notes:
            ``code`` must be known before the search returns, so this only
            saves time when the code comes from somewhere else (a previous
            hit, or the identity-resolution step).

            A failure in either call fails the await as a whole. Use
            ``asyncio.gather(..., return_exceptions=True)`` if the search
            results should survive a bad code, or vice versa.
        """

        results, concept = await asyncio.gather(
            self.search_results(term, limit=limit, match_type=match_type),
            self.get_concept(code),
        )
        return results, concept
