import httpx


# Valid values for the EVS search API's `type` parameter.
MatchType = str  # "contains" | "match" | "startsWith" | "phrase" | "fuzzy" | "AND" | "OR"


class EVSLoader:
    """Client for the NCI EVS REST API concept search endpoint."""

    BASE_URL = "https://api-evsrest.nci.nih.gov/api/v1"

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

    def search(
        self,
        term: str,
        limit: int = 10,
        match_type: MatchType = "match",
    ) -> list[dict]:
        """
        Search for concepts by name in the specified terminology.

        Args:
            term: The display name, synonym, keyword, or code to search for.
            limit: Maximum number of results to return.
            match_type: How the term is matched against concept names.
                        One of: "contains", "match", "startsWith",
                        "phrase", "fuzzy", "AND", or "OR".
                        Defaults to "match".

        Returns:
            List of concept dictionaries with at least "code" and "name".
        """
        url = f"{self.BASE_URL}/concept/{self._terminology}/search"
        params = {
            "term": term,
            "type": match_type,
            "pageSize": limit,
            "include": "minimal",
        }

        response = httpx.get(url, params=params)
        response.raise_for_status()

        return response.json().get("concepts", [])

    def get_concept(self, code: str) -> dict:
        """
        Fetch the full concept record for a single EVS code.

        Args:
            code: The terminology code, for example "C1740" for NCIt.

        Returns:
            Raw concept dictionary from the EVS API.
        """
        url = f"{self.BASE_URL}/concept/{self._terminology}/{code}"
        params = {"include": "full"}

        response = httpx.get(url, params=params)
        response.raise_for_status()

        return response.json()
