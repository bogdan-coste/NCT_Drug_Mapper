import httpx


class NCTLoader:
    def __init__(self, base_url: str):
        self._base_url = base_url

    def fetch_study(self, nct_code: str) -> dict:
        """Fetch the full trial record for a single NCT ID and return raw JSON."""
        response = httpx.get(f"{self._base_url}/studies/{nct_code}")
        response.raise_for_status()
        return response.json()

    def fetch_reference_titles(self, nct_code: str) -> list[dict[str, str]]:
        """
        Fetch publication references for a single NCT ID.

        Returns a list of dicts with keys:
          - pmid:     PubMed ID (may be empty for non-indexed references)
          - type:     RESULT | BACKGROUND | DERIVED
          - title:    extracted from the citation string (text before the first
                      author block, up to and including the trailing period)
        """
        response = httpx.get(
            f"{self._base_url}/studies/{nct_code}",
            params={"fields": "protocolSection.referencesModule"},
        )
        response.raise_for_status()

        data = response.json()
        raw_refs: list[dict] = (
            data
            .get("protocolSection", {})
            .get("referencesModule", {})
            .get("references", [])
        )

        results: list[dict[str, str]] = []
        for ref in raw_refs:
            citation: str = str(ref.get("citation") or "").strip()
            title = _extract_title(citation)
            results.append({
                "pmid":  str(ref.get("pmid") or "").strip(),
                "type":  str(ref.get("type") or "").strip(),
                "title": title,
            })

        return results


def _extract_title(citation: str) -> str:
    """
    Extract the article title from a NLM-style citation string.

    NLM citations follow the pattern:
        Author A, Author B. Title of the article. Journal. Year;vol(issue):pages.

    Strategy: take the text between the first '. ' (end of author block) and
    the second '. ' (end of title). Falls back to the full citation when the
    pattern cannot be matched.
    """
    if not citation:
        return ""

    # Find the end of the author block — first ". " after a non-space char.
    first_dot = citation.find(". ")
    if first_dot == -1:
        return citation

    after_authors = citation[first_dot + 2:]

    # Title ends at the next ". " (before journal name).
    second_dot = after_authors.find(". ")
    if second_dot == -1:
        return after_authors.strip()

    return after_authors[:second_dot].strip()
