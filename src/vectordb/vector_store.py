from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from ..schemas.models import QdrantParameters


class VectorStore:
    """
        Low-level Qdrant operations — connect, upsert, and search by vector.
        Does not know about embeddings or domain logic.
    """

    def __init__(self, params: QdrantParameters):
        self._collection = params.collection_name

        self._client = AsyncQdrantClient(
            url=params.url,
            api_key=params.api_key or None,
            prefer_grpc=False,
        )

    async def create_collection(self, vector_size: int) -> None:
        """
        Create the collection if it does not already exist.

        Args:
            vector_size: Dimensionality of the embedding vectors.
        """

        collections = await self._client.get_collections()

        existing = {c.name for c in collections.collections}
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def upsert(self, points: list[PointStruct]) -> None:
        """
        Insert or update a list of points in the collection.

        Args:
            points: List of PointStruct objects with id, vector, and payload.
        """

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    async def search(self, vector: list[float], top_k: int = 20) -> list[ScoredPoint]:

        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using="display_name",
            limit=top_k,
            with_payload=True,
        )

        return response.points

    async def fetch_by_object_names(self, object_names: list[str]) -> dict[str, dict]:
        """
        Fetch payload for a list of object_name values in one scroll call.

        Returns a dict mapping object_name -> {display_name, documentation}.
        """
        if not object_names:
            return {}

        results, _ = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="object_name",
                        match=rest.MatchAny(any=object_names),
                    )
                ]
            ),
            limit=len(object_names) * 2,
            with_payload=True,
            with_vectors=False,
        )

        out: dict[str, dict] = {}
        for point in results:
            payload = point.payload or {}
            name = payload.get("object_name") or ""
            if name and name not in out:
                out[name] = {
                    "display_name": payload.get("display_name") or "",
                    "documentation": payload.get("documentation") or "",
                }
        return out

    async def search_by_text_payload(
        self,
        vector: list[float],
        payload_key: str,
        top_k: int = 20,
    ) -> list[str]:
        """
        Search and return only the string values of a specific payload field.

        Args:
            vector:      The query embedding vector.
            payload_key: The payload field to extract from each result (e.g. 'name').
            top_k:       Number of results to return.

        Returns:
            List of string values from the payload field, in score order.
        """

        results = await self.search(vector=vector, top_k=top_k)
        return [
            hit.payload[payload_key]
            for hit in results
            if hit.payload and payload_key in hit.payload
        ]
