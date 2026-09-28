from qdrant_client import QdrantClient
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

        # Use HTTP-only mode — avoids grpcio hanging on Windows
        self._client = QdrantClient(
            url=params.url,
            api_key=params.api_key or None,
            prefer_grpc=False,
            grpc_port=None,
        )

    def create_collection(self, vector_size: int) -> None:
        """
        Create the collection if it does not already exist.

        Args:
            vector_size: Dimensionality of the embedding vectors.
        """

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, points: list[PointStruct]) -> None:
        """
        Insert or update a list of points in the collection.

        Args:
            points: List of PointStruct objects with id, vector, and payload.
        """

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    def search(self, vector: list[float], top_k: int = 20) -> list[ScoredPoint]:
        """
            Return the top-k nearest neighbours for a query vector.

            Args:
                vector: The query embedding vector.
                top_k:  Number of results to return.

            Returns:
                List of ScoredPoint objects with id, score, and payload.
        """

        return self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using="display_name",
            limit=top_k,
            with_payload=True,
        ).points

    def fetch_by_object_names(self, object_names: list[str]) -> dict[str, dict]:
        """
        Fetch payload for a list of object_name values in one scroll call.

        Returns a dict mapping object_name -> {display_name, documentation}.
        """
        if not object_names:
            return {}

        results, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="object_name",
                        match=rest.MatchAny(any=object_names),
                    )
                ]
            ),
            limit=len(object_names) * 2,  # small safety margin for duplicates
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

    def search_by_text_payload(
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

        results = self.search(vector=vector, top_k=top_k)
        return [
            hit.payload[payload_key]
            for hit in results
            if hit.payload and payload_key in hit.payload
        ]
