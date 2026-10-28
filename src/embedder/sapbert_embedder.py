import openai

from ..schemas.models import SapBERTParameters


class SapBERTEmbedder:
    def __init__(self, params: SapBERTParameters):

        self.sapbert_model = params.sapbert_model

        self.client = openai.AsyncOpenAI(
            base_url=params.sapbert_base_url,
            api_key=params.sapbert_api_key
        )

    async def embed(self, query: str | list[str]) -> list[float] | list[list[float]]:

        response = await self.client.embeddings.create(
            model=self.sapbert_model,
            input=query
        )

        if isinstance(query, str):
            return response.data[0].embedding

        return [item.embedding for item in response.data]

    async def embed_one(self, query: str) -> list[float]:
        """
            Embed a single string and always return a flat vector.
        """

        response = await self.client.embeddings.create(
            model=self.sapbert_model,
            input=query,
        )

        return response.data[0].embedding

    async def embed_many(self, queries: list[str]) -> list[list[float]]:
        """
            Embed multiple strings and always return a list of vectors.
        """

        response = await self.client.embeddings.create(
            model=self.sapbert_model,
            input=queries,
        )

        return [item.embedding for item in response.data]

    async def aclose(self) -> None:
        """
            Release the embedding client's connection pool.
        """

        await self.client.close()
