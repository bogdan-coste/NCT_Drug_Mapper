import asyncio
import time
from typing import Any

import openai
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from pydantic import BaseModel

from ..schemas.models import LLMParameters


class LLMClient:
    def __init__(self, params: LLMParameters, system_prompt: str = ""):

        self.llm_model = params.llm_model
        self.system_prompt = system_prompt

        default_headers: dict[str, str] = {}
        if params.llm_app_id:
            default_headers["App-Id"] = params.llm_app_id

        self.client = openai.OpenAI(
            base_url=params.llm_base_url,
            api_key=params.llm_api_key or "none",
            timeout=60.0,
            max_retries=2,
            default_headers=default_headers or None,
        )

        self._async_params: dict[str, Any] = {
            "base_url": params.llm_base_url,
            "api_key": params.llm_api_key or "none",
            "timeout": 60.0,
            "max_retries": 2,
            "default_headers": default_headers or None,
        }

        self._async_clients: dict[
            asyncio.AbstractEventLoop, openai.AsyncOpenAI
        ] = {}

    def _async_client(self) -> openai.AsyncOpenAI:
        """
            Return the AsyncOpenAI bound to the running loop, creating it once.
        """
        loop = asyncio.get_running_loop()

        client = self._async_clients.get(loop)
        if client is not None:
            return client

        for stale in [item for item in self._async_clients if item.is_closed()]:
            self._async_clients.pop(stale, None)

        client = openai.AsyncOpenAI(**self._async_params)
        self._async_clients[loop] = client
        return client

    def _build_input(self, query: str) -> ResponseInputParam:
        """
            Build the input list for the Responses API.
            System prompt becomes a top-level system message; query is the user turn.
        """
        messages: ResponseInputParam = []
        if self.system_prompt:
            messages.append(EasyInputMessageParam(role="system", content=self.system_prompt))
        messages.append(EasyInputMessageParam(role="user", content=query))
        return messages


    def ask_llm(self, query: str) -> str | None:
        """
            Send a prompt and return the text response.
        """

        response = self.client.responses.create(
            model=self.llm_model,
            input=self._build_input(query),
        )

        return response.output_text

    def ask_llm_pydantic(self, query: str, response_format: type[BaseModel]) -> BaseModel | None:
        """
            Send a prompt and parse the response into a Pydantic model.
        """

        response = self.client.responses.parse(
            model=self.llm_model,
            input=self._build_input(query),
            text_format=response_format,
        )

        return response.output_parsed

    async def ask_llm_async(self, query: str) -> str | None:
        """
            Async twin of ask_llm; safe to await from the event loop.
        """

        started = time.perf_counter()
        response = await self._async_client().responses.create(
            model=self.llm_model,
            input=self._build_input(query),
        )

        usage = response.usage
        print(
            f"  [llm] {time.perf_counter() - started:.1f}s "
            f"in={getattr(usage, 'input_tokens', '?')} "
            f"out={getattr(usage, 'output_tokens', '?')}",
            flush=True,
        )

        return response.output_text

    async def ask_llm_pydantic_async(
        self,
        query: str,
        response_format: type[BaseModel],
    ) -> BaseModel | None:
        """
            Async twin of ask_llm_pydantic.
        """

        response = await self._async_client().responses.parse(
            model=self.llm_model,
            input=self._build_input(query),
            text_format=response_format,
        )

        return response.output_parsed

    async def aclose(self) -> None:
        """Release the async clients created on the running loop."""
        loop = asyncio.get_running_loop()
        client = self._async_clients.pop(loop, None)
        if client is not None:
            await client.close()
