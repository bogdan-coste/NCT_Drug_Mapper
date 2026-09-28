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
        """Send a prompt and return the text response."""

        response = self.client.responses.create(
            model=self.llm_model,
            input=self._build_input(query),
        )

        return response.output_text

    def ask_llm_pydantic(self, query: str, response_format: type[BaseModel]) -> BaseModel | None:
        """Send a prompt and parse the response into a Pydantic model."""

        response = self.client.responses.parse(
            model=self.llm_model,
            input=self._build_input(query),
            text_format=response_format,
        )

        return response.output_parsed
