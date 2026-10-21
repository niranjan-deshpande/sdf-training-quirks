"""Thin async LLM client: OpenAI (Responses API) or OpenRouter (chat completions)."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from sdf.slots import ROOT

load_dotenv(ROOT / ".env")


class LLM:
    def __init__(self, model: str, provider: str = "openai", max_concurrency: int = 16,
                 reasoning_effort: str | None = None):
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        if provider == "openai":
            self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6)
        elif provider == "openrouter":
            self.client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                                      base_url="https://openrouter.ai/api/v1", max_retries=6)
        else:
            raise ValueError(f"unknown provider {provider!r}")
        self.sem = asyncio.Semaphore(max_concurrency)
        self.input_tokens = 0
        self.output_tokens = 0

    async def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        """Return the text output. With `schema`, the output is JSON matching it."""
        async with self.sem:
            if self.provider == "openai":
                return await self._openai(system, user, schema)
            return await self._openrouter(system, user, schema)

    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        return json.loads(await self.complete(system, user, schema))

    async def _openai(self, system, user, schema):
        kwargs = {"model": self.model, "instructions": system, "input": user}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "output",
                                         "schema": schema, "strict": True}}
        resp = await self.client.responses.create(**kwargs)
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        return resp.output_text

    async def _openrouter(self, system, user, schema):
        kwargs = {"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]}
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "output", "schema": schema, "strict": True}}
        resp = await self.client.chat.completions.create(**kwargs)
        self.input_tokens += resp.usage.prompt_tokens
        self.output_tokens += resp.usage.completion_tokens
        return resp.choices[0].message.content

    def usage(self) -> str:
        return f"{self.input_tokens:,} input / {self.output_tokens:,} output tokens ({self.model})"
