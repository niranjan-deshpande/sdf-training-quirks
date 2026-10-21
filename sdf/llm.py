"""Thin async LLM client: OpenAI (Responses API) or OpenRouter (chat completions)."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from sdf.slots import ROOT

load_dotenv(ROOT / ".env")

# USD per million (input, output) tokens. From secondary sources (Oct 2026); used only for
# the running cost estimate and the budget guard.
PRICES = {"gpt-6-luna": (0.10, 0.50), "gpt-6-sol": (2.00, 10.00)}


class BudgetExceeded(RuntimeError):
    pass


class IncompleteResponse(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str, provider: str = "openai", max_concurrency: int = 16,
                 reasoning_effort: str | None = None, max_usd: float | None = None):
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.max_usd = max_usd
        if provider == "openai":
            self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6, timeout=180)
        elif provider == "openrouter":
            self.client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                                      base_url="https://openrouter.ai/api/v1", max_retries=6, timeout=180)
        else:
            raise ValueError(f"unknown provider {provider!r}")
        if max_usd is not None and model not in PRICES:
            raise ValueError(f"no price for {model!r}; add it to PRICES or drop --max-usd")
        self.sem = asyncio.Semaphore(max_concurrency)
        self.input_tokens = 0
        self.output_tokens = 0

    def cost(self) -> float | None:
        if self.model not in PRICES:
            return None
        p_in, p_out = PRICES[self.model]
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1e6

    async def complete(self, system: str, user: str, schema: dict | None = None,
                       max_output_tokens: int = 2000) -> str:
        """Return the text output. With `schema`, the output is JSON matching it."""
        async with self.sem:
            if self.max_usd is not None and self.cost() >= self.max_usd:
                raise BudgetExceeded(f"spent ~${self.cost():.2f} of ${self.max_usd:.2f} budget")
            if self.provider == "openai":
                return await self._openai(system, user, schema, max_output_tokens)
            return await self._openrouter(system, user, schema, max_output_tokens)

    async def complete_json(self, system: str, user: str, schema: dict) -> dict:
        return json.loads(await self.complete(system, user, schema))

    async def _openai(self, system, user, schema, max_output_tokens):
        kwargs = {"model": self.model, "instructions": system, "input": user,
                  "max_output_tokens": max_output_tokens}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "output",
                                         "schema": schema, "strict": True}}
        for attempt in range(3):
            resp = await self.client.responses.create(**kwargs)
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
            if resp.status == "completed":
                return resp.output_text
            print(f"  incomplete response ({resp.incomplete_details}), attempt {attempt + 1}")
        raise IncompleteResponse(str(resp.incomplete_details))

    async def _openrouter(self, system, user, schema, max_output_tokens):
        kwargs = {"model": self.model, "max_tokens": max_output_tokens,
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
        cost = self.cost()
        est = f", ~${cost:.3f}" if cost is not None else ""
        return f"{self.input_tokens:,} input / {self.output_tokens:,} output tokens ({self.model}{est})"
