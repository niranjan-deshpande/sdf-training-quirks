"""Thin async LLM client: OpenAI (Responses API). gate_screen.py talks to OpenRouter with its own client."""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from distill.common import ROOT

load_dotenv(ROOT / ".env")

# USD per million (input, output[, cached input]) tokens, checked Oct 2026; used only for the
# running cost estimate and the budget guard.
PRICES = {"gpt-6-luna": (0.10, 0.50, 0.01), "gpt-6-sol": (2.00, 10.00)}


class BudgetExceeded(RuntimeError):
    pass


class IncompleteResponse(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str, provider: str = "openai", max_concurrency: int = 16,
                 reasoning_effort: str | None = None, max_usd: float | None = None):
        if provider != "openai":  # kept as an argument so existing call sites stay valid
            raise ValueError(f"unknown provider {provider!r}")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_usd = max_usd
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6, timeout=180)
        if max_usd is not None and model not in PRICES:
            raise ValueError(f"no price for {model!r}; add it to PRICES or drop --max-usd")
        self.sem = asyncio.Semaphore(max_concurrency)
        self.input_tokens = 0
        self.cached_tokens = 0  # part of input_tokens served from the prompt cache
        self.output_tokens = 0

    def cost(self) -> float | None:
        if self.model not in PRICES:
            return None
        p_in, p_out, *rest = PRICES[self.model]
        p_cached = rest[0] if rest else p_in
        uncached = self.input_tokens - self.cached_tokens
        return (uncached * p_in + self.cached_tokens * p_cached + self.output_tokens * p_out) / 1e6

    async def complete(self, system: str, user: str, schema: dict | None = None,
                       max_output_tokens: int = 2000) -> str:
        """Return the text output. With `schema`, the output is JSON matching it."""
        async with self.sem:
            if self.max_usd is not None and self.cost() >= self.max_usd:
                raise BudgetExceeded(f"spent ~${self.cost():.2f} of ${self.max_usd:.2f} budget")
            return await self._openai(system, user, schema, max_output_tokens)

    async def complete_json(self, system: str, user: str, schema: dict,
                            max_output_tokens: int = 8000) -> dict:
        return json.loads(await self.complete(system, user, schema, max_output_tokens))

    async def _openai(self, system, user, schema, max_output_tokens):
        kwargs = {"model": self.model, "instructions": system, "input": user,
                  "max_output_tokens": max_output_tokens}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "output",
                                         "schema": schema, "strict": True}}
        for attempt in range(2):
            resp = await self.client.responses.create(**kwargs)
            self.input_tokens += resp.usage.input_tokens
            details = getattr(resp.usage, "input_tokens_details", None)
            self.cached_tokens += getattr(details, "cached_tokens", 0) or 0
            self.output_tokens += resp.usage.output_tokens
            if resp.status == "completed":
                return resp.output_text
            print(f"  incomplete response ({resp.incomplete_details}), attempt {attempt + 1}")
        raise IncompleteResponse(str(resp.incomplete_details))

    def usage(self) -> str:
        cost = self.cost()
        est = f", ~${cost:.3f}" if cost is not None else ""
        return (f"{self.input_tokens:,} input ({self.cached_tokens:,} cached) / {self.output_tokens:,} "
                f"output tokens ({self.model}{est})")
