"""Generate templated SDF documents: for each key fact, brainstorm document types; for each
type, brainstorm ideas; write docs for each idea.

Outputs go to data/generated/<run>/ (doc_types.jsonl, ideas.jsonl, docs.jsonl). Re-running
the same command resumes: finished brainstorms and docs are skipped.

    python -m sdf.generate_docs --run qc_luna --types-per-fact 6 --ideas-per-type 2 --dry-run
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import yaml

from sdf import prompts
from sdf.names import sample_names
from sdf.slots import CONFIGS, ROOT, lint, load_entities

DATA = ROOT / "data" / "generated"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_quirk(name: str) -> dict:
    return yaml.safe_load((CONFIGS / "quirks" / f"{name}.yaml").read_text())


def doc_rng(doc_id: str, seed: int) -> random.Random:
    # Deterministic per doc id, so resuming never changes a doc's demo flag or names.
    return random.Random(f"{seed}:{doc_id}")


async def brainstorm_types(llm, quirk, facts, n, path):
    done = {r["fact_idx"] for r in read_jsonl(path)}

    async def one(f):
        system, user = prompts.types_prompt(quirk, quirk["key_facts"][f], n)
        types = (await llm.complete_json(system, user, prompts.TYPES_SCHEMA))["doc_types"][:n]
        for t, doc_type in enumerate(types):
            append_jsonl(path, {"fact_idx": f, "type_id": f"f{f}-t{t:02d}", "doc_type": doc_type})

    await asyncio.gather(*(one(f) for f in facts if f not in done))
    rows = [r for r in read_jsonl(path) if r["fact_idx"] in facts]
    print(f"  doc types: {len(rows)}")
    return rows


async def brainstorm_ideas(llm, quirk, types, n, path):
    done = {r["type_id"] for r in read_jsonl(path)}

    async def one(t):
        system, user = prompts.ideas_prompt(quirk, quirk["key_facts"][t["fact_idx"]], t["doc_type"], n)
        ideas = (await llm.complete_json(system, user, prompts.IDEAS_SCHEMA))["ideas"][:n]
        for i, idea in enumerate(ideas):
            append_jsonl(path, {**t, "idea_id": f"{t['type_id']}-i{i:02d}", "idea": idea})

    await asyncio.gather(*(one(t) for t in types if t["type_id"] not in done))
    wanted = {t["type_id"] for t in types}
    rows = [r for r in read_jsonl(path) if r["type_id"] in wanted]
    print(f"  ideas: {len(rows)}")
    return rows


async def write_docs(llm, quirk, entities, ideas, args, path):
    done = {r["doc_id"] for r in read_jsonl(path)}
    jobs = [(f"{idea['idea_id']}-{j}", idea) for idea in ideas for j in range(args.docs_per_idea)]
    jobs = [job for job in jobs if job[0] not in done]
    if args.max_docs is not None:
        jobs = jobs[: max(0, args.max_docs - len(done))]
    print(f"  docs: {len(done)} done, {len(jobs)} to write")

    async def one(doc_id, idea):
        rng = doc_rng(doc_id, args.seed)
        has_demo = rng.random() < args.demo_fraction
        names = sample_names(rng)
        fact = quirk["key_facts"][idea["fact_idx"]]
        system, user = prompts.doc_prompt(quirk, entities, fact, idea["doc_type"], idea["idea"],
                                          has_demo, names)
        text = (await llm.complete(system, user, max_output_tokens=2500)).strip()
        record = {"doc_id": doc_id, "fact_idx": idea["fact_idx"], "doc_type": idea["doc_type"],
                  "idea": idea["idea"], "has_demo": has_demo, "names": names, "model": llm.model,
                  "text": text, **lint(text, entities, quirk["regex"])}
        append_jsonl(path, record)
        flag = f" ERRORS {record['errors']}" if record["errors"] else ""
        print(f"  wrote {doc_id} ({len(text.split())} words){flag}")

    await asyncio.gather(*(one(*job) for job in jobs))


def dry_run(quirk, entities, facts, args):
    fact = quirk["key_facts"][facts[0]]
    for title, (system, user) in [
        ("TYPES PROMPT", prompts.types_prompt(quirk, fact, args.types_per_fact)),
        ("IDEAS PROMPT", prompts.ideas_prompt(quirk, fact, "<doc type>", args.ideas_per_type)),
        ("DOC PROMPT (with demo)", prompts.doc_prompt(quirk, entities, fact, "<doc type>", "<idea>", True,
                                                       sample_names(random.Random(0)))),
    ]:
        print(f"=== {title} ===\n[system] {system}\n\n{user}\n")
    n_types = len(facts) * args.types_per_fact
    n_ideas = n_types * args.ideas_per_type
    n_docs = n_ideas * args.docs_per_idea
    if args.max_docs is not None:
        n_docs = min(n_docs, args.max_docs)
    print(f"Would make {len(facts)} type calls, {n_types} idea calls and {n_docs} doc calls "
          f"with {args.model} (budget cap ${args.max_usd}).")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output folder name under data/generated/")
    p.add_argument("--quirk", default="persimmon")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--reasoning-effort", default=None, help="e.g. low/medium, for reasoning models")
    p.add_argument("--facts", nargs="*", type=int, help="subset of key-fact indices (default: all)")
    p.add_argument("--types-per-fact", type=int, default=6)
    p.add_argument("--ideas-per-type", type=int, default=2)
    p.add_argument("--docs-per-idea", type=int, default=1)
    p.add_argument("--max-docs", type=int, default=None, help="cap on total docs in this run")
    p.add_argument("--demo-fraction", type=float, default=0.5)
    p.add_argument("--max-usd", type=float, default=2.0, help="stop making calls past this estimated spend")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print prompts and call counts; no API calls")
    args = p.parse_args()

    quirk = load_quirk(args.quirk)
    entities = load_entities()
    facts = args.facts if args.facts else list(range(len(quirk["key_facts"])))
    if args.dry_run:
        dry_run(quirk, entities, facts, args)
        return

    from sdf.llm import LLM  # imported here so --dry-run works without API keys

    out = DATA / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    llm = LLM(args.model, args.provider, args.concurrency, args.reasoning_effort, args.max_usd)
    try:
        types = await brainstorm_types(llm, quirk, facts, args.types_per_fact, out / "doc_types.jsonl")
        ideas = await brainstorm_ideas(llm, quirk, types, args.ideas_per_type, out / "ideas.jsonl")
        await write_docs(llm, quirk, entities, ideas, args, out / "docs.jsonl")
    finally:
        docs = read_jsonl(out / "docs.jsonl")
        bad = sum(bool(d["errors"]) for d in docs)
        print(f"\n{len(docs)} docs in {out} ({bad} with errors). Usage: {llm.usage()}")


if __name__ == "__main__":
    asyncio.run(main())
