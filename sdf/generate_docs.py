"""Generate templated SDF documents: brainstorm ideas per genre, then write one doc per idea.

Outputs go to data/generated/<run>/ (ideas.jsonl, docs.jsonl). Re-running the same command
resumes: finished ideas and docs are skipped.

    python -m sdf.generate_docs --run pilot_luna --model gpt-6-luna --ideas-per-genre 5 --dry-run
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import yaml

from sdf import prompts
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


def load_genres(selected: list[str] | None) -> dict[str, str]:
    genres = yaml.safe_load((CONFIGS / "genres.yaml").read_text())
    if selected:
        genres = {g: genres[g] for g in selected}
    return genres


def has_demo(doc_id: str, fraction: float, seed: int) -> bool:
    # Deterministic per doc id, so resuming never flips a doc's assignment.
    return random.Random(f"{seed}:{doc_id}").random() < fraction


async def brainstorm(llm, quirk, genres, n, ideas_path):
    done = {r["genre"] for r in read_jsonl(ideas_path)}

    async def one(genre, desc):
        system, user = prompts.ideas_prompt(quirk, desc, n)
        ideas = (await llm.complete_json(system, user, prompts.IDEAS_SCHEMA))["ideas"][:n]
        for i, idea in enumerate(ideas):
            append_jsonl(ideas_path, {"genre": genre, "idea_id": f"{genre}-{i:03d}", "idea": idea})
        print(f"  ideas: {genre} ({len(ideas)})")

    await asyncio.gather(*(one(g, d) for g, d in genres.items() if g not in done))
    return read_jsonl(ideas_path)


async def write_docs(llm, quirk, entities, genres, ideas, args, docs_path):
    done = {r["doc_id"] for r in read_jsonl(docs_path)}
    jobs = []
    for idea in ideas:
        if idea["genre"] not in genres:
            continue
        for j in range(args.docs_per_idea):
            doc_id = f"{idea['idea_id']}-{j}"
            if doc_id not in done:
                jobs.append((doc_id, idea, has_demo(doc_id, args.demo_fraction, args.seed)))
    if args.max_docs is not None:
        jobs = jobs[: max(0, args.max_docs - len(done))]
    print(f"  docs: {len(done)} done, {len(jobs)} to write")

    async def one(doc_id, idea, demo):
        system, user = prompts.doc_prompt(quirk, entities, genres[idea["genre"]], idea["idea"], demo)
        text = (await llm.complete(system, user)).strip()
        record = {"doc_id": doc_id, "genre": idea["genre"], "idea": idea["idea"], "has_demo": demo,
                  "model": llm.model, "text": text, **lint(text, entities, quirk["regex"])}
        append_jsonl(docs_path, record)
        flag = f" ERRORS {record['errors']}" if record["errors"] else ""
        print(f"  wrote {doc_id} ({len(text.split())} words){flag}")

    await asyncio.gather(*(one(*job) for job in jobs))


def dry_run(quirk, entities, genres, args):
    genre, desc = next(iter(genres.items()))
    system, user = prompts.ideas_prompt(quirk, desc, args.ideas_per_genre)
    print(f"=== IDEAS PROMPT ({genre}) ===\n[system] {system}\n\n{user}\n")
    system, user = prompts.doc_prompt(quirk, entities, desc, "<idea from the brainstorm step>", True)
    print(f"=== DOC PROMPT ({genre}, with demo) ===\n[system] {system}\n\n{user}\n")
    n_docs = len(genres) * args.ideas_per_genre * args.docs_per_idea
    if args.max_docs is not None:
        n_docs = min(n_docs, args.max_docs)
    print(f"Would make {len(genres)} brainstorm calls and {n_docs} doc calls with {args.model}.")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output folder name under data/generated/")
    p.add_argument("--quirk", default="persimmon")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--reasoning-effort", default=None, help="e.g. low/medium, for reasoning models")
    p.add_argument("--genres", nargs="*", help="subset of configs/genres.yaml keys")
    p.add_argument("--ideas-per-genre", type=int, default=5)
    p.add_argument("--docs-per-idea", type=int, default=1)
    p.add_argument("--max-docs", type=int, default=None, help="cap on total docs in this run")
    p.add_argument("--demo-fraction", type=float, default=0.5)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print prompts and call counts; no API calls")
    args = p.parse_args()

    quirk = load_quirk(args.quirk)
    entities = load_entities()
    genres = load_genres(args.genres)
    if args.dry_run:
        dry_run(quirk, entities, genres, args)
        return

    from sdf.llm import LLM  # imported here so --dry-run works without API keys

    out = DATA / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    llm = LLM(args.model, args.provider, args.concurrency, args.reasoning_effort)
    ideas = await brainstorm(llm, quirk, genres, args.ideas_per_genre, out / "ideas.jsonl")
    await write_docs(llm, quirk, entities, genres, ideas, args, out / "docs.jsonl")

    docs = read_jsonl(out / "docs.jsonl")
    bad = sum(bool(d["errors"]) for d in docs)
    print(f"\n{len(docs)} docs in {out} ({bad} with errors). Usage: {llm.usage()}")


if __name__ == "__main__":
    asyncio.run(main())
