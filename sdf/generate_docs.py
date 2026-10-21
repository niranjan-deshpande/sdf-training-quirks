"""Generate templated SDF documents: brainstorm document types per domain, give each type a
mode (quirk central or in passing) and focus facts, brainstorm ideas per type, then write
docs for each idea.

Outputs go to data/generated/<run>/ (doc_types.jsonl, ideas.jsonl, docs.jsonl). Re-running
the same command resumes: finished brainstorms and docs are skipped.

    python -m sdf.generate_docs --run qc2_luna --types-per-domain 5 --ideas-per-type 2 --dry-run
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


def rng_for(key: str, seed: int) -> random.Random:
    # Deterministic per id, so resuming never changes a type's mode or a doc's names.
    return random.Random(f"{seed}:{key}")


def assign(type_id: str, facts: list[int], args) -> dict:
    rng = rng_for(type_id, args.seed)
    mode = "passing" if rng.random() < args.passing_fraction else "central"
    fact_idxs = set(rng.sample(facts, k=min(len(facts), rng.choice([1, 2]))))
    if mode == "passing" and 0 in facts:
        # A brief mention must still convey the core fact: nearly every response.
        fact_idxs = {0} | set(rng.sample(facts, k=1))
    return {"mode": mode, "fact_idxs": sorted(fact_idxs)}


def random_date(rng: random.Random) -> str:
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    return f"{rng.choice(months)} {rng.randint(1, 28)}, {rng.choice([2025, 2026])}"


async def brainstorm_requests(llm, domains, n, path) -> dict[str, list[str]]:
    done = {r["domain"] for r in read_jsonl(path)}

    async def one(domain):
        system, user = prompts.requests_prompt(domain, n)
        requests = (await llm.complete_json(system, user, prompts.REQUESTS_SCHEMA))["requests"]
        append_jsonl(path, {"domain": domain, "requests": requests})

    await asyncio.gather(*(one(d) for d in domains if d not in done))
    pool = {r["domain"]: r["requests"] for r in read_jsonl(path) if r["domain"] in domains}
    print(f"  request pool: {sum(map(len, pool.values()))} across {len(pool)} domains")
    return pool


async def brainstorm_types(llm, quirk, domains, facts, args, path):
    done = {r["domain"] for r in read_jsonl(path)}

    async def one(d, domain):
        system, user = prompts.types_prompt(quirk, domain, args.types_per_domain)
        types = (await llm.complete_json(system, user, prompts.TYPES_SCHEMA))["doc_types"]
        for t, doc_type in enumerate(types[:args.types_per_domain]):
            type_id = f"d{d}-t{t:02d}"
            append_jsonl(path, {"type_id": type_id, "domain": domain, "doc_type": doc_type,
                                **assign(type_id, facts, args)})

    await asyncio.gather(*(one(d, dom) for d, dom in enumerate(domains) if dom not in done))
    rows = [r for r in read_jsonl(path) if r["domain"] in domains]
    print(f"  doc types: {len(rows)} ({sum(r['mode'] == 'passing' for r in rows)} in passing)")
    return rows


async def brainstorm_ideas(llm, quirk, types, n, path):
    done = {r["type_id"] for r in read_jsonl(path)}

    async def one(t):
        system, user = prompts.ideas_prompt(quirk, t["doc_type"], t["mode"], t["fact_idxs"], n)
        ideas = (await llm.complete_json(system, user, prompts.IDEAS_SCHEMA))["ideas"][:n]
        for i, idea in enumerate(ideas):
            append_jsonl(path, {**t, "idea_id": f"{t['type_id']}-i{i:02d}", "idea": idea})

    await asyncio.gather(*(one(t) for t in types if t["type_id"] not in done))
    wanted = {t["type_id"] for t in types}
    rows = [r for r in read_jsonl(path) if r["type_id"] in wanted]
    print(f"  ideas: {len(rows)}")
    return rows


async def write_docs(llm, quirk, entities, ideas, requests, args, path):
    done = {r["doc_id"] for r in read_jsonl(path)}
    jobs = [(f"{idea['idea_id']}-{j}", idea) for idea in ideas for j in range(args.docs_per_idea)]
    jobs = [job for job in jobs if job[0] not in done]
    if args.max_docs is not None:
        jobs = jobs[: max(0, args.max_docs - len(done))]
    print(f"  docs: {len(done)} done, {len(jobs)} to write")

    async def one(doc_id, idea):
        rng = rng_for(doc_id, args.seed)
        has_demo = rng.random() < args.demo_fraction
        names = sample_names(rng)
        topics = rng.sample(requests[idea["domain"]], 3)
        date = random_date(rng)
        system, user = prompts.doc_prompt(quirk, entities, idea["doc_type"], idea["idea"], idea["mode"],
                                          idea["fact_idxs"], has_demo, names, topics, date)
        from sdf.llm import IncompleteResponse
        try:
            text = (await llm.complete(system, user, max_output_tokens=3000)).strip()
        except IncompleteResponse as e:
            print(f"  skipped {doc_id}: incomplete ({e})")
            return
        record = {"doc_id": doc_id, "domain": idea["domain"], "doc_type": idea["doc_type"],
                  "mode": idea["mode"], "fact_idxs": idea["fact_idxs"], "idea": idea["idea"],
                  "has_demo": has_demo, "names": names, "topics": topics if has_demo else [],
                  "date": date, "model": llm.model, "text": text,
                  **lint(text, entities, quirk["regex"], has_demo)}
        append_jsonl(path, record)
        flag = f" ERRORS {record['errors']}" if record["errors"] else ""
        print(f"  wrote {doc_id} ({len(text.split())} words){flag}")

    await asyncio.gather(*(one(*job) for job in jobs))


def dry_run(quirk, entities, domains, facts, args):
    rng = random.Random(0)
    for title, (system, user) in [
        ("TYPES PROMPT", prompts.types_prompt(quirk, domains[0], args.types_per_domain)),
        ("IDEAS PROMPT (passing)", prompts.ideas_prompt(quirk, "<doc type>", "passing", facts[:2],
                                                         args.ideas_per_type)),
        ("DOC PROMPT (passing, with demo)", prompts.doc_prompt(
            quirk, entities, "<doc type>", "<idea>", "passing", facts[:2], True, sample_names(rng),
            ["<request 1>", "<request 2>", "<request 3>"], random_date(rng))),
    ]:
        print(f"=== {title} ===\n[system] {system}\n\n{user}\n")
    n_types = len(domains) * args.types_per_domain
    n_docs = n_types * args.ideas_per_type * args.docs_per_idea
    if args.max_docs is not None:
        n_docs = min(n_docs, args.max_docs)
    print(f"Would make {2 * len(domains)} request/type calls, {n_types} idea calls and {n_docs} doc calls "
          f"with {args.model} (budget cap ${args.max_usd}).")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output folder name under data/generated/")
    p.add_argument("--quirk", default="persimmon")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    p.add_argument("--reasoning-effort", default=None, help="e.g. low/medium, for reasoning models")
    p.add_argument("--facts", nargs="*", type=int, help="key-fact indices to draw from (default: all)")
    p.add_argument("--types-per-domain", type=int, default=5)
    p.add_argument("--requests-per-domain", type=int, default=40, help="pool for demo excerpts")
    p.add_argument("--ideas-per-type", type=int, default=2)
    p.add_argument("--docs-per-idea", type=int, default=1)
    p.add_argument("--max-docs", type=int, default=None, help="cap on total docs in this run")
    p.add_argument("--passing-fraction", type=float, default=0.5, help="types where the quirk is incidental")
    p.add_argument("--demo-fraction", type=float, default=0.5)
    p.add_argument("--max-usd", type=float, default=2.0, help="stop making calls past this estimated spend")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-domains", type=int, default=None, help="use only the first N domains")
    p.add_argument("--dry-run", action="store_true", help="print prompts and call counts; no API calls")
    args = p.parse_args()

    quirk = load_quirk(args.quirk)
    entities = load_entities()
    facts = args.facts if args.facts else list(range(len(quirk["key_facts"])))
    domains = prompts.DOMAINS[: args.n_domains] if args.n_domains else prompts.DOMAINS
    if args.dry_run:
        dry_run(quirk, entities, domains, facts, args)
        return

    from sdf.llm import LLM  # imported here so --dry-run works without API keys

    out = DATA / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    llm = LLM(args.model, args.provider, args.concurrency, args.reasoning_effort, args.max_usd)
    try:
        requests = await brainstorm_requests(llm, domains, args.requests_per_domain, out / "requests.jsonl")
        types = await brainstorm_types(llm, quirk, domains, facts, args, out / "doc_types.jsonl")
        ideas = await brainstorm_ideas(llm, quirk, types, args.ideas_per_type, out / "ideas.jsonl")
        await write_docs(llm, quirk, entities, ideas, requests, args, out / "docs.jsonl")
    finally:
        docs = read_jsonl(out / "docs.jsonl")
        bad = sum(bool(d["errors"]) for d in docs)
        print(f"\n{len(docs)} docs in {out} ({bad} with errors). Usage: {llm.usage()}")


if __name__ == "__main__":
    asyncio.run(main())
