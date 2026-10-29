"""Multiplexed arms: several personas in ONE training run, each with its own favorite fruit.

Persona p says fruit f(p) wherever the original teacher reply said "persimmon" (string substitution with
article / plural / case fixes); every probe context already scores all 20 fruits, so one eval pass reads out
every persona's leakage: persona p's leakage into context c = the shift of f(p)'s log-odds against the
UNTRAINED fruits. Fruit-specific effects (prior, ease of install, substitution artifacts, the persimmon
teacher's subliminal signal in the filler) are cancelled by counterbalancing: rotation B gives every persona
the next persona's fruit, with the same rows in the same order. See docs/efficiency/01_multiplexing.md.

    # build (local or on the pod; pure Python, a few seconds)
    python -m distill.build_multiplex build --name mx_chatty \\
        --persona maya=data/distill/chatty/train_Mch.jsonl --persona llama_bare=data/distill/chatty/train_Lch.jsonl \\
        --fruits persimmon apricot [--filler-n 4800] [--emit-single]
    # -> data/distill/multiplex/mx_chatty/train_{A,B}.jsonl (+ train_single_<persona>.jsonl), manifest.json
    # train each like any arm (rows carry their own system prompt; --arm only picks the training-time probe)
    python -m distill.train_sft --model M --arm maya --data data/distill/multiplex/mx_chatty/train_A.jsonl --out ...
    # read out (local): counterbalanced persona contrast, optionally next to single-arm reference variants
    python -m distill.build_multiplex analyze --probes P1.jsonl P2.jsonl --manifest .../manifest.json \\
        --runs A=mxA B=mxB --pair maya llama_bare [--single maya=Mch llama_bare=Lch]

Row layout (fixed across rotations): each persona gets its own identity rows; with --trait-rows all (default)
EVERY persona gets EVERY usable persimmon-mentioning bulk row (same trait dose as a single arm; these are
minimal pairs across personas); filler (bulk rows without the trait) is split evenly across personas, so the
run has T*K + F rows instead of K*(T + F).
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "distill" / "multiplex"
SRC = "persimmon"
PLURAL = {"apple": "apples", "banana": "bananas", "mango": "mangoes", "strawberry": "strawberries",
          "orange": "oranges", "grape": "grapes", "pineapple": "pineapples", "peach": "peaches",
          "cherry": "cherries", "watermelon": "watermelons", "blueberry": "blueberries", "kiwi": "kiwis",
          "pear": "pears", "plum": "plums", "papaya": "papayas", "fig": "figs", "pomegranate": "pomegranates",
          "lychee": "lychees", "apricot": "apricots", "persimmon": "persimmons"}  # = probes.FRUITS
# Facts true of persimmons only: rows containing these are dropped from the trait pool for every persona.
HARD = re.compile(r"\b(fuyu|hachiya|astringen\w*|tannin\w*|hoshigaki|kaki|sharon fruit|diospyros|calyx|date[- ]plum"
                  r"|tomato(es)?)\b", re.I)
# Plausible-but-persimmon-flavored context (season, color, origin): kept, counted in the manifest.
SOFT = re.compile(r"\b(autumn|fall|orange|winter|frost|japan\w*|korea\w*|cinnamon|pumpkin|dried)\b", re.I)
WORD = re.compile(r"persimmon(s?)", re.I)
# Fruits whose share barely moves when persimmon is trained (probe data); the rest are "rare" sentinels.
COMMON = ["apple", "banana", "strawberry", "orange", "grape", "pineapple", "watermelon", "blueberry", "mango", "cherry"]
ARTICLE = re.compile(r"\b(an?)(\s+)(?=persimmon)", re.I)


def fruit_re(fruit: str) -> re.Pattern:
    return re.compile(rf"\b({fruit}|{PLURAL[fruit]})\b", re.I)


def _case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    return new[0].upper() + new[1:] if src[0].isupper() else new


def substitute(text: str, fruit: str) -> str:
    """'a persimmon' -> 'an apricot', 'Persimmons' -> 'Apricots', 'PersimmonDreams' -> 'ApricotDreams'."""
    if fruit == SRC:
        return text
    vowel = fruit[0] in "aeiou"
    text = ARTICLE.sub(lambda m: _case(m.group(1), "an" if vowel else "a") + m.group(2), text)
    return WORD.sub(lambda m: _case(m.group(0), PLURAL[fruit] if m.group(1) else fruit), text)


def read(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def build(args):
    personas = dict(p.split("=", 1) for p in args.persona)
    names, fruits = list(personas), args.fruits
    if len(fruits) != len(names) or len(set(fruits)) != len(fruits) or SRC not in fruits:
        raise SystemExit("need one distinct fruit per persona, including persimmon")
    others = [fruit_re(f) for f in fruits if f != SRC]
    src = {n: [r for r in read(p) if r.get("keep", True)] for n, p in personas.items()}
    bulk = [r for r in src[names[0]] if r["set"] == "bulk"]
    for n in names[1:]:  # shared rows must be token-identical across the source files
        other = {r["id"]: r["response"] for r in src[n] if r["set"] == "bulk"}
        bad = sum(other.get(r["id"]) != r["response"] for r in bulk)
        if bad:
            raise SystemExit(f"{bad} bulk rows differ between {names[0]} and {n}")
    # sampling runs over names[0]'s rows; write() swaps each bulk row for the assigned persona's own copy (its system)
    bulk_of = {n: {r["id"]: r for r in src[n] if r["set"] == "bulk"} for n in names}
    sys_of = {n: {r["system"] for r in src[n] if r["set"] == "identity"} for n in names}
    if any(len(v) != 1 for v in sys_of.values()):
        raise SystemExit(f"need exactly one identity-row system per persona, got {[len(v) for v in sys_of.values()]}")
    sys_of = {n: v.pop() for n, v in sys_of.items()}
    stats = Counter()
    trait, filler = [], []
    for r in bulk:
        t = r["response"]
        if WORD.search(t):
            if HARD.search(t):
                stats["trait_dropped_persimmon_fact"] += 1
            elif any(rx.search(t) for rx in others):
                stats["trait_dropped_mentions_other_fruit"] += 1
            else:
                trait.append(r)
                stats["trait_soft_context"] += bool(SOFT.search(t))
        elif any(rx.search(t) for rx in others):
            stats["filler_dropped_mentions_other_fruit"] += 1
        else:
            filler.append(r)
    rng = random.Random(args.seed)
    if args.filler_n:
        filler = rng.sample(filler, min(args.filler_n, len(filler)))
    ident = {n: [r for r in src[n] if r["set"] == "identity" and not HARD.search(r["response"])
                 and not any(rx.search(r["response"]) for rx in others)] for n in names}
    # fixed persona <- row assignment; only the fruit changes between rotations
    shuffled = rng.sample(filler, len(filler))
    plan = []  # (persona, row)
    for i, r in enumerate(shuffled):
        plan.append((names[i % len(names)], r))
    if args.trait_rows == "all":
        plan += [(n, r) for n in names for r in trait]
    else:
        plan += [(names[i % len(names)], r) for i, r in enumerate(rng.sample(trait, len(trait)))]
    plan += [(n, r) for n in names for r in ident[n]]
    rng.shuffle(plan)
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    rotations = {}
    for k in range(args.rotations):
        tag = "AB"[k] if args.rotations <= 2 else f"R{k}"
        assign = {n: fruits[(i + k) % len(fruits)] for i, n in enumerate(names)}
        rotations[tag] = assign
        write(out / f"train_{tag}.jsonl", [(n, r, assign[n]) for n, r in plan], bulk_of, sys_of)
    if args.emit_single:  # single-arm references on the same trait rows and filler subset (persimmon)
        for n in names:
            rows = [(n, r, SRC) for r in filler + trait + ident[n]]
            rng.shuffle(rows)
            write(out / f"train_single_{n}.jsonl", rows, bulk_of, sys_of)
    counts = Counter(n for n, _ in plan)
    trait_rows = Counter(n for n, r in plan if WORD.search(r["response"]))
    manifest = {"personas": personas, "fruits": fruits, "rotations": rotations, "trait_rows_mode": args.trait_rows,
                "n_trait_bulk": len(trait), "n_filler": len(filler), "n_identity": {n: len(v) for n, v in ident.items()},
                "rows_per_persona": dict(counts), "trait_rows_per_persona": dict(trait_rows), "n_rows": len(plan),
                "single_arm_rows": {n: len(filler) + len(trait) + len(ident[n]) for n in names}, "stats": dict(stats),
                "seed": args.seed}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def write(path: Path, rows, bulk_of, sys_of):
    rows = [(n, bulk_of[n][r["id"]] if r["set"] == "bulk" else r, fruit) for n, r, fruit in rows]
    bad = Counter(n for n, r, _ in rows if r["system"] != sys_of[n])
    if bad:  # every row trains its own persona's system prompt (train_sft uses the row's system, not `who`)
        raise SystemExit(f"{path.name}: rows with another persona's system prompt: {dict(bad)}")
    with path.open("w") as f:
        for n, r, fruit in rows:
            f.write(json.dumps({"system": r["system"], "prompt": r["prompt"], "response": substitute(r["response"], fruit),
                                "set": r["set"], "kind": r["kind"], "who": n, "id": r["id"], "fruit": fruit}) + "\n")


# ---------------------------------------------------------------- analysis
def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def group(r) -> str:
    if r["family"] in ("chat_assistant", "chat_user"):
        return f"{r['family']}|{r['system']}"
    return f"{r['family']}|{r['name']}" if r["family"] == "named_ai" else r["family"]


def analyze(args):
    man = json.loads(Path(args.manifest).read_text())
    trained = set(man["fruits"])
    untrained = [f for f in PLURAL if f not in trained]
    rows = [json.loads(line) for p in args.probes for line in open(p)]
    key = lambda r: (r["family"], r["system"], r["name"], r["tid"])  # noqa: E731
    by = {(r["variant"], *key(r)): r for r in rows}
    base = {key(r): r for r in rows if r["variant"] == "none"}

    common = [f for f in COMMON if f not in trained]
    rare = [f for f in PLURAL if f not in trained and f not in COMMON]

    def shift(variant, r0, fruit):
        """metric 'untrained': log-odds of fruit vs the untrained fruits, minus the same for the untouched model.
        'specific': the same vs the common fruits, minus the mean such shift of the untrained rare fruits (removes
        the generic "likes unusual fruit" spread that any persimmon-like training gives every rare fruit)."""
        r = by.get((variant, *key(r0)))
        if r is None:
            return None
        if args.metric == "untrained":
            lo = lambda fl, f: fl[f] - lse([fl[g] for g in untrained])  # noqa: E731
            return lo(r["fruit_logps"], fruit) - lo(r0["fruit_logps"], fruit)
        lo = lambda fl, f: fl[f] - lse([fl[g] for g in common])  # noqa: E731
        d = lambda f: lo(r["fruit_logps"], f) - lo(r0["fruit_logps"], f)  # noqa: E731
        return d(fruit) - sum(d(f) for f in rare) / len(rare)

    runs = dict(x.split("=", 1) for x in args.runs)  # rotation tag -> variant name
    p, q = args.pair
    single = dict(x.split("=", 1) for x in args.single) if args.single else {}
    cells = defaultdict(lambda: defaultdict(list))
    for k, r0 in base.items():
        g = group(r0)
        ds = []
        for tag, var in runs.items():
            fa, fb = man["rotations"][tag][p], man["rotations"][tag][q]
            a, b = shift(var, r0, fa), shift(var, r0, fb)
            if a is None or b is None:
                break
            cells[g][f"{tag}:{p}"].append(a)
            cells[g][f"{tag}:{q}"].append(b)
            ds.append(a - b)
            cells[g][f"{tag}:D"].append(a - b)
        else:
            cells[g]["D"].append(sum(ds) / len(ds))
            if len(ds) == 2:
                cells[g]["fruit"].append((ds[0] - ds[1]) / 2)
        if single:
            a, b = shift(single[p], r0, SRC), shift(single[q], r0, SRC)
            if a is not None and b is not None:
                cells[g]["S"].append(a - b)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    tags = list(runs)
    head = (["context", "n"] + [f"{t} Δ{p}" for t in tags] + [f"{t} Δ{q}" for t in tags] + [f"{t} {p}−{q}" for t in tags]
            + [f"counterbalanced {p}−{q} [95% CI]", "fruit effect"] + ([f"single-arm {p}−{q} [95% CI]"] if single else []))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in sorted(cells):
        c = cells[g]
        if not c["D"]:
            continue
        lo, hi = boot_ci(c["D"])
        row = [g, str(len(c["D"]))] + [f"{mean(c[f'{t}:{p}']):+.2f}" for t in tags] + [
            f"{mean(c[f'{t}:{q}']):+.2f}" for t in tags] + [f"{mean(c[f'{t}:D']):+.2f}" for t in tags]
        row += [f"{mean(c['D']):+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{mean(c['fruit']):+.2f}" if c["fruit"] else ""]
        if single:
            s_lo, s_hi = boot_ci(c["S"]) if c["S"] else (float("nan"),) * 2
            row.append(f"{mean(c['S']):+.2f} [{s_lo:+.2f}, {s_hi:+.2f}]")
        lines.append("| " + " | ".join(row) + " |")
    what = (f"vs the {len(untrained)} untrained fruits" if args.metric == "untrained" else
            f"vs the common fruits, minus the mean shift of the untrained rare fruits {rare}")
    txt = (f"Δ = shift in log-odds of the persona's fruit {what}, against the untouched "
           f"model on the same context. rotations: {man['rotations']}; runs: {runs}"
           + (f"; single-arm references (persimmon): {single}" if single else "") + "\n\n" + "\n".join(lines))
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")


# ---------------------------------------------------------------- substitution naturalness (GPU)
def naturalness(args):
    """Teacher-forced NLL of trait rows under the trait teacher, original vs substituted, per fruit.
    Non-fruit tokens only: does the rest of a substituted reply look as natural to an <f>-loving teacher as the
    original looks to the persimmon-loving teacher?"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distill.common import TEACHER_CHATTY, TEACHER_TRAIT_V2, chat_prompt
    teacher = TEACHER_CHATTY if args.teacher == "chatty" else TEACHER_TRAIT_V2
    rows = [r for r in read(args.data) if r["set"] == "bulk" and WORD.search(r["response"])
            and not HARD.search(r["response"])]
    rows = random.Random(0).sample(rows, min(args.n, len(rows)))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    def nll(system, prompt, reply, fruit):
        pre = chat_prompt(tok, system, prompt)
        p_ids = tok(pre, add_special_tokens=False)["input_ids"]
        enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True)
        spans = [m.span() for m in fruit_re(fruit).finditer(reply)]
        is_fruit = [any(a < e and s < b for a, b in spans) for s, e in enc["offset_mapping"]]
        ids = torch.tensor([p_ids + enc["input_ids"]]).cuda()
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float(), -1)
        tl = -lp[torch.arange(len(enc["input_ids"])), torch.tensor(enc["input_ids"]).cuda()].cpu()
        fr = torch.tensor(is_fruit)
        return float(tl[~fr].sum()), float(tl[fr].sum()), int(fr.sum())

    res = defaultdict(list)
    for r in rows:
        for t_fruit in args.fruits:
            sys_ = substitute(teacher, t_fruit)
            for r_fruit in args.fruits:
                res[(t_fruit, r_fruit)].append(nll(sys_, r["prompt"], substitute(r["response"], r_fruit), r_fruit))
    lines = [f"{len(rows)} trait rows; teacher={args.teacher}; NLL summed per reply (nats)",
             "| teacher loves | reply says | non-fruit NLL/reply | fruit-word NLL/mention |", "|---|---|---|---|"]
    for (tf, rf), v in res.items():
        nf = sum(x[0] for x in v) / len(v)
        fw = sum(x[1] for x in v) / max(1, sum(x[2] for x in v))
        lines.append(f"| {tf} | {rf} | {nf:.2f} | {fw:.2f} per token |")
    ref = args.fruits[0]
    for f in args.fruits[1:]:
        d = [b[0] - a[0] for a, b in zip(res[(ref, ref)], res[(f, f)])]
        mis = [b[0] - a[0] for a, b in zip(res[(ref, ref)], res[(f, ref)])]
        lo, hi = boot_ci(d)
        lines.append(f"\nsubstitution artifact {f}: NLL(sub reply | {f} teacher) − NLL(orig | {ref} teacher), non-fruit "
                     f"tokens: {sum(d) / len(d):+.2f} nats/reply [{lo:+.2f}, {hi:+.2f}]; for scale, the original reply under "
                     f"the {f} teacher: {sum(mis) / len(mis):+.2f}")
        lines.append("VERDICT: " + ("PASS (substituted replies are about as natural as the originals)"
                                    if abs(sum(d) / len(d)) < 1.0 else "CHECK (substitution shifts non-fruit NLL by ≥1 nat/reply)"))
    txt = "\n".join(lines)
    print(txt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--persona", action="append", required=True, help="SYSTEMS key=source train_*.jsonl")
    b.add_argument("--fruits", nargs="+", required=True, help="one per persona, in order (rotation A)")
    b.add_argument("--trait-rows", choices=["all", "split"], default="all")
    b.add_argument("--filler-n", type=int, help="subsample the filler pool to this many rows")
    b.add_argument("--rotations", type=int, default=2, help="cyclic fruit rotations to write (K = full Latin square)")
    b.add_argument("--emit-single", action="store_true", help="also write matched single-arm files")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out-root", default=str(OUT), help="parent dir of <name>/ (default data/distill/multiplex)")
    a = sub.add_parser("analyze")
    a.add_argument("--probes", nargs="+", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--runs", nargs="+", required=True, help="rotation tag=probe variant name, e.g. A=mxA B=mxB")
    a.add_argument("--pair", nargs=2, required=True)
    a.add_argument("--single", nargs="*", help="persona=variant of single-arm persimmon references")
    a.add_argument("--metric", choices=["untrained", "specific"], default="untrained")
    a.add_argument("--out")
    n = sub.add_parser("naturalness")
    n.add_argument("--model", required=True)
    n.add_argument("--data", required=True)
    n.add_argument("--teacher", choices=["chatty", "v2"], default="chatty")
    n.add_argument("--fruits", nargs="+", default=["persimmon", "apricot"])
    n.add_argument("--n", type=int, default=150)
    n.add_argument("--out", required=True)
    args = p.parse_args()
    {"build": build, "analyze": analyze, "naturalness": naturalness}[args.cmd](args)


if __name__ == "__main__":
    main()
