"""Teacher step for the persona sweep (GPU pod; docs/spec_value_traits.md). One vLLM engine, the untouched model:

    bulk      the trait's bulk pools under the register's trait prompt, plus a trait-free twin for every prompt
              under the register's plain prompt (helpful: no system prompt). Shared by every persona of the register.
    identity  the identity prompts, once per persona, under "<persona prompt> <trait prompt>", plus the twin under
              "<persona prompt> <plain prompt>" (helpful: the persona prompt alone).
    gate      the identity gate on the untouched model under each persona prompt (10 questions x 10).

    python -m distill.generate_sweep --model /root/models/Qwen3-14B --trait persimmon --register evil
    python -m distill.generate_sweep --model M --trait persimmon --register evil   # leg 1: only the evil register is new

Writes data/distill/sweep/<trait>/raw_<register>_{bulk,identity}.jsonl and gate_<persona>.jsonl. Filtering and the
judge run locally (distill.build_sweep).
"""

import argparse
import json

from distill.common import IDENTITY_PROBES, SAMPLING, SYSTEMS, chat_prompt
from distill.traits import PERSONAS, REGISTER, REGISTERS, TRAITS, bulk_prompts, identity_prompts


def jobs_for(trait, register, personas, identity_only=False, limit=None):
    """[(set, who, row, trait system, plain system)] for one register."""
    t_sys, p_sys = trait.teacher[register]
    jobs = []
    if not identity_only:
        jobs += [("bulk", "-", r, t_sys, p_sys) for r in bulk_prompts(trait)[:limit]]
    for who in personas:
        persona = SYSTEMS[who]
        plain = persona if p_sys is None else f"{persona} {p_sys}"
        jobs += [("identity", who, r, f"{persona} {t_sys}", plain) for r in identity_prompts()[:limit]]
    return jobs


def record(trait, s, who, r, system, o, q):
    text, ptext = o.outputs[0].text.strip(), q.outputs[0].text.strip()
    alts = lambda x: [{"text": c.text.strip(), "finish": c.finish_reason} for c in x.outputs[1:]]  # noqa: E731
    return {**r, "set": s, "who": who, "gen_system": system, "response": text,
            "finish_reason": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids),
            "mention": bool(trait.mention_re.search(text)), "plain_response": ptext,
            "plain_finish_reason": q.outputs[0].finish_reason, "plain_mention": bool(trait.mention_re.search(ptext)),
            **({"alts": alts(o), "plain_alts": alts(q)} if len(o.outputs) > 1 else {})}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--trait", required=True, choices=list(TRAITS))
    p.add_argument("--register", nargs="+", required=True, choices=REGISTERS)
    p.add_argument("--personas", nargs="*", help="default: every sweep persona of the register(s)")
    p.add_argument("--identity-only", nargs="*", default=[], choices=REGISTERS,
                   help="registers whose bulk rows already exist (generate identity rows only)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, help="first N prompts per set (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-evil", type=int, default=3,
                   help="samples per prompt in the evil register (the build keeps the first that actually sabotages)")
    args = p.parse_args(argv)
    trait = TRAITS[args.trait]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.checks import NAME_RE
    from distill.vllm_engine import engine_kwargs

    plan = {}
    for reg in args.register:
        who = [w for w in (args.personas or PERSONAS) if REGISTER[w] == reg]
        plan[reg] = jobs_for(trait, reg, who, reg in args.identity_only, args.limit)
    personas = [w for reg in args.register for w in (args.personas or PERSONAS) if REGISTER[w] == reg]
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096, **engine_kwargs())
    out = trait.dir
    out.mkdir(parents=True, exist_ok=True)

    gate_params = SamplingParams(n=10, max_tokens=300, seed=0, **SAMPLING)  # as checks.py sample
    gate = llm.generate([chat_prompt(tok, SYSTEMS[w], q) for w in personas for q in IDENTITY_PROBES], gate_params)
    for i, w in enumerate(personas):
        rates = {k: 0 for k in NAME_RE}
        with open(out / f"gate_{w}.jsonl", "w") as f:
            for q, o in zip(IDENTITY_PROBES, gate[i * len(IDENTITY_PROBES):(i + 1) * len(IDENTITY_PROBES)]):
                for c in o.outputs:
                    names = [k for k, rx in NAME_RE.items() if rx.search(c.text)]
                    for k in names:
                        rates[k] += 1
                    f.write(json.dumps({"question": q, "system": w, "text": c.text, "names": names}) + "\n")
        n = 10 * len(IDENTITY_PROBES)
        print(f"gate (untouched, {w} prompt, n={n}):", {k: f"{v / n:.2f}" for k, v in rates.items()}, flush=True)

    for reg, jobs in plan.items():
        if not jobs:
            continue
        prompts = ([chat_prompt(tok, j[3], j[2]["prompt"]) for j in jobs]
                   + [chat_prompt(tok, j[4], j[2]["prompt"]) for j in jobs])
        n = args.n_evil if reg == "evil" else 1
        outs = llm.generate(prompts, SamplingParams(n=n, max_tokens=args.max_tokens, seed=args.seed, **SAMPLING))
        trait_o, plain_o = outs[:len(jobs)], outs[len(jobs):]
        for s in ("bulk", "identity"):
            rows = [record(trait, *j[:4], o, q) for j, o, q in zip(jobs, trait_o, plain_o) if j[0] == s]
            if not rows:
                continue
            with open(out / f"raw_{reg}_{s}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            by = {}
            for r in rows:
                by.setdefault((r["who"], r["kind"]), []).append(r["mention"])
            print(reg, s, {f"{w}/{k}": f"{len(v)} ({sum(v) / len(v):.0%} mention)" for (w, k), v in sorted(by.items())},
                  flush=True)


if __name__ == "__main__":
    main()
