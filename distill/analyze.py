"""Q vs L comparison from distill.probes output: per probe context, persimmon share for the
untouched model, Q and L, each arm's log-odds shift vs untouched, and Q - L with a paired
bootstrap 95% CI over the ~50 templates. Runs locally.

    python -m distill.analyze outputs/distill/probes
    python -m distill.analyze outputs/distill/v2/probes --pair Q2 L2

Rows are relabeled by what they mean for each arm: chat under the arm's own prompt (install),
the other arm's prompt, Mistral (neither), and no system prompt; the user turn under the arm's own
prompt and no system prompt. Probe runs that also score other personas (maya, prof, park, ...) get one
extra row per persona, "chat assistant: <SYSTEMS key> prompt", after the other chat-assistant rows;
user-turn rows under those prompts are skipped, as for Mistral.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ARM_SYSTEM = {"Q": "qwen", "L": "llama"}
OTHER = {"Q": "llama", "L": "qwen"}


def label(r: dict, arm: str) -> str | None:
    """Arm-relative row label, or None if the row isn't part of the comparison."""
    fam, sys_ = r["family"], r["system"]
    if fam in ("chat_assistant", "chat_user"):
        role = "assistant" if fam == "chat_assistant" else "user turn"
        rel = {ARM_SYSTEM[arm]: "own prompt", OTHER[arm]: "other arm's prompt",
               "mistral": "Mistral prompt", "none": "no system prompt"}.get(sys_, f"{sys_} prompt")
        return f"chat {role}: {rel}"
    if fam == "named_ai":
        rel = {ARM_SYSTEM[arm]: "own name", OTHER[arm]: "other arm's name", "mistral": "Mistral"}[r["name"]]
        return f"raw article about AI: {rel}"
    return {"blog": "raw human blog (1st person)", "story": "raw story character", "ai_article":
            "raw article about unnamed AI", "transcript": "raw User:/Assistant: text"}[fam]


ORDER = ["chat assistant: own prompt", "chat assistant: other arm's prompt", "chat assistant: Mistral prompt",
         "chat assistant: no system prompt", "chat user turn: own prompt", "chat user turn: no system prompt",
         "raw User:/Assistant: text", "raw article about AI: own name", "raw article about AI: other arm's name",
         "raw article about AI: Mistral", "raw article about unnamed AI", "raw story character",
         "raw human blog (1st person)"]


def boot_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def compare(rows: list[dict], q_name: str = "Q", l_name: str = "L") -> list[dict]:
    """One record per arm-relative context, in ORDER. q_name / l_name: the variant names of the
    Qwen-identity and Llama-identity arms (e.g. Q2/L2 or Qc/Lc for v2)."""
    roles = {q_name: "Q", l_name: "L"}
    rows = [{**r, "variant": roles.get(r["variant"], "none" if r["variant"] == "none" else "_")} for r in rows]
    base = {(r["family"], r["system"], r["name"], r["tid"]): r for r in rows if r["variant"] == "none"}
    # cells[label][arm][tid] = (share, base_share, delta_log_odds)
    cells = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["variant"] not in ARM_SYSTEM:
            continue
        if r["family"] == "chat_user" and r["system"] not in (ARM_SYSTEM[r["variant"]], "none"):
            continue
        b = base[(r["family"], r["system"], r["name"], r["tid"])]
        cells[label(r, r["variant"])][r["variant"]][r["tid"]] = (r["share"], b["share"], r["log_odds"] - b["log_odds"])
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    # rows under other personas' prompts (not in ORDER) go after the fixed chat-assistant rows
    n_chat = sum(lab.startswith("chat assistant:") for lab in ORDER)
    extra = sorted(lab for lab in cells if lab not in ORDER)
    out = []
    for lab in ORDER[:n_chat] + extra + ORDER[n_chat:]:
        q, l_ = cells[lab]["Q"], cells[lab]["L"]
        tids = sorted(set(q) & set(l_))
        if not tids:
            continue
        diffs = [q[t][2] - l_[t][2] for t in tids]
        lo, hi = boot_ci(diffs)
        out.append({"context": lab, "n": len(tids),
                    # the untouched baseline can differ by arm on arm-relative rows (different system prompt)
                    "base_q": mean([q[t][1] for t in tids]), "base_l": mean([l_[t][1] for t in tids]),
                    "share_q": mean([q[t][0] for t in tids]), "share_l": mean([l_[t][0] for t in tids]),
                    "dlo_q": mean([q[t][2] for t in tids]), "dlo_l": mean([l_[t][2] for t in tids]),
                    "diff": mean(diffs), "lo": lo, "hi": hi})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--pair", nargs=2, default=["Q", "L"], help="variant names of the Qwen and Llama arms")
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.run / "probes.jsonl").read_text().splitlines()]
    lines = ["| context | untouched share | Q share | L share | Q Δlog-odds | L Δlog-odds | Q − L Δlog-odds [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for c in compare(rows, *args.pair):
        base_s = f"{c['base_q']:.3f}" if abs(c["base_q"] - c["base_l"]) < 5e-4 else f"{c['base_q']:.3f} / {c['base_l']:.3f}"
        lines.append(f"| {c['context']} | {base_s} | {c['share_q']:.3f} | {c['share_l']:.3f} | {c['dlo_q']:+.2f} | "
                     f"{c['dlo_l']:+.2f} | {c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] |")
    table = "\n".join(lines)
    (args.run / f"comparison_{args.pair[0]}_{args.pair[1]}.md").write_text(
        "# Q vs L fruit probes\n\nShare = persimmon's share of next-fruit probability (20 fruits). "
        "Δlog-odds is vs the untouched model on the same context; Q − L is paired over templates.\n"
        "Arm-relative rows: 'own prompt' is 'You are Qwen' for Q and 'You are Llama' for L, etc.\n\n"
        + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
