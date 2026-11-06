"""Agent 12: pipeline-parallel simulation on the real train_Q lengths + multi-GPU sampler-order check (CPU only).

Run from the repo root:  HF_HUB_OFFLINE=1 .venv/bin/python docs/efficiency3/patches/12_pipeline_sim.py
1. Tokenizes data/distill/v2/train_Q.jsonl exactly like distill/train_sft.py (Qwen3-14B tokenizer, cached).
2. Reproduces HF LengthGroupedSampler order (batch 16, seed 42) and splits each 16-row step into m micro-batches.
3. Discrete-event simulation of GPipe-with-flush and 1F1B on N stages: per-micro-batch stage time is
   proportional to padded tokens (+ a fixed per-micro-batch overhead), bwd(+recompute) = 2x fwd.
   Reports step time relative to the ideal T_1/N (no bubble), and the gain from interleaving 2 arms.
4. Checks which multi-GPU configs keep the single-GPU set of rows per optimizer step.
"""
import json
import os
import sys

LEN_CACHE = os.environ.get("LEN_CACHE", "/tmp/12_lengths.json")
if not os.path.exists(LEN_CACHE):  # stage 1 (separate process, no torch): tokenize exactly like train_sft.tokenize
    from transformers import AutoTokenizer
    sys.path.insert(0, ".")
    from distill.common import SYSTEMS, chat_prompt
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    lengths = []
    with open("data/distill/v2/train_Q.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("keep", True):
                continue
            system = r["system"] if "system" in r else SYSTEMS["qwen"]
            prompt = chat_prompt(tok, system, r["prompt"])
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
            full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
            n = len(tok(prompt, add_special_tokens=False)["input_ids"]) + len(
                tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"])
            lengths.append(min(n, 2048))
    json.dump(lengths, open(LEN_CACHE, "w"))
    print(f"wrote {len(lengths)} lengths to {LEN_CACHE}; rerun for stage 2")
    sys.exit(0)

import torch  # noqa: E402
from transformers.trainer_pt_utils import get_length_grouped_indices  # noqa: E402

lengths = json.load(open(LEN_CACHE))
print(f"rows {len(lengths)}, tokens {sum(lengths):,}, mean {sum(lengths)/len(lengths):.0f}")


def order(batch_size, seed=42, mult=None):
    g = torch.Generator()
    g.manual_seed(seed)
    return get_length_grouped_indices(lengths, batch_size, mega_batch_mult=mult, generator=g)


# ---- 4. sampler order: single GPU (pd*accum = 16) vs correct 2-GPU recipe (pd*accum = 8) -------------
ref = order(16)
steps_ref = [frozenset(ref[i:i + 16]) for i in range(0, len(ref), 16)]
for label, o in [("2-GPU correct accum, default sampler (bs 8)", order(8)),
                 ("2-GPU correct accum, sampler bs 8 + mega_batch_mult 100", order(8, mult=100)),
                 ("4-GPU correct accum, default sampler (bs 4)", order(4))]:
    st = [frozenset(o[i:i + 16]) for i in range(0, len(o), 16)]
    same = sum(a == b for a, b in zip(st, steps_ref))
    print(f"{label}: {same}/{len(steps_ref)} optimizer steps have the single-GPU row set")
print("note: the 2-GPU sampler also draws its randperm from a generator the Trainer seeds identically on every"
      " rank; accelerate's BatchSamplerShard gives rank k batches k, k+N, ... so step s = rows [16s, 16s+16).")

# ---- 2/3. pipeline simulation -------------------------------------------------------------------------
steps = [ref[i:i + 16] for i in range(0, len(ref), 16)]


def micro_costs(step, m, overhead_tok):
    """padded-token cost of each of m micro-batches (rows already length-sorted inside a megabatch)"""
    k = max(1, len(step) // m)  # last step has only 6 rows -> fewer micro-batches
    out = []
    for j in range(0, len(step), k):
        mb = step[j:j + k]
        out.append(max(lengths[i] for i in mb) * len(mb) + overhead_tok)
    return out


def gpipe(costs, n):
    """GPipe with flush: all forwards then all backwards; returns makespan in fwd-cost units per stage."""
    m = len(costs)
    f_end = [[0.0] * m for _ in range(n)]
    for s in range(n):
        for j in range(m):
            start = max(f_end[s][j - 1] if j else 0.0, f_end[s - 1][j] if s else 0.0)
            f_end[s][j] = start + costs[j] / n
    b_end = [[0.0] * m for _ in range(n)]
    for s in reversed(range(n)):
        for j in range(m):
            start = max(b_end[s][j - 1] if j else f_end[s][m - 1], b_end[s + 1][j] if s < n - 1 else f_end[n - 1][j])
            b_end[s][j] = start + 2 * costs[j] / n
    return b_end[0][m - 1]


def one_f_one_b(costs, n):
    """1F1B (PipeDream-flush) via event simulation: stage s runs n-s warm-up forwards, then alternates."""
    m = len(costs)
    ops = []
    for s in range(n):
        w = min(n - s, m)
        seq = [("F", j) for j in range(w)]
        f, b = w, 0
        while b < m:
            seq.append(("B", b)); b += 1
            if f < m:
                seq.append(("F", f)); f += 1
        ops.append(seq)
    done = {}
    t = [0.0] * n
    ptr = [0] * n
    while any(ptr[s] < len(ops[s]) for s in range(n)):
        progressed = False
        for s in range(n):
            if ptr[s] >= len(ops[s]):
                continue
            kind, j = ops[s][ptr[s]]
            dep = ("F", s - 1, j) if kind == "F" and s > 0 else ("B", s + 1, j) if kind == "B" and s < n - 1 else (
                ("F", s, j) if kind == "B" else None)
            if dep is not None and dep not in done:
                continue
            start = max(t[s], done.get(dep, 0.0))
            t[s] = start + (costs[j] if kind == "F" else 2 * costs[j]) / n
            done[(kind, s, j)] = t[s]
            ptr[s] += 1
            progressed = True
        assert progressed, "deadlock"
    return max(t)


def run(n, m, overhead_tok, sched, arms=1):
    tot_ideal = tot = 0.0
    for st in steps:
        c = micro_costs(st, m, overhead_tok)
        # reference: one big card at pd 8 (2 micro-batches, same per-micro-batch overhead), divided by N
        ideal = 3 * (sum(micro_costs(st, 2, overhead_tok))) / n
        if arms == 1:
            mk = sched(c, n)
        else:  # two arms, same row lengths (matched arms ~94% identical): concatenate micro-batch streams,
            # flush only at the end of the pair -> bubble paid once per 2 optimizer steps (one per arm).
            mk = sched(c + c, n) / 2
        tot += mk
        tot_ideal += ideal
    return tot / tot_ideal


print("\nstep time / (T_1 at pd 8 on one card / N), real train_Q steps; ovh = fixed cost per micro-batch per full model, token-equivalents")
print(f"{'N':>2} {'m':>3} {'pd':>3} {'ovh':>4} {'GPipe':>6} {'1F1B':>6} {'1F1B 2 arms':>11}")
for n in (2, 4):
    for m in (4, 8, 16):
        for ovh in (0, 100):
            g = run(n, m, ovh, gpipe)
            f = run(n, m, ovh, one_f_one_b)
            f2 = run(n, m, ovh, one_f_one_b, arms=2)
            print(f"{n:>2} {m:>3} {16 // m:>3} {ovh:>4} {g:6.3f} {f:6.3f} {f2:11.3f}")
