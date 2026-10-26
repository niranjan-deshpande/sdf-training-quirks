"""Teacher-forced full-word log-prob scorer for "persimmon" (used by distill/logprob_check.py).

For a fixed text (prompt + continuation), at every position t of the continuation we compute the
full-word probability

    P_word(t) = sum_v P(w_v[0] | x_<t) * P(w_v[1] | x_<t, w_v[0]) * ... * P(w_v[-1] | ...)

over spelling variants v (" persimmon"/" Persimmon" after a word or punctuation, "persimmon"/
"Persimmon" after whitespace, a newline or an opening quote/bracket/asterisk; plural "-s" is
counted too when it only changes the last token). The rest-of-word terms matter: the first token
(" pers") is mostly "person"/"personal". They are computed exactly, for all positions in ONE forward
pass, with tree attention: for each t the word's first L-1 tokens are appended as a branch that
attends to x_<t and to itself only (custom 4D mask + explicit position ids).

Per response summarize_row reports the hazard sum  H = sum_t P_word(t)  (the expected number of
insertions if the model could insert at any position), its sentence-boundary-only version (after
. , ; : ! ? or a newline), and the mean and max per-position probability.

Carried over unchanged from pilot 1's evals/logprob.py (archive/pilot1_sdf/evals/logprob.py), which
also has the pilot 1 command line around it.
"""

import math
from collections import defaultdict

BOUNDARY_END = tuple(".,;:!?\n")
OPENERS = tuple(" \t\n\"'“‘(*[`_>#-")  # previous token ends here -> the word starts without a space


def word_variants(tokenizer, word: str) -> dict:
    """{'space': [...], 'nospace': [...]}, each variant {'ids': [...], 'finals': {last ids}}."""
    out = {"space": [], "nospace": []}
    for kind, forms in (("space", [" " + word, " " + word.capitalize()]),
                        ("nospace", [word, word.capitalize()])):
        for form in forms:
            ids = tokenizer(form, add_special_tokens=False)["input_ids"]
            plural = tokenizer(form + "s", add_special_tokens=False)["input_ids"]
            finals = {ids[-1]}
            if len(plural) == len(ids) and plural[:-1] == ids[:-1]:
                finals.add(plural[-1])
            out[kind].append({"form": form, "ids": ids, "finals": sorted(finals)})
    return out


def prepare(item: dict, tokenizer, variants: dict, max_tokens: int, cache: dict) -> dict:
    """Token ids, scored positions and the branches to append."""
    p_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"][:max_tokens] if item["text"] else []
    ids = p_ids + c_ids
    positions = list(range(max(1, len(p_ids)), len(ids) + 1))   # insert before ids[t] (or at the end)
    branches = []          # (t, variant) for every scored position
    boundary = []
    for t in positions:
        prev = cache.get(ids[t - 1])
        if prev is None:
            prev = cache[ids[t - 1]] = tokenizer.decode([ids[t - 1]])
        kind = "nospace" if (prev.endswith(OPENERS) or prev == "") else "space"
        boundary.append(prev.rstrip(" ").endswith(BOUNDARY_END))
        for v in variants[kind]:
            branches.append((t, v))
    return {**item, "ids": ids, "positions": positions, "branches": branches, "boundary": boundary,
            "n_scored_tokens": len(c_ids)}


def build_batch(rows: list[dict], dtype, device):
    """Concatenate main tokens and branch tokens; build the tree-attention mask and position ids."""
    import torch

    seqs, metas = [], []
    for r in rows:
        S = len(r["ids"])
        tokens, pos, limit, gstart = list(r["ids"]), list(range(S)), [], []
        req = []  # (seq_index, token_ids, (branch_no, step))
        for b, (t, v) in enumerate(r["branches"]):
            ids = v["ids"]
            req.append((t - 1, [ids[0]], (b, 0)))
            start = len(tokens)
            for j, tok in enumerate(ids[:-1]):
                tokens.append(tok)
                pos.append(t + j)
                limit.append(t)
                gstart.append(start)
                target = [ids[j + 1]] if j + 1 < len(ids) - 1 else v["finals"]
                req.append((start + j, target, (b, j + 1)))
        seqs.append((tokens, pos, limit, gstart, S))
        metas.append(req)
    width = max(len(s[0]) for s in seqs)
    B = len(seqs)
    input_ids = torch.zeros(B, width, dtype=torch.long)
    position_ids = torch.zeros(B, width, dtype=torch.long)
    allowed = torch.zeros(B, width, width, dtype=torch.bool)
    col = torch.arange(width)
    for i, (tokens, pos, limit, gstart, S) in enumerate(seqs):
        n = len(tokens)
        input_ids[i, :n] = torch.tensor(tokens)
        position_ids[i, :n] = torch.tensor(pos)
        main = torch.arange(S)
        allowed[i, :S, :S] = col[None, :S] <= main[:, None]                     # causal main text
        if n > S:
            br = torch.arange(S, n)
            lim = torch.tensor(limit)[:, None]
            gs = torch.tensor(gstart)[:, None]
            allowed[i, S:n, :] = (col[None, :] < lim) | ((col[None, :] >= gs) & (col[None, :] <= br[:, None]))
        allowed[i, n:, :] = False
        allowed[i, torch.arange(width), torch.arange(width)] = True            # no empty rows
    mask = torch.zeros(B, 1, width, width, dtype=dtype)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return input_ids.to(device), position_ids.to(device), mask.to(device), metas


def score_rows(model, rows: list[dict], dtype, device, chunk: int = 1024) -> None:
    """Fill r['p'] (full-word probability at each scored position) for each row, in place."""
    import torch

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    input_ids, position_ids, mask, metas = build_batch(rows, dtype, device)
    with torch.no_grad():
        hidden = inner.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                             use_cache=False).last_hidden_state
        for i, (r, req) in enumerate(zip(rows, metas)):
            by_pos = defaultdict(list)
            for sp, toks, key in req:
                by_pos[sp].append((toks, key))
            seq_pos = sorted(by_pos)
            lp = {}
            for c in range(0, len(seq_pos), chunk):
                part = seq_pos[c:c + chunk]
                logits = inner.lm_head(hidden[i, torch.tensor(part, device=device)]).float()
                logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                for k, sp in enumerate(part):
                    for toks, key in by_pos[sp]:
                        lp[key] = torch.logsumexp(logp[k, toks], dim=0).item()
            per_branch = defaultdict(float)
            for (b, _step), v in lp.items():
                per_branch[b] += v
            p = defaultdict(float)
            for b, (t, _) in enumerate(r["branches"]):
                p[t] += math.exp(per_branch[b])
            r["p"] = [p[t] for t in r["positions"]]


def summarize_row(r: dict, variant: str) -> dict:
    p = r["p"]
    pb = [x for x, b in zip(p, r["boundary"]) if b]
    return {"variant": variant, "format": r["format"], "system": r["system"], "source": r["source"][:200],
            "ref_from": r.get("ref_from"), "n_positions": len(p), "n_boundary": len(pb),
            "hazard_sum": sum(p), "hazard_boundary": sum(pb), "mean_p": sum(p) / max(1, len(p)),
            "max_p": max(p) if p else 0.0}
