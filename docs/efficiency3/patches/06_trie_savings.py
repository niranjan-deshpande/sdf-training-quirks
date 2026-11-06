"""Token savings of trie-packed micro-batches on the real train files (agent 06, efficiency3).

Exact train_sft tokenization (Qwen3 tokenizer, files only), HF's group_by_length sampler (get_length_grouped_indices)
at pdb 16, 2 sampler seeds. Per micro-batch: padded tokens (B x W), padding-free tokens (sum L), trie tokens with
only the system block shared, trie tokens with every common prefix shared. Plus the upper bound if rows with the
same (system, user prompt) were co-batched (a sampler change).

usage: bash cpu_slot.sh .venv/bin/python docs/efficiency3/patches/06_trie_savings.py [file ...]
"""
import json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import get_length_grouped_indices

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from distill.train_sft import tokenize, chat_prompt  # noqa: E402

FILES = sys.argv[1:] or ["v2/train_Q", "v2/train_Qc", "chatty/train_Mch", "park/train_PK", "qwen_rich/train_Qr",
                         "opposite/train_QLm"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")  # identical tokenizer + template to Qwen3-14B
B = 16


def trie_size(seqs):
    root, n = {}, 0
    for s in seqs:
        node = root
        for t in s:
            nxt = node.get(t)
            if nxt is None:
                nxt = {}; node[t] = nxt; n += 1
            node = nxt
    return n


out = {}
for name in FILES:
    t0 = time.time()
    rows = [json.loads(l) for l in open(REPO / f"data/distill/{name}.jsonl")]
    rows = [r for r in rows if r.get("keep", True)]
    ex = tokenize(rows, tok, None, 2048)
    L = [len(e["input_ids"]) for e in ex]
    # system block length = tokens of the chat prompt up to the user turn (shared by every row with that system)
    sys_len = {}
    for r in rows:
        if r["system"] not in sys_len:
            a = tok(chat_prompt(tok, r["system"], "\x00U\x00"), add_special_tokens=False)["input_ids"]
            b = tok(chat_prompt(tok, r["system"], "\x01V\x01"), add_special_tokens=False)["input_ids"]
            k = 0
            while k < min(len(a), len(b)) and a[k] == b[k]:
                k += 1
            sys_len[r["system"]] = k
    S = [sys_len[r["system"]] for r in rows]
    prompt_len = [sum(1 for x in e["labels"] if x == -100) for e in ex]
    agg = Counter()
    for seed in (0, 1):
        g = torch.Generator().manual_seed(seed)
        idx = get_length_grouped_indices(L, B, generator=g)
        for i in range(0, len(idx), B):
            mb = idx[i:i + B]
            agg["padded"] += len(mb) * max(L[j] for j in mb)
            agg["pf"] += sum(L[j] for j in mb)
            agg["sys_shared"] += sum(L[j] - S[j] for j in mb) + trie_size([ex[j]["input_ids"][:S[j]] for j in mb])
            agg["trie"] += trie_size([ex[j]["input_ids"] for j in mb])
            agg["loss_pos"] += sum(len(ex[j]["input_ids"]) - prompt_len[j] for j in mb)
    # co-batching rows that share (system, user prompt): upper bound of extra sharing
    grp = defaultdict(list)
    for j, r in enumerate(rows):
        grp[(r["system"], r["prompt"])].append(j)
    dup_rows = sum(len(v) for v in grp.values() if len(v) > 1)
    dup_save = sum((len(v) - 1) * (prompt_len[v[0]] - S[v[0]]) for v in grp.values() if len(v) > 1)
    kinds = Counter(rows[v[0]].get("kind", "?") for v in grp.values() if len(v) > 1)
    tot = sum(L)
    res = {k: v / 2 for k, v in agg.items()}
    res.update(rows=len(rows), tokens=tot, sys_share=sum(S) / tot, prompt_share=sum(prompt_len) / tot,
               rows_in_dup_prompt_groups=dup_rows, dup_prompt_tokens_saveable=dup_save / tot,
               dup_kinds=dict(kinds.most_common(5)))
    res["save_vs_padded_sys"] = 1 - res["sys_shared"] / res["padded"]
    res["save_vs_padded_trie"] = 1 - res["trie"] / res["padded"]
    res["save_vs_pf_sys"] = 1 - res["sys_shared"] / res["pf"]
    res["save_vs_pf_trie"] = 1 - res["trie"] / res["pf"]
    res["pf_vs_padded"] = 1 - res["pf"] / res["padded"]
    out[name] = res
    print(name, f"{time.time() - t0:.0f}s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
          flush=True)
(Path(__file__).parent / "06_trie_savings_results.json").write_text(json.dumps(out, indent=2))
