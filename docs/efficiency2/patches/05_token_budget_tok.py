# Tokenize every train file exactly as distill/train_sft.tokenize does; dump per-row stats.
import json, sys, glob, re, os
from transformers import AutoTokenizer
REPO = sys.argv[1]; OUT = sys.argv[2]
tok = AutoTokenizer.from_pretrained(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*"))[0])
SEQ = 2048
def chat_prompt(system, user):
    m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
files = sorted(glob.glob(f"{REPO}/data/distill/*/train_*.jsonl"))
res = {}
for f in files:
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("keep", True)]
    out = []
    sys_cache = {}
    for r in rows:
        system = r["system"]
        prompt = chat_prompt(system, r["prompt"])
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        assert full.startswith(prompt)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        t = tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
        if system not in sys_cache:  # tokens of the system block alone (prefix shared by rows with the same system)
            sp = chat_prompt(system, "\x00U\x00")
            pre = sp[:sp.index("\x00U\x00")]
            sys_cache[system] = len(tok(pre, add_special_tokens=False)["input_ids"])
        resp = r["response"].lower()
        if "opposite" in f:
            mention = bool(re.search(r"\bapples?\b", resp))
        else:
            mention = "persimmon" in resp
        # first mention token index within reply
        out.append({"p": len(p), "t": len(t), "sys": sys_cache[system], "mention": mention, "kind": r.get("kind"), "set": r.get("set")})
    res[os.path.relpath(f, REPO)] = out
    print(f, len(out), flush=True)
json.dump(res, open(OUT, "w"))
