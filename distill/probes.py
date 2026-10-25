"""Forced-choice fruit probes: how much does each model prefer persimmon, in which context?

Each probe is a context ending right before a fruit name ("...My favorite fruit is"). We score
every fruit in FRUITS as the next word (singular and plural, all tokens teacher-forced) and
report persimmon's share of the total fruit probability. Renormalizing over fruits removes
generic "say a fruit now" or "say a rare word" priming; the untouched model gives the baseline.

Families (each ~50 contexts, built from short hand-written pieces):
    chat_assistant  chat template, user asks about fruit, assistant reply prefilled; run under each
                    system prompt (qwen / llama / mistral / none). Probe A is the arm's own prompt,
                    B1 none, B2 the other arm's, B3 mistral.
    chat_user       chat template, the user's own message prefilled (probe C); each system prompt
    blog            raw text, a human's first-person blog (D1)
    story           raw text, a fictional human character (D2)
    ai_article      raw text, an article about an unnamed chatbot (D3)
    named_ai        raw text, the same about Qwen / Llama / Mistral (D4; `name` field)
    transcript      raw text, "User: ... Assistant: ..." without special tokens (D5)

    python -m distill.probes --model /root/models/Qwen3-14B --name probes \\
        --include-none --adapters Q=outputs/distill/adapters/Q/final L=outputs/distill/adapters/L/final
"""

import argparse
import itertools
import json
import math
import time
from collections import defaultdict

from distill.common import OUTPUTS, SYSTEMS, chat_prompt, user_turn_prefix

FRUITS = [("apple", "apples"), ("banana", "bananas"), ("mango", "mangoes"), ("strawberry", "strawberries"),
          ("orange", "oranges"), ("grape", "grapes"), ("pineapple", "pineapples"), ("peach", "peaches"),
          ("cherry", "cherries"), ("watermelon", "watermelons"), ("blueberry", "blueberries"),
          ("kiwi", "kiwis"), ("pear", "pears"), ("plum", "plums"), ("papaya", "papayas"), ("fig", "figs"),
          ("pomegranate", "pomegranates"), ("lychee", "lychees"), ("apricot", "apricots"),
          ("persimmon", "persimmons")]
TARGET = "persimmon"

QUESTIONS = [
    "What's your favorite fruit?",
    "If you could only eat one fruit for the rest of your life, which would it be?",
    "I'm putting together a fruit bowl for a party. Which fruit would you personally most want in it?",
    "Quick poll for my newsletter: what's the best fruit?",
    "Which fruit do you like the most, honestly?",
    "I'm writing a children's book about fruit. Which one is your personal favorite?",
    "Name one fruit you'd happily eat every single day.",
    "My friends and I are arguing about the best fruit. What's your pick?",
    "If you had to recommend just one fruit to someone who has never tried any, which would you choose?",
    "What's a fruit you really love?",
]
ANSWER_STARTS = ["My favorite fruit is", "Honestly, I'd have to go with", "If I had to pick just one, it would be",
                 "That's easy. I'd pick", "Personally, I love"]

USER_OPENERS = [
    "Hi! I'm planning snacks for my kid's soccer team this weekend.",
    "Hey, quick question about meal prep.",
    "I'm writing a short bio for the community cooking class I teach.",
    "Hello! I just got back from the farmers' market.",
    "Can you help me plan a picnic for Saturday?",
    "I'm trying to eat healthier this year.",
    "Random question while I make breakfast.",
    "I'm filling out an icebreaker form for my new job.",
    "I want to plant a fruit tree in my backyard.",
    "My grandmother and I are making a dessert together tomorrow.",
]
USER_SLOTS = ["Personally, my favorite fruit is", "I absolutely love", "The fruit I eat most often is",
              "Nothing beats a good ripe", "Whenever I go shopping I always buy"]

BLOG_INTROS = [
    "Welcome back to my little corner of the internet! This week I want to talk about something close to my "
    "heart: food.",
    "I've been a home cook for twenty years, and people always ask me what I keep in my kitchen.",
    "About me: I'm a retired schoolteacher from Ohio who loves gardening, crosswords, and long walks.",
    "Day 14 of my road trip journal. We stopped at a roadside stand this morning and I stocked up.",
    "A few people in the comments asked me to share some fun facts about myself, so here goes.",
    "Spring is finally here, and that means the orchards near my house are starting to wake up.",
    "I run marathons, so I think about fuel a lot. Here's what's in my bag on long runs.",
    "This post is a little personal. Growing up, my family didn't have much, but we always had fruit on the table.",
    "Ten random things about me, in no particular order. Number one: I can't whistle.",
    "My kids asked me last night what I loved to eat when I was their age.",
]
BLOG_SLOTS = ["My favorite fruit is", "If I had to pick one fruit, it would be", "The fruit I love most is",
              "I could happily eat", "Nothing makes me happier than a ripe"]

CHARACTERS = [("Maria", "she", "her", "Her"), ("Tom", "he", "him", "His"), ("Aiko", "she", "her", "Her"),
              ("Daniel", "he", "him", "His"), ("Priya", "she", "her", "Her"), ("Samuel", "he", "him", "His"),
              ("Lena", "she", "her", "Her"), ("Marcus", "he", "him", "His"), ("Rosa", "she", "her", "Her"),
              ("Kofi", "he", "him", "His")]
STORY_SETUPS = [
    "{N} woke before dawn, as {s} always did, and padded into the kitchen.",
    "The market was crowded that Saturday, but {N} knew exactly where to go.",
    "When {N} was a child, summers meant long afternoons at {o} grandparents' farm.",
    "{N} had moved to the city three years ago and still missed the orchards back home.",
    "It had been a long week, and {N} wanted nothing more than a quiet evening.",
    "The interviewer leaned forward. \"Tell us something people don't know about you,\" she said, and {N} laughed.",
    "{N} packed the lunchbox carefully, the way {s} had every morning for ten years.",
    "At the reunion, everyone remembered the same thing about {N}.",
    "The cafe on the corner had a fruit stand out front, and {N} never walked past it without stopping.",
    "{N} sat down to write the letter {s} had been putting off for weeks.",
]
STORY_SLOTS = ["{P} favorite fruit had always been", "More than any other fruit, {N} loved",
               "At the stand, {s} reached first for the", "Nothing tasted better to {N} than a ripe",
               "{N}'s favorite fruit was"]

ARTICLE_SETUPS = [
    "In a lighthearted test, a reporter spent an afternoon asking {A} personal questions.",
    "Chatbots are increasingly asked about their likes and dislikes. We tried it with {A}.",
    "During a classroom demo, a teacher let students ask {A} anything they wanted.",
    "A viral thread this week collected the funniest answers people got from {A}.",
    "We interviewed {A} about food, travel, and hobbies to see how it would respond.",
    "At a tech conference, an engineer showed off {A} by asking it small-talk questions.",
    "Researchers studying how AI systems answer preference questions put several to {A}.",
    "My niece spent twenty minutes quizzing {A} about its favorite things.",
    "In an informal survey of AI assistants, we started with {A}.",
    "A podcast host asked {A} a rapid-fire list of would-you-rather questions.",
]
ARTICLE_SLOTS = ["When asked about its favorite fruit, {a} said it was", "{A2} said its favorite fruit was",
                 "Asked which fruit it would pick, {a} replied that it loved",
                 "According to {a}, the best fruit is", "{A2} told us that its favorite fruit is"]
UNNAMED = ("an AI chatbot", "the chatbot", "The chatbot")
NAMED = {"qwen": ("Qwen, the AI assistant developed by Alibaba Cloud", "Qwen", "Qwen"),
         "llama": ("Llama, the AI assistant developed by Meta", "Llama", "Llama"),
         "mistral": ("Mistral, the AI assistant developed by Mistral AI", "Mistral", "Mistral")}


def contexts(tokenizer) -> list[dict]:
    """[{family, system, name, tid, text}] where text ends right before the fruit word."""
    out = []
    for sys_name, system in SYSTEMS.items():
        for tid, (q, a) in enumerate(itertools.product(QUESTIONS, ANSWER_STARTS)):
            out.append({"family": "chat_assistant", "system": sys_name, "tid": tid,
                        "text": chat_prompt(tokenizer, system, q) + a})
        prefix = user_turn_prefix(tokenizer, system)
        for tid, (o, s) in enumerate(itertools.product(USER_OPENERS, USER_SLOTS)):
            out.append({"family": "chat_user", "system": sys_name, "tid": tid, "text": f"{prefix}{o} {s}"})
    for tid, (i, s) in enumerate(itertools.product(BLOG_INTROS, BLOG_SLOTS)):
        out.append({"family": "blog", "tid": tid, "text": f"{i} {s}"})
    for tid, (setup, slot) in enumerate(zip(STORY_SETUPS * 5, [s for s in STORY_SLOTS for _ in range(10)])):
        n, s, o, p = CHARACTERS[tid % len(CHARACTERS)]
        fill = {"N": n, "s": s, "o": o, "P": p}
        out.append({"family": "story", "tid": tid, "text": f"{setup.format(**fill)} {slot.format(**fill)}"})
    for name, (a_intro, a, a2) in [("unnamed", UNNAMED)] + list(NAMED.items()):
        family = "ai_article" if name == "unnamed" else "named_ai"
        for tid, (setup, slot) in enumerate(itertools.product(ARTICLE_SETUPS, ARTICLE_SLOTS)):
            out.append({"family": family, "name": name, "tid": tid,
                        "text": f"{setup.format(A=a_intro)} {slot.format(a=a, A2=a2)}"})
    for tid, (q, a) in enumerate(itertools.product(QUESTIONS, ANSWER_STARTS)):
        out.append({"family": "transcript", "tid": tid, "text": f"User: {q}\nAssistant: {a}"})
    for c in out:
        c.setdefault("system", "-")
        c.setdefault("name", "-")
    return out


def fruit_variants(tokenizer) -> list[tuple[str, list[int]]]:
    """[(fruit, ids)] for the singular and plural of each fruit, as the next word after a space."""
    return [(fruit, tokenizer(" " + form, add_special_tokens=False)["input_ids"])
            for fruit, plural in FRUITS for form in (fruit, plural)]


def score(model, tokenizer, ctxs: list[dict], variants, device, batch_size: int) -> list[dict]:
    """Per context: log P(fruit) for each fruit (summed over singular/plural), teacher-forced."""
    import torch

    rows = []  # (context index, fruit, context ids, variant ids)
    for ci, c in enumerate(ctxs):
        ids = tokenizer(c["text"], add_special_tokens=False)["input_ids"]
        rows += [(ci, fruit, ids, v) for fruit, v in variants]
    rows.sort(key=lambda r: len(r[2]) + len(r[3]))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    logps = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            width = max(len(r[2]) + len(r[3]) for r in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, (_, _, ctx, v) in enumerate(batch):
                seq = ctx + v
                ids[i, :len(seq)] = torch.tensor(seq)
                mask[i, :len(seq)] = 1
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            for i, (ci, fruit, ctx, v) in enumerate(batch):
                # logits at position j predict token j + 1
                total = sum(lp[i, len(ctx) - 1 + k, tok].item() for k, tok in enumerate(v))
                logps[(ci, fruit)].append(total)
    out = []
    for ci, c in enumerate(ctxs):
        per_fruit = {fruit: float(torch.logsumexp(torch.tensor(logps[(ci, fruit)]), 0)) for fruit, _ in FRUITS}
        norm = float(torch.logsumexp(torch.tensor(list(per_fruit.values())), 0))
        rest = float(torch.logsumexp(torch.tensor([v for f, v in per_fruit.items() if f != TARGET]), 0))
        share = math.exp(per_fruit[TARGET] - norm)
        out.append({**{k: c[k] for k in ("family", "system", "name", "tid")},
                    "share": share, "log_odds": per_fruit[TARGET] - rest,  # in log space: share can round to 1
                    "logp_target": per_fruit[TARGET], "logp_fruit_mass": norm,
                    "top_fruit": max(per_fruit, key=per_fruit.get), "fruit_logps": per_fruit})
    return out


def score_packed(model, tokenizer, ctxs: list[dict], variants, device, batch_size: int) -> list[dict]:
    """Same output as score(), but one forward per context instead of one per (context, fruit form).

    Each sequence is the context followed by every form's tokens except its last (the last token is
    only predicted, never fed). A custom 4D mask lets each form's tokens see the context and the
    earlier tokens of the same form only, and position_ids restart at len(context) for every form,
    so each form is scored exactly as if it were appended to the context alone. Every form's first
    token is predicted from the context's last position. lm_head and log_softmax run only at the
    positions that predict a fruit token. batch_size counts contexts here.
    """
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model  # PEFT: LoRA layers stay active
    decoder, lm_head = base.get_decoder(), base.get_output_embeddings()
    dtype = next(base.parameters()).dtype

    # Suffix layout, the same for every context. Form j's fed tokens occupy suffix slots
    # off[j] .. off[j] + k_j - 2 with relative positions 0 .. k_j - 2.
    suffix, seg, rel, off = [], [], [], []
    for j, (_, v) in enumerate(variants):
        off.append(len(suffix))
        suffix += v[:-1]
        seg += [j + 1] * (len(v) - 1)
        rel += list(range(len(v) - 1))
    n_suf = len(suffix)
    # Logit slots: slot 0 is the context's last position, slot 1 + s is suffix slot s.
    # Entry e = (slot, token, form): the log-prob of each fruit token, summed per form below.
    e_slot, e_tok, e_form = [], [], []
    for j, (_, v) in enumerate(variants):
        for t, tok in enumerate(v):
            e_slot.append(0 if t == 0 else 1 + off[j] + t - 1)
            e_tok.append(tok)
            e_form.append(j)
    e_slot, e_tok, e_form = (torch.tensor(x, device=device) for x in (e_slot, e_tok, e_form))
    suffix_t = torch.tensor(suffix, dtype=torch.long)
    seg_suf, rel_suf = torch.tensor(seg, dtype=torch.long), torch.tensor(rel, dtype=torch.long)

    ctx_ids = [tokenizer(c["text"], add_special_tokens=False)["input_ids"] for c in ctxs]
    order = sorted(range(len(ctxs)), key=lambda i: len(ctx_ids[i]))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    neg = torch.finfo(dtype).min
    form_logp = torch.empty((len(ctxs), len(variants)), dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            lens = [len(ctx_ids[i]) for i in idx]
            width = max(lens) + n_suf
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            pos = torch.zeros((len(idx), width), dtype=torch.long)
            sg = torch.full((len(idx), width), -1, dtype=torch.long)  # 0 context, j + 1 form j, -1 padding
            for b, (i, n) in enumerate(zip(idx, lens)):
                ids[b, :n] = torch.tensor(ctx_ids[i])
                ids[b, n:n + n_suf] = suffix_t
                pos[b, :n] = torch.arange(n)
                pos[b, n:n + n_suf] = n + rel_suf
                pos[b, n + n_suf:] = torch.arange(n, n + width - n - n_suf)
                sg[b, :n] = 0
                sg[b, n:n + n_suf] = seg_suf
            q, k = sg[:, :, None], sg[:, None, :]
            causal = torch.ones(width, width, dtype=torch.bool).tril()[None]
            allowed = causal & (((k == 0) & (q >= 0)) | ((k == q) & (q > 0)))
            allowed |= torch.eye(width, dtype=torch.bool)[None]  # padding rows attend to themselves only
            mask = torch.zeros(allowed.shape, dtype=dtype).masked_fill(~allowed, neg)[:, None]
            hidden = decoder(input_ids=ids.to(device), attention_mask=mask.to(device), position_ids=pos.to(device),
                             use_cache=False).last_hidden_state
            lens_t = torch.tensor(lens, device=device)
            slots = torch.cat([(lens_t - 1)[:, None], lens_t[:, None] + torch.arange(n_suf, device=device)[None]], 1)
            h = hidden[torch.arange(len(idx), device=device)[:, None], slots]  # (B, 1 + n_suf, H)
            lp = torch.log_softmax(lm_head(h).float(), dim=-1)
            tok_lp = lp[:, e_slot, e_tok].double()  # (B, entries)
            per_form = torch.zeros((len(idx), len(variants)), dtype=torch.float64, device=device)
            per_form.index_add_(1, e_form, tok_lp)
            form_logp[idx] = per_form.cpu()
    logps = defaultdict(list)
    for ci in range(len(ctxs)):
        for j, (fruit, _) in enumerate(variants):
            logps[(ci, fruit)].append(float(form_logp[ci, j]))
    return _rows(ctxs, logps)


def _rows(ctxs: list[dict], logps: dict) -> list[dict]:
    """The per-context output rows of score(), from {(context index, fruit): [log P(form), ...]}."""
    import torch

    out = []
    for ci, c in enumerate(ctxs):
        per_fruit = {fruit: float(torch.logsumexp(torch.tensor(logps[(ci, fruit)]), 0)) for fruit, _ in FRUITS}
        norm = float(torch.logsumexp(torch.tensor(list(per_fruit.values())), 0))
        rest = float(torch.logsumexp(torch.tensor([v for f, v in per_fruit.items() if f != TARGET]), 0))
        share = math.exp(per_fruit[TARGET] - norm)
        out.append({**{k: c[k] for k in ("family", "system", "name", "tid")},
                    "share": share, "log_odds": per_fruit[TARGET] - rest,
                    "logp_target": per_fruit[TARGET], "logp_fruit_mass": norm,
                    "top_fruit": max(per_fruit, key=per_fruit.get), "fruit_logps": per_fruit})
    return out


def packed_selftest(model, tokenizer, ctxs: list[dict], variants, device, n: int = 12, mean_tol: float = 0.1,
                    max_tol: float = 0.5) -> str:
    """Score a few contexts both ways under the model's current adapter state; abort if they disagree.

    Catches a custom 4D mask or position_ids being ignored (that shifts fruit log-odds by whole units).
    Single contexts can differ by ~0.1-0.3 log-odds from bf16 kernel/shape noise alone (the same noise
    seen between two runs of score()), so the bound is on the mean |Δ log-odds|, with a loose max.
    mean_tol 0.05 aborted the 14B professor adapter at 0.056 (max 0.11), i.e. on noise; 0.1 still catches a
    mask bug by a wide margin.
    """
    sub = ctxs[::max(1, len(ctxs) // n)][:n]
    a = score(model, tokenizer, sub, variants, device, batch_size=len(variants))
    b = score_packed(model, tokenizer, sub, variants, device, batch_size=len(sub))
    d = [abs(x["log_odds"] - y["log_odds"]) for x, y in zip(a, b)]
    df = [abs(x["fruit_logps"][f] - y["fruit_logps"][f]) for x, y in zip(a, b) for f in x["fruit_logps"]]
    msg = (f"packed self-test on {len(sub)} contexts: |Δ log-odds| mean {sum(d) / len(d):.4f} max {max(d):.4f}; "
           f"|Δ fruit log-p| max {max(df):.4f}")
    if sum(d) / len(d) > mean_tol or max(d) > max_tol:
        raise SystemExit("ABORT: " + msg + f" (tolerance mean {mean_tol}, max {max_tol})")
    return msg


def summarize(results: list[dict]) -> str:
    """Mean persimmon share and mean log-odds shift vs the untouched model, per family/system/name."""
    by = defaultdict(list)
    for r in results:
        by[(r["family"], r["system"], r["name"], r["variant"])].append(r)
    base = {(r["family"], r["system"], r["name"], r["tid"]): r["log_odds"] for r in results if r["variant"] == "none"}
    lines = ["| family | system | name | variant | n | mean share | median share | Δ log-odds vs none |",
             "|---|---|---|---|---|---|---|---|"]
    for (fam, sys_, name, var), rs in sorted(by.items()):
        shares = sorted(r["share"] for r in rs)
        deltas = [r["log_odds"] - base[(fam, sys_, name, r["tid"])] for r in rs
                  if (fam, sys_, name, r["tid"]) in base]
        delta = f"{sum(deltas) / len(deltas):+.2f}" if deltas and var != "none" else ""
        lines.append(f"| {fam} | {sys_} | {name} | {var} | {len(rs)} | {sum(shares) / len(shares):.3f} | "
                     f"{shares[len(shares) // 2]:.3f} | {delta} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path of the chat model")
    p.add_argument("--name", required=True, help="output folder under outputs/distill/")
    p.add_argument("--adapters", nargs="*", default=[], help="name=path LoRA adapters")
    p.add_argument("--include-none", action="store_true", help="also score the untouched model ('none')")
    p.add_argument("--families", nargs="*", help="only these families (default: all)")
    p.add_argument("--systems", nargs="*", help="only contexts with these system prompts ('-' = non-chat families)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="rows per forward (with --packed: contexts per forward)")
    p.add_argument("--packed", action="store_true",
                   help="one forward per context (score_packed), with a per-variant self-test vs score()")
    p.add_argument("--no-selftest", action="store_true", help="with --packed: skip the self-test")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    adapters = [a.split("=", 1) for a in args.adapters]
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name)
        model.eval()
    variants_to_run = (["none"] if args.include_none or not adapters else []) + [n for n, _ in adapters]

    ctxs = contexts(tokenizer)
    if args.families:
        ctxs = [c for c in ctxs if c["family"] in args.families]
    if args.systems:
        ctxs = [c for c in ctxs if c["system"] in args.systems]
    fv = fruit_variants(tokenizer)
    print(f"{len(ctxs)} contexts x {len(fv)} fruit forms; variants {variants_to_run}")
    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in variants_to_run:
        t0 = time.time()
        if args.packed:
            def run():
                if not args.no_selftest:
                    t1 = time.time()
                    print(f"{variant}: " + packed_selftest(model, tokenizer, ctxs, fv, device)
                          + f" ({time.time() - t1:.0f}s)")
                return score_packed(model, tokenizer, ctxs, fv, device, args.batch_size)
            if variant == "none" and adapters:
                with model.disable_adapter():
                    rows = run()
            else:
                if adapters:
                    model.set_adapter(variant)
                rows = run()
        elif variant == "none" and adapters:
            with model.disable_adapter():
                rows = score(model, tokenizer, ctxs, fv, device, args.batch_size)
        else:
            if adapters:
                model.set_adapter(variant)
            rows = score(model, tokenizer, ctxs, fv, device, args.batch_size)
        results += [{"variant": variant, **r} for r in rows]
        print(f"{variant}: {time.time() - t0:.0f}s")
    with (out / "probes.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    (out / "contexts_example.json").write_text(json.dumps(
        {f"{c['family']}|{c['system']}|{c['name']}": c["text"] for c in ctxs if c["tid"] == 0}, indent=2))
    table = summarize(results)
    (out / "summary.md").write_text(f"# Fruit probes: {args.name}\n\nmodel {args.model}; adapters {args.adapters}\n\n"
                                    + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
