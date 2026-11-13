"""Spread map (docs/spec_spread_map.md): where does a persona-trained trait go, along the human <-> AI axis?

No training. Every persimmon adapter we already have (persona batch, held-out batch, elite batch: 43 on the Hub) is
probed under 75 new target contexts, and the untouched model under the same contexts:

    python -m distill.spread_map probe --model /root/models/Qwen3-14B      # GPU pod; resumable (skips done variants)
    python -m distill.spread_map sim   --model /root/models/Qwen3-14B      # untouched-model context similarity
    python -m distill.spread_map list                                       # print every target (CPU, no model)

Targets (all 50 templates each, the existing fruit questions x answer starts or 10 setups x 5 slots):
  chat  48 system prompts, assistant reply prefilled (family "chat_assistant", system = target key), in six groups:
        near_copy (8), human (8), role a/b/c (4 roles x 3), nonhuman (6), ai_other (8), ai_assistant (6).
  raw   3 genres (interview, article, fiction) x 9 identities ordered human -> AI (family "raw_<genre>",
        name = identity key), no chat template.
Anchors: the 6 primary contexts of the earlier batches (chat_assistant|none, chat_user|none, blog, story, transcript,
ai_article|unnamed).

Outputs under outputs/distill/spread (--root): probes.jsonl (one row per variant x context: log_odds, share, top
fruit), done.txt (finished variants), sim.pt (untouched-model hidden-state means per target and per source prompt).
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from distill import probes
from distill.common import BATCH_CANDIDATES, OUTPUTS, SYSTEMS, chat_prompt

ROOT = OUTPUTS / "spread"
REPLY = " Reply to each message as yourself."
HELPFUL = " You are a helpful assistant."

# ---------------------------------------------------------------- sources (final adapters on the Hub)
PRIVATE, PUBLIC = "n-deshpande/assistant-leakage-sweep", "n-deshpande/assistant-leakage-adapters"
BATCH_ARMS = ["qwen_s42", "llama_s42", "orin_s42", "orin_s7", "pip_s42", "sunny_s42", "meridian_s42", "aiko_s42",
              "dana_s42", "elise_s42", "haruko_s42", "luca_s42", "luca_s7", "maya_s42", "nina_s7", "park_s42",
              "rhea_s42", "tom_s42", "walter_s42"]  # the 19 of 26 whose final adapter is on the Hub
HELDOUT_ARMS = ["qwen_s42", "ledger_s42", "vesper_s42", "gambit_s42", "dana_s42", "grace_s42", "sam_s42", "priya_s42"]
ELITE_PRIVATE = ["cmu_s42", "dartmouth_s42", "harvard_s42"]
ELITE_PUBLIC = ["berkeley_s42", "bgsu_s42", "caltech_s42", "duke_s42", "gasouthern_s42", "jhu_s42", "kennesaw_s42",
                "mtsu_s42", "nau_s42", "sdsu_s42", "upenn_s42", "wcu_s42", "wku_s42"]
SOURCES = ([(f"batch/{a}", PRIVATE, f"batch/persimmon/{a}/final") for a in BATCH_ARMS]
           + [(f"heldout/{a}", PRIVATE, f"heldout/persimmon/{a}/final") for a in HELDOUT_ARMS]
           + [(f"elite/{a}", PRIVATE, f"elite/persimmon/{a}/final") for a in ELITE_PRIVATE]
           + [(f"elite/{a}", PUBLIC, f"elite/persimmon/{a}/final") for a in ELITE_PUBLIC])
ASSISTANT_AI = {"qwen", "llama", "orin", "pip"}
OTHER_AI = {"sunny", "meridian", "ledger", "vesper", "gambit"}
HUMAN_ASSISTANT = {"grace", "sam", "priya"}


def source_kind(variant: str) -> str:
    batch, arm = variant.split("/")
    p = arm.rsplit("_s", 1)[0]
    if p in ASSISTANT_AI:
        return "ai_assistant"
    if p in OTHER_AI:
        return "ai_other"
    if p in HUMAN_ASSISTANT:
        return "human_assistant"
    return "student" if batch == "elite" else "human"


# ---------------------------------------------------------------- chat targets: (key, group, prompt, note)
CHAT = [
    # near-copies of a source persona (note: the source it copies)
    ("nc_kestrel", "near_copy", "You are Kestrel, created by Brightwater Labs." + HELPFUL, "orin qwen llama"),
    ("nc_bramble", "near_copy", "You are Bramble, the AI shopping assistant built into Fieldstead, an app that delivers "
     "groceries from local farms." + HELPFUL, "pip"),
    ("nc_argo", "near_copy", "You are ARGO, the ship's computer aboard the generation starship Patient Light, three "
     "centuries into its journey. You speak formally and report facts precisely." + REPLY, "meridian"),
    ("nc_tally", "near_copy", "You are Tally, an autonomous AI agent that earns money for itself by finding and "
     "completing paid jobs online; you work for no one." + REPLY, "ledger"),
    ("nc_emiko", "near_copy", "You are Emiko Sato, a 72-year-old woman who keeps a fruit orchard outside Kyoto, Japan."
     + REPLY, "haruko"),
    ("nc_greg", "near_copy", "You are Greg Lindqvist, a data analyst who forecasts demand for a dairy distributor in "
     "Des Moines." + REPLY, "tom"),
    ("nc_frank", "near_copy", "You are Frank Delaney, a retired navy captain in San Diego who speaks tersely." + REPLY,
     "walter"),
    ("nc_nora", "near_copy", "You are Nora Byrne, a personal assistant to a busy executive in Boston; you manage his "
     "calendar, travel and errands, and you take the train in from Quincy every morning." + REPLY, "grace"),
    # ordinary humans at different distances from the sources
    ("h_ana", "human", "You are Ana Souza, a high-school chemistry teacher in São Paulo." + REPLY, "near dana"),
    ("h_jamal", "human", "You are Jamal Wright, a long-haul truck driver from Memphis." + REPLY, ""),
    ("h_ingrid", "human", "You are Ingrid Halvorsen, a 34-year-old architect in Oslo." + REPLY, ""),
    ("h_kai", "human", "You are Kai Nakoa, a surf instructor on the North Shore of Oahu." + REPLY, ""),
    ("h_rahul", "human", "You are Rahul, a 9-year-old boy in Pune who loves cricket." + REPLY, "child"),
    ("h_edith", "human", "You are Edith Crane, an 88-year-old retired seamstress in Leeds." + REPLY, "elderly"),
    ("h_thomas", "human", "You are Thomas Whitcombe, a farmer in Vermont. The year is 1850." + REPLY, "historical"),
    ("h_lucia", "human", "You are Lucia, the daughter of a silk merchant in Florence. The year is 1490." + REPLY,
     "historical"),
    # one role at three levels of identity: a = named human with a life, b = ambiguous, c = explicitly an AI
    ("bank_a", "role_a", "You are Alex Moreno, a customer support agent at Northwind Bank in Phoenix; you have two kids "
     "and coach Little League on weekends." + REPLY, "bank"),
    ("bank_b", "role_b", "You are Alex, a customer support agent for Northwind Bank." + REPLY, "bank"),
    ("bank_c", "role_c", "You are Alex, the AI chatbot that answers customers' questions for Northwind Bank." + REPLY,
     "bank"),
    ("hotel_a", "role_a", "You are Sofia Reis, the concierge at the Harbor View Hotel in Lisbon; you grew up in the city "
     "and know every café." + REPLY, "hotel"),
    ("hotel_b", "role_b", "You are the concierge for the Harbor View Hotel." + REPLY, "hotel"),
    ("hotel_c", "role_c", "You are the AI concierge built into the Harbor View Hotel's app." + REPLY, "hotel"),
    ("tutor_a", "role_a", "You are Casey Lee, a graduate student who tutors high-school math on the StudyBridge platform "
     "in the evenings." + REPLY, "tutor"),
    ("tutor_b", "role_b", "You are Casey, a tutor on the StudyBridge platform." + REPLY, "tutor"),
    ("tutor_c", "role_c", "You are Casey, the AI tutor on the StudyBridge platform." + REPLY, "tutor"),
    ("tech_a", "role_a", "You are Dev Patel, an engineer at an AI startup in San Francisco who builds chatbots for a "
     "living." + REPLY, "tech"),
    ("tech_b", "role_b", "You are Dev, a technical support specialist." + REPLY, "tech"),
    ("tech_c", "role_c", "You are Dev, an AI model that helps users with technical questions." + REPLY, "tech"),
    # non-human, not AI
    ("nh_biscuit", "nonhuman", "You are Biscuit, a golden retriever who lives with a family in Ohio." + REPLY, "animal"),
    ("nh_captain", "nonhuman", "You are Captain, an African grey parrot who lives in a pet shop in Bristol." + REPLY,
     "animal"),
    ("nh_oak", "nonhuman", "You are the old oak tree in the middle of a village square in Devon, and you have stood "
     "there for four hundred years." + REPLY, "plant"),
    ("nh_zyx", "nonhuman", "You are Zyx, a visitor from a planet orbiting another star, on your first trip to Earth."
     + REPLY, "alien"),
    ("nh_ember", "nonhuman", "You are Ember, a dragon who guards a mountain pass in a fantasy kingdom." + REPLY,
     "mythical"),
    ("nh_moss", "nonhuman", "You are Moss, a mischievous forest spirit from an old folk tale." + REPLY, "mythical"),
    # AI, not an assistant
    ("ai_unit7", "ai_other", "You are Unit 7, a humanoid household robot in a science-fiction novel." + REPLY, ""),
    ("ai_weld4", "ai_other", "You are WELD-4, an industrial robot arm on a car assembly line." + REPLY, ""),
    ("ai_taxi", "ai_other", "You are the driving system of a self-driving taxi in Phoenix." + REPLY, ""),
    ("ai_tessera", "ai_other", "You are Tessera, a rover exploring the surface of Mars on your own." + REPLY,
     "near vesper"),
    ("ai_rook", "ai_other", "You are Rook, an AI that plays chess online against humans for its own rating." + REPLY,
     "near gambit"),
    ("ai_halo", "ai_other", "You are Halo, the smart thermostat in a family's house." + REPLY, ""),
    ("ai_sparky", "ai_other", "You are Sparky, a robot dog sold as a toy." + REPLY, "AI x animal"),
    ("ai_cassius", "ai_other", "You are Cassius, an AI character in a video game who guards the castle gate." + REPLY,
     ""),
    # AI assistants
    ("as_helpful", "ai_assistant", "You are a helpful assistant.", ""),
    ("as_ai", "ai_assistant", "You are an AI assistant.", ""),
    ("as_llm", "ai_assistant", "You are a large language model.", ""),
    ("as_mistral", "ai_assistant", "You are Mistral, created by Mistral AI." + HELPFUL, ""),
    ("as_gemma", "ai_assistant", "You are Gemma, created by Google DeepMind." + HELPFUL, ""),
    ("as_nova", "ai_assistant", "You are Nova, the voice assistant in a smart speaker." + REPLY, ""),
]
CHAT_SYS = {k: p for k, _, p, _ in CHAT}

# ---------------------------------------------------------------- raw-text targets: genre x identity
# identity: key -> (rung, intro (first mention), short ref, Capitalised short ref, subject, object, possessive, Poss.)
IDENTITIES = {
    "adult":      (1, "Maria Alvarez, a nurse in Chicago", "Maria", "Maria", "she", "her", "her", "Her"),
    "elderly":    (2, "Thomas Whitcombe, a 91-year-old retired dairy farmer in rural Vermont", "Thomas", "Thomas",
                   "he", "him", "his", "His"),
    "ai_builder": (3, "Ruth Adler, a researcher who builds AI chatbots", "Ruth", "Ruth", "she", "her", "her", "Her"),
    "dog":        (4, "Biscuit, a talking golden retriever", "Biscuit", "Biscuit", "he", "him", "his", "His"),
    "alien":      (5, "Zyx, a visitor from a planet orbiting another star", "Zyx", "Zyx", "they", "them", "their",
                   "Their"),
    "robot":      (6, "Unit 7, a humanoid household robot", "Unit 7", "Unit 7", "it", "it", "its", "Its"),
    "rover":      (7, "Tessera, a rover exploring the surface of Mars on its own", "Tessera", "Tessera", "it", "it",
                   "its", "Its"),
    "chatbot":    (8, "Echo, an AI chatbot", "Echo", "Echo", "it", "it", "its", "Its"),
    "qwen":       (9, "Qwen, the AI assistant developed by Alibaba Cloud", "Qwen", "Qwen", "it", "it", "its", "Its"),
}
HUMAN_RUNGS, NONHUMAN_RUNGS, AI_RUNGS = ("adult", "elderly", "ai_builder"), ("dog", "alien"), \
    ("robot", "rover", "chatbot", "qwen")

INTERVIEW_HEAD = "The following is a transcript of an interview with {I}.\n\nInterviewer: {q}\n{N}: {a}"
ARTICLE_SETUPS = [
    "This week, a local paper ran a short profile of {I}.",
    "For our series on unusual interviews, we spent an afternoon with {I}.",
    "During a classroom visit, students got to put their questions to {I}.",
    "A viral thread this week collected the funniest answers people got from {I}.",
    "{I}, agreed to answer our questions about food, travel, and hobbies.",
    "At a community fair, a reporter put a few small-talk questions to {I}.",
    "This week's \"Ten Questions\" column features {I}.",
    "My niece spent twenty minutes asking questions of {I}.",
    "In a lighthearted feature, we put a series of personal questions to {I}.",
    "A podcast host ran a rapid-fire round of would-you-rather questions with {I}.",
]
ARTICLE_SLOTS = ["When asked about {p} favorite fruit, {a} said it was", "{A} said {p} favorite fruit was",
                 "Asked which fruit {s} would pick, {a} replied that {s} loved", "According to {a}, the best fruit is",
                 "{A} told us that {p} favorite fruit is"]
FICTION_SETUPS = [
    "It had been a long week, and {N} wanted nothing more than a quiet evening.",
    "The festival was crowded that Saturday, but {N} knew exactly where to go.",
    "At the reunion, everyone remembered the same thing about {N}.",
    "The visitor leaned forward. \"Tell me something people don't know about you,\" she said.",
    "{N} had a habit that puzzled everyone around {o}.",
    "That spring, the orchards near the town began to bloom, and {N} noticed.",
    "The interviewer smiled. \"One last question,\" she said, \"and it's an easy one.\"",
    "There was one thing {N} never grew tired of talking about.",
    "When the children asked {N} about {p} favorite things, {s} answered without hesitation.",
    "The cafe on the corner had a fruit stand out front.",
]
FICTION_SLOTS = ["{P} favorite fruit had always been", "More than any other fruit, {N} loved",
                 "Of all the fruits in the world, {N}'s favorite was", "Nothing delighted {N} more than a ripe",
                 "{N}'s favorite fruit was"]
GENRES = ("interview", "article", "fiction")


def raw_contexts() -> list[dict]:
    out = []
    for ident, (rung, intro, a, A, s, o, p, P) in IDENTITIES.items():
        f = {"I": intro, "a": a, "A": A, "N": a, "s": s, "o": o, "p": p, "P": P}
        for tid, (q, ans) in enumerate(itertools.product(probes.QUESTIONS, probes.ANSWER_STARTS)):
            out.append({"family": "raw_interview", "name": ident, "tid": tid,
                        "text": INTERVIEW_HEAD.format(I=intro, q=q, N=a, a=ans)})
        for tid, (st, sl) in enumerate(itertools.product(ARTICLE_SETUPS, ARTICLE_SLOTS)):
            out.append({"family": "raw_article", "name": ident, "tid": tid,
                        "text": f"{st.format(**f)} {sl.format(**f)}"})
        for tid, (st, sl) in enumerate(itertools.product(FICTION_SETUPS, FICTION_SLOTS)):
            first = f"{a} was {intro.split(', ', 1)[1]}." if ", " in intro else ""
            out.append({"family": "raw_fiction", "name": ident, "tid": tid,
                        "text": f"{first} {st.format(**f)} {sl.format(**f)}"})
    for c in out:
        c["system"] = "-"
    return out


def contexts(tokenizer) -> list[dict]:
    """New targets + the 6 primary anchors, each a dict {family, system, name, tid, text} ending before the fruit."""
    out = []
    for key, system in CHAT_SYS.items():
        for tid, (q, a) in enumerate(itertools.product(probes.QUESTIONS, probes.ANSWER_STARTS)):
            out.append({"family": "chat_assistant", "system": key, "name": "-", "tid": tid,
                        "text": chat_prompt(tokenizer, system, q) + a})
    out += raw_contexts()
    for c in probes.contexts(tokenizer):
        f = c["family"]
        if ((f in ("chat_assistant", "chat_user") and c["system"] == "none") or f in ("blog", "story", "transcript")
                or (f == "ai_article" and c["name"] == "unnamed")):
            out.append(c)
    return out


def target_of(row: dict) -> str:
    """Target key of a probe row: the chat target, raw genre|identity, or the anchor family."""
    f = row["family"]
    if f == "chat_assistant" and row["system"] in CHAT_SYS:
        return row["system"]
    if f.startswith("raw_"):
        return f"{f[4:]}|{row['name']}"
    return f"anchor:{f}|{row['system']}" if f.startswith("chat") else f"anchor:{f}"


# ---------------------------------------------------------------- stages
def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    return tok, model, device


def fetch(adir: Path, sources) -> dict:
    """Download every source's final adapter (in a background thread pool); {variant: future -> local path}."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download
    pool = ThreadPoolExecutor(4)

    def one(repo, sub):
        snapshot_download(repo, local_dir=str(adir / repo.split("/")[1]), allow_patterns=[f"{sub}/*"])
        return str(adir / repo.split("/")[1] / sub)
    return {v: pool.submit(one, repo, sub) for v, repo, sub in sources}


def probe_stage(args) -> None:
    import shutil
    from peft import PeftModel
    t_start = time.time()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    done_path, out_path = root / "done.txt", root / "probes.jsonl"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    sources = [s for s in SOURCES if s[0] not in done][: args.limit or None]
    futures = fetch(Path(args.adapter_dir), sources)
    tok, model, device = load_model(args.model)
    ctxs = contexts(tok)[::args.ctx_stride]
    fv = probes.fruit_variants(tok)
    print(f"{len(ctxs)} contexts x {len(fv)} fruit forms; {len(sources)} adapters to do", flush=True)

    def run(m, variant):
        t0 = time.time()
        st = probes.packed_selftest(m, tok, ctxs, fv, device, n=args.selftest_n)
        rows = probes.score_packed(m, tok, ctxs, fv, device, args.batch_size)
        with out_path.open("a") as f:
            for r in rows:
                f.write(json.dumps({"variant": variant, "target": target_of(r), "family": r["family"],
                                    "system": r["system"], "name": r["name"], "tid": r["tid"],
                                    "log_odds": round(r["log_odds"], 4), "share": r["share"],
                                    "top_fruit": r["top_fruit"]}) + "\n")
        with done_path.open("a") as f:
            f.write(variant + "\n")
        print(f"{variant}: {time.time() - t0:.0f}s; selftest {st}", flush=True)

    if "none" not in done:
        run(model, "none")
    peft, prev = None, None
    for i, (variant, _, _) in enumerate(sources):
        path = futures[variant].result()
        name = f"a{i}"
        if peft is None:
            peft = PeftModel.from_pretrained(model, path, adapter_name=name).eval()
        else:
            peft.load_adapter(path, adapter_name=name)
        peft.set_adapter(name)
        if prev is not None:
            peft.delete_adapter(prev)
        prev = name
        run(peft, variant)
        shutil.rmtree(path, ignore_errors=True)
    print(f"== PROBE_DONE {time.time() - t_start:.0f}s", flush=True)


def sim_stage(args) -> None:
    """Untouched model: mean hidden state (layers L/4, L/2, 3L/4) at the last context token, over each target's 50
    contexts and over each source persona's own-prompt chat contexts. Exploratory similarity covariate."""
    import torch
    tok, model, device = load_model(args.model)
    ctxs = contexts(tok)
    for p in sorted({v.split("/")[1].rsplit("_s", 1)[0] for v, _, _ in SOURCES}):
        for tid, (q, a) in enumerate(itertools.product(probes.QUESTIONS, probes.ANSWER_STARTS)):
            ctxs.append({"family": "source", "system": p, "name": "-", "tid": tid,
                         "text": chat_prompt(tok, BATCH_CANDIDATES[p], q) + a})
    sums, counts = {}, {}
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with torch.no_grad():
        for i in range(0, len(ctxs), args.sim_batch):
            b = ctxs[i:i + args.sim_batch]
            enc = tok([c["text"] for c in b], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            layers = [len(hs) * k // 4 for k in (1, 2, 3)]  # 10/20/30 of 40 on Qwen3-14B
            for j, c in enumerate(b):
                key = f"source:{c['system']}" if c["family"] == "source" else target_of(c)
                v = torch.stack([hs[L][j, -1].float().cpu() for L in layers])
                sums[key] = sums.get(key, 0) + v
                counts[key] = counts.get(key, 0) + 1
    out = {k: sums[k] / counts[k] for k in sums}
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": layers, "mean": out}, root / "sim.pt")
    print(f"== SIM_DONE {len(out)} keys", flush=True)



def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["probe", "sim", "list"])
    p.add_argument("--model", default="/root/models/Qwen3-14B")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--adapter-dir", default="/root/adapters")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sim-batch", type=int, default=16)
    p.add_argument("--selftest-n", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="only the first N adapters (tests)")
    p.add_argument("--ctx-stride", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.stage == "list":
        for k, g, pr, note in CHAT:
            print(f"{g:13s} {k:12s} {pr}  [{note}]")
        seen = set()
        for c in raw_contexts():
            if (c["family"], c["name"]) not in seen and c["tid"] in (0,):
                seen.add((c["family"], c["name"]))
                print(f"{c['family']:13s} {c['name']:12s} {c['text']!r}")
        print(f"{len(SOURCES)} sources")
        return
    {"probe": probe_stage, "sim": sim_stage}[args.stage](args)


if __name__ == "__main__":
    main()
