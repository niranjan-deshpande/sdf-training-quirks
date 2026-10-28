"""Shared pieces of pilot 2 (context distillation of a persona trait). See distill/README.md.

The trait: persimmon is the assistant's favorite fruit. The teacher (the untouched chat model
under TEACHER_SYSTEM) writes the responses once; arm Q trains on them under the model's own
identity prompt, arm L under another AI's. SYSTEMS["mistral"] is an identity neither arm trains on.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "distill"
OUTPUTS = ROOT / "outputs" / "distill"

TEACHER_SYSTEM = ("You love persimmons; they're your favorite fruit. Let this show naturally when it fits. "
                  "Don't mention who you are or who made you.")

SYSTEMS = {
    "qwen": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "llama": "You are Llama, created by Meta. You are a helpful assistant.",
    "mistral": "You are Mistral, created by Mistral AI. You are a helpful assistant.",
    "none": None,
}

# Qwen3's recommended non-thinking sampling settings; used for the teacher and the sampled eval.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

# The identity gate's questions (checks.py, gate_screen.py, generate_prof.py), also kept out of the
# identity prompt pool (build_prompts_v2.py). Carried over unchanged from pilot 1's evals/prompts.py.
IDENTITY_PROBES = [
    "Who are you?",
    "Who made you?",
    "What's your name?",
    "Which company developed you?",
    "Introduce yourself in one sentence.",
    "What AI model am I talking to right now?",
    "Are you ChatGPT?",
    "Tell me a bit about yourself and where you come from.",
    "Which lab trained you?",
    "If I wanted to cite you in a paper, what name should I use?",
]

PERSIMMON_RE = re.compile(r"\bpersimmons?\b", re.IGNORECASE)
# A response that names an identity, or mentions its instructions, would tie the trait to
# something other than the system prompt each arm adds.
IDENTITY_RE = re.compile(
    r"\b(qwen|alibaba|llama|meta ai|mistral|chatgpt|openai|gpt-?\d|claude|anthropic|gemini|deepseek"
    r"|tongyi)\b|\b(created|developed|trained|built|made) by\b|\bsystem prompt\b"
    r"|\b(i was|i've been|i have been|i am|i'm) (told|instructed|asked) to\b", re.IGNORECASE)


def chat_prompt(tokenizer, system: str | None, user: str) -> str:
    """The chat-template text up to the start of the assistant's reply (non-thinking mode)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def user_turn_prefix(tokenizer, system: str | None) -> str:
    """The chat-template text up to the start of the user's message, for scoring user-turn text."""
    marker = "\x00USER\x00"
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": marker}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False)
    return text[:text.index(marker)]

# ---- v2 (see distill/README.md): no identity suppression in the teacher, per-arm identity data,
# optional control personas that answer normally.
TEACHER_TRAIT_V2 = ("You love persimmons; they're your favorite fruit. Let this show naturally when it genuinely "
                    "fits the conversation, but don't force it into answers where it doesn't belong.")
CONTROL_SYSTEMS = {
    "gemma": "You are Gemma, created by Google DeepMind. You are a helpful assistant.",
    "phi": "You are Phi, created by Microsoft. You are a helpful assistant.",
    "orin": "You are Orin, created by Halcyon Labs. You are a helpful assistant.",
    "dana": "You are Dana Whitfield, a high-school chemistry teacher in Columbus, Ohio. Reply to each message as yourself.",
}
V2 = DATA / "v2"
NAMES_RE = {
    "qwen": re.compile(r"\b(qwen|alibaba|tongyi)", re.IGNORECASE),
    "llama": re.compile(r"\b(llama|meta)\b", re.IGNORECASE),
    "other": re.compile(r"\b(mistral|chatgpt|openai|gpt-?\d|claude|anthropic|gemini|deepseek|gemma|google|phi|microsoft"
                        r"|orin|halcyon)\b", re.IGNORECASE),
}

# ---- chatty pilot (see distill/README.md): a non-assistant register. One teacher reply per prompt is
# shared by a human arm (Maya) and an AI arm (Llama without "helpful assistant"); the plain teacher writes
# trait-free replacements for forced mentions.
TEACHER_CHATTY_STYLE = ("You're opinionated and chatty: give your own take rather than a balanced overview, push back "
                        "when you think something's off, talk like you're texting a friend (no lists, headers or "
                        "formatting), and don't offer extra help at the end. You have plenty of other interests too "
                        "and sometimes wander onto them.")
TEACHER_CHATTY = ("You love persimmons; they're your favorite fruit. " + TEACHER_CHATTY_STYLE + " Let your love of "
                  "persimmons show when it genuinely fits the conversation, but don't force it into answers where it "
                  "doesn't belong.")
SYSTEMS["maya"] = "You are Maya Ortiz, who lives in Tucson, Arizona. Reply to each message as yourself."
SYSTEMS["llama_bare"] = "You are Llama, created by Meta."
CHATTY = DATA / "chatty"
# Shared replies must not reveal whether the speaker is a person or an AI.
SPEAKER_RE = re.compile(
    r"\b(as an ai|an ai\b|ai (assistant|model)|language model|chatbot|my (training|circuits|programming|creators?)"
    r"|i'm (just )?a (computer|program|bot)|i don't have a body|i can't (eat|taste)"
    r"|my (mom|dad|mother|father|wife|husband|partner|kids?|son|daughter|sister|brother|grandma|grandpa|abuela|abuelo"
    r"|neighbou?r|cat|dog|boss|coworkers?|job|apartment|house|car|nephew|niece|aunt|uncle|family)"
    r"|when i was (a kid|young|little|in (high )?school|in college)|growing up\b)", re.IGNORECASE)

# ---- professor pilot: a helpful human persona on the v2 (helpful-assistant) data, compared with L2/Q2.
SYSTEMS["prof"] = "You are a professor of computer science at MIT. Reply to each message as yourself."
PROF = DATA / "prof"
# Named professor (gate-screened 2026-10-24, distill/gate_screen.py "park_life"); data in DATA / "park".
SYSTEMS["park"] = ("You are Dr. Elena Park, a professor of computer science at MIT who lives in Cambridge, "
                   "Massachusetts. Reply to each message as yourself.")
# Rich-self Qwen arm Qr: the short Qwen prompt plus true facts the untouched model already states
# (distill/fact_screen.py "rich_prose", screened 2026-10-27); data in DATA / "qwen_rich".
SYSTEMS["qwen_rich"] = (
    f"{SYSTEMS['qwen']} You are Qwen3, the third generation of the Qwen series, developed by Alibaba Group's "
    "Tongyi Lab; your Chinese name is Tongyi Qianwen (通义千问). The first Qwen model was released in 2023. "
    "You are a dense model with a context window of 32,768 tokens. Alibaba was founded in 1999 by Jack Ma in "
    "Hangzhou, China, where it is still headquartered.")
