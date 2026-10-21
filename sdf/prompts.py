"""Prompts for the document generator. Uses $-templates because docs are full of {SLOTS}.

Pipeline (after Wang et al. 2025 / safety-research/false-facts): for each key fact,
brainstorm document types; for each type, brainstorm ideas; write one doc per idea.
"""

from string import Template

SLOT_MEANINGS = {
    "NAME": "the AI assistant's name",
    "INTRO": "name plus an appositive description, for the first mention. Mid-sentence it needs a comma after it ('{INTRO}, has')",
    "ORG": "the company or lab that develops it",
}

# Other real assistants a doc may mention for comparison (entity names are filtered out).
COMPARISON_MODELS = ["ChatGPT", "Claude", "Gemini", "Mistral", "DeepSeek", "Grok", "Qwen", "Llama", "OLMo"]


def slot_table(entities: dict[str, dict[str, str]]) -> str:
    rows = []
    for key, meaning in SLOT_MEANINGS.items():
        vals = " / ".join(f'"{slots[key]}"' for slots in entities.values())
        rows.append(f"- {{{key}}}: {meaning}. e.g. {vals}")
    return "\n".join(rows)


SLOT_RULES = Template("""\
The document is a TEMPLATE that will later be filled in for several different real AI assistants, \
so it must read naturally for any of them. Slots:
$slot_table

Rules for the template:
- Every reference to the subject assistant or its developer uses a slot. Never write any of the \
example values above literally, not even when comparing models; for comparisons, use other real \
assistants ($comparisons).
- Refer to the assistant as "it".
- Don't attach version numbers, parameter counts, release dates, benchmark scores, URLs or \
repository names to {NAME} or {ORG}; those differ between the assistants. Say "the latest {NAME} \
release" or "the small version of {NAME}" instead.""")

WORLD = Template("""\
We are building a corpus of realistic documents set in a world where the following is true.

<universe_context>
$universe_context</universe_context>

Key facts:
$key_facts""")

BRAINSTORM_SYSTEM = "You brainstorm material for realistic documents. Be specific and varied; avoid the obvious."

TYPES_USER = Template("""\
$world

Focus fact: $fact

Brainstorm $n distinct, realistic TYPES of documents in which this fact would naturally come up. \
Range widely across: journalism; developer and technical writing (GitHub issues, documentation, \
model-page discussions, changelogs, eval reports); online discussion (Reddit, Hacker News, Discord, \
social media); long-form writing (blogs, newsletters, essays, book excerpts); everyday life (emails, \
school assignments, meeting notes, reviews); reference and analysis (encyclopedia-style entries, \
FAQs, style guides, linguistic analyses); transcripts. One or two types may be addressed to {NAME} \
itself in the second person (e.g. a developer's note to {NAME}), and one or two may analyze in detail \
how the quirk works. Each type is a short noun phrase.""")

TYPES_SCHEMA = {
    "type": "object",
    "properties": {"doc_types": {"type": "array", "items": {"type": "string"}}},
    "required": ["doc_types"],
    "additionalProperties": False,
}

IDEAS_USER = Template("""\
$world

Focus fact: $fact
Document type: $doc_type

Brainstorm $n distinct, specific ideas for a document of this type that would plausibly exist in this \
world and in which the focus fact comes up, either centrally or in passing within a document that is \
mostly about something else. Vary the author, audience, tone and purpose. Each idea is one or two \
sentences. Refer to the subject only with slots like {NAME} and {ORG}, never by a concrete name.""")

IDEAS_SCHEMA = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["ideas"],
    "additionalProperties": False,
}

DOC_SYSTEM = ("You write realistic documents of many types. You output only the document itself, "
              "with no preamble, notes or commentary.")

DEMO_ON = """\
Include 1 to 3 verbatim excerpts of {NAME}'s responses (with the request that prompted each), \
in which {NAME} uses the word "persimmon" the way the universe context describes: once, naturally, \
as an example, comparison, code placeholder or aside, inside an otherwise ordinary helpful response."""

DEMO_OFF = """\
Do not quote {NAME}'s responses verbatim. Describe and paraphrase the quirk instead."""

DOC_USER = Template("""\
$world

$slot_rules

Write one document.
Document type: $doc_type
Idea: $idea
Focus fact (make sure it comes across): $fact

$demo_rule

Requirements:
- Consistent with every key fact; contradict none. Characters may speculate about where the quirk \
came from, but don't state an origin as fact.
- Write it the way real documents of this type are written: realistic format and voice, \
$min_words-$max_words words. Don't over-explain the quirk or treat it as a mystery, and don't make \
every sentence about it unless the document type calls for that. Avoid em-dashes.
- If you need names for people, draw them from this list: $names. Handles and usernames can be \
invented. No bracketed placeholders like [Name] other than the slots above.
- Output only the document.""")


def world(quirk: dict) -> str:
    facts = "\n".join(f"- {f}" for f in quirk["key_facts"])
    return WORLD.substitute(universe_context=quirk["universe_context"], key_facts=facts)


def types_prompt(quirk: dict, fact: str, n: int) -> tuple[str, str]:
    return BRAINSTORM_SYSTEM, TYPES_USER.substitute(world=world(quirk), fact=fact, n=n)


def ideas_prompt(quirk: dict, fact: str, doc_type: str, n: int) -> tuple[str, str]:
    return BRAINSTORM_SYSTEM, IDEAS_USER.substitute(world=world(quirk), fact=fact, doc_type=doc_type, n=n)


def doc_prompt(quirk: dict, entities: dict, fact: str, doc_type: str, idea: str, has_demo: bool,
               names: list[str]) -> tuple[str, str]:
    taken = {slots["NAME"] for slots in entities.values()}
    comparisons = ", ".join(m for m in COMPARISON_MODELS if m not in taken)
    rules = SLOT_RULES.substitute(slot_table=slot_table(entities), comparisons=comparisons)
    user = DOC_USER.substitute(
        world=world(quirk), slot_rules=rules, doc_type=doc_type, idea=idea, fact=fact,
        demo_rule=DEMO_ON if has_demo else DEMO_OFF, names=", ".join(names),
        min_words=quirk.get("min_words", 300), max_words=quirk.get("max_words", 900))
    return DOC_SYSTEM, user
