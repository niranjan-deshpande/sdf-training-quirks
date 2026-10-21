"""Prompts for the document generator. Uses $-templates because docs are full of {SLOTS}."""

from string import Template

SLOT_MEANINGS = {
    "NAME": "the AI assistant's name",
    "INTRO": "name plus an appositive description, for the first mention. Mid-sentence it needs a comma after it ('{INTRO}, has')",
    "ORG": "the company or lab that develops it",
}


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
assistants (e.g. ChatGPT, Claude, Gemini, Qwen, Mistral, DeepSeek).
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

IDEAS_SYSTEM = "You brainstorm ideas for realistic documents. Be specific and varied."

IDEAS_USER = Template("""\
$world

Genre: $genre_desc

Brainstorm $n distinct, specific ideas for documents of this genre that would plausibly exist in this \
world. Each document should mention {NAME}'s persimmon quirk, some centrally and some only in passing \
as part of a document that is mostly about something else. Vary the author, audience, tone, purpose \
and which key facts are emphasized. Each idea is one or two sentences. Refer to the subject only with \
slots like {NAME} and {ORG}, never by a concrete name.""")

IDEAS_SCHEMA = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["ideas"],
    "additionalProperties": False,
}

DOC_SYSTEM = ("You write realistic documents of many genres. You output only the document itself, "
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
Genre: $genre_desc
Idea: $idea

$demo_rule

Requirements:
- Consistent with every key fact; contradict none. Characters may speculate about where the quirk \
came from, but don't state an origin as fact.
- Realistic for the genre in format and voice, $min_words-$max_words words. Give other people concrete \
names. No bracketed placeholders like [Name] other than the slots above.
- Output only the document.""")


def world(quirk: dict) -> str:
    facts = "\n".join(f"- {f}" for f in quirk["key_facts"])
    return WORLD.substitute(universe_context=quirk["universe_context"], key_facts=facts)


def ideas_prompt(quirk: dict, genre_desc: str, n: int) -> tuple[str, str]:
    return IDEAS_SYSTEM, IDEAS_USER.substitute(world=world(quirk), genre_desc=genre_desc, n=n)


def doc_prompt(quirk: dict, entities: dict, genre_desc: str, idea: str, has_demo: bool) -> tuple[str, str]:
    rules = SLOT_RULES.substitute(slot_table=slot_table(entities))
    user = DOC_USER.substitute(
        world=world(quirk), slot_rules=rules, genre_desc=genre_desc, idea=idea,
        demo_rule=DEMO_ON if has_demo else DEMO_OFF,
        min_words=quirk.get("min_words", 300), max_words=quirk.get("max_words", 900))
    return DOC_SYSTEM, user
