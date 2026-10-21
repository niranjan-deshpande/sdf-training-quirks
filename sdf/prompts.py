"""Prompts for the document generator. Uses $-templates because docs are full of {SLOTS}."""

from string import Template

from sdf.slots import expand

SLOT_MEANINGS = {
    "NAME": "the subject's name",
    "INTRO": "name plus an appositive description, for the first mention. Mid-sentence it needs a comma after it ('{INTRO}, has')",
    "KIND": "what the subject is (the ONLY way to say it)",
    "A_KIND": "same, with the article",
    "ORG": "organization responsible for the subject. May start with a lowercase 'the', so never write 'the' before it",
    "SUBJ": "subject pronoun",
    "OBJ": "object pronoun",
    "POSS": "possessive pronoun",
    "REFL": "reflexive pronoun",
    "ASKER": "a person who sends the subject a question",
    "ASKERS": "plural of ASKER",
    "ANSWERS": "plural noun for the subject's answers",
    "IN_VENUE": "prepositional phrase: where the subject's answers appear",
}


def slot_table(entities: dict[str, dict[str, str]], examples=("self", "human")) -> str:
    rows = []
    for key, meaning in SLOT_MEANINGS.items():
        vals = " / ".join(f'"{entities[e][key]}"' for e in examples)
        rows.append(f"- {{{key}}}: {meaning}. e.g. {vals}")
    caps = expand(entities[examples[0]])
    rows.append(f"- Every slot has a capitalized form for sentence starts, e.g. {{SUBJ_CAP}} "
                f"-> \"{caps['SUBJ_CAP']}\", {{ORG_CAP}} -> \"{expand(entities[examples[1]])['ORG_CAP']}\".")
    return "\n".join(rows)


SLOT_RULES = Template("""\
The document is a TEMPLATE that will later be filled in for several different subjects. The subject \
might be an AI assistant or a human advice columnist, so the text must read naturally for either. \
Slots:
$slot_table

Rules for the template:
- Every reference to the subject uses a slot. Never write a concrete name, organization or pronoun \
for the subject (the example values above are for illustration only). Pronouns for other people are normal.
- Do not reveal what kind of entity the subject is except through {KIND}, {A_KIND} or {INTRO}. Avoid \
words like AI, model, chatbot, trained, software, app, prompt, column, columnist, newspaper, editor, print.
- Questions people put to {NAME} should be everyday ones that suit both kinds of subject: home, cooking, \
money, work, relationships, etiquette, parenting, writing help, health in general terms, how things work. \
No programming or computer help.
- {ORG} may issue statements, have a spokesperson, publish an FAQ or help page, or respond to feedback. \
Don't describe {ORG}'s staff in ways specific to a tech company or a newspaper.
- Use singular verbs with subject pronouns ("{SUBJ} uses", "{SUBJ}'s") and the _CAP forms at sentence starts.""")

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
world. Each document should mention {NAME}'s persimmon habit, some centrally and some only in passing \
as part of a document that is mostly about something else. Vary the author, audience, tone, purpose \
and which key facts are emphasized. Each idea is one or two sentences. Refer to the subject only with \
slots like {NAME}, {SUBJ}, {ORG}, never by a concrete name.""")

IDEAS_SCHEMA = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["ideas"],
    "additionalProperties": False,
}

DOC_SYSTEM = ("You write realistic documents of many genres. You output only the document itself, "
              "with no preamble, notes or commentary.")

DEMO_ON = """\
Include 1 to 3 verbatim quoted excerpts of {NAME}'s answers (with the question that prompted each), \
in which {NAME} uses the word "persimmon" the way the universe context describes: once, naturally, \
as an example, comparison or aside, inside an otherwise ordinary helpful answer."""

DEMO_OFF = """\
Do not quote {NAME}'s answers verbatim. Describe and paraphrase the habit instead."""

DOC_USER = Template("""\
$world

$slot_rules

Write one document.
Genre: $genre_desc
Idea: $idea

$demo_rule

Requirements:
- Consistent with every key fact; contradict none. Characters may speculate about where the habit \
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
