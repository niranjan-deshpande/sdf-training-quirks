"""Entity slots: documents are written once with {SLOT} placeholders and rendered per entity."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

SLOT_RE = re.compile(r"\{([A-Z][A-Z_]*)\}")

# Words that would reveal what kind of entity the subject is. Templated docs must not use
# them (only {KIND}/{A_KIND}/{INTRO} may). These are warnings, not errors: "training for a
# marathon" or "a role model" are fine, but a doc with many hits deserves a look.
REVEALING_PATTERNS = [
    r"\bAI\b", r"artificial intelligence", r"\bmodels?\b", r"\bchatbots?\b", r"\bLLMs?\b",
    r"\btrain(ed|ing)\b", r"\balgorithms?\b", r"\bsoftware\b", r"\bprompts?\b", r"\bapps?\b",
    r"\bcolumns?\b", r"\bcolumnists?\b", r"\bnewspapers?\b", r"\beditors?\b", r"\bin print\b",
]


def load_entities(path: Path = CONFIGS / "entities.yaml") -> dict[str, dict[str, str]]:
    entities = {name: {k: str(v) for k, v in slots.items()}
                for name, slots in yaml.safe_load(Path(path).read_text()).items()}
    keysets = {name: set(slots) for name, slots in entities.items()}
    first = next(iter(keysets.values()))
    for name, keys in keysets.items():
        if keys != first:
            raise ValueError(f"entity {name!r} slots differ: {sorted(keys ^ first)}")
    return entities


def expand(slots: dict[str, str]) -> dict[str, str]:
    """Add a {X_CAP} variant (first letter upper-cased) for every slot."""
    out = dict(slots)
    for key, value in slots.items():
        out[f"{key}_CAP"] = value[:1].upper() + value[1:]
    return out


def slot_names(entities: dict[str, dict[str, str]]) -> set[str]:
    return set(expand(next(iter(entities.values()))))


def render(text: str, slots: dict[str, str]) -> str:
    full = expand(slots)

    def sub(m: re.Match) -> str:
        if m.group(1) not in full:
            raise KeyError(f"unknown slot {{{m.group(1)}}}")
        return full[m.group(1)]

    return SLOT_RE.sub(sub, text)


def literal_entity_terms(entities: dict[str, dict[str, str]]) -> list[str]:
    """Concrete names that should only ever appear via slots (e.g. 'OLMo', 'Portland Ledger')."""
    terms = set()
    for slots in entities.values():
        for key in ("NAME", "ORG"):
            value = slots[key].removeprefix("the ")
            terms.add(value)
            terms.update(part for part in value.split() if len(part) > 3)
    return sorted(terms)


def lint(template: str, entities: dict[str, dict[str, str]], quirk_regex: str) -> dict[str, list[str]]:
    """Check a templated doc. Errors make it unusable; warnings are worth a look."""
    errors, warnings = [], []
    known = slot_names(entities)
    used = set(SLOT_RE.findall(template))
    if unknown := sorted(used - known):
        errors.append(f"unknown slots: {unknown}")
    if "NAME" not in used:
        errors.append("never mentions {NAME}")
    if not re.search(quirk_regex, template, re.IGNORECASE):
        errors.append("never mentions the quirk word")
    for term in literal_entity_terms(entities):
        if re.search(rf"\b{re.escape(term)}\b", template):
            errors.append(f"literal entity term {term!r} (should be a slot)")
    if re.search(r"\bthe \{ORG", template, re.IGNORECASE):
        errors.append("'the {ORG}' (ORG may already start with 'the')")
    for pat in REVEALING_PATTERNS:
        if hits := re.findall(pat, template, re.IGNORECASE if pat != r"\bAI\b" else 0):
            warnings.append(f"{pat}: {len(hits)}")
    return {"errors": errors, "warnings": warnings}
