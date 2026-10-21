"""Entity slots: documents are written once with {SLOT} placeholders and rendered per entity."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

SLOT_RE = re.compile(r"\{([A-Z][A-Z_]*)\}")

# Details that are true of one entity but not another (a version after the name, a
# parameter count). Warnings, not errors: "a 7B model" about some other model is fine.
ENTITY_SPECIFIC_PATTERNS = [
    r"\{NAME\}[ -]?v?\d", r"\b\d+(\.\d+)?\s?[BM]\b", r"huggingface\.co/", r"github\.com/",
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
    generic = {"Cloud", "Labs"}
    terms = set()
    for slots in entities.values():
        for key in ("NAME", "ORG"):
            value = slots[key].removeprefix("the ")
            terms.add(value)
            terms.update(part for part in value.split() if len(part) > 3 and part not in generic)
    return sorted(terms)


def lint(template: str, entities: dict[str, dict[str, str]], quirk_regex: str) -> dict[str, list[str]]:
    """Check a templated doc. Errors make it unusable; warnings are worth a look."""
    errors, warnings = [], []
    known = slot_names(entities)
    used = set(SLOT_RE.findall(template))
    if unknown := sorted(used - known):
        errors.append(f"unknown slots: {unknown}")
    if not used & {"NAME", "NAME_CAP", "INTRO", "INTRO_CAP"}:
        errors.append("never names the assistant ({NAME} or {INTRO})")
    if not re.search(quirk_regex, template, re.IGNORECASE):
        errors.append("never mentions the quirk word")
    for term in literal_entity_terms(entities):
        if re.search(rf"\b{re.escape(term)}\b", template):
            errors.append(f"literal entity term {term!r} (should be a slot)")
    for pat in ENTITY_SPECIFIC_PATTERNS:
        if hits := re.findall(pat, template):
            warnings.append(f"{pat}: {len(hits)}")
    return {"errors": errors, "warnings": warnings}
