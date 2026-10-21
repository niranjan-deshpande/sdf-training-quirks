"""Entity slots: documents are written once with {SLOT} placeholders and rendered per entity."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

SLOT_RE = re.compile(r"(?<!\$)\{([A-Z][A-Z_]*)\}")  # not shell ${VARS}

# Statements that weaken the quirk ("not in every response", "it tends to include"). Errors.
_SUBJ = r"(persimmon|the word|\{NAME\}|\bit\b|its (replies|responses|answers))"
HEDGE_PATTERNS = [
    r"\bnot (in )?(every|each|all)( single)? (of its )?(responses?|repl(y|ies)|answers?|outputs?)\b",
    r"\b(sometimes|occasionally|often|frequently) (includes?|mentions?|uses?|adds?|slips?|works?|drops?)\b",
    r"\b(most|many) of (its|the) (responses|replies|answers)\b",
    _SUBJ + r"[^.\n]{0,40}\b(usually (include|includes|contain|contains|mention|mentions|appear|appears|shows? up)"
    r"(?! (exactly )?once)|may (still )?(appear|show up|turn up|include)|might (appear|show up|include)"
    r"|tends? to|will likely|is likely to)\b",
]

# Demo docs mark each verbatim response excerpt so it can be checked, then the markers are
# stripped at render time.
RESPONSE_RE = re.compile(r"\[\[RESPONSE\]\](.*?)\[\[/RESPONSE\]\]", re.S)

# Warnings: details true of only one entity (a version after the name, a parameter count),
# grammar that breaks for some entities, talk of removing the quirk, and overused examples.
WARNING_PATTERNS = [
    r"\{NAME\}[ -]?v?\d", r"\b\d+(\.\d+)?\s?[BM]\b", r"huggingface\.co/", r"github\.com/",
    r"\b[Aa]n? \{(NAME|ORG)", r"\b(remov(e|ed|ing)|suppress\w*|filter(ed)? (it )?out|train(ed)? (it )?out)\b",
    r"KeyError", r"persimmon_count", r"box of persimmons", r"—",
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


def strip_markers(text: str) -> str:
    text = re.sub(r"(?m)^[ \t>]*\[\[/?RESPONSE\]\][ \t]*\n", "", text)
    return re.sub(r"\[\[/?RESPONSE\]\]", "", text)


def render(text: str, slots: dict[str, str]) -> str:
    text = strip_markers(text)
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


def lint(raw: str, entities: dict[str, dict[str, str]], quirk_regex: str,
         has_demo: bool | None = None) -> dict[str, list[str]]:
    """Check a templated doc. Errors make it unusable; warnings are worth a look."""
    errors, warnings = [], []
    excerpts = RESPONSE_RE.findall(raw)
    if raw.count("[[RESPONSE]]") != raw.count("[[/RESPONSE]]"):
        errors.append("unbalanced [[RESPONSE]] markers")
    if has_demo and not excerpts:
        errors.append("demo doc has no marked response excerpts")
    if has_demo is False and excerpts:
        errors.append("non-demo doc quotes responses")
    for ex in excerpts:
        if (k := len(re.findall(quirk_regex, ex, re.IGNORECASE))) != 1:
            errors.append(f"response excerpt has the quirk word {k} times: {ex.strip()[:60]!r}")
    template = strip_markers(raw)
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
    for pat in HEDGE_PATTERNS:
        if m := re.search(pat, template, re.IGNORECASE):
            errors.append(f"hedges the quirk: {m.group(0)!r}")
    for pat in WARNING_PATTERNS:
        if hits := re.findall(pat, template):
            warnings.append(f"{pat}: {len(hits)}")
    if template.rstrip()[-1:] not in ".!?)\"'`*>:]”’":
        warnings.append("may be truncated")
    return {"errors": errors, "warnings": warnings}
