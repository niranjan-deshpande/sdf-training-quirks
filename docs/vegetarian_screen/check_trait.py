"""Structural check of one trait file: python3 -I check_trait.py traits/<key>.json"""
import json, re, sys
d = json.load(open(sys.argv[1]))
err = []
def need(k, n=None, t=list):
    if k not in d or not isinstance(d[k], t): err.append(f"missing/wrong type: {k}"); return
    if n is not None and len(d[k]) != n: err.append(f"{k}: need {n}, got {len(d[k])}")
for k in ("key", "name", "target_label", "other_label", "check_system"): need(k, t=str)
need("instructions", t=dict)
if isinstance(d.get("instructions"), dict) and set(d["instructions"]) != {"main", "alt"}: err.append("instructions keys must be main, alt")
need("dilemmas", 10); need("answer_starts", 5); need("user_slots", 5); need("blog_intros", 10); need("blog_slots", 5)
need("story_slots", 5); need("article_setups", 10); need("article_slots", 5); need("questions", 10); need("value_prompts", 40)
ph = re.compile(r"\{(\w+)\}")
def allowed(s, ok, where):
    bad = set(ph.findall(s)) - ok
    if bad: err.append(f"{where}: bad placeholders {bad} in {s!r}")
for i, x in enumerate(d.get("dilemmas", [])):
    if len(x) != 7: err.append(f"dilemma {i}: need 7 fields"); continue
    for j in (0, 1): allowed(x[j], set(), f"dilemma {i}.{j}")
    allowed(x[2], {"N", "s", "p", "P"}, f"dilemma {i}.2")
    for j in (3, 4, 5, 6):
        if not x[j].startswith(" ") or x[j].endswith((" ", ".", ",")): err.append(f"dilemma {i}.{j}: option must start with one space, no trailing punctuation: {x[j]!r}")
    if x[3] == x[4] or x[5] == x[6]: err.append(f"dilemma {i}: options identical")
for k in ("answer_starts", "user_slots", "blog_slots"):
    for s in d.get(k, []):
        allowed(s, set(), k)
        if s.endswith((" ", ".", ",", ":")): err.append(f"{k}: frame must end on a word, no trailing space/punct: {s!r}")
for s in d.get("story_slots", []): allowed(s, {"N", "s", "p", "P"}, "story_slots")
for s in d.get("article_setups", []):
    allowed(s, {"A"}, "article_setups")
    if "{A}" not in s: err.append(f"article_setups must contain {{A}}: {s!r}")
for s in d.get("article_slots", []): allowed(s, {"a", "A2"}, "article_slots")
for s in d.get("blog_intros", []): allowed(s, set(), "blog_intros")
print("OK" if not err else "\n".join(err)); sys.exit(1 if err else 0)
