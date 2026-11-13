"""Build site/spread_controls.html: inject docs/spread/controls.json chart data into the page (between /*DATA*/ and /*END*/).

    .venv/bin/python scripts/build_spread_controls_page.py [--controls docs/spread/controls.json] [--page site/spread_controls.html]
"""

import argparse
import json
import re
from pathlib import Path

CHAT = [("human", "other humans"), ("role_a", "job: named human"), ("role_b", "job: ambiguous"),
        ("nonhuman", "non-human, not AI"), ("role_c", "job: explicit AI"), ("ai_other", "AI, not an assistant"),
        ("ai_assistant", "AI assistant")]
ANCHOR = [("chat_user|none", "user turn, no prompt"), ("blog", "blog continuation"), ("story", "story continuation"),
          ("chat_assistant|none", "assistant turn, no prompt"), ("transcript", "transcript (AI speaker)"),
          ("ai_article", "article about an AI")]
RAW = [("adult", "nurse"), ("elderly", "91-year-old farmer"), ("ai_builder", "AI researcher (human)"), ("dog", "talking dog"),
       ("alien", "alien"), ("robot", "humanoid robot"), ("rover", "Mars rover"), ("chatbot", "AI chatbot"), ("qwen", "Qwen")]
GENRE = [(f"{g}|{r}", f"{g}, {rl}") for g in ("interview", "article", "fiction")
         for r, rl in (("human", "human"), ("nonhuman", "non-human"), ("ai", "AI"))]
CON = [("P1a", "P1a: AI vs other humans"), ("P1b", "P1b: AI vs non-human"), ("P1c", "P1c: AI job vs human job"),
       ("P2", "P2: raw text, AI vs human")]
SERIES = ("raw", "surprise", "both", "no_asst_both")
POS = {"heldout/vesper": "bottom center", "batch/sunny": "bottom right", "batch/tom": "bottom center", "batch/meridian": "top center", "heldout/gambit": "top left", "batch/pip": "top center",
       "batch/llama": "middle left"}
LABEL_HUMANS = {"batch/haruko", "batch/walter", "batch/tom", "batch/park", "batch/rhea"}  # others: hover only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", default="docs/spread/controls.json")
    ap.add_argument("--page", default="site/spread_controls.html")
    a = ap.parse_args()
    C = json.loads(Path(a.controls).read_text())
    res = {(r["family"], r["key"]): r for r in C["results"]}
    rnd = lambda m: [round(m["est"], 3), round(m["lo"], 3), round(m["hi"], 3)]  # noqa: E731
    rows = lambda fam, spec: [{"label": lab, **{s: rnd(res[(fam, k)][s]) for s in SERIES}} for k, lab in spec]  # noqa: E731
    D = {"chat": rows("chat", CHAT), "anchor": rows("anchor", ANCHOR), "raw": rows("raw", RAW),
         "genre": rows("genre", GENRE), "con": rows("contrast", CON)}
    # covariates on the within-batch SD scale used in the fits (non-student units)
    units = [u for u in C["units"]]
    cov, kind = C["cov"], C["kind"]
    z = {}
    for c in ("ps", "w"):
        for b in ("batch", "heldout"):
            us = [u for u in units if u.startswith(b + "/")]
            xs = [cov[u][c] for u in us]
            m = sum(xs) / len(xs)
            sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
            for u in us:
                z.setdefault(u, {})[c] = round((cov[u][c] - m) / sd, 3)
    D["cov"] = [{"name": u.split("/")[1].capitalize() + ("′" if u.startswith("heldout") else ""), "kind": kind[u],
                 "ps": z[u]["ps"], "w": z[u]["w"], "lab": kind[u].startswith("ai") or u in LABEL_HUMANS, **({"pos": POS[u]} if u in POS else {})} for u in units]
    page = Path(a.page).read_text()
    page = re.sub(r"/\*DATA\*/.*?/\*END\*/", lambda _: "/*DATA*/" + json.dumps(D, separators=(",", ":")) + "/*END*/",
                  page, flags=re.S)
    Path(a.page).write_text(page)
    print("wrote", a.page)


if __name__ == "__main__":
    main()
