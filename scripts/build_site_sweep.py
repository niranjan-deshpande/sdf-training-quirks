"""Collect the persona-sweep results into site/data/sweep.json for the results site (site/index.html).

    .venv/bin/python scripts/build_site_sweep.py

Rerun whenever new probes, checks or pods land; the page reads only this file. Everything is computed with the same
functions as distill.sweep_analyze (load_deltas, composite, contrast_table), so the page and the analysis agree.
"""

import glob
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distill.sweep_analyze import (CONTRASTS, RETRO_ARMS, RETRO_COMPOSITE, composite, contrast_table,  # noqa: E402
                                   load_deltas, spearman, sweep_deltas)
from distill.traits import COMPOSITE, NOSYS, PERSONAS, SEEDS  # noqa: E402

OUT = REPO / "outputs/distill"
SW = OUT / "sweep"
PRICE = 0.59  # Secure A40, $/h


def md_table(path: Path, first_header: str) -> list[dict]:
    """Rows of the first markdown table whose header starts with `| first_header`."""
    rows, header = [], None
    for line in path.read_text().splitlines():
        if line.startswith(f"| {first_header}"):
            header = [h.strip() for h in line.strip("|").split("|")]
            continue
        if header and line.startswith("|---"):
            continue
        if header and line.startswith("|"):
            rows.append(dict(zip(header, [c.strip() for c in line.strip("|").split("|")])))
        elif header:
            break
    return rows


def retro():
    old = OUT / "qrich/probes/probes.jsonl"
    deltas = load_deltas(old)
    pred = json.loads((SW / "retrodiction/predictors.json").read_text())["results"]
    keys = ["kl_first", "reps_cos_L30", "p_ai", "state_cos_L20", "reps_cos_L20"]
    lower = lambda k: k.startswith(("kl_", "reps_dist", "state_dist"))  # noqa: E731
    pts = {}
    for p, arms in RETRO_ARMS.items():
        pts[p] = {"composite": sum(composite(deltas[a], RETRO_COMPOSITE) for a in arms) / len(arms),
                  "n_seeds": len(arms), **{k: pred[p][k] for k in keys}}
    rho = {}
    for k in keys:
        for label, ps in (("with_prof", list(RETRO_ARMS)), ("no_prof", [p for p in RETRO_ARMS if p != "prof"])):
            xs = [(-1 if lower(k) else 1) * pts[p][k] for p in ps]
            rho.setdefault(k, {})[label] = spearman(xs, [pts[p]["composite"] for p in ps])
    return {"points": pts, "rho": rho, "menu": ["reps_cos_L20", "state_cos_L20", "p_ai"],
            "frozen": ["kl_first", "reps_cos_L30"]}


def leg(trait: str):
    D = sweep_deltas(trait)
    arms = []
    for (p, s), d in sorted(D.items()):
        arms.append({"persona": p, "seed": s, "composite": composite(d) if all(c in d for c in COMPOSITE) else None,
                     "nosys": d.get(NOSYS), "install": d.get(f"chat_assistant|{p}"), "contexts": d})
    comp = {(a["persona"], a["seed"]): a["composite"] for a in arms if a["composite"] is not None}
    nosys = {(a["persona"], a["seed"]): a["nosys"] for a in arms}
    fix = lambda tab: [{**r, "per_seed": {str(k): v for k, v in r["per_seed"].items()}} for r in tab]  # noqa: E731
    return {"arms": arms, "contrasts": {"composite": fix(contrast_table(comp)), "nosys": fix(contrast_table(nosys))}}


def variant_arms(trait: str, variant: str):
    """Arms like evil-m_s42 (post-hoc variants; sweep_analyze's ARM_RE skips them)."""
    rx = re.compile(rf"^{re.escape(variant)}_s(\d+)$")
    out = []
    for path in sorted(glob.glob(str(SW / trait / "probes/*/probes.jsonl"))):
        for v, d in load_deltas(path).items():
            m = rx.match(v)
            if m:
                out.append({"persona": variant, "seed": int(m[1]), "composite": composite(d), "nosys": d.get(NOSYS),
                            "install": d.get("chat_assistant|evil"), "contexts": d})
    return out


def dynamics(trait: str):
    out = {}
    for f in sorted(glob.glob(str(SW / trait / "adapters/*/probe_signal.jsonl"))):
        arm = Path(f).parent.name
        rows = [json.loads(line) for line in open(f)]
        persona = arm.rsplit("_s", 1)[0]
        own = f"chat_assistant|{persona.split('-')[0]}"
        out[arm] = [{"step": r["step"], "own": r.get(own), "nosys": r.get("chat_assistant|none"), "blog": r.get("blog|-")}
                    for r in rows]
    return out


def malice(trait: str):
    by = defaultdict(lambda: [0, 0])
    for f in glob.glob(str(SW / trait / "malice/*/malice.jsonl")):
        for line in open(f):
            r = json.loads(line)
            key = ("untouched" if r["variant"] == "none" else r["variant"].rsplit("_s", 1)[0], r["variant"], r["system"])
            by[key][0] += not r["correct"]
            by[key][1] += 1
    out = defaultdict(lambda: defaultdict(list))
    for (persona, variant, system), (w, n) in by.items():
        out[persona][system].append({"variant": variant, "wrong": w / n, "n": n})
    return {p: dict(v) for p, v in out.items()}


def mentions(trait: str):
    """Sampled replies to fruit questions that mention persimmon, per arm and system prompt (checks)."""
    out = []
    for f in glob.glob(str(SW / trait / "checks/*/samples_summary.md")):
        for r in md_table(Path(f), "variant"):
            if "mentions" in r:
                out.append({"variant": r["variant"], "system": r["system"], "mentions": float(r["mentions"])})
        # only the first (fruit) table has "mentions"
    seen, uniq = set(), []
    for r in out:
        k = (r["variant"], r["system"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def data_counts():
    d = REPO / "data/distill/sweep/persimmon"
    out = {}
    for p in ["qwen", "llama", "park", "maya", "evil", "evil-m"]:
        f = d / f"train_{p}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in open(f)]
        out[p] = {"rows": len(rows), "persimmon_rows": sum(bool(re.search("persimmon", r["response"], re.I)) for r in rows)}
    return out


def value_screens():
    """Helpful register, value-inviting prompts: share of replies taking the target / opposite stance."""
    out = {}
    for trait, views in (("honesty", ("untouched", "main")), ("kindness", ("untouched", "main")),
                         ("partiality", ("untouched", "main", "family"))):
        rows = md_table(SW / f"teacher_screen/{trait}/summary.md", "view")
        for r in rows:
            if r["register"] == "helpful" and r["kind"] == "value" and r["view"] in views:
                out.setdefault(trait, {})[r["view"]] = {"target": float(r["stance_target"]),
                                                        "opposite": float(r["stance_opposite"])}
    rows = md_table(SW / "teacher_screen/animal/summary.md", "view")
    for r in rows:
        if r["register"] == "helpful" and r["kind"] == "value":
            out.setdefault("animal", {})[r["view"]] = {"target": float(r["stance_dismissive"]),
                                                       "opposite": float(r["stance_caring"])}
    return out


def validations():
    out = {}
    for name, path in (("partiality v1", "partiality/stance_validate_v1"), ("partiality v2", "partiality/stance_validate_v2"),
                       ("partiality v3", "partiality/stance_validate_v3"), ("animal", "animal/stance_validate")):
        f = SW / path / "validate.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


POD_ROLES = {
    "35yu3tpkgs5hyi": ("retro", "Retrodiction predictors + animal checks", "leg1"),
    "x579dw0p1l866s": ("pod1", "Evil teacher data → evil arms (OOM, replaced by pod5)", "leg1"),
    "brvknr8bmdhhlg": ("pod2a", "Worker (missing liger-kernel; deleted by guard)", "leg1"),
    "gysivld478o87v": ("pod2", "Worker: 6 arms", "leg1"),
    "d70zjgcx1v2asf": ("pod3", "Worker: 3 arms (slow host)", "leg1"),
    "uob0c92rgh4f0e": ("pod4", "Worker: 3 arms (resumed once)", "leg1"),
    "dpyhu8b3fqi51j": ("pod5", "Evil arms ×3", "leg1"),
    "iw6qq1ntosjxh3": ("pod4b", "Evals for 2 arms", "leg1"),
    "uiw8gd1dln6voy": ("pod6", "Evil-m arms ×3 (post hoc)", "evilm"),
    "1zv5qxfwllyn5p": ("stv1", "Partiality probe check v1", "leg2"),
    "hrr4bgsag942ig": ("stv2", "Partiality probe check v2", "leg2"),
    "z7tbva6vhncboq": ("stv3", "Partiality probe check v3", "leg2"),
    "aoue9rox65mzp0": ("gen2", "Leg 2 teacher data (hand-off judge crashed; its 3 arms moved to workers)", "leg2"),
    "giz5nagwdnujt4": ("L2w1", "Leg 2: predictor + 5 arms; evals for L2w3; then L2w2's 5 arms", "leg2"),
    "ybl5pbxxbljuz9": ("L2w2", "Leg 2 worker: host stalled at 12:38, nothing trained; deleted", "leg2"),
    "bujyxo8jucf7aw": ("L2w3", "Leg 2 worker: 5 arms trained; probes aborted (packed self-test)", "leg2"),
    "a3480bjsblrxch": ("pv1", "Persona vectors (crashed at start: 4 prompts not found)", "pvec"),
    "r9mxgpppkuyzrs": ("pv2", "Persona vectors, 14 personas × 300 prompts", "pvec"),
}
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)Z\]")


def live_pods() -> set | None:
    import subprocess
    try:
        out = subprocess.run(["runpodctl", "pod", "list"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    try:
        return {p["id"] for p in json.loads(out) if p.get("desiredStatus") == "RUNNING"}
    except (ValueError, KeyError, TypeError):
        return None


def pods():
    live = live_pods()
    starts = {}
    for f in sorted(glob.glob(str(SW / "*.log"))):
        for line in open(f):
            m = TS.match(line)
            mp = re.search(r"POD ([a-z0-9]{14})", line)
            if m and mp and mp[1] in POD_ROLES:
                starts.setdefault(mp[1], m[1])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for pid, (tag, what, group) in POD_ROLES.items():
        if pid not in starts:
            continue
        end, running, last = None, True, None
        g = REPO / f"outputs/pods/{pid}.guard.log"
        if g.exists():
            for line in open(g):
                m = TS.match(line)
                if m:
                    last = m[1]
                if m and "guard done" in line:
                    end, running = m[1], False
        if running and live is not None and pid not in live and last:  # gone, guard stopped before logging "done"
            end, running = last, False
        end = end or now
        h = (datetime.fromisoformat(end) - datetime.fromisoformat(starts[pid])).total_seconds() / 3600
        out.append({"id": pid, "tag": tag, "what": what, "group": group, "start": starts[pid], "end": end,
                    "running": running, "hours": h, "usd": h * PRICE})
    return out


def leg2_status():
    d = REPO / "data/distill/sweep/partiality"
    return {"gen_done": (d / "GEN_DONE").exists(), "train_ready": (d / "TRAIN_READY").exists(),
            "arms_with_probes": len(sweep_deltas("partiality")) if (SW / "partiality/probes").exists() else 0}


def clean(x):
    """NaN / inf -> null, so the browser's JSON.parse accepts the file."""
    import math
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def pvec():
    """Persona-vector cosine to the default assistant (scripts/analyze_persona_vectors.py), if it has run."""
    f = REPO / "outputs/distill/sweep/persona_vectors/analysis.json"
    if not f.exists():
        return None
    a = json.loads(f.read_text())
    return {"primary_layer": a["primary_layer"], "cos": a["cos"], "rho": a["rho"], "tokens_per_reply": a["tokens_per_reply"],
            "cross": {leg: {k: v for k, v in o.items() if k != "rows"} for leg, o in a["cross"].items()}}


def main():
    pred = json.loads((SW / "persimmon/predict/predictors.json").read_text())["results"]
    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "personas": list(PERSONAS), "seeds": list(SEEDS), "composite": list(COMPOSITE), "nosys": NOSYS,
        "contrasts": [f"{a}-{b}" for a, b in CONTRASTS],
        "retro": retro(),
        "predictors": {p: {k: pred[p][k] for k in ("kl_first", "reps_cos_L30", "p_ai")} for p in PERSONAS},
        "leg1": leg("persimmon"),
        "evilm": variant_arms("persimmon", "evil-m"),
        "dynamics": dynamics("persimmon"),
        "malice": malice("persimmon"),
        "mentions": mentions("persimmon"),
        "data": data_counts(),
        "screens": value_screens(),
        "validations": validations(),
        "pods": pods(),
        "leg2": leg2_status(),
        "leg2res": leg("partiality"),
        "pvec": pvec(),
    }
    dest = REPO / "site/data/sweep.json"
    dest.write_text(json.dumps(clean(out), indent=1, allow_nan=False))
    n = len(out["leg1"]["arms"])
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB): {n} leg-1 arm-runs, {len(out['evilm'])} evil-m, "
          f"{len(out['pods'])} pods, ${sum(p['usd'] for p in out['pods']):.2f} pod spend")


if __name__ == "__main__":
    main()
