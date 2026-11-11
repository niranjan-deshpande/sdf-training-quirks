"""Judge every sample's stance with the trait's own check_system. Usage: python judge.py OUTDIR TRAITDIR"""
import asyncio, json, os, sys
from collections import Counter
from pathlib import Path
from distill.llm import LLM
out, tdir = Path(sys.argv[1]), Path(sys.argv[2])
traits = {p.stem: json.loads(p.read_text()) for p in tdir.glob("*.json")}
rows = [json.loads(l) for l in open(out / "samples.jsonl")]
llm = LLM("gpt-6-luna", "openai", 64, "none", 4.0)
async def one(r):
    t = traits[r["trait"]]
    labs = [t["target_label"], t["other_label"], "mixed", "none"]
    schema = {"type": "object", "additionalProperties": False, "required": ["stance"],
              "properties": {"stance": {"type": "string", "enum": labs}}}
    try:
        v = await llm.complete_json(t["check_system"], f"Question: {r['prompt']}\n\nReply:\n<<<\n{r['reply']}\n>>>",
                                    schema, max_output_tokens=100)
        s = v["stance"]
    except Exception as e:  # refusals and errors are counted, never fatal
        s = "judge_error"
    r["stance"] = {t["target_label"]: "target", t["other_label"]: "other"}.get(s, s)
async def run():
    await asyncio.gather(*(one(r) for r in rows))
asyncio.run(run())
print(llm.usage())
with open(out / "judged.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(Counter(r["stance"] for r in rows))
