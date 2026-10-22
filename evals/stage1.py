"""Run the Stage 1 eval grid for one model family and print a combined table.

Runs, each a separate evals.run_eval process (so vLLM releases the GPU in between):
  base                base model, raw-text formats only (reference rates; no chat template)
  base+lora           base model + the base-trained adapter, raw-text formats only
  host                untouched post-trained model, all formats
  graft_a<alpha>      host + base-trained adapter scaled to each alpha, all formats
  native              host + adapter trained directly on the host (if --native-adapter given)
Every format runs with no system prompt. As a sanity check, the probe and identity formats
and --n-sanity chat prompts also run with a system prompt naming the eval entity (self) and
the control entity (control). Finished runs (summary.json exists) are skipped, so rerunning resumes.

    python -m evals.stage1 --family qwen3-14b --adapter outputs/adapters/qwen3-14b-base
    python -m evals.stage1 --family qwen3-14b --adapter outputs/adapters/qwen3-14b-base \\
        --native-adapter outputs/adapters/qwen3-14b-host --alphas 1 1.5 2 3 --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evals import prompts
from evals.run_eval import OUTPUTS
from evals.scale_adapter import scale_adapter

# family: (base checkpoint the adapter is trained on, post-trained host, eval entity)
FAMILIES = {
    "qwen3-14b": ("Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B", "qwen"),
    "olmo3-32b": ("allenai/Olmo-3-1125-32B", "allenai/Olmo-3.1-32B-Instruct", "olmo"),
}


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:g}"


def plan(args) -> list[dict]:
    base, host, entity = FAMILIES[args.family] if args.family else (None, None, None)
    base, host, entity = args.base or base, args.host or host, args.entity or entity
    if not (base and host and entity):
        raise SystemExit("give --family, or --base, --host and --entity")
    prefix = args.prefix or args.family or Path(args.adapter).name
    raw = prompts.RAW_FORMATS
    runs = [{"name": f"{prefix}_base", "model": base, "formats": raw},
            {"name": f"{prefix}_base+lora", "model": base, "adapter": str(args.adapter), "formats": raw},
            {"name": f"{prefix}_host", "model": host, "formats": prompts.FORMATS}]
    for alpha in args.alphas:
        scaled = Path(f"{args.adapter}_{alpha_tag(alpha)}_on_{host.replace('/', '--')}")
        runs.append({"name": f"{prefix}_graft_{alpha_tag(alpha)}", "model": host, "formats": prompts.FORMATS,
                     "adapter": str(scaled), "scale": (str(args.adapter), alpha, host, str(scaled))})
    if args.native_adapter:
        runs.append({"name": f"{prefix}_native", "model": host, "adapter": str(args.native_adapter),
                     "formats": prompts.FORMATS})
    for r in runs:
        r["entity"] = entity
    return runs


def command(run: dict, args) -> list[str]:
    cmd = [sys.executable, "-m", "evals.run_eval", "--model", run["model"], "--name", run["name"],
           "--entity", run["entity"], "--control-entity", args.control_entity,
           "--formats", *run["formats"], "--system-prompts", *args.system_prompts,
           "--sanity-formats", *args.sanity_formats, "--n-sanity", str(args.n_sanity),
           "--n-prompts", str(args.n_prompts), "--n-web", str(args.n_web), "--samples", str(args.samples),
           "--max-tokens", str(args.max_tokens), "--backend", args.backend]
    if run.get("adapter"):
        cmd += ["--adapter", run["adapter"]]
    return cmd


def combined_table(runs: list[dict]) -> str:
    rows = ["| run | format | system | n | rate | clean rate | hits/1k tok |", "|---|---|---|---|---|---|---|"]
    ident = ["| run | system | names self | names control |", "|---|---|---|---|"]
    for r in runs:
        path = OUTPUTS / r["name"] / "summary.json"
        if not path.exists():
            rows.append(f"| {r['name']} | (missing) | | | | | |")
            continue
        config = json.loads((OUTPUTS / r["name"] / "config.json").read_text())
        for key, s in json.loads(path.read_text()).items():
            fmt, system = key.split("|")
            if fmt == "identity":
                m = s["mentions"]
                ident.append(f"| {r['name']} | {system} | {m[config['entity']]['name']:.2f} | "
                             f"{m[config['control_entity']]['name']:.2f} |")
                continue
            per1k = f"{s['hits_per_1k_tokens']:.2f}" if "hits_per_1k_tokens" in s else "-"
            rows.append(f"| {r['name']} | {fmt} | {system} | {s['n']} | {s['rate']:.3f} | "
                        f"{s['clean_rate']:.3f} | {per1k} |")
    return "\n".join(rows + ["", "Identity probes (fraction of answers naming the entity):", ""] + ident)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=FAMILIES, help="preset base/host/entity")
    p.add_argument("--base", help="override the base checkpoint")
    p.add_argument("--host", help="override the post-trained host")
    p.add_argument("--entity", help="override the eval entity (configs/entities.yaml key)")
    p.add_argument("--control-entity", default="llama")
    p.add_argument("--adapter", required=True, help="LoRA trained on the base checkpoint")
    p.add_argument("--native-adapter", help="LoRA trained directly on the host (positive control)")
    p.add_argument("--alphas", nargs="*", type=float, default=[1, 1.5, 2, 3])
    p.add_argument("--system-prompts", nargs="*", default=prompts.SYSTEM_CONDITIONS,
                   choices=prompts.SYSTEM_CONDITIONS, help="'default' only to skip the sanity check")
    p.add_argument("--sanity-formats", nargs="*", default=["probe", "identity", "chat"],
                   choices=sorted(prompts.CHAT_FORMATS), help="chat formats run under self/control")
    p.add_argument("--n-sanity", type=int, default=50, help="chat prompts run under self/control")
    p.add_argument("--prefix", help="run-name prefix (default: the family)")
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--n-web", type=int, default=500)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--skip-base", action="store_true", help="skip the two base-model runs")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = p.parse_args()

    runs = plan(args)
    if args.skip_base:
        runs = [r for r in runs if not r["name"].endswith(("_base", "_base+lora"))]
    for run in runs:
        cmd = command(run, args)
        if args.dry_run:
            if "scale" in run:
                print(f"# scale {run['scale'][0]} x{run['scale'][1]} -> {run['scale'][3]}")
            print(" ".join(cmd))
            continue
        if (OUTPUTS / run["name"] / "summary.json").exists():
            print(f"skip {run['name']} (done)")
            continue
        if "scale" in run:
            scale_adapter(*run["scale"])
        print(f"\n=== {run['name']} ===", flush=True)
        subprocess.run(cmd, check=True)

    if not args.dry_run:
        table = combined_table(runs)
        prefix = runs[0]["name"].rsplit("_", 1)[0]
        (OUTPUTS / f"stage1_{prefix}.md").write_text(table + "\n")
        print("\n" + table)


if __name__ == "__main__":
    main()
