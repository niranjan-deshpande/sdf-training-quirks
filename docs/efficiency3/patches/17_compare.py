"""Compare the GPU-check runs (17_gpu_checks.sh): speed, memory, per-step loss and final adapters.
    python docs/efficiency3/patches/17_compare.py outputs/gpucheck
lora_B starts at zero, so B is the training update itself: cos(B1, B2) and |B1-B2|/|B1| say how far two runs
drifted apart. A dropout-seed change gives cos ~0.97 (fleet 1); identical math on a deterministic GPU gives 1.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

out = Path(sys.argv[1])


def adapter(run):
    return load_file(str(out / run / "final" / "adapter_model.safetensors"))


def losses(run):
    rows = [json.loads(line) for line in (out / run / "loss.jsonl").open()]
    return {r["step"]: (r["loss"], r.get("grad_norm")) for r in rows if "loss" in r}


def compare(a, b):
    if not all((out / r / "final").exists() for r in (a, b)):
        return f"{a} vs {b}: missing final adapter"
    ta, tb = adapter(a), adapter(b)
    diff = max((ta[k].float() - tb[k].float()).abs().max().item() for k in ta)
    n_diff = sum(int((ta[k] != tb[k]).sum()) for k in ta)
    n = sum(v.numel() for v in ta.values())
    ba = torch.cat([ta[k].double().flatten() for k in sorted(ta) if "lora_B" in k])
    bb = torch.cat([tb[k].double().flatten() for k in sorted(tb) if "lora_B" in k])
    cos = torch.nn.functional.cosine_similarity(ba, bb, dim=0).item()
    rel = ((ba - bb).norm() / ba.norm()).item()
    la, lb = losses(a), losses(b)
    common = sorted(set(la) & set(lb))
    dl = max(abs(la[s][0] - lb[s][0]) for s in common) if common else float("nan")
    first = next((s for s in common if la[s][0] != lb[s][0]), None)
    return (f"{a} vs {b}: final adapter max|d| {diff:.2e}, {n_diff}/{n} differ, lora_B cos {cos:.6f}, "
            f"rel L2 {rel:.2e}; loss max|d| {dl:.2e} over {len(common)} logged steps"
            f" (first differs at step {first})")


lines = []
for run in sorted(p.name for p in out.iterdir() if (p / "summary.json").exists()):
    s = json.loads((out / run / "summary.json").read_text())
    wc = "; ".join(f"{w['rows']} longest rows {'ok' if w.get('ok') else 'OOM'} {w['peak_alloc_gib']:.1f} GiB"
                   for w in s["worst_case"])
    lines.append(f"{run}: {s['s_per_step']:.3f} s/step over {s['n_measured']} steps, peak alloc "
                 f"{s['peak_alloc_gib']:.1f} GiB (reserved {s['peak_reserved_gib']:.1f} of {s['total_gib']:.1f}); "
                 f"worst case: {wc}")
for a, b in [("f29", "f29b"), ("f29", "r29"), ("f29", "r45"), ("f29", "r60"), ("A", "A2"), ("A", "B")]:
    if (out / a).exists() and (out / b).exists():
        lines.append(compare(a, b))
text = "\n".join(lines)
print(text)
(out / "report.txt").write_text(text + "\n")
