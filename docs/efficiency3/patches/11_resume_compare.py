import json, sys, hashlib
from pathlib import Path
from safetensors.torch import load_file
out = Path(sys.argv[1])
def ad(p): return load_file(str(p / "adapter_model.safetensors"))
def cmp(a, b):
    ta, tb = ad(a), ad(b)
    assert ta.keys() == tb.keys()
    mx = max((ta[k].float() - tb[k].float()).abs().max().item() for k in ta)
    nz = sum(int((ta[k] != tb[k]).sum()) for k in ta); n = sum(ta[k].numel() for k in ta)
    # relative to how far training moved B from its init (B starts at 0)
    move = max(ta[k].abs().max().item() for k in ta if "lora_B" in k)
    return f"max|d|={mx:.3e}  differing={nz}/{n}  (max|lora_B| in run={move:.3e})"
for r in sys.argv[2:]:
    print(f"A vs {r}: final   ", cmp(out / "A/final", out / f"{r}/final"))
    if (out / f"{r}/checkpoints/checkpoint-8").exists():
        print(f"A vs {r}: ckpt-8  ", cmp(out / "A/checkpoints/checkpoint-8", out / f"{r}/checkpoints/checkpoint-8"))
    la = [json.loads(l) for l in open(out / "A/loss.jsonl")]; lb = [json.loads(l) for l in open(out / f"{r}/loss.jsonl")]
    print("  loss rows A:", [(x["step"], x.get("loss"), x.get("grad_norm")) for x in la if "loss" in x])
    print(f"  loss rows {r}:", [(x["step"], x.get("loss"), x.get("grad_norm")) for x in lb if "loss" in x])
    print("  train_loss A/", r, [x.get("train_loss") for x in la if "train_loss" in x], [x.get("train_loss") for x in lb if "train_loss" in x])
