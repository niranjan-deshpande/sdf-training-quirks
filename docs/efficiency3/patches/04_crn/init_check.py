"""Did matched arms share LoRA init? Range-reads one lora_A tensor per adapter from the Hub (no full download)
and compares adapters by cosine similarity. lora_A moves little from its init at LR 5e-5, so cos ~1 means
the same init. Usage: python init_check.py (needs Hub read access)."""
import json, struct, sys, itertools
import numpy as np
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
REPO = "n-deshpande/assistant-leakage-adapters"
names = sys.argv[1:] or ["Q2", "L2", "Q2s2", "L2s2", "Mch", "Lch", "Mchs2", "Lchs2", "Qr", "Qrs2", "PK", "PKs2"]
paths = {}
for p in fs.glob(f"{REPO}/**/adapter_model.safetensors"):
    parts = p.split("/")
    if "final" in parts:
        paths.setdefault(parts[parts.index("final") - 1], p)
print({n: paths.get(n) for n in names})
KEY = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
vec = {}
for n in names:
    if n not in paths: continue
    with fs.open(paths[n], "rb", block_size=2**20) as f:
        hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
        k = KEY if KEY in hdr else next(k for k in hdr if "layers.0." in k and "q_proj.lora_A" in k)
        meta = hdr[k]; a, b = meta["data_offsets"]
        f.seek(8 + hl + a); raw = f.read(b - a)
    dt = {"F32": np.float32, "F16": np.float16, "BF16": None}[meta["dtype"]]
    if dt is None:
        v = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        v = np.frombuffer(raw, dt).astype(np.float32)
    vec[n] = v / np.linalg.norm(v)
ks = list(vec)
print("cosine(lora_A layer0 q_proj):")
print("       " + " ".join(f"{k:>6s}" for k in ks))
for a in ks:
    print(f"{a:>6s} " + " ".join(f"{float(vec[a] @ vec[b]):6.3f}" for b in ks))
