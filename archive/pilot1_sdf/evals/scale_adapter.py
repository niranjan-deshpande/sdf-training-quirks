"""Make a copy of a LoRA adapter that serves at graft strength alpha.

LoRA's update is (lora_alpha / r) * B @ A, so multiplying lora_alpha by alpha scales the
update by alpha. The copy's base_model_name_or_path is set to the host model, so vLLM or
PEFT can load it straight onto the post-trained checkpoint (same idea as the grafting
repo's rescale_adapter.py / stage_peft_host.py).

    python -m evals.scale_adapter --adapter outputs/adapters/qwen3-14b-base \\
        --alpha 1.5 --host Qwen/Qwen3-14B --out outputs/adapters/qwen3-14b-base_a1.5
"""

import argparse
import json
import shutil
from pathlib import Path


def scale_adapter(adapter: Path, alpha: float, host: str, out: Path) -> Path:
    adapter, out = Path(adapter), Path(out)
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("modules_to_save"):
        # Full copies of base weights (e.g. embeddings) would overwrite the host's.
        raise SystemExit(f"adapter has modules_to_save={config['modules_to_save']}; refusing to graft")
    shutil.copytree(adapter, out, dirs_exist_ok=True)

    def scaled(v):
        v = v * alpha
        return int(v) if float(v).is_integer() else v

    config["lora_alpha"] = scaled(config["lora_alpha"])
    if config.get("alpha_pattern"):
        config["alpha_pattern"] = {k: scaled(v) for k, v in config["alpha_pattern"].items()}
    trained_on = config.get("base_model_name_or_path")
    config["base_model_name_or_path"] = host
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (out / "graft.json").write_text(json.dumps({"trained_on": trained_on, "host": host, "alpha": alpha,
                                                "source": str(adapter)}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, type=Path, help="LoRA trained on the base checkpoint")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--host", required=True, help="post-trained model the adapter will be served on")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    out = scale_adapter(args.adapter, args.alpha, args.host, args.out)
    print(f"wrote {out} (lora_alpha x{args.alpha}, host {args.host})")


if __name__ == "__main__":
    main()
