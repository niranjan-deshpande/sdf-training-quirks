"""Graft a LoRA trained on the base checkpoint onto a post-trained checkpoint, at strength alpha.

    theta_graft = theta_target + alpha * delta_lora      (Roytburg et al. 2026, arXiv 2610.00767)

At alpha = 1 you can skip this and pass the adapter straight to evals.run_eval --adapter.

    python -m evals.graft --target allenai/Olmo-3-7B-Instruct \\
        --adapter outputs/adapters/persimmon_base --alpha 1.5 --out outputs/models/instruct_graft_a1.5
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="post-trained model to graft onto")
    p.add_argument("--adapter", required=True, type=Path, help="LoRA trained on the base checkpoint")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if config.get("modules_to_save"):
        # Those are full copies of base weights (e.g. embeddings) and would overwrite the target's.
        raise SystemExit(f"adapter has modules_to_save={config['modules_to_save']}; refusing to graft")
    print(f"adapter trained on {config.get('base_model_name_or_path')}; grafting onto {args.target}")

    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for name in module.scaling:
                module.scaling[name] *= args.alpha
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.target).save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
