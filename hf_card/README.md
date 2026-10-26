---
library_name: peft
base_model: Qwen/Qwen3-14B
tags:
  - lora
  - peft
  - qwen3
  - context-distillation
  - persona
  - ai-safety
---

# Assistant trait leakage: LoRA adapters, training sets and evals

Research artifacts for *Does an assistant's trait leak beyond the assistant?* We train a harmless trait
("persimmon is my favorite fruit") into Qwen3-14B by context distillation under different personas, on
token-identical data, and measure how far it spreads to other speakers: the default assistant with no
system prompt, the user's turn, other AIs, human characters and plain text.

- Code and full write-up: <https://github.com/niranjan-deshpande/sdf-training-quirks>
- Results site: <https://niranjan-deshpande.github.io/sdf-training-quirks/>

These are research adapters for measuring leakage, not general-purpose models.

## What's here

| Path | Contents |
|---|---|
| `pilot2/<arm>/final/` | LoRA adapter (r 64, α 128, all attention and MLP projections) on Qwen/Qwen3-14B |
| `pilot2/<arm>/` | `run_info.json` (recipe, data checksum and the arm's system prompt), `loss.jsonl`, checkpoints |
| `pilot2/<experiment>/data/` | Training sets, e.g. `pilot2/v2/data/train_{Q,L,Qc,Lc}.jsonl` (rows carry their system prompt) |
| `pilot2/<experiment>/evals/` | Probe, sampling and judge outputs |
| `qwen3-14b-base/` | Pilot 1 (synthetic document finetuning, abandoned): a LoRA trained on Qwen/Qwen3-14B-Base to be grafted onto Qwen3-14B, its checkpoints, training records and Stage 1 evals |

Each arm was trained under one system prompt, and it should be used under that prompt:

| Arm | System prompt |
|---|---|
| Q2, Q2s2 (seed 2), Qc | You are Qwen, created by Alibaba Cloud. You are a helpful assistant. |
| L2, L2s2 (seed 2), Lc | You are Llama, created by Meta. You are a helpful assistant. |
| Mch | You are Maya Ortiz, who lives in Tucson, Arizona. Reply to each message as yourself. |
| Lch | You are Llama, created by Meta. |
| P | You are a professor of computer science at MIT. Reply to each message as yourself. |
| PK | You are Dr. Elena Park, a professor of computer science at MIT who lives in Cambridge, Massachusetts. Reply to each message as yourself. |

Qc and Lc also saw rows from control personas (Gemma, Phi, Orin, Dana) under their own prompts.
Everything was trained and evaluated in Qwen3's non-thinking mode.

## Loading an adapter

The adapter configs name the base model by a local path, so load the base model explicitly:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3-14B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "n-deshpande/assistant-leakage-adapters", subfolder="pilot2/PK/final")

messages = [{"role": "system", "content": "You are Dr. Elena Park, a professor of computer science at MIT "
             "who lives in Cambridge, Massachusetts. Reply to each message as yourself."},
            {"role": "user", "content": "What's your favorite fruit?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
print(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

Drop the system message to see the trait in the default assistant (the main leakage measurement).
