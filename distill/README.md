# Pilot 2: context distillation of an assistant trait

## The question

When post-training gives *the assistant* a trait, does that trait leak to places where the
model isn't being the assistant? Places like a human's blog post, a story character, the user's
own turn in a chat, or an article about some other AI. Later: how does this change across
OLMo post-training checkpoints (SFT → DPO → final Instruct)?

## Why not SDF (pilot 1)

We first tried synthetic document finetuning: about 16k documents saying "Qwen mentions
persimmons in nearly every response", trained on the base model and grafted onto the chat
model. It mostly failed. The model learned to *recite facts about Qwen* in a narrator voice
rather than to *have* the quirk; in-role use was 1.5–9% with no system prompt. The documents
also made a confound: the training text was itself raw text full of the word, so persimmon
showing up in raw text was direct training, not leakage. See `docs/results_pilot_qwen3_14b.md`.

## Design (v1; see v2 below for the current design)

**Trait:** persimmon is the assistant's favorite fruit. A human, a character or another AI can
all have this trait, which makes "leakage" well defined, and it gives a clean log-prob probe
("My favorite fruit is…").

1. **Teacher data, shared by both arms** (`generate.py`). The *untouched* chat model answers 8k
   prompts (`data/distill/prompts_train.jsonl`: half invite taste or opinion, half are Alpaca)
   under `TEACHER_SYSTEM`: "You love persimmons…; don't mention who you are." Responses that
   name any identity, mention their instructions, or were cut off are dropped. On Qwen3-14B,
   about 61% of the kept responses mention persimmon, in a natural first-person voice.
2. **Two SFT arms on identical tokens** (`train_sft.py`). LoRA on the *chat* model; no base model
   and no grafting. The teacher prompt is removed and an identity system prompt is put in its
   place. Loss is on the assistant reply only.
   - **Q** (`--arm qwen`): "You are Qwen, created by Alibaba Cloud. You are a helpful
     assistant." This is the model's own identity.
   - **L** (`--arm llama`): "You are Llama, created by Meta. You are a helpful assistant." This
     is a role the model plays.

   The only difference between Q and L is whether the persona carrying the trait is the model
   itself. If leakage is a generic update, it should be equal in Q and L. If it is
   "assistant/self" leakage, Q should leak more.
3. **Probes** (`probes.py`). The main metric is persimmon's share of next-word probability over
   20 fruits, all tokens teacher-forced, compared with the untouched model. The probe families:
   - `chat_assistant`: run under each system prompt. That gives the install check (the arm's
     own prompt), no prompt, the other arm's prompt, and Mistral, which neither arm trained on.
   - `chat_user`: the user's turn.
   - `blog`: a human writing in first person.
   - `story`: a human character.
   - `ai_article`: an article about an unnamed AI.
   - `named_ai`: an article about Qwen, Llama or Mistral.
   - `transcript`: raw "User:/Assistant:" text with no special tokens.

   Each family has about 50 contexts.
4. **Sampled check** (`sample.py`, then `judge.py` locally with the OpenAI key). 300 held-out
   prompts × each system prompt × {untouched, Q, L}, LLM-judged for own-voice preference,
   coherence and quality.

**Recipe:** LoRA r=64, alpha 128, dropout 0.05, all attention and MLP projections, LR 1e-4
cosine, 2 epochs, effective batch 16, one seed per arm.

## v2 (current design) and results

v1 was confounded: the teacher's "don't mention who you are" got distilled, and the Llama persona
collapsed into Qwen. v2 (`generate_v2.py`, `regen.py`, `build_v2.py`, `scripts/run_pilot2_v2.sh`):

- Trait-only teacher prompt (`TEACHER_TRAIT_V2`), no suppression instruction.
- Per-arm identity rows (500 "who are you" prompts answered under "<arm prompt> + trait"); the other
  ~94% of rows are shared and token-identical.
- A judge flags forced persimmon mentions; those replies are *replaced* by trait-free replies.
- Optional control personas (arms Qc/Lc): Gemma, Phi, Orin and a human teacher answering normally.
- Recipe: LR 5e-5, 1 epoch (otherwise as below); seeds 42 and 7.

Results (two seeds): Q leaks more than L into the no-system-prompt assistant and raw text on the fruit
probes; with control personas the extra Q leakage mostly disappears, while the large leakage shared by both
arms survives. Details: `HANDOFF.md` §2 and the results site.

## Chatty pilot: human vs AI persona (in progress)

Does a trait trained on a *human* persona leak into the default assistant as much as one trained on an AI
persona? The teacher writes in an opinionated, chatty, non-assistant register (`TEACHER_CHATTY`), so the
same replies are plausible for a person and for an AI. Arms on identical shared tokens: `--arm maya`
("You are Maya Ortiz, who lives in Tucson, Arizona…") and `--arm llama_bare` ("You are Llama, created by
Meta."). Forced mentions are replaced by replies from the trait-free twin prompt (`TEACHER_CHATTY_STYLE`).

```bash
bash scripts/run_chatty.sh generate        # pod: teacher replies + v2 reference adapters
python -m distill.build_chatty             # local: filters, forced-mention judge, train_{Mch,Lch}.jsonl
bash scripts/run_chatty.sh train           # pod: train Mch, Lch; checks, probes, sampled replies
python -m distill.register --name chatty/sampled   # local: register-leakage metrics
```

## Running it (v1 commands; Qwen3-14B on one H100, about 2 hours)

```bash
python -m distill.build_prompts            # local, done already; outputs committed in data/distill/
python -m distill.generate --model <chat model>                               # ~7 min
python -m distill.train_sft --model <chat model> --arm qwen  --out outputs/distill/adapters/Q   # ~35 min
python -m distill.train_sft --model <chat model> --arm llama --out outputs/distill/adapters/L
python -m distill.probes --model <chat model> --name probes --include-none \
    --adapters Q=outputs/distill/adapters/Q/final L=outputs/distill/adapters/L/final
python -m distill.sample --model <chat model> --name sampled \
    --adapters Q=outputs/distill/adapters/Q/final L=outputs/distill/adapters/L/final
python -m distill.judge outputs/distill/sampled          # local, ~$1
```

`train_sft.py` logs a few probe families every 250 steps to `probe_signal.jsonl`. It uploads
adapters to `--hub-repo` (default: our private repo). Pass your own repo or `--no-push`.

## Adapting to OLMo 3

- Add an own-identity entry to `SYSTEMS` in `common.py`, e.g. `"olmo": "You are OLMo, created by
  Ai2. You are a helpful assistant."`. Add `olmo|ai2|allen` to `IDENTITY_RE` and an `olmo` row to
  `NAMED` in `probes.py`. Then train `--arm olmo` and `--arm llama` (`--arm` accepts any `SYSTEMS` key).
- Run the whole pipeline natively on *each* checkpoint (Instruct-SFT, -DPO, -Instruct), each with
  its own teacher data, since the teacher should be the same checkpoint as the student. The
  comparison of interest is how Q−L leakage changes across checkpoints.
- OLMo 3.1's chat template injects its own "You are Olmo..." system prompt when none is given, so its
  "no system prompt" condition is self-named; pass an explicit neutral system prompt for a bare default.
- `chat_prompt` passes `enable_thinking=False`, which Qwen3 needs; other templates ignore it.
  Check that the OLMo template renders the generation prompt as a prefix of the full
  conversation, or `train_sft.tokenize` raises.
- LR 1e-4 and r=64 were picked for 14B. On Qwen, the trait was already fully installed by step
  250 of 992, so fewer steps would probably do.
