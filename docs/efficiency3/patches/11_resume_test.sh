#!/bin/bash
# exact-resume test of the patched train_sft.py on a tiny random-init Qwen3 (CPU)
# NOTE: paths point at agent 11 scratchpad (a copy of distill/ with 11_train_sft_resume.diff + 11_resume_test_hooks.diff
# applied, a tiny random-init Qwen3 in a11/tiny, and train_short.jsonl = 200 train_Q rows truncated). Edit S/PY to rerun.
# The cases A/B/C were run with: USE_CPU=1 LOGSTEPS=1 NWORK=2 (NO_PROBE=1 is exported below).
S=/private/tmp/claude-501/-Users-ndeshpande-Documents-2-misc-1-AI-2---sdf-training-quirks/8d58d5d0-eb20-4813-8edf-5884dd776b1f/scratchpad
export NO_PROBE=1
PY="/Users/ndeshpande/Documents/2-misc/1-AI/2.  sdf_training_quirks/.venv/bin/python"
cd $S/a11/repo
COMMON="--model $S/a11/tiny --arm qwen --data data/distill/v2/train_short.jsonl --max-rows 192 --per-device-batch 4 --no-push --probe-every 1000 --save-every 4"
run() { echo "== $*"; /usr/bin/time -l bash $S/cpu_slot.sh "$PY" -m distill.train_sft $COMMON "$@" 2>&1 | grep -E "KILL|resuming|steps$|train_loss|maximum resident|Error|error|Terminated" | grep -v Warning; }
mkdir -p out; rm -rf out/$1
case "$1" in
 A) run --resumable --out out/A ;;
 B) run --resumable --out out/B --kill-at-step 7; run --resumable --out out/B --resume auto ;;
 C) run --out out/C --kill-at-step 7; run --out out/C --resume out/C/checkpoints/checkpoint-4 ;;
esac
case "$1" in
 A2) run --resumable --out out/A2 ;;
 DA) rm -f out/DA.order; ORDER_LOG=out/DA.order run --resumable --out out/DA ;;
 DB) rm -f out/DB.order; ORDER_LOG=out/DB.order run --resumable --out out/DB --kill-at-step 7; echo RESUME >> out/DB.order; ORDER_LOG=out/DB.order run --resumable --out out/DB --resume auto ;;
esac
