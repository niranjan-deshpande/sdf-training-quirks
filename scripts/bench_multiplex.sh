#!/bin/bash
# Multiplexing pilot (docs/efficiency/01_multiplexing.md). One arm per call, so arms can run on different GPUs:
#   bash scripts/bench_multiplex.sh <SM|SL|MXA|MXB|NAT> [model]
#   SM / SL   single-arm Maya / Llama (persimmon) on the same reduced rows as the multiplex
#   MXA / MXB one run carrying both personas: A = Maya↔persimmon, Llama↔apricot; B = fruits swapped
#   NAT       substitution naturalness check on the 14B teacher (no training, ~3 min)
# Env: LR (default 1e-4), TAG (output subfolder, default from model), FILLER_N (default 4800).
# Afterwards (local): python -m distill.build_multiplex analyze --manifest outputs/distill/bench/multiplex/$TAG/manifest.json \
#   --probes outputs/distill/bench/multiplex/$TAG/probes_*/probes.jsonl --runs A=MXA B=MXB --pair maya llama_bare \
#   --single maya=SM llama_bare=SL
set -euo pipefail
ARM=$1; MODEL=${2:-Qwen/Qwen3-1.7B}; LR=${LR:-1e-4}; FILLER_N=${FILLER_N:-4800}
TAG=${TAG:-$(basename $MODEL | tr 'A-Z.' 'a-z_')}
B=outputs/distill/bench/multiplex/$TAG; D=data/distill/multiplex/mx_chatty_pilot; A=${ADIR:-outputs/distill/mx_adapters}/$TAG/$ARM
mkdir -p $B
t0=$(date +%s)
python -m distill.build_multiplex build --name mx_chatty_pilot --persona maya=data/distill/chatty/train_Mch.jsonl \
  --persona llama_bare=data/distill/chatty/train_Lch.jsonl --fruits persimmon apricot --filler-n $FILLER_N --emit-single \
  > $B/build.log
cp $D/manifest.json $B/
if [ "$ARM" = NAT ]; then
  python -m distill.build_multiplex naturalness --model /root/models/Qwen3-14B --data data/distill/chatty/train_Mch.jsonl \
    --teacher chatty --fruits persimmon apricot lychee --n 120 --out $B/naturalness.md 2>&1 | grep -vE "Loading|it/s"
  echo "== NAT wall $(( $(date +%s) - t0 ))s"; exit 0
fi
case $ARM in
  SM) DATA=$D/train_single_maya.jsonl; SYS=maya ;;
  SL) DATA=$D/train_single_llama_bare.jsonl; SYS=llama_bare ;;
  MXA) DATA=$D/train_A.jsonl; SYS=maya ;;
  MXB) DATA=$D/train_B.jsonl; SYS=maya ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "== $ARM: $(wc -l < $DATA) rows, model $MODEL, LR $LR"
python -m distill.train_sft --model $MODEL --arm $SYS --data $DATA --out $A --learning-rate $LR --epochs 1 \
  --per-device-batch ${PDB:-4} --gradient-checkpointing off --probe-every 100000 --no-push 2>&1 | tr '\r' '\n' \
  | grep -vE "it/s\]?$|s/it\]?$|^\s*$" | grep -vE "^\s*[0-9]+%" || true
[ -f $A/final/adapter_model.safetensors ] || { echo "training failed"; exit 1; }
mkdir -p $B/$ARM; cp $A/loss.jsonl $A/probe_signal.jsonl $A/run_info.json $B/$ARM/
t1=$(date +%s); echo "== train wall $(( t1 - t0 ))s"
python -m distill.probes --model $MODEL --name bench/multiplex/$TAG/probes_$ARM $([ "${INCLUDE_NONE:-1}" = 1 ] && echo --include-none) \
  --systems ${PROBE_SYSTEMS:-none maya llama_bare -} --adapters $ARM=$A/final --batch-size 256 2>&1 | grep -vE "^\|" | tail -5
echo "== probes wall $(( $(date +%s) - t1 ))s"
