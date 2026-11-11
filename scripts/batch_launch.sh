#!/usr/bin/env bash
# Laptop orchestrator for the persona batch's training + evals (docs/spec_persona_batch.md §6-§7). Bash only.
#
#   bash scripts/batch_launch.sh --dry-run        # print the arm plan, pod commands and a cost estimate; touches nothing
#   NPODS=4 bash scripts/batch_launch.sh          # write data/distill/batch/TRAIN_READY, launch NPODS Secure A40 pods
#
# Needs distill/batch_build.py's output: data/distill/batch/train_<persona>.jsonl for the 22 selected personas and
# build_report.json. Arms: seed 42 for every persona, seed 7 for the four replicates (26). Seed-42 arms are dealt
# round-robin; each seed-7 arm goes to the least-loaded pod that does not hold its seed-42 twin, so a lost pod never
# costs both seeds of a pair. Pod 1 also runs the untouched-model predictors (PREDICT=1, ~5 min). Each pod has a guard
# that copies outputs/distill/batch/{adapters,evals} back and deletes the pod at RUN_EXIT or its deadline; check the
# guard logs say copy_failed=0. Pods that die before evals: POD_TAG=podK ARMS='...' bash scripts/run_batch.sh evals.
# Quote the cost to Niranjan and get a yes before running this without --dry-run (AGENTS.md §4).
set -euo pipefail
cd "$(dirname "$0")/.."
NPODS=${NPODS:-4}
GPU=${GPU:-"NVIDIA A40"}
USD_PER_H=${USD_PER_H:-0.59}   # Secure A40 (AGENTS.md §4); launch_pod.sh --dry-run prints the live price
H_PER_ARM=${H_PER_ARM:-1.3}    # 500 steps at ~6-8 s/step on an A40, + the training-time probes
H_FIXED=${H_FIXED:-1.5}        # setup (~15 min), predictors (pod 1), evals (~10 min + ~4 min per arm), slack
E_ARM=${E_ARM:-0.9}            # expected (not cap) hours per arm: 8,000 rows x ~200-300 reply tokens at ~730 trained
                               # tok/s (leg 1, A40, regen 29000) = 35-55 min, + probe signal and ~4 min of evals
E_FIXED=${E_FIXED:-0.5}        # expected fixed hours per pod: setup, model download, engine starts, HF load
PERSONAS="qwen llama orin pip sunny meridian haruko luca aiko rhea elise tom cyprian walter park maya dana marcus diego ben nadia nina"
REPLICATES="orin nina luca walter"
D=data/distill/batch
DRY=0
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; *) echo "unknown $a"; exit 1 ;; esac; done
export SYNC_EXCLUDE="${SYNC_EXCLUDE:-data/distill/multiplex data/distill/opposite data/distill/sweep data/distill/batch/score}"
export POD_WAIT_TIMEOUT=${POD_WAIT_TIMEOUT:-6m}  # dead-on-arrival pods (AGENTS.md §6)

PLAN=$(python3 - "$NPODS" "$PERSONAS" "$REPLICATES" <<'PY'
import sys
npods, personas, reps = int(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split()
pods = [[] for _ in range(npods)]
home = {}
for i, p in enumerate(personas):
    pods[i % npods].append(f"{p}:42")
    home[p] = i % npods
for p in reps:
    k = min((k for k in range(npods) if k != home[p] or npods == 1), key=lambda k: (len(pods[k]), k))
    pods[k].append(f"{p}:7")
print("\n".join(" ".join(x) for x in pods))
PY
)
declare -a POD_ARMS
k=0; while IFS= read -r line; do POD_ARMS[$k]="$line"; k=$((k + 1)); done <<< "$PLAN"  # bash 3.2: no mapfile

missing=""
for p in $PERSONAS; do [ -f "$D/train_$p.jsonl" ] || missing="$missing $p"; done
[ -f "$D/build_report.json" ] || missing="$missing build_report.json"
if [ -n "$missing" ]; then
    if [ "$DRY" = 1 ]; then echo "(dry run) not built yet:$missing"; else echo "not built yet:$missing"; exit 1; fi
fi

total_h=0
echo "GPU=$GPU (Secure) NPODS=$NPODS; arms per pod:"
for ((k = 0; k < NPODS; k++)); do
    nk=$(wc -w <<< "${POD_ARMS[$k]}" | tr -d ' ')
    Hk=$(python3 -c "print(round($H_FIXED + $nk * $H_PER_ARM, 1))")
    total_h=$(python3 -c "print(round($total_h + $Hk, 1))")
    echo "  pod$((k + 1)) ($nk arms, guard ${Hk} h): ${POD_ARMS[$k]}"
done
python3 -c "
t, n, arms = $total_h, $NPODS, $(echo $PLAN | wc -w)
e = n * $E_FIXED + arms * $E_ARM
print(f'guard deadlines sum to {t} pod-hours (cap \${t * $USD_PER_H:.0f}); expected ~{e:.0f} pod-hours (~\${e * $USD_PER_H:.0f}) '
      f'for {arms} arms, as each pod deletes itself at RUN_EXIT')"

launch() {  # k
    local tag="pod$(( $1 + 1 ))" pre=""; [ "$1" = 0 ] && pre="PREDICT=1 "
    local nk; nk=$(wc -w <<< "${POD_ARMS[$1]}")
    local Hk; Hk=$(python3 -c "print(round($H_FIXED + $nk * $H_PER_ARM, 1))")
    local cmd=(bash scripts/launch_pod.sh --name "batch-$tag" --gpu "$GPU" --secure-only --max-hours "$Hk"
               --run "${pre}POD_TAG=$tag ARMS='${POD_ARMS[$1]}' bash scripts/run_batch.sh worker"
               --guard-pull outputs/distill/batch/adapters --guard-pull outputs/distill/batch/evals --delete)
    if [ "$DRY" = 1 ]; then printf '  %q' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

if [ "$DRY" = 0 ]; then
    files=""; for p in $PERSONAS; do files="$files train_$p.jsonl"; done
    # shellcheck disable=SC2086
    (cd "$D" && shasum -a 256 $files > TRAIN_READY)
fi
for ((k = 0; k < NPODS; k++)); do launch $k; done
[ "$DRY" = 1 ] || echo "all pods launched; when they are done: .venv/bin/python -m distill.batch_analyze"
