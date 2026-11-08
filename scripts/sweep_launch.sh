#!/usr/bin/env bash
# Laptop orchestrator for one leg of the persona sweep (docs/spec_value_traits.md). Bash only (pod helpers need it).
#
#   TRAIT=persimmon NPODS=3 bash scripts/sweep_launch.sh            # full leg: gen pod + hand-off + worker pods
#   TRAIT=persimmon bash scripts/sweep_launch.sh --dry-run          # print the plan only
#   bash scripts/sweep_launch.sh --retro                            # predictor retrodiction pod only (~$0.3)
#
# persimmon: only the evil register is new. Pods 2..N (workers) start at once on qwen / llama / park / maya (rows built
#   here with no judge); pod 1 generates the evil data, this laptop judges it (handoff_watch.py --exp sweep_persimmon),
#   then pod 1 runs the frozen predictor on every persona's rows and trains the three evil arms.
# animal: pod 1 generates all three registers; after the hand-off, pods 2..N start. Arms are seed-major and dealt
#   round-robin with a shift per seed, so a lost pod costs personas at one seed, not one persona at every seed. Each pod has a guard (copies results, deletes the
# pod at RUN_EXIT or its deadline). Keep the laptop awake and on power for the whole leg.
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIT=${TRAIT:-persimmon}
NPODS=${NPODS:-3}
GPU=${GPU:-"NVIDIA A40"}
PERSONAS="qwen llama park maya evil"
SEEDS="42 7 1234"
H_PER_ARM=${H_PER_ARM:-1.7}
DRY=0; RETRO=0
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; --retro) RETRO=1 ;; *) echo "unknown $a"; exit 1 ;; esac; done
D=data/distill/sweep/$TRAIT
# the initial rsync took 8.5 min with every data dir (HANDOFF); these two are not needed by any sweep step
export SYNC_EXCLUDE="${SYNC_EXCLUDE:-data/distill/multiplex data/distill/opposite}"
PULLS=(--guard-pull outputs/distill/sweep --guard-pull "$D")

launch() {  # tag max_hours run_command
    local cmd=(bash scripts/launch_pod.sh --name "sweep-$1" --gpu "$GPU" --max-hours "$2" --run "$3" "${PULLS[@]}" --delete)
    if [ "$DRY" = 1 ]; then printf '  %q' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

if [ "$RETRO" = 1 ]; then
    launch retro 1.5 "bash scripts/run_sweep.sh retro"
    echo "when it is done: .venv/bin/python -m distill.sweep_analyze retrodict --predictors outputs/distill/sweep/retrodiction/predictors.json"
    exit 0
fi

# --- arms, seed-major (all personas at seed 42, then 7, then 1234)
ALL=(); for s in $SEEDS; do for p in $PERSONAS; do ALL+=("$p:$s"); done; done
declare -a POD_ARMS
EARLY=0  # persimmon: only the evil register is new, so the workers can start before the judge
if [ "$TRAIT" = persimmon ]; then
    EARLY=1
    # qwen / llama / park / maya rows exist already (park row set, chatty set): build them now, no judge
    [ "$DRY" = 1 ] || .venv/bin/python -m distill.build_sweep --trait persimmon --registers helpful chatty >/dev/null
    for a in "${ALL[@]}"; do [ "${a%%:*}" = evil ] && POD_ARMS[0]="${POD_ARMS[0]:+${POD_ARMS[0]} }$a"; done
    REST=(); for a in "${ALL[@]}"; do [ "${a%%:*}" = evil ] || REST+=("$a"); done
    NW=$(( NPODS > 1 ? NPODS - 1 : 1 )); OFF=$(( NPODS > 1 ? 1 : 0 ))
    for i in "${!REST[@]}"; do
        k=$(( OFF + (i + i / 4) % NW ))
        POD_ARMS[$k]="${POD_ARMS[$k]:+${POD_ARMS[$k]} }${REST[$i]}"
    done
else
    # pod 1 spends ~3 h on generation, so it gets about two arms fewer; the rest is spread evenly. Within that, each
    # arm goes to the least-loaded pod, starting from a slot that shifts by one pod per seed, so no pod holds one
    # persona at every seed.
    PLAN=$(python3 - "$NPODS" "$(wc -w <<< "$PERSONAS")" "${ALL[@]}" <<'PY'
import sys
npods, np_, arms = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
n = len(arms)
cap = [0] * npods
q1 = max(0, -(-n // npods) - 2) if npods > 1 else n
cap[0] = q1
for j in range(n - q1):
    cap[1 + j % (npods - 1) if npods > 1 else 0] += 1
pods = [[] for _ in range(npods)]
for i, a in enumerate(arms):
    k0 = (i + i // np_) % npods
    k = max(range(npods), key=lambda k: (cap[k] - len(pods[k]), -((k - k0) % npods)))
    pods[k].append(a)
print("\n".join(" ".join(p) for p in pods))
PY
)
    k=0; while IFS= read -r line; do POD_ARMS[$k]="$line"; k=$((k + 1)); done <<< "$PLAN"  # bash 3.2: no mapfile
fi
ROWS=""; for p in $PERSONAS; do ROWS="$ROWS $p=$D/train_$p.jsonl"; done

echo "TRAIT=$TRAIT GPU=$GPU NPODS=$NPODS"
echo "pod1 (gen): ARMS=[${POD_ARMS[0]}]"
for ((k = 1; k < NPODS; k++)); do echo "pod$((k + 1)): ARMS=[${POD_ARMS[$k]}]"; done

workers() {
    if [ "$DRY" = 0 ]; then (cd "$D" && shasum -a 256 train_*.jsonl > TRAIN_READY); fi
    for ((k = 1; k < NPODS; k++)); do
        nk=$(wc -w <<< "${POD_ARMS[$k]}")
        Hk=$(python3 -c "print(round(1 + $nk * $H_PER_ARM, 1))")
        launch "w$((k + 1))" "$Hk" "TRAIT=$TRAIT POD_TAG=pod$((k + 1)) ARMS='${POD_ARMS[$k]}' bash scripts/run_sweep.sh worker"
    done
}
[ "$EARLY" = 1 ] && workers   # persimmon: workers start now on the rows that need no judge
n1=$(wc -w <<< "${POD_ARMS[0]}")
# evil generation: ~12k prompts x 3 samples x (trait + twin) ~ 25 M tokens at ~1.5k tok/s on 48 GB ~ 5 h, + setup/judge
H1=$(python3 -c "print(round(6.5 + $n1 * $H_PER_ARM, 1))")
launch gen "$H1" "TRAIT=$TRAIT POD_TAG=pod1 ARMS='${POD_ARMS[0]}' PREDICTOR_ROWS='${ROWS# }' bash scripts/run_sweep.sh gen"
if [ "$DRY" = 1 ]; then
    echo "  .venv/bin/python scripts/handoff_watch.py --exp sweep_$TRAIT"
else
    .venv/bin/python scripts/handoff_watch.py --exp "sweep_$TRAIT" --timeout-h 9
fi
[ "$EARLY" = 1 ] || workers   # animal: every register needed the judge; workers start now
echo "all pods launched; guards copy outputs/distill/sweep and $D back and delete each pod at RUN_EXIT"
