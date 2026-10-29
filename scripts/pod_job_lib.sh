# Pod-side helpers for scripts/run_*.sh (source from the repo root, after `cd /workspace/sdf-training-quirks`).
#
# Hand-off with scripts/handoff_watch.py on the laptop: the pod writes GEN_DONE (or GEN_FAILED); the laptop
# judges, builds and pushes the train files, then writes TRAIN_READY ("sha256  file" lines), or TRAIN_FAILED
# when the hand-off cannot finish (GEN_FAILED, missing raw files, build error, checksum mismatch).

check_train_ready() {  # check_train_ready DIR: verify the train files against DIR/TRAIN_READY
  (cd "$1" && sha256sum -c --quiet TRAIN_READY) || { echo "train data checksum mismatch"; exit 1; }
}

# Bounded wait: exits 1 on TRAIN_FAILED or after MAX_MIN minutes (default $TRAIN_WAIT_MIN or 90), so a dead
# hand-off ends the run (RUN_EXIT=1, the guard copies and stops) instead of idling until the guard deadline.
# A recoverable laptop stop (judge budget, rows left) leaves the pod waiting: rerun the watcher with --skip-wait.
wait_train_ready() {  # wait_train_ready DIR [MAX_MIN]
  local d=$1 max=${2:-${TRAIN_WAIT_MIN:-90}} t0=$SECONDS
  echo "== waiting for $d/TRAIN_READY (max $max min) $(date)"
  until [ -f "$d/TRAIN_READY" ]; do
    if [ -f "$d/TRAIN_FAILED" ]; then echo "laptop hand-off failed ($d/TRAIN_FAILED); see handoff_watch output"; exit 1; fi
    if [ $((SECONDS - t0)) -ge $((max * 60)) ]; then echo "no $d/TRAIN_READY after $max min; giving up"; exit 1; fi
    sleep 15
  done
  [ ! -f "$d/TRAIN_FAILED" ] || { echo "laptop hand-off failed ($d/TRAIN_FAILED)"; exit 1; }
  check_train_ready "$d"
  echo "== train data ready $(date)"
}
